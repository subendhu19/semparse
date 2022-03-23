import torch
import os
import pickle
import argparse
from datetime import datetime

from transformers import AutoTokenizer, AutoModel

import random
from random import shuffle
from src.fastsp.utils import slot_descriptions


random.seed(1100)


tag_entity_name_dict = {
    "PlayMusic": ["genre", "year", "sort", "service", "music item", "playlist", "album", "artist", "track", "none"],
    "RateBook": ["object name", "rating unit", "best rating", "rating value", "object type", "object select",
                 "object part of series type", "none"],
    "SearchCreativeWork": ["object name", "object type", "none"],
    "GetWeather": ["state", "spatial relation", "condition description", "country", "timeRange", "city",
                   "condition temperature", "current location", "geographic poi", "none"],
    "BookRestaurant": ["state", "spatial relation", "party size number", "sort", "country", "timeRange",
                       "restaurant type", "served dish", "restaurant name", "city", "cuisine", "poi", "facility",
                       "party size description", "none"],
    "SearchScreeningEvent": ["spatial relation", "object type", "timeRange", "movie name", "movie type",
                             "location name", "object location type", "none"],
    "AddToPlaylist": ["music item", "entity name", "playlist", "artist", "playlist owner", "none"]
}


class BaseScorer(torch.nn.Module):
    def __init__(self, ckpt, model_style, slot_vecs=None):
        super(BaseScorer, self).__init__()
        self.bert = AutoModel.from_pretrained(ckpt)
        # self.bias = torch.nn.ParameterDict({i: torch.nn.Parameter(torch.rand(1, len(tag_entity_name_dict[i])))
        #                                     for i in tag_entity_name_dict})
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model_style = model_style
        if self.model_style == 'ff':
            self.ff = torch.nn.Linear(2 * self.bert.config.hidden_size, 1)
        elif self.model_style == 'wdot':
            self.wdot = torch.nn.Bilinear(self.bert.config.hidden_size, self.bert.config.hidden_size, 1)
        self.slot_vecs = slot_vecs

    def forward(self, inputs, c_intent, use_descriptions=False):
        outs = self.bert(**inputs)

        if self.slot_vecs is None:
            slot_list = [s for s in tag_entity_name_dict[c_intent]]
            if use_descriptions:
                for i in range(len(slot_list)):
                    if slot_list[i] != "none":
                        slot_list[i] += ' : ' + slot_descriptions[c_intent][slot_list[i]]

            slot_tensors = self.tokenizer(slot_list, return_tensors="pt", padding=True,
                                          add_special_tokens=True).to(device=self.bert.device)

            slot_outs = self.bert(**slot_tensors)
            slot_vectors = slot_outs['last_hidden_state'][:, 0, :]
        else:
            if use_descriptions:
                slot_vectors = self.slot_vecs[c_intent]['desc'].to(self.bert.device)
            else:
                slot_vectors = self.slot_vecs[c_intent]['no_desc'].to(self.bert.device)

        token_level_outputs = outs['last_hidden_state']

        # mags = torch.clamp(torch.einsum('bp,r->bpr', torch.norm(token_level_outputs, dim=2),
        #                                 torch.norm(slot_vectors, dim=1)),
        #                    min=1e-08)

        if self.model_style == 'dot':
            ret = torch.matmul(token_level_outputs, slot_vectors.T)  # / mags  # + self.bias[c_intent]
        elif self.model_style == 'wdot':
            tok_mod = token_level_outputs.unsqueeze(2).repeat(1, 1, slot_vectors.shape[0], 1)
            slot_mod = slot_vectors.unsqueeze(0).unsqueeze(1).repeat(tok_mod.shape[0], tok_mod.shape[1], 1, 1)
            ret = self.wdot(tok_mod, slot_mod).squeeze()
        else:
            tok_mod = token_level_outputs.unsqueeze(2).repeat(1, 1, slot_vectors.shape[0], 1)
            slot_mod = slot_vectors.unsqueeze(0).unsqueeze(1).repeat(tok_mod.shape[0], tok_mod.shape[1], 1, 1)
            ret = self.ff(torch.cat([tok_mod, slot_mod], dim=3)).squeeze()

        return ret


def process_data_with_tags(split, intent_list, tokenizer, batch_size):
    processed = []
    for intent in intent_list:
        all_examples = []
        for i in range(len(split[intent]['utterances'])):
            utt = split[intent]['utterances'][i]
            tags = split[intent]['tag_indices'][i]
            tag_offsets = torch.tensor(split[intent]['tag_offsets'][i])
            t_utt = tokenizer(utt, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
            new_offsets = t_utt['offset_mapping'][0]
            new_tags = []
            for k in range(new_offsets.shape[0]):
                idx = 0
                while new_offsets[k][0] >= tag_offsets[idx][0]:
                    idx += 1
                    if idx == len(tag_offsets):
                        break
                new_tags.append(tags[idx - 1])

            all_examples.append([utt, new_tags])

        for i in range(0, len(all_examples), batch_size):
            mini_batch = all_examples[i:i + batch_size]
            processed.append((mini_batch, intent))

    return processed


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training models for fast semantic parsing")

    parser.add_argument('--data_folder', type=str, default='/home/srongali/data/snips')
    parser.add_argument('--save_folder', type=str, default='/mnt/nfs/scratch1/srongali/semparse/snips')

    parser.add_argument('--held_out_intent', type=str, required=True)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--log_every', type=int, default=10)

    parser.add_argument('--use_descriptions', action='store_true')

    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--validate_every', type=int, default=1)
    parser.add_argument('--model_style', type=str, choices=['dot', 'ff', 'wdot'], default='dot')

    parser.add_argument('--precompute_slotvecs', action='store_true')
    parser.add_argument('--model_checkpoint', type=str, default='bert-base-uncased')

    args = parser.parse_args()

    model_checkpoint = args.model_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Model params
    MAX_SEQ_LEN = 128
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # Data
    data_folder = args.data_folder
    save_folder = args.save_folder

    train_data = pickle.load(open(os.path.join(data_folder, 'base_train_data.p'), 'rb'))
    val_data = pickle.load(open(os.path.join(data_folder, 'base_val_data.p'), 'rb'))
    intents = list(train_data.keys())

    held_out_intent = args.held_out_intent
    train_intents = [i for i in intents if i != held_out_intent]

    epochs = args.epochs
    batch_size = args.batch_size
    log_every = args.log_every
    device = "cuda:0"

    slot_vectors = None
    if args.precompute_slotvecs:
        slot_vectors = pickle.load(open(os.path.join(args.data_folder, 'slot_vecs.p'), 'rb'))

    model = BaseScorer(ckpt=model_checkpoint,
                       model_style=args.model_style,
                       slot_vecs=slot_vectors).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    margin = 0.7

    CE = torch.nn.CrossEntropyLoss()

    start_time = datetime.now()

    val_processed_1 = process_data_with_tags(val_data, train_intents, tokenizer, batch_size)

    val_processed_2 = process_data_with_tags(val_data, [held_out_intent], tokenizer, batch_size)

    train_processed = process_data_with_tags(train_data, train_intents, tokenizer, batch_size)

    # Training metrics
    ind_accuracies = [0]
    ood_accuracies = [0]
    patience_count = 0

    model_name = args.model_style + '_pc' if args.precompute_slotvecs else args.model_style
    model_name = model_name + '_{}'.format(model_checkpoint) if model_checkpoint != 'bert-base-uncased' else model_name

    print('Begin Training...', flush=True)
    model.train()

    for epoch in range(epochs):
        shuffle(train_processed)

        update = 0
        total_updates = len(train_processed)

        # Training
        model.train()
        for i in range(0, len(train_processed)):
            mini_batch, intent = train_processed[i]

            sents = [a[0] for a in mini_batch]
            sent_tensors = tokenizer(sents, return_tensors="pt", padding=True,
                                     add_special_tokens=False).to(device=device)

            scores = model(sent_tensors, intent, args.use_descriptions)

            tags = [a[1] for a in mini_batch]
            pad = len(max(tags, key=len))
            tags = torch.tensor([i + [-100]*(pad-len(i)) for i in tags]).to(device=device)

            scores = scores.reshape(-1, scores.shape[2])
            tags = tags.reshape(-1)
            loss = CE(scores, tags)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update += 1

            if update % log_every == 0:
                print("Epoch: {}/{} \t Update: {}/{} \t Loss: {} \t Time elapsed: {}".
                      format(epoch+1, epochs, update, total_updates, loss.item(), datetime.now() - start_time),
                      flush=True)

        # Validation
        if (epoch + 1) % args.validate_every == 0:
            print("End of epoch {}. Running validation...".format(epoch+1), flush=True)
            model.eval()

            correct = 0
            total = 0

            for i in range(0, len(val_processed_1)):
                mini_batch, intent = val_processed_1[i]

                sents = [a[0] for a in mini_batch]
                sent_tensors = tokenizer(sents, return_tensors="pt", padding=True,
                                         add_special_tokens=False).to(device=device)

                with torch.no_grad():
                    scores = model(sent_tensors, intent, args.use_descriptions)

                tags = [a[1] for a in mini_batch]
                pad = len(max(tags, key=len))
                tags = torch.tensor([i + [-100]*(pad-len(i)) for i in tags]).to(device=device)

                scores = scores.reshape(-1, scores.shape[2])
                preds = torch.argmax(scores, dim=1)
                tags = tags.reshape(-1)

                mask_1 = tags >= 0
                mask_2 = tags != (len(tag_entity_name_dict[intent]) - 1)
                mask = mask_1 * mask_2
                total += torch.sum(mask).item()
                correct += torch.sum(mask * (preds == tags)).item()

            acc = correct * 100.0 / total
            print('Same domain tagging accuracy: {:.2f}'.format(acc), flush=True)
            if acc > max(ind_accuracies):
                print('BEST SO FAR! Saving model...')
                state_dict = {'model_state_dict': model.state_dict()}
                if args.use_descriptions:
                    save_path = os.path.join(save_folder, 'base_{}_wo_{}_desc_best.pt'.format(model_name,
                                                                                              held_out_intent))
                else:
                    save_path = os.path.join(save_folder, 'base_{}_wo_{}_best.pt'.format(model_name,
                                                                                         held_out_intent))
                torch.save(state_dict, save_path)
                print('Best checkpoint saved to {}'.format(save_path), flush=True)
                patience_count = 0
            else:
                patience_count += 1

            ind_accuracies.append(acc)

            correct = 0
            total = 0

            for i in range(0, len(val_processed_2)):
                mini_batch, intent = val_processed_2[i]

                sents = [a[0] for a in mini_batch]
                sent_tensors = tokenizer(sents, return_tensors="pt", padding=True,
                                         add_special_tokens=False).to(device=device)

                with torch.no_grad():
                    scores = model(sent_tensors, intent, args.use_descriptions)

                tags = [a[1] for a in mini_batch]
                pad = len(max(tags, key=len))
                tags = torch.tensor([i + [-100]*(pad-len(i)) for i in tags]).to(device=device)

                scores = scores.reshape(-1, scores.shape[2])
                preds = torch.argmax(scores, dim=1)
                tags = tags.reshape(-1)

                mask_1 = tags >= 0
                mask_2 = tags != (len(tag_entity_name_dict[intent]) - 1)
                mask = mask_1 * mask_2
                total += torch.sum(mask).item()
                correct += torch.sum(mask * (preds == tags)).item()

            acc = correct * 100.0 / total
            print('Out of domain tagging accuracy: {:.2f}'.format(acc), flush=True)
            ood_accuracies.append(acc)

            if patience_count > args.patience:
                print('Ran out of patience. Exiting training.', flush=True)
                break

    print('Done. Total time taken: {}'.format(datetime.now() - start_time), flush=True)

    state_dict = {'model_state_dict': model.state_dict()}
    if args.use_descriptions:
        save_path = os.path.join(save_folder, 'base_{}_wo_{}_desc_latest.pt'.format(model_name, held_out_intent))
    else:
        save_path = os.path.join(save_folder, 'base_{}_wo_{}_latest.pt'.format(model_name, held_out_intent))
    torch.save(state_dict, save_path)
    print('Latest checkpoint saved to {}'.format(save_path), flush=True)







