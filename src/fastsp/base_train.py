import torch
import os
import pickle
import argparse
from datetime import datetime

from transformers import BertTokenizerFast, BertModel

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
    def __init__(self):
        super(BaseScorer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bias = torch.nn.ParameterDict({i: torch.nn.Parameter(torch.rand(1, len(tag_entity_name_dict[i])))
                                            for i in tag_entity_name_dict})

    def forward(self, inputs, c_intent):
        outs = self.bert(**inputs)

        slot_tensors = tokenizer(tag_entity_name_dict[c_intent], return_tensors="pt", padding=True,
                                 add_special_tokens=True).to(device=device)

        slot_outs = self.bert(**slot_tensors)

        token_level_outputs = outs['last_hidden_state']
        slot_vectors = slot_outs['last_hidden_state'][:, 0, :]

        ret = torch.matmul(token_level_outputs, slot_vectors.T) + self.bias[c_intent]

        return ret


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training models for fast semantic parsing")

    parser.add_argument('--data_folder', type=str, default='/home/srongali/data/snips')
    parser.add_argument('--save_folder', type=str, default='/mnt/nfs/scratch1/srongali/semparse/snips')

    parser.add_argument('--held_out_intent', type=str, required=True)

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--log_every', type=int, default=10)

    parser.add_argument('--use_descriptions', action='store_true')

    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Model params
    MAX_SEQ_LEN = 128
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # Data
    data_folder = args.data_folder
    save_folder = args.save_folder

    train_data = pickle.load(open(os.path.join(data_folder, 'base_train_data.p'), 'rb'))
    intents = list(train_data.keys())

    held_out_intent = args.held_out_intent
    train_intents = [i for i in intents if i != held_out_intent]

    epochs = args.epochs
    batch_size = args.batch_size
    log_every = args.log_every
    device = "cuda:0"

    model = BaseScorer().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    margin = 0.7

    CE = torch.nn.CrossEntropyLoss()

    print('Begin Training...', flush=True)
    model.train()

    start_time = datetime.now()

    for epoch in range(epochs):

        train_processed = []
        for intent in train_intents:
            all_examples = []
            for i in range(len(train_data[intent]['utterances'])):

                utt = train_data[intent]['utterances'][i]
                utt_tok = tokenizer(utt, return_tensors="pt", add_special_tokens=False)
                utt_ids = utt_tok.word_ids()

                ti = train_data[intent]['tag_indices'][i]

                nti = [ti[a] for a in utt_ids]

                all_examples.append([utt, nti])

            shuffle(all_examples)

            for i in range(0, len(all_examples), batch_size):
                mini_batch = all_examples[i:i + batch_size]
                train_processed.append((mini_batch, intent))

        shuffle(train_processed)

        update = 0
        total_updates = len(train_processed)

        for i in range(0, len(train_processed)):
            mini_batch, intent = train_processed[i]

            sents = [a[0] for a in mini_batch]
            sent_tensors = tokenizer(sents, return_tensors="pt", padding=True,
                                     add_special_tokens=False).to(device=device)

            scores = model(sent_tensors, intent)

            tags = [a[1] for a in mini_batch]
            pad = len(max(tags, key=len))
            tags = torch.tensor([i + [-100]*(pad-len(i)) for i in tags]).to(device=device)

            loss = CE(scores, tags)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update += 1

            if update % log_every == 0:
                print("Epoch: {}/{} \t Update: {}/{} \t Loss: {} \t Time elapsed: {}".
                      format(epoch+1, epochs, update, total_updates, loss.item(), datetime.now() - start_time),
                      flush=True)

    print('Done. Total time taken: {}'.format(datetime.now() - start_time), flush=True)

    state_dict = {'model_state_dict': model.state_dict()}
    if args.use_descriptions:
        save_path = os.path.join(save_folder, 'base_wo_{}_desc.pt'.format(held_out_intent))
    else:
        save_path = os.path.join(save_folder, 'base_wo_{}.pt'.format(held_out_intent))
    torch.save(state_dict, save_path)
    print('Checkpoint saved to {}'.format(save_path), flush=True)







