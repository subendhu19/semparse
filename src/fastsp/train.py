import torch
import os
import pickle
import argparse
from datetime import datetime

from transformers import BertTokenizerFast, BertForSequenceClassification, BertModel

import random
from random import shuffle, sample
from src.fastsp.utils import slot_descriptions

random.seed(1100)


def find_all_spans(words, threshold):
    all_spans = []
    all_sids = []
    for i in range(1, threshold):
        for j in range(0, len(words) - i + 1):
            all_spans.append(' '.join(words[j:j+i]))
            all_sids.append((j, j+i))
    return all_spans, all_sids


entity_name_dict = {
    "PlayMusic": ["genre", "year", "sort", "service", "music item", "playlist", "album", "artist", "track"],
    "RateBook": ["object name", "rating unit", "best rating", "rating value", "object type", "object select",
                 "object part of series type"],
    "SearchCreativeWork": ["object name", "object type"],
    "GetWeather": ["state", "spatial relation", "condition description", "country", "timeRange", "city",
                   "condition temperature", "current location", "geographic poi"],
    "BookRestaurant": ["state", "spatial relation", "party size number", "sort", "country", "timeRange",
                       "restaurant type", "served dish", "restaurant name", "city", "cuisine", "poi", "facility",
                       "party size description"],
    "SearchScreeningEvent": ["spatial relation", "object type", "timeRange", "movie name", "movie type",
                             "location name", "object location type"],
    "AddToPlaylist": ["music item", "entity name", "playlist", "artist", "playlist owner"]
}


def get_indices(span, ips, sid):
    word_ids = ips.word_ids()
    n_span = [word_ids.index(span[0]) + sid, word_ids.index(span[1]-1) + sid]
    return n_span


class ImplicitScorer(torch.nn.Module):
    def __init__(self):
        super(ImplicitScorer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
        self.scorer = torch.nn.Linear(2 * self.bert.config.hidden_size, 1)

    def forward(self, inputs, spans):
        outs = self.bert(**inputs)
        token_level_outputs = outs['last_hidden_state']

        span_vecs = token_level_outputs[torch.arange(token_level_outputs.shape[0]).unsqueeze(-1), spans]
        score_vecs = self.dropout(torch.cat([span_vecs[:, 0, :], span_vecs[:, 1, :]], dim=1))

        scores = self.scorer(score_vecs)

        return scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training models for fast semantic parsing")

    parser.add_argument('--data_folder', type=str, default='/home/srongali/data/snips')
    parser.add_argument('--save_folder', type=str, default='/mnt/nfs/scratch1/srongali/semparse/snips')

    parser.add_argument('--held_out_intent', type=str, required=True)

    parser.add_argument('--span_threshold', type=int, default=6)

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--log_every', type=int, default=10)

    parser.add_argument('--model_style', type=str, choices=['base', 'context', 'implicit'], default='base')
    parser.add_argument('--use_descriptions', action='store_true')

    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--validate_every_update', type=int, default=500)

    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Model params
    MAX_SEQ_LEN = 128
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # Data
    data_folder = args.data_folder
    save_folder = args.save_folder

    train_entity_data = pickle.load(open(os.path.join(data_folder, 'train_entity_data.p'), 'rb'))
    # margin_train_data = pickle.load(open(os.path.join(data_folder, 'margin_train_data.p'), 'rb'))
    intents = list(train_entity_data.keys())

    val_data = pickle.load(open(os.path.join(data_folder, 'val_data.p'), 'rb'))

    held_out_intent = args.held_out_intent
    train_intents = [i for i in intents if i != held_out_intent]

    epochs = args.epochs
    batch_size = args.batch_size
    log_every = args.log_every
    validate_every_update = args.validate_every_update
    device = "cuda:0"
    span_threshold = args.span_threshold

    if args.model_style in ['base', 'context']:
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1).to(device)
    else:
        model = ImplicitScorer().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    margin = 0.7

    print('Begin Training...', flush=True)
    model.train()

    top1_scores = [0]
    patience_count = 0

    start_time = datetime.now()

    for epoch in range(epochs):

        exit_training = False

        margin_train_data = {i: [] for i in intents}
        for intent in intents:
            for et in train_entity_data[intent]:
                utt = et["utterance"]
                name = et["name"].replace('_', " ")
                pos = et["positive"]
                min_ns = min(len(et["negative_non_overlap_spans"]), 5)
                min_os = min(len(et["negative_overlap_spans"]), 5)
                negs = (et["negative_other_entites"] + sample(et["negative_overlap_spans"], min_os)
                        + sample(et["negative_non_overlap_spans"], min_ns))
                for neg in negs:
                    margin_train_data[intent].append((name, pos, neg, utt))
            shuffle(margin_train_data[intent])

        train_processed = []
        for intent in train_intents:
            for ex in margin_train_data[intent]:

                if args.use_descriptions:
                    ent_span = ex[0] + ' : ' + slot_descriptions[intent][ex[0]]
                else:
                    ent_span = ex[0]

                if args.model_style == 'base':
                    train_processed.append(['[CLS] ' + ent_span + ' [SEP] ' + ex[1][0],
                                            '[CLS] ' + ent_span + ' [SEP] ' + ex[2][0]])
                elif args.model_style == 'context':
                    train_processed.append(['[CLS] ' + ent_span + ' [SEP] ' + ex[1][0] + ' [SEP] ' + ex[3],
                                            '[CLS] ' + ent_span + ' [SEP] ' + ex[2][0] + ' [SEP] ' + ex[3]])
                else:
                    start_id = len(tokenizer.tokenize(ent_span)) + 2
                    inp = tokenizer(ex[3], return_tensors="pt", add_special_tokens=False)
                    train_processed.append(['[CLS] ' + ent_span + ' [SEP] ' + ex[3],
                                            get_indices(ex[1][1], inp, start_id),
                                            get_indices(ex[2][1], inp, start_id)])
        shuffle(train_processed)

        update = 0
        total_updates = int(len(train_processed) / batch_size)
        for i in range(0, len(train_processed), batch_size):
            mini_batch = train_processed[i:i+batch_size]

            if args.model_style == 'implicit':
                sents = [a[0] for a in mini_batch]
                sent_tensors = tokenizer(sents, return_tensors="pt", padding=True,
                                         add_special_tokens=False).to(device=device)
                pos_spans = torch.tensor([a[1] for a in mini_batch]).to(device=device)
                neg_spans = torch.tensor([a[2] for a in mini_batch]).to(device=device)

                pos_scores = torch.sigmoid(model(sent_tensors, pos_spans))
                neg_scores = torch.sigmoid(model(sent_tensors, neg_spans))

            else:
                pos_ex = [a[0] for a in mini_batch]
                neg_ex = [a[1] for a in mini_batch]

                pos_tensors = tokenizer(pos_ex, return_tensors="pt", padding=True,
                                        add_special_tokens=False).to(device=device)
                neg_tensors = tokenizer(neg_ex, return_tensors="pt", padding=True,
                                        add_special_tokens=False).to(device=device)

                pos_outputs = model(**pos_tensors)
                neg_outputs = model(**neg_tensors)

                pos_scores = torch.sigmoid(pos_outputs.logits)
                neg_scores = torch.sigmoid(neg_outputs.logits)

            margin_scores = torch.max(torch.zeros_like(pos_scores), margin - pos_scores + neg_scores)

            loss = margin_scores.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update += 1

            if update % log_every == 0:
                print("Epoch: {}/{} \t Update: {}/{} \t Loss: {} \t Time elapsed: {}".
                      format(epoch+1, epochs, update, total_updates, loss.item(), datetime.now() - start_time),
                      flush=True)

            if update % validate_every_update == 0:
                print("Reached update {}. Running validation...".format(update), flush=True)
                model.eval()

                pred_ranks = []
                # In domain validation
                for eval_intent in train_intents:
                    entity_names = entity_name_dict[eval_intent]

                    for k in range(len(val_data[eval_intent]['utterances'])):
                        utt = val_data[eval_intent]['utterances'][k]
                        ets = val_data[eval_intent]['entities'][k]

                        spans, span_ids = find_all_spans(utt.split(), span_threshold)

                        for ent in entity_names:
                            if args.use_descriptions:
                                ent_span = ent + ' : ' + slot_descriptions[eval_intent][ent]
                            else:
                                ent_span = ent

                            if args.model_style == 'context':
                                inputs = ['[CLS] ' + ent_span + ' [SEP] ' + s + ' [SEP] ' + utt for s in spans]
                            elif args.model_style == 'base':
                                inputs = ['[CLS] ' + ent_span + ' [SEP] ' + s for s in spans]
                            else:
                                start_id = len(tokenizer.tokenize(ent_span)) + 2
                                inp = tokenizer(utt, return_tensors="pt", add_special_tokens=False)
                                inputs = [['[CLS] ' + ent_span + ' [SEP] ' + utt,
                                           get_indices(spid, inp, start_id)] for spid in span_ids]

                            if args.model_style == 'implicit':
                                with torch.no_grad():
                                    sents = [inputs[0][0]]
                                    sent_tensors = tokenizer(sents, return_tensors="pt", padding=True,
                                                             add_special_tokens=False).to(device=device)
                                    pos_spans = torch.tensor([a[1] for a in inputs]).to(device=device)

                                    scores = torch.sigmoid(model(sent_tensors, pos_spans))
                            else:
                                with torch.no_grad():
                                    input_tensor = tokenizer(inputs, return_tensors="pt", padding=True,
                                                             add_special_tokens=False).to(device=device)
                                    scores = torch.sigmoid(model(**input_tensor).logits)

                            spans_w_scores = list(zip(spans, list(scores.squeeze()), span_ids))
                            spans_w_scores.sort(key=lambda x: x[1], reverse=True)

                            for a in ets:
                                if a[1].replace('_', ' ') == ent:
                                    index = -1
                                    sspans = [b[0] for b in spans_w_scores]
                                    if a[0] in sspans:
                                        index = sspans.index(a[0])
                                    pred_ranks.append(index)

                correct = len([r for r in pred_ranks if r == 0])
                total = len(pred_ranks)
                acc = correct * 100.0 / total

                print('Same domain Top1 Accuracy: {:.2f}'.format(acc), flush=True)

                if acc > max(top1_scores):
                    print('BEST SO FAR! Saving model...', flush=True)
                    state_dict = {'model_state_dict': model.state_dict()}
                    if args.use_descriptions:
                        save_path = os.path.join(save_folder, 'joint_{}_wo_{}_desc_best.pt'.format(args.model_style,
                                                                                                   held_out_intent))
                    else:
                        save_path = os.path.join(save_folder, 'joint_{}_wo_{}_best.pt'.format(args.model_style,
                                                                                              held_out_intent))
                    torch.save(state_dict, save_path)
                    print('Best checkpoint saved to {}'.format(save_path), flush=True)
                    patience_count = 0
                else:
                    patience_count += 1

                top1_scores.append(acc)

                # OOD validation
                eval_intent = held_out_intent
                entity_names = entity_name_dict[eval_intent]

                pred_ranks = []
                for k in range(len(val_data[eval_intent]['utterances'])):
                    utt = val_data[eval_intent]['utterances'][k]
                    ets = val_data[eval_intent]['entities'][k]

                    spans, span_ids = find_all_spans(utt.split(), span_threshold)

                    for ent in entity_names:
                        if args.use_descriptions:
                            ent_span = ent + ' : ' + slot_descriptions[eval_intent][ent]
                        else:
                            ent_span = ent

                        if args.model_style == 'context':
                            inputs = ['[CLS] ' + ent_span + ' [SEP] ' + s + ' [SEP] ' + utt for s in spans]
                        elif args.model_style == 'base':
                            inputs = ['[CLS] ' + ent_span + ' [SEP] ' + s for s in spans]
                        else:
                            start_id = len(tokenizer.tokenize(ent_span)) + 2
                            inp = tokenizer(utt, return_tensors="pt", add_special_tokens=False)
                            inputs = [['[CLS] ' + ent_span + ' [SEP] ' + utt,
                                       get_indices(spid, inp, start_id)] for spid in span_ids]

                        if args.model_style == 'implicit':
                            with torch.no_grad():
                                sents = [inputs[0][0]]
                                sent_tensors = tokenizer(sents, return_tensors="pt", padding=True,
                                                         add_special_tokens=False).to(device=device)
                                pos_spans = torch.tensor([a[1] for a in inputs]).to(device=device)

                                scores = torch.sigmoid(model(sent_tensors, pos_spans))
                        else:
                            with torch.no_grad():
                                input_tensor = tokenizer(inputs, return_tensors="pt", padding=True,
                                                         add_special_tokens=False).to(device=device)
                                scores = torch.sigmoid(model(**input_tensor).logits)

                        spans_w_scores = list(zip(spans, list(scores.squeeze()), span_ids))
                        spans_w_scores.sort(key=lambda x: x[1], reverse=True)

                        for a in ets:
                            if a[1].replace('_', ' ') == ent:
                                index = -1
                                sspans = [b[0] for b in spans_w_scores]
                                if a[0] in sspans:
                                    index = sspans.index(a[0])
                                pred_ranks.append(index)

                correct = len([r for r in pred_ranks if r == 0])
                total = len(pred_ranks)

                print('Out of domain Top1 Accuracy: {:.2f}'.format(correct * 100.0 / total), flush=True)

                if patience_count > args.patience:
                    print('Ran out of patience. Exiting training.', flush=True)
                    exit_training = True
                    break

        if exit_training:
            break

    print('Done. Total time taken: {}'.format(datetime.now() - start_time), flush=True)

    state_dict = {'model_state_dict': model.state_dict()}
    if args.use_descriptions:
        save_path = os.path.join(save_folder, 'joint_{}_wo_{}_desc_latest.pt'.format(args.model_style,
                                                                                     held_out_intent))
    else:
        save_path = os.path.join(save_folder, 'joint_{}_wo_{}_latest.pt'.format(args.model_style,
                                                                                held_out_intent))
    torch.save(state_dict, save_path)
    print('Latest checkpoint saved to {}'.format(save_path), flush=True)







