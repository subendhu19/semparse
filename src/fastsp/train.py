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

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--log_every', type=int, default=10)

    parser.add_argument('--model_style', type=str, choices=['base', 'context', 'implicit'], default='base')
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

    train_entity_data = pickle.load(open(os.path.join(data_folder, 'train_entity_data.p'), 'rb'))
    # margin_train_data = pickle.load(open(os.path.join(data_folder, 'margin_train_data.p'), 'rb'))
    intents = list(train_entity_data.keys())

    held_out_intent = args.held_out_intent
    train_intents = [i for i in intents if i != held_out_intent]

    epochs = args.epochs
    batch_size = args.batch_size
    log_every = args.log_every
    device = "cuda:0"

    if args.model_style in ['base', 'context']:
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1).to(device)
    else:
        model = ImplicitScorer().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    margin = 0.7

    print('Begin Training...', flush=True)
    model.train()

    start_time = datetime.now()

    for epoch in range(epochs):

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

    print('Done. Total time taken: {}'.format(datetime.now() - start_time), flush=True)

    state_dict = {'model_state_dict': model.state_dict()}
    if args.use_descriptions:
        save_path = os.path.join(save_folder, 'bert_wo_{}_{}_desc.pt'.format(held_out_intent, args.model_style))
    else:
        save_path = os.path.join(save_folder, 'bert_wo_{}_{}.pt'.format(held_out_intent, args.model_style))
    torch.save(state_dict, save_path)
    print('Checkpoint saved to {}'.format(save_path), flush=True)







