import torch
import os
import pickle
import argparse
from datetime import datetime
import sys

from transformers import BertTokenizer, BertForSequenceClassification, BertForTokenClassification

from random import shuffle
from fastsp.utils import slot_descriptions


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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Model params
    MAX_SEQ_LEN = 128
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # Data
    data_folder = args.data_folder
    save_folder = args.save_folder

    margin_train_data = pickle.load(open(os.path.join(data_folder, 'margin_train_data.p'), 'rb'))
    intents = list(margin_train_data.keys())

    held_out_intent = args.held_out_intent
    train_intents = [i for i in intents if i != held_out_intent]

    train_processed = []

    for intent in train_intents:
        for ex in margin_train_data[intent]:

            if args.use_descriptions:
                ent_span = ex[0] + ' : ' + slot_descriptions[intent][ex]
            else:
                ent_span = ex[0]

            if args.model_style == 'base':
                train_processed.append(['[CLS] ' + ent_span + ' [SEP] ' + ex[1][0],
                                        '[CLS] ' + ent_span + ' [SEP] ' + ex[2][0]])
            elif args.model_style == 'context':
                train_processed.append(['[CLS] ' + ent_span + ' [SEP] ' + ex[1][0] + ' [SEP] ' + ex[3],
                                        '[CLS] ' + ent_span + ' [SEP] ' + ex[2][0] + ' [SEP] ' + ex[3]])
            else:
                train_processed.append(['[CLS] ' + ent_span + ' [SEP] ' + ex[3],
                                        ex[1][1], ex[2][1]])

    epochs = args.epochs
    batch_size = args.batch_size
    log_every = args.log_every
    device = "cuda:0"

    if args.model_style in ['base', 'context']:
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1).to(device)
    else:
        sys.exit("Implicit model not implemented yet")
        # model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    margin = 0.7

    print('Begin Training...')
    model.train()

    start_time = datetime.now()

    for epoch in range(epochs):
        shuffle(train_processed)

        update = 0
        total_updates = int(len(train_processed) / batch_size)
        for i in range(0, len(train_processed), batch_size):
            mini_batch = train_processed[i:i+batch_size]

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
                      format(epoch+1, epochs, update, total_updates, loss.item(), datetime.now() - start_time))

    print('Done. Total time taken: {}'.format(datetime.now() - start_time))

    state_dict = {'model_state_dict': model.state_dict()}
    if args.save_descriptions:
        save_path = os.path.join(save_folder, 'bert_wo_{}_{}_desc.pt'.format(held_out_intent, args.model_style))
    else:
        save_path = os.path.join(save_folder, 'bert_wo_{}_{}.pt'.format(held_out_intent, args.model_style))
    torch.save(state_dict, save_path)
    print('Checkpoint saved to {}'.format(save_path))







