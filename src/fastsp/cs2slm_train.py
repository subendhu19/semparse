import torch
import os
import pickle
import argparse
from datetime import datetime
from torch import nn

from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.nn import TransformerDecoder, TransformerDecoderLayer

import random
from random import shuffle
import math
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

random.seed(1100)
target_vocab = ['<PAD>', '<MASK>'] + ['@ptr_{}'.format(i) for i in range(64)]


def process_s2s_data(path, domain_list, split, bsize, tokenizer, schema):
    processed = []
    for domain in domain_list:
        all_examples = []
        with open(os.path.join(path, '{}_{}.tsv'.format(domain, split))) as inf:
            for line in inf:
                fields = line.strip().split('\t')
                target = fields[2].split()
                target_indices = [target_vocab.index(a) if a in target_vocab else
                                  66 + (schema[domain]['intents'] + schema[domain]['slots']).index(a)
                                  for a in target]
                all_examples.append((fields[0], target_indices))

        for i in range(0, len(all_examples), bsize):
            mini_batch = all_examples[i: i + bsize]

            sents = [a[0] for a in mini_batch]
            sent_tensors = tokenizer(sents, return_tensors="pt", padding=True,
                                     add_special_tokens=False)

            target = [a[1] for a in mini_batch]
            pad = len(max(target, key=len))
            target = torch.tensor([i + [0] * (pad - len(i)) for i in target])

            processed.append((sent_tensors, target, domain))

    return processed


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_slot_expression(token):
    if '[' in token:
        if token[:3] == 'IN:':
            return ' '.join(['begin'] + token[4:].lower().split('_') + ['intent'])
        else:
            return ' '.join(['begin'] + token[4:].lower().split('_') + ['slot'])
    else:
        if token[:3] == 'IN:':
            return ' '.join(['end'] + token[3:-1].lower().split('_') + ['intent'])
        else:
            return ' '.join(['end'] + token[3:-1].lower().split('_') + ['slot'])


class CustomSeq2Seq(nn.Module):
    def __init__(self, enc, dec, schema, max_target_len=60, tag_model=None):
        super(CustomSeq2Seq, self).__init__()
        self.dropout = 0.1
        self.d_model = enc.config.hidden_size
        self.device = enc.device

        self.encoder = enc
        self.decoder = dec

        self.tag_model = tag_model

        if tag_model:
            self.tag_encoder = AutoModel.from_pretrained(tag_model).to(self.device)
            self.tag_tokenizer = AutoTokenizer.from_pretrained(tag_model)
        else:
            self.tag_encoder = enc
            self.tag_tokenizer = AutoTokenizer.from_pretrained(enc.config._name_or_path)

        self.position = PositionalEncoding(self.d_model, self.dropout).to(self.device)
        self.decoder_emb = Embeddings(d_model=self.d_model, vocab=2,
                                      padding_idx=target_vocab.index('<PAD>')).to(self.device)
        self.fix_len = 66

        self.length_module = torch.nn.Linear(self.d_model, max_target_len).to(self.device)

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.schema = schema
        self.fixed_tag_embeddings = None
        self.beam_width = 5

    def forward(self, inputs, target, domain, decode=False):

        encoder_outputs = self.encoder(**inputs)
        enc_hidden_states = encoder_outputs['last_hidden_state']

        enc_len_module_states = mean_pooling(encoder_outputs, inputs['attention_mask'])
        target_length_logits = self.length_module(enc_len_module_states)

        if not decode:
            target_inputs = torch.ones_like(target).long() * (target > 0)
            target_embeddings = self.decoder_emb(target_inputs)
            pos_target_embeddings = self.position(target_embeddings)
            decoder_output = self.decoder(tgt=pos_target_embeddings,
                                          tgt_key_padding_mask=(target == 0),
                                          memory=enc_hidden_states,
                                          memory_key_padding_mask=(inputs['attention_mask'] == 0))

            tag_list = [get_slot_expression(a) for a in self.schema[domain]['intents'] + self.schema[domain]['slots']]
            tag_tensors = self.tag_tokenizer(tag_list, return_tensors="pt", padding=True,
                                             add_special_tokens=False).to(device=self.device)
            tag_outs = self.tag_encoder(**tag_tensors)

            if self.tag_model[:4] == 'bert':
                tag_embeddings = tag_outs['last_hidden_state'][:, 0, :]
            else:
                tag_embeddings = mean_pooling(tag_outs, tag_tensors['attention_mask'])

            tag_target_scores = torch.einsum('abc, dc -> abd', decoder_output, tag_embeddings)

            fixed_scores = torch.zeros(tag_target_scores.shape[0], tag_target_scores.shape[1],
                                       self.fix_len).to(device=self.device)

            src_ptr_scores = torch.einsum('abc, adc -> abd', decoder_output,
                                          enc_hidden_states)  # / np.sqrt(decoder_output.shape[-1])
            src_ptr_scores = src_ptr_scores * inputs['attention_mask'].unsqueeze(1)

            fixed_scores[:, :, 2:src_ptr_scores.shape[-1] + 2] = src_ptr_scores

            final_scores = torch.cat((fixed_scores, tag_target_scores), dim=2)

            label_loss = self.loss(final_scores.contiguous().view(-1, final_scores.shape[-1]),
                                   target.contiguous().view(-1))

            gold_target_lengths = (target > 0).long().sum(dim=1)
            length_loss = self.loss(target_length_logits, gold_target_lengths)

            loss = label_loss + (0.1 * length_loss)

            return loss, final_scores

        else:
            _, top_lengths = torch.topk(target_length_logits, self.beam_width)
            sentence_shape = top_lengths.shape
            largest_length = torch.max(top_lengths).int().item()

            target_inputs = (torch.arange(largest_length).expand(top_lengths.shape[0],
                                                                 top_lengths.shape[1],
                                                                 largest_length).to(device=self.device)
                             < top_lengths.unsqueeze(2)).long()
            target_inputs = target_inputs.reshape(target_inputs.shape[0] * target_inputs.shape[1], -1)

            target_embeddings = self.decoder_emb(target_inputs)
            pos_target_embeddings = self.position(target_embeddings)

            beam_enc_hidden_states = enc_hidden_states.unsqueeze(1).repeat(1, self.beam_width, 1, 1)
            beam_enc_hidden_states = beam_enc_hidden_states.reshape(-1,
                                                                    beam_enc_hidden_states.shape[2],
                                                                    beam_enc_hidden_states.shape[3])

            beam_input_mask = inputs['attention_mask'].unsqueeze(1).repeat(1, self.beam_width, 1)
            beam_input_mask = beam_input_mask.reshape(-1, beam_input_mask.shape[2])

            decoder_output = self.decoder(tgt=pos_target_embeddings,
                                          tgt_key_padding_mask=(target_inputs == 0),
                                          memory=beam_enc_hidden_states,
                                          memory_key_padding_mask=(beam_input_mask == 0))

            if self.fixed_tag_embeddings is None:
                tag_list = [get_slot_expression(a) for a in self.schema[domain]['intents'] +
                            self.schema[domain]['slots']]
                tag_tensors = self.tag_tokenizer(tag_list, return_tensors="pt", padding=True,
                                                 add_special_tokens=False).to(device=self.device)
                tag_outs = self.tag_encoder(**tag_tensors)
                if self.tag_model[:4] == 'bert':
                    tag_embeddings = tag_outs['last_hidden_state'][:, 0, :]
                else:
                    tag_embeddings = mean_pooling(tag_outs, tag_tensors['attention_mask'])
            else:
                tag_embeddings = self.fixed_tag_embeddings[domain]

            tag_target_scores = torch.einsum('abc, dc -> abd', decoder_output, tag_embeddings)

            fixed_scores = torch.zeros(tag_target_scores.shape[0], tag_target_scores.shape[1],
                                       self.fix_len).to(device=self.device)

            src_ptr_scores = torch.einsum('abc, adc -> abd', decoder_output,
                                          beam_enc_hidden_states)  # / np.sqrt(decoder_output.shape[-1])
            src_ptr_scores = src_ptr_scores * beam_input_mask.unsqueeze(1)

            fixed_scores[:, :, 2:src_ptr_scores.shape[-1] + 2] = src_ptr_scores

            final_scores = torch.cat((fixed_scores, tag_target_scores), dim=2)

            final_scores = final_scores.reshape(sentence_shape[0], sentence_shape[1],
                                                final_scores.shape[1], final_scores.shape[2])
            predictions = torch.argmax(final_scores, dim=3)

            ys = []
            for i in range(predictions.shape[0]):
                all_beams = []
                for j in range(predictions.shape[1]):
                    all_beams.append(list(predictions[i][j][: top_lengths[i][j]].cpu().numpy()))
                ys.append(all_beams)
            return ys


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, padding_idx=0):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training models for fast semantic parsing")

    parser.add_argument('--data_folder', type=str, default='/home/srongali/data/top/seq2seq_sp')
    parser.add_argument('--save_folder', type=str, default='/mnt/nfs/scratch1/srongali/semparse/top')

    parser.add_argument('--held_out_domain', type=str, required=True)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--log_every', type=int, default=10)

    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--validate_every', type=int, default=1)

    parser.add_argument('--model_checkpoint', type=str, default='roberta-base')
    parser.add_argument('--use_span_encoder', action='store_true')
    parser.add_argument('--span_encoder_checkpoint', type=str, default='bert-base-uncased')

    args = parser.parse_args()

    model_checkpoint = args.model_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    data_folder = args.data_folder
    save_folder = args.save_folder

    schema = pickle.load(open(os.path.join(data_folder, 'schema.p'), 'rb'))

    domains = list(schema.keys())
    held_out_domain = args.held_out_domain
    train_domains = [d for d in domains if d != held_out_domain]

    epochs = args.epochs
    batch_size = args.batch_size
    log_every = args.log_every
    device = "cuda:0"

    train_processed = process_s2s_data(data_folder, train_domains, 'train', batch_size, tokenizer, schema)
    val_processed_1 = process_s2s_data(data_folder, train_domains, 'eval', batch_size, tokenizer, schema)
    val_processed_2 = process_s2s_data(data_folder, [held_out_domain], 'eval', batch_size, tokenizer, schema)

    encoder = AutoModel.from_pretrained(model_checkpoint).to(device)
    d_model = encoder.config.hidden_size
    decoder = TransformerDecoder(TransformerDecoderLayer(d_model=d_model, nhead=8, batch_first=True),
                                 num_layers=6).to(device)

    tag_model = None
    if args.use_span_encoder:
        tag_model = args.span_encoder_checkpoint

    model = CustomSeq2Seq(enc=encoder, dec=decoder, schema=schema, tag_model=tag_model)

    warmup_proportion = 0.1
    learning_rate = 2e-5
    adam_epsilon = 1e-8
    weight_decay = 0.01

    num_train_optimization_steps = len(train_processed) * epochs

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight', 'norm.a_2', 'norm.b_2']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    # Training metrics
    ind_accuracies = [0]
    ood_accuracies = [0]
    patience_count = 0

    print('Begin Training...', flush=True)
    model.train()

    start_time = datetime.now()

    for epoch in range(epochs):
        shuffle(train_processed)

        update = 0
        total_updates = len(train_processed)

        # Training
        model.train()
        for i in range(0, len(train_processed)):
            inp, tgt, domain = train_processed[i]
            inp = inp.to(device=device)
            tgt = tgt.to(device=device)

            loss, logits = model(inp, tgt, domain)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
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
                inp, tgt, domain = val_processed_1[i]
                inp = inp.to(device=device)
                tgt = tgt.to(device=device)

                with torch.no_grad():
                    loss, logits = model(inp, tgt, domain)

                scores = logits.reshape(-1, logits.shape[2])
                preds = torch.argmax(scores, dim=1)
                tags = tgt[:, 1:].reshape(-1)

                mask = tags > 0
                total += torch.sum(mask).item()
                correct += torch.sum(mask * (preds == tags)).item()

            acc = correct * 100.0 / total
            print('Same domain sequence accuracy: {:.2f}'.format(acc), flush=True)
            if acc > max(ind_accuracies):
                print('BEST SO FAR! Saving model...')
                state_dict = {'model_state_dict': model.state_dict()}
                save_path = os.path.join(save_folder, 's2slm_wo_{}_best.pt'.format(held_out_domain))
                torch.save(state_dict, save_path)
                print('Best checkpoint saved to {}'.format(save_path), flush=True)
                patience_count = 0
            else:
                patience_count += 1

            ind_accuracies.append(acc)

            correct = 0
            total = 0

            for i in range(0, len(val_processed_2)):
                inp, tgt, domain = val_processed_2[i]
                inp = inp.to(device=device)
                tgt = tgt.to(device=device)

                with torch.no_grad():
                    loss, logits = model(inp, tgt, domain)

                scores = logits.reshape(-1, logits.shape[2])
                preds = torch.argmax(scores, dim=1)
                tags = tgt[:, 1:].reshape(-1)

                mask = tags > 0
                total += torch.sum(mask).item()
                correct += torch.sum(mask * (preds == tags)).item()

            acc = correct * 100.0 / total
            print('Out of domain sequence accuracy: {:.2f}'.format(acc), flush=True)
            ood_accuracies.append(acc)

            if patience_count > args.patience:
                print('Ran out of patience. Exiting training.', flush=True)
                break

    print('Done. Total time taken: {}'.format(datetime.now() - start_time), flush=True)

    state_dict = {'model_state_dict': model.state_dict()}
    save_path = os.path.join(save_folder, 's2slm_wo_{}_latest.pt'.format(held_out_domain))
    torch.save(state_dict, save_path)
    print('Latest checkpoint saved to {}'.format(save_path), flush=True)

