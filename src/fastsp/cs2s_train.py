import torch
import os
import pickle
import argparse
from datetime import datetime
from torch import nn
from queue import PriorityQueue

from transformers import AutoTokenizer, AutoModel
from torch.nn import TransformerDecoder, TransformerDecoderLayer

import random
from random import shuffle
import math
import operator
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

random.seed(1100)
target_vocab = ['<PAD>', '<START>', '<END>'] + ['@ptr_{}'.format(i) for i in range(64)]
schema = {}


def process_s2s_data(path, domain_list, split, bsize, tokenizer):
    processed = []
    for domain in domain_list:
        all_examples = []
        with open(os.path.join(path, '{}_{}.tsv'.format(domain, split))) as inf:
            for line in inf:
                fields = line.strip().split('\t')
                target = fields[2].split()
                target_indices = [1] + [target_vocab.index(a) if a in target_vocab else
                                        67 + (schema[domain]['intents'] + schema[domain]['slots']).index(a)
                                        for a in target] + [2]
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
    def __init__(self, enc, dec, tok, tag_enc=None):
        super(CustomSeq2Seq, self).__init__()
        self.dropout = 0.1
        self.d_model = enc.config.hidden_size
        self.tokenizer = tok
        self.device = enc.device

        self.encoder = enc
        self.decoder = dec

        if tag_enc:
            self.tag_encoder = tag_enc
        else:
            self.tag_encoder = enc

        self.position = PositionalEncoding(self.d_model, self.dropout).to(self.device)
        self.decoder_emb = Embeddings(d_model=self.d_model, vocab=len(target_vocab),
                                      padding_idx=target_vocab.index('<PAD>')).to(self.device)
        self.fix_len = 67

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, inputs, target, domain, decode=False):

        encoder_outputs = self.encoder(**inputs)
        enc_hidden_states = encoder_outputs['last_hidden_state']

        if not decode:
            fixed_target_mask = target < self.fix_len
            target_mask = (target > 0).float()

            tag_list = [get_slot_expression(a) for a in schema[domain]['intents'] + schema[domain]['slots']]
            tag_tensors = self.tokenizer(tag_list, return_tensors="pt", padding=True,
                                         add_special_tokens=False).to(device=self.device)
            tag_outs = self.tag_encoder(**tag_tensors)
            tag_embeddings = mean_pooling(tag_outs, tag_tensors['attention_mask'])

            fixed_target_embeddings = self.decoder_emb(target * fixed_target_mask)

            tag_target_tokens = (target * ~fixed_target_mask - self.fix_len) * ~fixed_target_mask
            tag_target_embeddings = F.embedding(tag_target_tokens, tag_embeddings)

            target_embeddings = ((fixed_target_embeddings * fixed_target_mask.unsqueeze(2)) +
                                 (tag_target_embeddings * ~fixed_target_mask.unsqueeze(2)))

            pos_target_embeddings = self.position(target_embeddings)

            decoder_output = self.decoder(pos_target_embeddings, enc_hidden_states,
                                          inputs['attention_mask'].float(), target_mask)

            tag_target_scores = torch.einsum('abc, dc -> abd', decoder_output, tag_embeddings)

            fixed_scores = torch.zeros(tag_target_scores.shape[0], tag_target_scores.shape[1],
                                       self.fix_len).to(device=self.device)

            src_ptr_scores = torch.einsum('abc, adc -> abd', decoder_output,
                                          enc_hidden_states)  # / np.sqrt(decoder_output.shape[-1])
            src_ptr_scores = src_ptr_scores * inputs['attention_mask']

            fixed_scores[:, :, 3:src_ptr_scores.shape[-1]] = src_ptr_scores

            fix_spl_tokens = torch.range(0, 2).long().repeat(src_ptr_scores.shape[0]).to(device=self.device)
            fix_spl_embeddings = self.decoder_emb(fix_spl_tokens)

            fixed_scores[:, :, :3] = torch.einsum('abc, dc -> abd', decoder_output, fix_spl_embeddings)

            final_scores = torch.cat((fixed_scores, tag_target_scores), dim=2)[:, :, :-1]

            target_y = target[:, :, 1:]

            loss = self.loss(final_scores.contiguous().view(-1, final_scores.shape[-1]),
                             target_y.contiguous().view(-1))
            return loss, final_scores

        else:
            ys = beam_decode(inputs, enc_hidden_states, self, domain)
            return ys


class BeamSearchNode(object):
    def __init__(self, ys, previousNode, wordId, logProb, length):
        self.ys = ys
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return True


def subsequent_mask(size, batch_size=1):
    attn_shape = (batch_size, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def beam_decode(inp, enc_hid, cur_model, domain):
    beam_width = 5
    topk = 1  # how many sentence do you want to generate
    decoded_batch = []
    start_symbol = 1
    end_symbol = 2
    batch_size = 1
    fix_len = cur_model.fix_len

    # decoding goes sentence by sentence
    for idx in range(enc_hid.size(0)):

        encoder_output = enc_hid[idx, :, :].unsqueeze(0)  # [1, 128, 768]
        src_mask = inp['attention_mask']

        # Number of sentence to generate
        endnodes = []

        ys = torch.ones(batch_size, 1).fill_(start_symbol).long().to(device=cur_model.device)

        node = BeamSearchNode(ys, None, start_symbol, 0, 1)
        nodes = PriorityQueue()
        nodes.put((-node.eval(), node))
        breaknow = False

        while not breaknow:
            nextnodes = PriorityQueue()
            while nodes.qsize() > 0:
                score, n = nodes.get()
                ys = n.ys

                if ys.shape[1] > 100:
                    breaknow = True
                    break

                if n.wordid == end_symbol and n.prevNode is not None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) == beam_width:
                        breaknow = True
                        break
                    else:
                        continue

                fixed_target_mask = ys < fix_len

                tag_list = [get_slot_expression(a) for a in schema[domain]['intents'] + schema[domain]['slots']]
                tag_tensors = cur_model.tokenizer(tag_list, return_tensors="pt", padding=True,
                                                  add_special_tokens=False).to(device=cur_model.device)
                tag_outs = cur_model.tag_encoder(**tag_tensors)
                tag_embeddings = mean_pooling(tag_outs, tag_outs['attention_mask'])

                fixed_target_embeddings = cur_model.decoder_emb(ys * fixed_target_mask)

                tag_target_tokens = (ys * ~fixed_target_mask - fix_len) + (
                            torch.ones_like(ys) * fixed_target_mask)
                tag_target_embeddings = F.embedding(tag_target_tokens, tag_embeddings)

                target_embeddings = ((fixed_target_embeddings * fixed_target_mask.unsqueeze(2)) +
                                     (tag_target_embeddings * ~fixed_target_mask.unsqueeze(2)))

                pos_target_embeddings = cur_model.position(target_embeddings)

                decoder_output = cur_model.decoder(pos_target_embeddings, enc_hid,
                                                   src_mask.float(),
                                                   Variable(subsequent_mask(ys.size(1), batch_size=batch_size).long().
                                                            to(device=cur_model.device)))

                tag_target_scores = torch.einsum('ac, dc -> ad', decoder_output[:, -1], tag_embeddings)

                fixed_scores = torch.zeros(tag_target_scores.shape[0], fix_len).to(device=cur_model.device)

                src_ptr_scores = torch.einsum('ac, adc -> ad', decoder_output[:, -1],
                                              encoder_output)  # / np.sqrt(decoder_output.shape[-1])
                src_ptr_scores = src_ptr_scores * src_mask

                fixed_scores[:, 3:src_ptr_scores.shape[-1]] = src_ptr_scores

                fix_spl_tokens = torch.range(0, 2).long().repeat(src_ptr_scores.shape[0]).to(device=cur_model.device)
                fix_spl_embeddings = cur_model.decoder_emb(fix_spl_tokens)

                fixed_scores[:, :3] = torch.einsum('ac, dc -> ad', decoder_output[:, -1], fix_spl_embeddings)

                all_scores = torch.cat((fixed_scores, tag_target_scores), dim=1)

                all_prob = F.log_softmax(all_scores, dim=-1)

                top_log_prob, top_indexes = torch.topk(all_prob, beam_width)

                for new_k in range(beam_width):
                    decoded_t = top_indexes[0][new_k].view(1, -1).type_as(ys)
                    log_prob = top_log_prob[0][new_k].item()

                    ys2 = torch.cat([ys, decoded_t], dim=1)

                    node = BeamSearchNode(ys2, n, decoded_t.item(), n.logp + log_prob, n.leng + 1)
                    score = -node.eval()
                    nextnodes.put((score, node))

                for i in range(beam_width):
                    if nextnodes.qsize() > 0:
                        score, nnode = nextnodes.get()
                        nodes.put((score, nnode))

        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]
        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = [n.wordid]
            while n.prevNode is not None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)
            if len(utterances) == topk:
                break

        decoded_batch.append(utterances)

    return decoded_batch


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

    parser.add_argument('--data_folder', type=str, default='/home/srongali/data/top/seq2seq')
    parser.add_argument('--save_folder', type=str, default='/mnt/nfs/scratch1/srongali/semparse/top')

    parser.add_argument('--held_out_domain', type=str, required=True)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--log_every', type=int, default=10)

    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--validate_every', type=int, default=1)

    parser.add_argument('--model_checkpoint', type=str, default='roberta-base')

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

    train_processed = process_s2s_data(data_folder, train_domains, 'train', batch_size, tokenizer)
    val_processed_1 = process_s2s_data(data_folder, train_domains, 'eval', batch_size, tokenizer)
    val_processed_2 = process_s2s_data(data_folder, [held_out_domain], 'eval', batch_size, tokenizer)

    encoder = AutoModel.from_pretrained(model_checkpoint).to(device)
    d_model = encoder.config.hidden_size
    decoder = TransformerDecoder(TransformerDecoderLayer(d_model=d_model, nhead=8), num_layers=6).to(device)

    model = CustomSeq2Seq(enc=encoder, dec=decoder, tok=tokenizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

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

                loss, logits = model(inp, tgt, domain)

                scores = logits.reshape(-1, logits.shape[2])
                preds = torch.argmax(scores, dim=1)
                tags = tgt.reshape(-1)

                mask = tags > 0
                total += torch.sum(mask).item()
                correct += torch.sum(mask * (preds == tags)).item()

            acc = correct * 100.0 / total
            print('Same domain sequence accuracy: {:.2f}'.format(acc), flush=True)
            if acc > max(ind_accuracies):
                print('BEST SO FAR! Saving model...')
                state_dict = {'model_state_dict': model.state_dict()}
                save_path = os.path.join(save_folder, 's2s_wo_{}_best.pt'.format(held_out_domain))
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

                loss, logits = model(inp, tgt, domain)

                scores = logits.reshape(-1, logits.shape[2])
                preds = torch.argmax(scores, dim=1)
                tags = tgt.reshape(-1)

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
    save_path = os.path.join(save_folder, 's2s_wo_{}_latest.pt'.format(held_out_domain))
    torch.save(state_dict, save_path)
    print('Latest checkpoint saved to {}'.format(save_path), flush=True)
