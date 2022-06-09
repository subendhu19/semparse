import torch
import os
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
from queue import PriorityQueue
import operator

random.seed(1100)
target_vocab = ['<PAD>', '<START>', '<END>'] + ['@ptr_{}'.format(i) for i in range(64)]
target_vocab_dict = {target_vocab[i]: i for i in range(len(target_vocab))}
descriptions = None
schema = None


def process_s2s_data(path, file_name, bsize, tokenizer, schema):
    processed = []
    all_examples = []

    error_count = 0
    with open(os.path.join(path, file_name)) as inf:
        for line in inf:
            skip = False
            fields = line.strip().split('\t')

            # Skip sentences longer than 64 tokens
            if len(fields[1].split()) > 64:
                continue

            target = fields[2].split()
            for t in target:
                if (t not in target_vocab_dict) and (t not in schema):
                    error_count += 1
                    skip = True
                    break
            if skip:
                continue
            tags_present = list(set([a for a in target if a in schema]))
            all_examples.append((fields[0], target, tags_present))
    print('Errors in {}: {}'.format(file_name, error_count), flush=True)

    for i in range(0, len(all_examples), bsize):
        mini_batch = all_examples[i: i + bsize]

        sents = [a[0] for a in mini_batch]
        sent_tensors = tokenizer(sents, return_tensors="pt", padding=True,
                                 add_special_tokens=False)

        all_tags = []
        for mb_item in mini_batch:
            for x in mb_item[2]:
                if x not in all_tags:
                    all_tags.append(x)
        all_tags_dict = {all_tags[j]: j for j in range(len(all_tags))}

        target = []
        for mb_item in mini_batch:
            target_indices = [1] + [target_vocab_dict[a] if a in target_vocab_dict else
                                    67 + all_tags_dict[a]
                                    for a in mb_item[1]] + [2]
            target.append(target_indices)

        pad = len(max(target, key=len))
        target = torch.tensor([i + [0] * (pad - len(i)) for i in target])

        processed.append((sent_tensors, target, all_tags))

    return processed


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_slot_expression(tag_item, descriptions):
    if '[' in tag_item:
        return 'begin ' + descriptions[tag_item[1:]]
    else:
        return 'end ' + descriptions[tag_item[:-1]]


class CustomSeq2Seq(nn.Module):
    def __init__(self, enc, dec, schema, tag_model=None):
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
        self.decoder_emb = Embeddings(d_model=self.d_model, vocab=len(target_vocab),
                                      padding_idx=target_vocab.index('<PAD>')).to(self.device)
        self.fix_len = 67

        self.schema = schema
        self.fixed_tag_embeddings = None
        self.beam_width = 5

    def forward(self, input_ids, attention_mask, target, all_tags, decode=False):

        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        enc_hidden_states = encoder_outputs['last_hidden_state']

        if not decode:
            fixed_target_mask = target < self.fix_len
            target_mask = (target > 0).float()

            tag_list = [get_slot_expression(a, self.schema) for a in all_tags]
            tag_tensors = self.tag_tokenizer(tag_list, return_tensors="pt", padding=True,
                                             add_special_tokens=False).to(device=input_ids.device)
            tag_outs = self.tag_encoder(**tag_tensors)

            if self.tag_model[:4] == 'bert':
                tag_embeddings = tag_outs['last_hidden_state'][:, 0, :]
            else:
                tag_embeddings = mean_pooling(tag_outs, tag_tensors['attention_mask'])

            fixed_target_embeddings = self.decoder_emb(target * fixed_target_mask)

            tag_target_tokens = (target * ~fixed_target_mask - self.fix_len) * ~fixed_target_mask
            tag_target_embeddings = F.embedding(tag_target_tokens, tag_embeddings)

            target_embeddings = ((fixed_target_embeddings * fixed_target_mask.unsqueeze(2)) +
                                 (tag_target_embeddings * ~fixed_target_mask.unsqueeze(2)))

            pos_target_embeddings = self.position(target_embeddings)

            decoder_output = self.decoder(tgt=pos_target_embeddings,
                                          tgt_mask=subsequent_mask(target.size(1)).to(device=input_ids.device),
                                          tgt_key_padding_mask=(target_mask == 0),
                                          memory=enc_hidden_states,
                                          memory_mask=full_mask(target.size(1),
                                                                enc_hidden_states.size(1)).to(device=input_ids.device),
                                          memory_key_padding_mask=(attention_mask == 0))

            tag_target_scores = torch.einsum('abc, dc -> abd', decoder_output, tag_embeddings)

            fixed_scores = torch.zeros(tag_target_scores.shape[0], tag_target_scores.shape[1],
                                       self.fix_len).to(device=input_ids.device)

            src_ptr_scores = torch.einsum('abc, adc -> abd', decoder_output,
                                          enc_hidden_states)  # / np.sqrt(decoder_output.shape[-1])
            src_ptr_scores = src_ptr_scores * attention_mask.unsqueeze(1)

            fixed_scores[:, :, 3:src_ptr_scores.shape[-1]+3] = src_ptr_scores

            fix_spl_tokens = torch.arange(0, 3).long().to(device=input_ids.device)
            fix_spl_embeddings = self.decoder_emb(fix_spl_tokens)

            fixed_scores[:, :, :3] = torch.einsum('abc, dc -> abd', decoder_output, fix_spl_embeddings)

            final_scores = torch.cat((fixed_scores, tag_target_scores), dim=2)[:, :-1, :]

            return final_scores
        else:
            ys = beam_decode(attention_mask, enc_hidden_states, self, all_tags)
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


def subsequent_mask(size):
    attn_shape = (size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 1


def full_mask(size1, size2):
    return torch.ones((size1, size2)) == 0


def beam_decode(inp_att, enc_hid, cur_model, all_tags):
    beam_width = cur_model.beam_width
    topk = 1  # how many sentence do you want to generate
    decoded_batch = []
    start_symbol = 1
    end_symbol = 2
    batch_size = 1
    fix_len = cur_model.fix_len

    # decoding goes sentence by sentence
    for idx in range(enc_hid.size(0)):

        encoder_output = enc_hid[idx, :, :].unsqueeze(0)  # [1, 128, 768]

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

                if (n.wordid == end_symbol and n.prevNode is not None) or ys.shape[1] > 60:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) == beam_width:
                        breaknow = True
                        break
                    else:
                        continue

                fixed_target_mask = ys < fix_len

                if cur_model.fixed_tag_embeddings is None:
                    tag_list = [get_slot_expression(a, cur_model.schema) for a in all_tags]
                    tag_tensors = cur_model.tag_tokenizer(tag_list, return_tensors="pt", padding=True,
                                                          add_special_tokens=False).to(device=cur_model.device)
                    tag_outs = cur_model.tag_encoder(**tag_tensors)
                    if cur_model.tag_model[:4] == 'bert':
                        tag_embeddings = tag_outs['last_hidden_state'][:, 0, :]
                    else:
                        tag_embeddings = mean_pooling(tag_outs, tag_tensors['attention_mask'])
                else:
                    tag_embeddings = cur_model.fixed_tag_embeddings

                fixed_target_embeddings = cur_model.decoder_emb(ys * fixed_target_mask)

                tag_target_tokens = (ys * ~fixed_target_mask - fix_len) * ~fixed_target_mask
                tag_target_embeddings = F.embedding(tag_target_tokens, tag_embeddings)

                target_embeddings = ((fixed_target_embeddings * fixed_target_mask.unsqueeze(2)) +
                                     (tag_target_embeddings * ~fixed_target_mask.unsqueeze(2)))

                pos_target_embeddings = cur_model.position(target_embeddings)

                decoder_output = cur_model.decoder(tgt=pos_target_embeddings,
                                                   memory=encoder_output,
                                                   memory_mask=full_mask(ys.size(1),
                                                                         encoder_output.size(1)).to(
                                                       device=cur_model.device),
                                                   memory_key_padding_mask=(inp_att[idx].unsqueeze(0) == 0),
                                                   tgt_mask=subsequent_mask(ys.size(1)).to(device=cur_model.device))

                tag_target_scores = torch.einsum('ac, dc -> ad', decoder_output[:, -1], tag_embeddings)

                fixed_scores = torch.zeros(tag_target_scores.shape[0], fix_len).to(device=cur_model.device)

                src_ptr_scores = torch.einsum('ac, adc -> ad', decoder_output[:, -1],
                                              encoder_output)  # / np.sqrt(decoder_output.shape[-1])
                src_ptr_scores = src_ptr_scores * inp_att[idx].unsqueeze(0)

                fixed_scores[:, 3:src_ptr_scores.shape[-1]+3] = src_ptr_scores

                fix_spl_tokens = torch.arange(0, 3).long().to(device=cur_model.device)
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

    parser.add_argument('--data_folder', type=str, default='/mnt/nfs/scratch1/srongali/semparse/wikidata')
    parser.add_argument('--save_folder', type=str, default='/mnt/nfs/scratch1/srongali/semparse/wikidata')

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--log_every', type=int, default=10)

    parser.add_argument('--validate_every', type=int, default=1000)

    parser.add_argument('--model_checkpoint', type=str, default='roberta-base')
    parser.add_argument('--use_span_encoder', action='store_true')
    parser.add_argument('--span_encoder_checkpoint', type=str, default='bert-base-uncased')
    parser.add_argument('--patience', type=int, default=10)

    args = parser.parse_args()

    model_checkpoint = args.model_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    data_folder = args.data_folder
    save_folder = args.save_folder

    descriptions = {}
    with open(os.path.join(data_folder, 'type_desc.txt')) as inf:
        for line in inf:
            fields = line.strip().split('\t')
            descriptions[fields[0]] = fields[1]

    schema = {}
    tag_count = 0
    for tag_item in ['[' + a for a in list(descriptions.keys())] + [a + ']' for a in list(descriptions.keys())]:
        schema[tag_item] = tag_count
        tag_count += 1

    epochs = args.epochs
    batch_size = args.batch_size

    ngpus = torch.cuda.device_count()
    batch_size = ngpus * batch_size

    log_every = args.log_every
    device = "cuda:0"

    test_processed = process_s2s_data(data_folder, 'test.proc.tsv', batch_size, tokenizer, schema)
    train_shards = ['shards/train.shard0{}'.format(i) for i in range(10)]

    encoder = AutoModel.from_pretrained(model_checkpoint).to(device)
    d_model = encoder.config.hidden_size
    decoder = TransformerDecoder(TransformerDecoderLayer(d_model=d_model, nhead=8, batch_first=True),
                                 num_layers=6).to(device)

    tag_model = None
    if args.use_span_encoder:
        tag_model = args.span_encoder_checkpoint

    model = CustomSeq2Seq(enc=encoder, dec=decoder, schema=descriptions, tag_model=tag_model)
    model = nn.DataParallel(model)

    warmup_proportion = 0.1
    learning_rate = args.lr
    adam_epsilon = 1e-8
    weight_decay = 0.01
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    num_train_optimization_steps = 75000 * epochs

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
        shuffle(train_shards)
        print('Training with shards: {}'.format(str(train_shards)), flush=True)
        for sc, shard in enumerate(train_shards):
            train_processed = process_s2s_data(data_folder, shard, batch_size, tokenizer, schema)
            shuffle(train_processed)
            print('Loaded shard {}'.format(shard), flush=True)

            update = 0
            total_updates = len(train_processed)

            # Training
            model.train()
            for i in range(0, len(train_processed)):
                inp, tgt, all_tags = train_processed[i]
                inp_ids = inp['input_ids'].to(device=device)
                att_mask = inp['attention_mask'].to(device=device)
                tgt = tgt.to(device=device)

                logits = model(inp_ids, att_mask, tgt, all_tags)
                target_y = tgt[:, 1:]

                loss = loss_fn(logits.contiguous().view(-1, logits.shape[-1]),
                               target_y.contiguous().view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                update += 1

                if update % log_every == 0:
                    print("Epoch: {}/{} \t Shard: {}/{} \t Update: {}/{} \t Loss: {} \t Time elapsed: {}".
                          format(epoch+1, epochs, sc+1, len(train_shards), update, total_updates, loss.item(),
                                 datetime.now() - start_time),
                          flush=True)

                # Validation
                if update % args.validate_every == 0:
                    print("Finished {} updates. Running validation...".format(update), flush=True)
                    model.eval()

                    correct = 0
                    total = 0

                    for i in range(0, len(test_processed)):
                        inp, tgt, all_tags = test_processed[i]
                        inp_ids = inp['input_ids'].to(device=device)
                        att_mask = inp['attention_mask'].to(device=device)
                        tgt = tgt.to(device=device)

                        with torch.no_grad():
                            logits = model(inp_ids, att_mask, tgt, all_tags)

                        scores = logits.reshape(-1, logits.shape[2])
                        preds = torch.argmax(scores, dim=1)
                        tags = tgt[:, 1:].reshape(-1)

                        mask = tags > 0
                        total += torch.sum(mask).item()
                        correct += torch.sum(mask * (preds == tags)).item()

                    acc = correct * 100.0 / total
                    print('Sequence accuracy: {:.2f}'.format(acc), flush=True)
                    if acc > max(ind_accuracies):
                        print('BEST SO FAR! Saving model...', flush=True)
                        state_dict = {'model_state_dict': model.state_dict()}
                        save_path = os.path.join(save_folder, 'wikidata_best.pt')
                        torch.save(state_dict, save_path)
                        print('Best checkpoint saved to {}'.format(save_path), flush=True)
                        patience_count = 0
                    else:
                        patience_count += 1

                    ind_accuracies.append(acc)

                    if patience_count > args.patience:
                        print('Ran out of patience. Exiting training.', flush=True)
                        break

    print('Done. Total time taken: {}'.format(datetime.now() - start_time), flush=True)

    state_dict = {'model_state_dict': model.state_dict()}
    save_path = os.path.join(save_folder, 'wikidata_latest.pt')
    torch.save(state_dict, save_path)
    print('Latest checkpoint saved to {}'.format(save_path), flush=True)

