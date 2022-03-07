import torch
import os
import pickle
import argparse

from transformers import BertTokenizerFast

import random
from src.fastsp.base_train import BaseScorer, tag_entity_name_dict

random.seed(1100)


def get_entities_from_tags(pred_tags, gold_tags, utterance, c_intent):
    golds = []
    preds = []

    none_idx = len(tag_entity_name_dict[c_intent]) - 1

    g_current_slot = gold_tags[0]
    g_current_vals = [0]

    p_current_slot = pred_tags[0]
    p_current_vals = [0]

    for i in range(1, len(gold_tags)):
        if gold_tags[i] == -100:
            if g_current_slot != none_idx:
                golds.append((tag_entity_name_dict[c_intent][g_current_slot], ' '.join([utterance[b]
                                                                                        for b in g_current_vals])))
            if p_current_slot != none_idx:
                preds.append((tag_entity_name_dict[c_intent][p_current_slot], ' '.join([utterance[b]
                                                                                        for b in p_current_vals])))
            break

        if gold_tags[i] != g_current_slot:
            if g_current_slot != none_idx:
                golds.append((tag_entity_name_dict[c_intent][g_current_slot], ' '.join([utterance[b]
                                                                                        for b in g_current_vals])))
            g_current_slot = gold_tags[i]
            g_current_vals = [i]
        else:
            g_current_vals.append(i)

        if pred_tags[i] != p_current_slot:
            if p_current_slot != none_idx:
                preds.append((tag_entity_name_dict[c_intent][p_current_slot], ' '.join([utterance[b]
                                                                                        for b in p_current_vals])))
            p_current_slot = pred_tags[i]
            p_current_vals = [i]
        else:
            p_current_vals.append(i)

    return preds, golds


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluating models for fast semantic parsing")

    parser.add_argument('--data_folder', type=str, default='/home/srongali/data/snips')
    parser.add_argument('--save_folder', type=str, default='/mnt/nfs/scratch1/srongali/semparse/snips')

    parser.add_argument('--eval_intent', type=str, required=True)
    parser.add_argument('--held_out_intent', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--use_descriptions', action='store_true')
    parser.add_argument('--model_style', type=str, choices=['dot', 'ff', 'wdot'], default='dot')

    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Model params
    MAX_SEQ_LEN = 128
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # Data
    data_folder = args.data_folder
    save_folder = args.save_folder

    val_data = pickle.load(open(os.path.join(data_folder, 'base_val_data.p'), 'rb'))
    intents = list(val_data.keys())

    held_out_intent = args.held_out_intent
    eval_intent = args.eval_intent

    batch_size = args.batch_size
    device = "cuda:0"

    model = BaseScorer(model_style=args.model_style).to(device)

    if args.use_descriptions:
        model.load_state_dict(torch.load(os.path.join(save_folder, 'base_{}_wo_{}_desc_best.pt'.
                                                      format(args.model_style, held_out_intent)))['model_state_dict'])
    else:
        model.load_state_dict(torch.load(os.path.join(save_folder, 'base_{}_wo_{}_best.pt'.
                                                      format(args.model_style, held_out_intent)))['model_state_dict'])

    val_processed = []
    for intent in [eval_intent]:
        all_examples = []
        for i in range(len(val_data[intent]['utterances'])):
            utt = val_data[intent]['utterances'][i]
            utt_tok = tokenizer(utt, return_tensors="pt", add_special_tokens=False)
            utt_ids = utt_tok.word_ids()

            ti = val_data[intent]['tag_indices'][i]

            nti = [ti[a] for a in utt_ids]

            all_examples.append([utt, nti])

        for i in range(0, len(all_examples), batch_size):
            mini_batch = all_examples[i:i + batch_size]
            val_processed.append((mini_batch, intent))

    model.eval()

    predictions = []
    golds = []

    for i in range(0, len(val_processed)):
        mini_batch, intent = val_processed[i]

        sents = [a[0] for a in mini_batch]
        sent_tensors = tokenizer(sents, return_tensors="pt", padding=True,
                                 add_special_tokens=False).to(device=device)

        scores = model(sent_tensors, intent, args.use_descriptions)

        tags = [a[1] for a in mini_batch]
        pad = len(max(tags, key=len))
        tags = torch.tensor([i + [-100]*(pad-len(i)) for i in tags]).to(device=device)

        preds = torch.argmax(scores, dim=2)

        for j in range(len(preds)):
            pred, gold = get_entities_from_tags(preds[j], tags[j], tokenizer.tokenize(sents[j]), intent)
            predictions.append(pred)
            golds.append(gold)

    for i in range(10):
        print(golds[i])
        print(predictions[i])





