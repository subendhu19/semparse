import torch
import os
import pickle
import argparse

from transformers import AutoTokenizer

import random
from src.fastsp.base_train import BaseScorer, tag_entity_name_dict, process_data_with_tags
import statistics

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

    val_data = pickle.load(open(os.path.join(data_folder, 'base_val_data.p'), 'rb'))
    intents = list(val_data.keys())

    held_out_intent = args.held_out_intent
    eval_intent = args.eval_intent

    batch_size = args.batch_size
    device = "cuda:0"

    slot_vectors = None
    if args.precompute_slotvecs:
        slot_vectors = pickle.load(open(os.path.join(args.data_folder, 'slot_vecs.p'), 'rb'))

    model = BaseScorer(ckpt=model_checkpoint,
                       model_style=args.model_style,
                       slot_vecs=slot_vectors).to(device)

    model_name = args.model_style + '_pc' if args.precompute_slotvecs else args.model_style
    model_name = model_name + '_{}'.format(model_checkpoint) if model_checkpoint != 'bert-base-uncased' else model_name

    if args.use_descriptions:
        model.load_state_dict(torch.load(os.path.join(save_folder, 'base_{}_wo_{}_desc_best.pt'.
                                                      format(model_name, held_out_intent)))['model_state_dict'])
    else:
        model.load_state_dict(torch.load(os.path.join(save_folder, 'base_{}_wo_{}_best.pt'.
                                                      format(model_name, held_out_intent)))['model_state_dict'])

    val_processed = process_data_with_tags(val_data, [eval_intent], tokenizer, batch_size)

    model.eval()

    predictions = []
    golds = []

    for i in range(0, len(val_processed)):
        mini_batch, intent = val_processed[i]

        sents = [a[0] for a in mini_batch]
        sent_tensors = tokenizer(sents, return_tensors="pt", padding=True,
                                 add_special_tokens=False).to(device=device)

        with torch.no_grad():
            scores = model(sent_tensors, intent, args.use_descriptions)

        tags = [a[1] for a in mini_batch]
        pad = len(max(tags, key=len))
        tags = torch.tensor([i + [-100]*(pad-len(i)) for i in tags]).to(device=device)

        preds = torch.argmax(scores, dim=2)

        for j in range(len(preds)):
            pred, gold = get_entities_from_tags(preds[j], tags[j], tokenizer.tokenize(sents[j]), intent)
            predictions.append(pred)
            golds.append(gold)

    precision_n = 0
    precision_d = 0
    recall_n = 0
    recall_d = 0
    correct = 0

    for i in range(len(predictions)):
        for p in predictions[i]:
            if p in golds[i]:
                precision_n += 1
            precision_d += 1

        for p in golds[i]:
            if p in predictions[i]:
                recall_n += 1
            recall_d += 1

        if sorted(predictions[i]) == sorted(golds[i]):
            correct += 1

    precision = precision_n / precision_d * 100.0
    recall = recall_n / recall_d * 100.0
    em = correct / len(predictions) * 100.0

    print('Precision: {:.2f}'.format(precision))
    print('Recall: {:.2f}'.format(recall))
    print('F1: {:.2f}'.format(statistics.harmonic_mean([precision, recall])))
    print('EM Accuracy: {:.2f}'.format(em))





