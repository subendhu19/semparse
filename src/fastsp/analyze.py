import torch
import os
import pickle
import argparse
from datetime import datetime

from transformers import BertTokenizerFast, BertForSequenceClassification
from src.fastsp.utils import slot_descriptions
from src.fastsp.train import ImplicitScorer, get_indices


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training models for fast semantic parsing")

    parser.add_argument('--data_folder', type=str, default='/home/srongali/data/snips')
    parser.add_argument('--save_folder', type=str, default='/mnt/nfs/scratch1/srongali/semparse/snips')

    parser.add_argument('--eval_intent', type=str, required=True)
    parser.add_argument('--held_out_intent', type=str, required=True)

    parser.add_argument('--span_threshold', type=int, default=6)
    parser.add_argument('--log_every', type=int, default=100)

    parser.add_argument('--save_metrics', action='store_true')
    parser.add_argument('--save_beam_search_file', action='store_true')

    parser.add_argument('--model_style', type=str, choices=['base', 'context', 'implicit'], default='base')
    parser.add_argument('--use_descriptions', action='store_true')

    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Data
    data_folder = args.data_folder
    save_folder = args.save_folder

    val_data = pickle.load(open(os.path.join(data_folder, 'val_data.p'), 'rb'))

    held_out_intent = args.held_out_intent
    eval_intent = args.eval_intent
    span_threshold = args.span_threshold
    device = "cuda:0"

    beam_search_utils = []

    if args.use_descriptions:
        analysis_file = open(
            os.path.join(save_folder, 'analysis', 'ho_{}_ev_{}_{}_desc_analysis.txt'.format(held_out_intent, eval_intent,
                                                                                args.model_style)), 'w')
    else:
        analysis_file = open(
            os.path.join(save_folder, 'analysis', 'ho_{}_ev_{}_{}_analysis.txt'.format(held_out_intent, eval_intent,
                                                                           args.model_style)), 'w')

    if args.model_style in ['base', 'context']:
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1).to(device)
    else:
        model = ImplicitScorer().to(device)

    if args.use_descriptions:
        model.load_state_dict(torch.load(os.path.join(save_folder, 'joint_{}_wo_{}_desc_best.pt'.
                                                      format(args.model_style, held_out_intent)))['model_state_dict'])
    else:
        model.load_state_dict(torch.load(os.path.join(save_folder, 'joint_{}_wo_{}_best.pt'.
                                                      format(args.model_style, held_out_intent)))['model_state_dict'])

    model.eval()

    print('Saved model loaded.', flush=True)

    entity_names = entity_name_dict[eval_intent]

    metrics_counts = {k: [] for k in entity_names}
    metrics_counts['all'] = []

    tot = len(val_data[eval_intent]['utterances'])
    # tot = 10

    log_every = args.log_every
    start_time = datetime.now()

    for i in range(tot):
        utt = val_data[eval_intent]['utterances'][i]
        ets = val_data[eval_intent]['entities'][i]

        spans, span_ids = find_all_spans(utt.split(), span_threshold)

        analysis_file.write('UTTERANCE: {}\n\n'.format(utt))
        analysis_file.write('GOLD SPANS: \n\t{}\n\n'.format('\n\t'.join([a[0] + ' ## ' + a[1] for a in ets])))

        beam_search_ents = {}

        for ent in entity_names:
            analysis_file.write('ENTITY NAME: {}\n\t'.format(ent))

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
                    metrics_counts[ent].append(index)
                    metrics_counts['all'].append(index)

            analysis_file.write('\n\t'.join([a[0] + ' ## ' + str(a[1]) for a in spans_w_scores[:5]]) + '\n\n')

            beam_search_ents[ent] = [(a[0], a[1].cpu(), a[2]) for a in spans_w_scores[:5]]

        beam_search_utils.append(beam_search_ents)

        analysis_file.write('\n\n############################################\n\n')

        if i % log_every == 0:
            print('Processed {}/{} item(s) \t Time elapsed: {}'.format(i, tot, datetime.now() - start_time), flush=True)

    analysis_file.close()

    if args.save_metrics:
        if args.use_descriptions:
            pickle.dump(metrics_counts, open(
                os.path.join(save_folder, 'metrics', 'ho_{}_ev_{}_{}_desc_metrics.p'.format(held_out_intent, eval_intent,
                                                                                 args.model_style)), 'wb'))
        else:
            pickle.dump(metrics_counts, open(
                os.path.join(save_folder, 'metrics', 'ho_{}_ev_{}_{}_metrics.p'.format(held_out_intent, eval_intent,
                                                                            args.model_style)), 'wb'))

    if args.save_beam_search_file:
        if args.use_descriptions:
            pickle.dump(beam_search_utils, open(
                os.path.join(save_folder, 'beamsearch', 'ho_{}_ev_{}_{}_desc_bs.p'.format(held_out_intent, eval_intent,
                                                                            args.model_style)), 'wb'))
        else:
            pickle.dump(beam_search_utils, open(
                os.path.join(save_folder, 'beamsearch', 'ho_{}_ev_{}_{}_bs.p'.format(held_out_intent, eval_intent,
                                                                       args.model_style)), 'wb'))

    print('Done. Total time taken: {}'.format(datetime.now() - start_time), flush=True)
