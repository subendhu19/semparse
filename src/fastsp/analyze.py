import torch
import os
import pickle
import argparse
from datetime import datetime

from transformers import BertTokenizer, BertForSequenceClassification


def find_all_spans(words, threshold):
    all_spans = []
    for i in range(1, threshold):
        for j in range(0, len(words) - i + 1):
            all_spans.append(' '.join(words[j:j+i]))
    return all_spans


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

    parser.add_argument('--model_style', type=str, choices=['base', 'context', 'implicit'], default='base')

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Data
    data_folder = args.data_folder
    save_folder = args.save_folder

    val_data = pickle.load(open(os.path.join(data_folder, 'val_data.p'), 'rb'))

    held_out_intent = args.held_out_intent
    eval_intent = args.eval_intent
    span_threshold = args.span_threshold
    device = "cuda:0"

    analysis_file = open(os.path.join(save_folder, 'ho_{}_ev_{}_{}_analysis.txt'.format(held_out_intent, eval_intent,
                                                                                        args.model_style)),
                         'w')

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1).to(device)
    model.load_state_dict(torch.load(os.path.join(save_folder, 'bert_wo_{}_{}.pt'.
                                                  format(held_out_intent, args.model_style)))['model_state_dict'])
    model.eval()

    print('Saved model loaded.')

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

        spans = find_all_spans(utt.split(), span_threshold)

        analysis_file.write('UTTERANCE: {}\n\n'.format(utt))
        analysis_file.write('GOLD SPANS: \n\t{}\n\n'.format('\n\t'.join([a[0] + ' ## ' + a[1] for a in ets])))

        for ent in entity_names:
            analysis_file.write('ENTITY NAME: {}\n\t'.format(ent))

            inputs = ['[CLS] ' + ent + ' [SEP] ' + s for s in spans]

            with torch.no_grad():
                input_tensor = tokenizer(inputs, return_tensors="pt", padding=True,
                                         add_special_tokens=False).to(device=device)
                scores = torch.sigmoid(model(**input_tensor).logits)

            spans_w_scores = list(zip(spans, list(scores.squeeze())))
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

        analysis_file.write('\n\n############################################\n\n')

        if i % log_every == 0:
            print('Processed {}/{} item(s) \t Time elapsed: {}'.format(i, tot, datetime.now() - start_time))

    analysis_file.close()

    if args.save_metrics:
        pickle.dump(metrics_counts, open(os.path.join(save_folder, 'ho_{}_ev_{}_{}_metrics.p'.format(held_out_intent,
                                                                                                     eval_intent,
                                                                                                     args.model_style)),
                                         'wb'))

    print('Done. Total time taken: {}'.format(datetime.now() - start_time))
