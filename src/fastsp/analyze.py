import torch
import os
import pickle
import argparse

from transformers import BertTokenizer, BertForSequenceClassification


def find_all_spans(words, threshold):
    all_spans = []
    for i in range(1, threshold):
        for j in range(0, len(words) - i + 1):
            all_spans.append(' '.join(words[j:j+i]))
    return all_spans


entity_name_dict = {
    "AddtoPlaylist": ["music item", "entity name", "playlist", "artist", "playlist owner"]
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training models for fast semantic parsing")

    parser.add_argument('--data_folder', type=str, default='/home/srongali/data/snips')
    parser.add_argument('--save_folder', type=str, default='/mnt/nfs/scratch1/srongali/semparse/snips')

    parser.add_argument('--eval_intent', type=str, required=True)
    parser.add_argument('--held_out_intent', type=str, required=True)

    parser.add_argument('--span_threshold', type=int, default=6)

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Data
    data_folder = args.data_folder
    save_folder = args.save_folder

    val_data = pickle.load(open(os.path.join(data_folder, 'val_data.p'), 'rb'))

    held_out_intent = args.held_out_intent
    eval_intent = args.held_out_intent
    span_threshold = args.span_threshold
    device = "cuda:0"

    analysis_file = open(os.path.join(save_folder, 'ho_{}_ev_{}_analysis.txt'.format(held_out_intent, eval_intent)), 'w')

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1).to(device)
    model.load_state_dict(torch.load(os.path.join(save_folder, 'bert_wo_{}.pt'.
                                                  format(held_out_intent)))['model_state_dict'])
    model.eval()

    print('Saved model loaded.')

    entity_names = entity_name_dict[eval_intent]

    tot = len(val_data[held_out_intent]['utterances'])
    # tot = 10

    for i in range(tot):
        utt = val_data[held_out_intent]['utterances'][i]
        ets = val_data[held_out_intent]['entities'][i]

        spans = find_all_spans(utt.split(), span_threshold)

        analysis_file.write('UTTERANCE: {}\n\n'.format(utt))

        analysis_file.write('GOLD SPANS: \n\t{}\n\n'.format('\n\t'.join([a[0] + ' ## ' + a[1] for a in ets])))

        for ent in entity_names:
            analysis_file.write('ENTITY NAME: {}\n\t'.format(ent))

            inputs = ['[CLS] ' + ent + ' [SEP] ' + s for s in spans]

            with torch.no_grad():
                input_tensor = tokenizer(inputs, return_tensors="pt", padding=True).to(device=device)
                scores = torch.sigmoid(model(**input_tensor).logits)

            spans_w_scores = list(zip(spans, list(scores.squeeze())))
            spans_w_scores.sort(key=lambda x: x[1], reverse=True)

            analysis_file.write('\n\t'.join([a[0] + ' ## ' + str(a[1]) for a in spans_w_scores[:5]]))

        analysis_file.write('\n\n############################################\n\n')

        print('Processed {} item(s)'.format(i))

    analysis_file.close()
