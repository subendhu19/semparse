import argparse
import statistics

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Metric computation")

    parser.add_argument('--gold_file', type=str, required=True)
    parser.add_argument('--pred_file', type=str, required=True)

    args = parser.parse_args()

    golds = []
    with open(args.gold_file) as inf:
        for line in inf:
            golds.append(line.strip().split('\t')[-1])

    preds = []
    with open(args.pred_file) as inf:
        for line in inf:
            words = line.strip().split(' ')
            if words[0] == '<START>':
                words = words[1:]
            preds.append(' '.join(words))

    gold_spans = []
    pred_spans = []

    for lf in golds:
        words = lf.split()
        ent_spans = {}
        ent_count = 0
        ent_list = []
        tok_count = 0
        for word in words:
            if '[' in word:
                ent_id = word[1:] + '_{}'.format(ent_count)
                ent_spans[ent_id] = {'toks': [], 'done': False, 'start_tok': -1}
                ent_list.append(ent_id)
                ent_count += 1
            elif ']' in word:
                ent_spans[ent_list[-1]]['done'] = True
                ent_list = ent_list[:-1]
            else:
                start_tok = tok_count
                for e in ent_list:
                    if not ent_spans[e]['done']:
                        ent_spans[e]['toks'] += word
                        if ent_spans[e]['start_tok'] == -1:
                            ent_spans[e]['start_tok'] = start_tok
                tok_count += 1

        gold_spans.append([('_'.join(k.split('_')[:-1]), ' '.join(ent_spans[k]['toks']),
                            ent_spans[k]['start_tok']) for k in ent_spans])

    for lf in preds:
        words = lf.split()
        ent_spans = {}
        ent_count = 0
        ent_list = []
        tok_count = 0
        for word in words:
            if '[' in word:
                ent_id = word[1:] + '_{}'.format(ent_count)
                ent_spans[ent_id] = {'toks': [], 'done': False, 'start_tok': -1}
                ent_list.append(ent_id)
                ent_count += 1
            elif ']' in word:
                ent_spans[ent_list[-1]]['done'] = True
                ent_list = ent_list[:-1]
            else:
                start_tok = tok_count
                for e in ent_list:
                    if not ent_spans[e]['done']:
                        ent_spans[e]['toks'] += word
                        if ent_spans[e]['start_tok'] == -1:
                            ent_spans[e]['start_tok'] = start_tok
                tok_count += 1

        pred_spans.append([('_'.join(k.split('_')[:-1]), ' '.join(ent_spans[k]['toks']),
                            ent_spans[k]['start_tok']) for k in ent_spans])

    em_n = 0
    em_d = 0
    precision_n = 0
    precision_d = 0
    recall_n = 0
    recall_d = 0

    for eid in range(len(golds)):
        if golds[eid] == preds[eid]:
            em_n += 1
        em_d += 1

        for p in pred_spans[eid]:
            if p in gold_spans[eid]:
                precision_n += 1
            precision_d += 1

        for p in gold_spans[eid]:
            if p in pred_spans[eid]:
                recall_n += 1
        recall_d += 1

    em = em_n / em_d * 100.0
    precision = precision_n / precision_d * 100.0
    recall = recall_n / recall_d * 100.0
    print('Precision: {:.2f}'.format(precision))
    print('Recall: {:.2f}'.format(recall))
    print('F1: {:.2f}'.format(statistics.harmonic_mean([precision, recall])))
    print('EM: {:.2f}'.format(em))