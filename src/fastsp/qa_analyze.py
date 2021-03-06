import os
import argparse

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator

import random
from src.fastsp.utils import slot_descriptions

from src.fastsp.train import entity_name_dict
from src.fastsp.qa_train import question_prefix
import collections
import numpy as np
from datasets import Dataset
import statistics
import json


random.seed(1100)

pad_on_right = True
use_descriptions = False
max_length = 48
doc_stride = 16
squad_v2 = False
threshold = 0

pose_question = False


def prepare_validation_features(examples):
    slot_questions = examples["question"]
    og_slot_questions = [a for a in slot_questions]

    if pose_question:
        mod_slot_questions = [question_prefix(s) + s for s in og_slot_questions]
        slot_questions = mod_slot_questions

    if use_descriptions:
        intents = examples["title"]
        descriptions = [slot_descriptions[intents[i]][og_slot_questions[i]] for i in range(len(og_slot_questions))]
        mod_slot_questions = [slot_questions[i] + ' : ' + descriptions[i] for i in range(len(slot_questions))]
        slot_questions = mod_slot_questions

    tokenized_examples = tokenizer(
        slot_questions if pad_on_right else examples["context"],
        examples["context"] if pad_on_right else slot_questions,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()
    original_predictions = collections.OrderedDict()

    # Logging.
    # print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(examples):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None  # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char],
                            "start_char": start_char,
                            "end_char": end_char
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0, "start_char": 0, "end_char": 0}

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)

        if example["original_id"] not in original_predictions:
            original_predictions[example["original_id"]] = []

        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
            original_predictions[example["original_id"]].append((example["question"],
                                                                 best_answer["text"],
                                                                 best_answer["start_char"],
                                                                 best_answer["end_char"],
                                                                 best_answer["score"]))
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            ascore = best_answer["score"] if best_answer["score"] > min_null_score else min_null_score
            predictions[example["id"]] = answer
            original_predictions[example["original_id"]].append((example["question"],
                                                                 answer,
                                                                 best_answer["start_char"],
                                                                 best_answer["end_char"],
                                                                 ascore))

    return predictions, original_predictions


def check_invalid(spans, span):
    for cspan in spans:
        if cspan[2] <= span[2] < cspan[3]:
            return True
        if cspan[2] < span[3] <= cspan[3]:
            return True
        if span[2] <= cspan[2] and span[3] >= cspan[3]:
            return True
        if span[0] == cspan[0]:
            return True

    if span[-1] < threshold:
        return True

    if span[1] == "":
        return True

    return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyzing QA models for fast semantic parsing")

    parser.add_argument('--data_folder', type=str, default='/home/srongali/data/snips/raw')
    parser.add_argument('--save_folder', type=str, default='/mnt/nfs/scratch1/srongali/semparse/snips')

    parser.add_argument('--held_out_intent', type=str, required=True)
    parser.add_argument('--train_held_out_intent', type=str)
    parser.add_argument('--use_descriptions', action='store_true')
    parser.add_argument('--use_negative_examples', action='store_true')
    parser.add_argument('--score_threshold', type=int, default=0)

    parser.add_argument('--override_model_checkpoint', type=str)

    parser.add_argument('--pose_question', action='store_true')

    args = parser.parse_args()

    intent = args.held_out_intent
    train_intent = args.train_held_out_intent if args.train_held_out_intent else intent
    threshold = args.score_threshold

    pose_question = args.pose_question

    model_name = 'qa_wo_{}'.format(train_intent)
    if args.use_descriptions:
        model_name += '_desc'
    if args.use_negative_examples:
        model_name += '_neg'
    if args.pose_question:
        model_name += '_pq'
    model_name += '-best'

    model_checkpoint = os.path.join(args.save_folder, model_name)

    if args.override_model_checkpoint:
        model_checkpoint = args.override_model_checkpoint

    val_json = json.load(open(os.path.join(args.data_folder,
                                           'validate_{}.json'.format(intent)), 'rb'))

    example_id = 0
    original_id = 0
    val_qa_data = {'question': [], 'answers': [], 'context': [], 'id': [], 'title': [], 'original_id': []}
    gold_entities = {}
    for i in range(len(val_json[intent])):
        text_so_far = ""
        context = ''.join([a['text'] for a in val_json[intent][i]['data']])

        for ent in entity_name_dict[intent]:
            val_qa_data['question'].append(ent)
            val_qa_data['answers'].append({'answer_start': [0], 'text': [""]})
            val_qa_data['context'].append(context)
            val_qa_data['id'].append(example_id)
            val_qa_data['original_id'].append(original_id)
            val_qa_data['title'].append(intent)

            example_id += 1

        for a in val_json[intent][i]['data']:
            if 'entity' in a:
                start_idx = len(text_so_far)
                if original_id not in gold_entities:
                    gold_entities[original_id] = []

                gold_entities[original_id].append((a['entity'].replace('_', ' '), a['text'], start_idx))

            text_so_far = text_so_far + a['text']

        original_id += 1

    val_dataset = Dataset.from_dict(val_qa_data)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    if args.use_descriptions:
        max_length = 64
        use_descriptions = True

    if args.use_negative_examples:
        squad_v2 = True

    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    targs = TrainingArguments(
        f"{args.save_folder}/{model_name}",
        per_device_eval_batch_size=64,
        push_to_hub=False,
    )

    data_collator = default_data_collator

    trainer = Trainer(
        model,
        targs,
        tokenizer=tokenizer,
    )

    validation_features = val_dataset.map(
        prepare_validation_features,
        batched=True,
        remove_columns=val_dataset.column_names
    )

    raw_predictions = trainer.predict(validation_features)

    validation_features.set_format(type=validation_features.format["type"],
                                   columns=list(validation_features.features.keys()))

    final_predictions, final_original_predictions = postprocess_qa_predictions(val_dataset,
                                                                               validation_features,
                                                                               raw_predictions.predictions)

    metric_gold = []
    metric_preds = []

    for k in final_original_predictions:
        # print('Example ID: {}'.format(k))
        # print('GOLD: ')
        # print(gold_entities[k])
        # print()
        # print('PRED: ')

        greedy_decode = []
        sorted_preds = sorted(final_original_predictions[k], key=lambda x: x[-1], reverse=True)
        for sp in sorted_preds:
            if check_invalid(greedy_decode, sp) is False:
                greedy_decode.append(sp)

        # for pred_ent in greedy_decode:
        #     print(pred_ent)

        metric_gold.append([(a[0], a[1]) for a in gold_entities[k]])
        metric_preds.append([(a[0], a[1]) for a in greedy_decode])

        # print()

    precision_n = 0
    precision_d = 0
    recall_n = 0
    recall_d = 0

    for eid in range(len(metric_gold)):

        for p in metric_preds[eid]:
            if p in metric_gold[eid]:
                precision_n += 1
            precision_d += 1

        for p in metric_gold[eid]:
            if p in metric_preds[eid]:
                recall_n += 1
            recall_d += 1

    precision = precision_n / precision_d * 100.0
    recall = recall_n / recall_d * 100.0
    print('Precision: {:.2f}'.format(precision))
    print('Recall: {:.2f}'.format(recall))
    print('F1: {:.2f}'.format(statistics.harmonic_mean([precision, recall])))



