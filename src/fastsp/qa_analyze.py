import torch
import os
import argparse

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator

import random
from src.fastsp.utils import slot_descriptions
import datasets

from src.fastsp.qa_train import tag_entity_name_dict, prepare_train_features
import collections
from tqdm.auto import tqdm
import numpy as np
from datasets import load_metric
import pickle


random.seed(1100)

pad_on_right = True
use_descriptions = False
max_length = 48
doc_stride = 16
squad_v2 = False


def prepare_validation_features(examples):
    slot_questions = [e.replace('_', ' ') for e in examples["question"]]

    if use_descriptions:
        intents = examples["title"]
        descriptions = [slot_descriptions[intents[i]][slot_questions[i]] for i in range(len(slot_questions))]
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

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
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
                            "text": context[start_char: end_char]
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyzing QA models for fast semantic parsing")

    parser.add_argument('--data_folder', type=str, default='/home/srongali/data/snips/qa')
    parser.add_argument('--save_folder', type=str, default='/mnt/nfs/scratch1/srongali/semparse/snips')
    parser.add_argument('--checkpoint_name', type=str, default='bert-base-uncased-finetuned-squad')

    parser.add_argument('--held_out_intent', type=str, required=True)
    parser.add_argument('--use_descriptions', action='store_true')
    parser.add_argument('--use_negative_examples', action='store_true')

    args = parser.parse_args()

    model_checkpoint = os.path.join(args.save_folder, args.checkpoint_name, 'checkpoint-500')

    ood_dataset = datasets.load_from_disk(os.path.join(args.data_folder, args.held_out_intent, 'ood'))
    sd_dataset = datasets.load_from_disk(os.path.join(args.data_folder, args.held_out_intent, 'sd'))

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    if args.use_descriptions:
        max_length = 64
        use_descriptions = True

    if args.use_negative_examples:
        squad_v2 = True

    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    model_name = args.checkpoint_name
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

    validation_features = ood_dataset["validation"].map(
        prepare_validation_features,
        batched=True,
        remove_columns=ood_dataset["validation"].column_names
    )

    raw_predictions = trainer.predict(validation_features)

    validation_features.set_format(type=validation_features.format["type"],
                                   columns=list(validation_features.features.keys()))

    final_predictions = postprocess_qa_predictions(ood_dataset["validation"], validation_features,
                                                   raw_predictions.predictions)

    metric = load_metric("squad_v2" if squad_v2 else "squad")

    if squad_v2:
        formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in
                                 final_predictions.items()]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in ood_dataset["validation"]]
    # metric.compute(predictions=formatted_predictions, references=references)

    pickle.dump(formatted_predictions, open(os.path.join(args.save_folder, 'preds.p'), 'wb'))
    pickle.dump(references, open(os.path.join(args.save_folder, 'refs.p'), 'wb'))

