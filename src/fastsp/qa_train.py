import os
import argparse

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator

import random
from src.fastsp.utils import slot_descriptions
import datasets


random.seed(1100)


pad_on_right = True
use_descriptions = False
max_length = 48
doc_stride = 16

pose_question = False


def question_prefix(slot_val):
    if slot_val in ['artist', 'playlist owner']:
        return 'who is the '
    else:
        return 'what is the '


def prepare_train_features(examples):
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
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                prev_index = -1
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    if prev_index == start_char and offsets[token_start_index][0] == 0:
                        break
                    prev_index = offsets[token_start_index][0]
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training QA models for fast semantic parsing")

    parser.add_argument('--data_folder', type=str, default='/home/srongali/data/snips/qa')
    parser.add_argument('--save_folder', type=str, default='/mnt/nfs/scratch1/srongali/semparse/snips')

    parser.add_argument('--held_out_intent', type=str, required=True)

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--use_descriptions', action='store_true')

    parser.add_argument('--use_negative_examples', action='store_true')

    parser.add_argument('--model_checkpoint', type=str, default='bert-base-uncased')

    parser.add_argument('--pose_question', action='store_true')

    args = parser.parse_args()

    model_checkpoint = args.model_checkpoint
    pose_question = args.pose_question

    folder_name = args.data_folder
    if args.use_negative_examples:
        folder_name += "-neg"

    ood_dataset = datasets.load_from_disk(os.path.join(folder_name, args.held_out_intent, 'ood'))
    sd_dataset = datasets.load_from_disk(os.path.join(folder_name, args.held_out_intent, 'sd'))

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    if args.use_descriptions:
        max_length = 64
        use_descriptions = True

    ood_tokenized_datasets = ood_dataset.map(prepare_train_features, batched=True,
                                             remove_columns=ood_dataset["train"].column_names)

    sd_tokenized_datasets = sd_dataset.map(prepare_train_features, batched=True,
                                           remove_columns=sd_dataset["train"].column_names)

    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    model_name = 'qa_wo_{}'.format(args.held_out_intent)
    if args.use_descriptions:
        model_name += '_desc'
    if args.use_negative_examples:
        model_name += '_neg'
    if args.pose_question:
        model_name += '_pq'

    if args.model_checkpoint != 'bert-base-uncased':
        model_name += args.model_checkpoint.split('/')[-1]

    targs = TrainingArguments(
        f"{args.save_folder}/{model_name}-checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        push_to_hub=False,
        load_best_model_at_end=True
    )

    data_collator = default_data_collator

    trainer = Trainer(
        model,
        targs,
        train_dataset=ood_tokenized_datasets["train"],
        eval_dataset=ood_tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(f"{args.save_folder}/{model_name}-best")









