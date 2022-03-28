import os
import argparse

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator

import random
import datasets
from datasets import concatenate_datasets


random.seed(1100)


pad_on_right = True
max_length = 48
doc_stride = 16


def prepare_train_features(examples):
    slot_questions = examples["question"]

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

    parser.add_argument('--data_folder', type=str, default='/home/srongali/data/top/qa')
    parser.add_argument('--save_folder', type=str, default='/mnt/nfs/scratch1/srongali/semparse/top')

    parser.add_argument('--held_out_domain', type=str, required=True)

    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--use_negative_examples', action='store_true')
    parser.add_argument('--neg_ex_pct', type=float, default=0.05)

    parser.add_argument('--model_checkpoint', type=str, default='deepset/roberta-base-squad2')

    args = parser.parse_args()

    model_checkpoint = args.model_checkpoint

    folder_name = args.data_folder

    domains = ['messaging', 'music', 'event', 'navigation', 'reminder', 'alarm', 'timer', 'weather']

    train_domains = [d for d in domains if d != args.held_out_domain]
    train_datasets = {}
    for d in train_domains:
        train_datasets[d] = datasets.load_from_disk(os.path.join(folder_name, d))

    print('Loaded Dataset Details:', flush=True)
    print(train_datasets)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    final_dataset = {'train': [], 'test': [], 'eval': []}

    d_train_neg_subsampled = {}
    if args.use_negative_examples:
        for d in train_datasets:
            d_train_neg_subsampled[d] = {}
            for split in final_dataset:
                d_neg_full = train_datasets[d][split + '_neg'].shuffle(seed=81)
                d_len = len(d_neg_full)
                max_ind = int(args.neg_ex_pct * d_len)
                d_neg_sub = d_neg_full.filter(lambda ex, ind: ind < max_ind, with_indices=True)
                d_neg_sub = d_neg_sub.cast(train_datasets[d][split].features)
                d_train_neg_subsampled[d][split] = d_neg_sub

    for split in final_dataset:
        all_parts = [train_datasets[d][split] for d in train_datasets]
        if args.use_negative_examples:
            all_parts += [d_train_neg_subsampled[d][split] for d in train_datasets]
        final_dataset[split] = concatenate_datasets(all_parts)

    final_dataset = datasets.DatasetDict(final_dataset)
    print('Final Training Dataset Details:', flush=True)
    print(final_dataset, flush=True)

    tokenized_datasets = final_dataset.map(prepare_train_features, batched=True,
                                           remove_columns=final_dataset["train"].column_names)

    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    model_name = 'qa_wo_{}_ngp_{}'.format(args.held_out_domain, args.neg_ex_pct if args.use_negative_examples else 0)

    model_name += '_' + args.model_checkpoint.split('/')[-1]

    targs = TrainingArguments(
        f"{args.save_folder}/{model_name}-checkpoints",
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
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
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(f"{args.save_folder}/{model_name}-best")









