import os
from datetime import datetime
import logging

import pandas as pd
from simpletransformers.seq2seq import Seq2SeqArgs
from src.finetune.utils import ConstrainedSeq2SeqModel
import pickle
import torch

import argparse

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)


def get_vnt_function(dict_path):
    vnt_dict = pickle.load(open(dict_path, "rb"))

    def get_valid_tokens(batch_id: int, prefix: torch.Tensor):
        key = " ## ".join(list(prefix.numpy()))
        if key in vnt_dict:
            return vnt_dict[key]
        else:
            return [2]

    return get_valid_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BART seq2seq model training for semantic parsing")

    parser.add_argument('--data_prefix', type=str, default="/Users/subendhu/Documents/Amazon 2021"
                                                           "/jt_data/overnight/basketball/original/basketball")
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--out_folder', type=str, default="/Users/subendhu/Documents/Amazon 2021")
    parser.add_argument('--constrained_dict_path', type=str)

    args = parser.parse_args()

    out_folder = args.out_folder

    # Data Processing
    with open(args.data_prefix + '.train.utterances.txt') as inf:
        train_utt = [l.strip() for l in inf.readlines()]

    with open(args.data_prefix + '.train.canonical.txt') as inf:
        train_can = [l.strip() for l in inf.readlines()]

    train_data = zip(train_utt, train_can)
    train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

    with open(args.data_prefix + '.test.utterances.txt') as inf:
        test_utt = [l.strip() for l in inf.readlines()]

    with open(args.data_prefix + '.test.canonical.txt') as inf:
        test_can = [l.strip() for l in inf.readlines()]

    test_data = zip(test_utt, test_can)
    test_df = pd.DataFrame(test_data, columns=["input_text", "target_text"])

    print(train_df)

    print(test_df)

    # Model Training
    model_args = Seq2SeqArgs()
    model_args.do_sample = False
    model_args.eval_batch_size = 32
    model_args.evaluate_during_training = True
    # model_args.evaluate_during_training_steps = 500
    model_args.evaluate_during_training_verbose = True
    model_args.fp16 = False
    model_args.learning_rate = 5e-5
    model_args.max_length = 64
    model_args.max_seq_length = 64
    model_args.num_beams = 4
    model_args.num_return_sequences = 1
    model_args.num_train_epochs = 5
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.save_eval_checkpoints = False
    model_args.save_steps = -1
    model_args.top_k = 50
    model_args.top_p = 0.95
    model_args.train_batch_size = 8
    model_args.use_multiprocessing = False
    constraining_function = None
    if args.constrained_dict_path is not None:
        constraining_function = get_vnt_function(args.constrained_dict_path)
    # model_args.wandb_project = "Semantic Parsing with BART"

    model = ConstrainedSeq2SeqModel(
        constraining_function=constraining_function,
        encoder_decoder_type="bart",
        encoder_decoder_name="facebook/bart-large",
        args=model_args,
        use_cuda=args.gpu
    )

    model.train_model(train_df, eval_data=test_df)

    to_predict = test_df["input_text"].tolist()
    truth = test_df["target_text"].tolist()

    preds = model.predict(to_predict)

    with open(f"{out_folder}/predictions_{datetime.now()}.txt", "w") as f:
        for i, text in enumerate(test_df["input_text"].tolist()):
            f.write(str(text) + "\n\n")

            f.write("Truth:\n")
            f.write(truth[i] + "\n\n")

            f.write("Prediction:\n")
            f.write(str(preds[i]) + "\n")
            f.write(
                "________________________________________________________________________________\n"
            )
