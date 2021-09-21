import logging

import pandas as pd
from src.finetune.utils import ConstrainedSeq2SeqModel
from src.finetune.bart_s2s import get_vnt_function
from datetime import datetime

import argparse

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BART seq2seq model evaluation for semantic parsing")

    parser.add_argument('--data_prefix', type=str, default="/Users/subendhu/Documents/Amazon 2021"
                                                           "/jt_data/overnight/basketball/original/basketball")
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--out_folder', type=str, default="/Users/subendhu/Documents/Amazon 2021")
    parser.add_argument('--constrained_dict_path', type=str)
    parser.add_argument('--model_cp', type=str)

    args = parser.parse_args()

    out_folder = args.out_folder

    constraining_function = None
    if args.constrained_dict_path is not None:
        constraining_function = get_vnt_function(args.constrained_dict_path)

    with open(args.data_prefix + '.test.utterances.txt') as inf:
        test_utt = [l.strip() for l in inf.readlines()]

    with open(args.data_prefix + '.test.canonical.txt') as inf:
        test_can = [l.strip() for l in inf.readlines()]

    test_data = zip(test_utt, test_can)
    test_df = pd.DataFrame(test_data, columns=["input_text", "target_text"])

    print(test_df)

    model = ConstrainedSeq2SeqModel(
        constraining_function=constraining_function,
        encoder_decoder_type="bart",
        encoder_decoder_name="args.model_cp",
        use_cuda=args.gpu
    )

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
