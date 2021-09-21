import warnings

import pandas as pd

from simpletransformers.seq2seq import Seq2SeqModel
import torch
from tqdm.auto import tqdm
import logging
from multiprocessing import Pool


logger = logging.getLogger(__name__)


def load_data(
    file_path, input_text_column, target_text_column, label_column, keep_label=1
):
    df = pd.read_csv(file_path, sep="\t", error_bad_lines=False)
    df = df.loc[df[label_column] == keep_label]
    df = df.rename(
        columns={input_text_column: "input_text", target_text_column: "target_text"}
    )
    df = df[["input_text", "target_text"]]
    df["prefix"] = "paraphrase"

    return df


def clean_unnecessary_spaces(out_string):
    if not isinstance(out_string, str):
        warnings.warn(f">>> {out_string} <<< is not a string.")
        out_string = str(out_string)
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string


class ConstrainedSeq2SeqModel(Seq2SeqModel):

    def __init__(self, constraining_function: None, **kwargs):
        super().__init__(**kwargs)
        self.constraining_function = constraining_function

    def predict(self, to_predict):
        """
        Performs constrained predictions on a list of text.
        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction. Note that the prefix should be prepended to the text.
        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        self._move_model_to_device()

        all_outputs = []
        all_retrieved = []
        all_doc_scores = []
        # Batching
        for batch in tqdm(
                [
                    to_predict[i: i + self.args.eval_batch_size]
                    for i in range(0, len(to_predict), self.args.eval_batch_size)
                ],
                desc="Generating outputs",
                disable=self.args.silent,
        ):
            if self.args.model_type == "marian":
                input_ids = self.encoder_tokenizer.prepare_seq2seq_batch(
                    batch,
                    max_length=self.args.max_seq_length,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                )["input_ids"]
            elif self.args.model_type in ["mbart"]:
                input_ids = self.encoder_tokenizer.prepare_seq2seq_batch(
                    src_texts=batch,
                    max_length=self.args.max_seq_length,
                    pad_to_max_length=True,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                    src_lang=self.args.src_lang,
                )["input_ids"]
            elif self.args.model_type in ["rag-token", "rag-sequence"]:
                input_ids = self.encoder_tokenizer(
                    batch, truncation=True, padding="longest", return_tensors="pt"
                )["input_ids"].to(self.device)

                question_hidden_states = self.model.question_encoder(input_ids)[0]

                docs_dict = self.retriever(
                    input_ids.cpu().numpy(),
                    question_hidden_states.detach().cpu().numpy(),
                    return_tensors="pt",
                )
                doc_scores = torch.bmm(
                    question_hidden_states.unsqueeze(1),
                    docs_dict["retrieved_doc_embeds"]
                        .float()
                        .transpose(1, 2)
                        .to(self.device),
                ).squeeze(1)
            else:
                input_ids = self.encoder_tokenizer.batch_encode_plus(
                    batch,
                    max_length=self.args.max_seq_length,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                )["input_ids"]
            input_ids = input_ids.to(self.device)

            if self.args.model_type in ["bart", "marian"]:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    early_stopping=self.args.early_stopping,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    num_return_sequences=self.args.num_return_sequences,
                    prefix_allowed_tokens_fn=self.constraining_function
                )
            elif self.args.model_type in ["mbart"]:
                tgt_lang_token = self.decoder_tokenizer._convert_token_to_id(
                    self.args.tgt_lang
                )

                outputs = self.model.generate(
                    input_ids=input_ids,
                    decoder_start_token_id=tgt_lang_token,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    early_stopping=self.args.early_stopping,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    num_return_sequences=self.args.num_return_sequences,
                    prefix_allowed_tokens_fn=self.constraining_function
                )
            elif self.args.model_type in ["rag-token", "rag-sequence"]:
                outputs = self.model.generate(
                    context_input_ids=docs_dict["context_input_ids"].to(self.device),
                    context_attention_mask=docs_dict["context_attention_mask"].to(
                        self.device
                    ),
                    doc_scores=doc_scores,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    early_stopping=self.args.early_stopping,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    num_return_sequences=self.args.num_return_sequences,
                    prefix_allowed_tokens_fn=self.constraining_function
                )
                retrieved_docs = [
                    doc
                    for doc in self.retriever.index.get_doc_dicts(docs_dict["doc_ids"])
                ]
            else:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    decoder_start_token_id=self.model.config.decoder.pad_token_id,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    early_stopping=self.args.early_stopping,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    num_return_sequences=self.args.num_return_sequences,
                    prefix_allowed_tokens_fn=self.constraining_function
                )

            all_outputs.extend(outputs.cpu().numpy())
            if self.args.model_type in ["rag-token", "rag-sequence"]:
                all_retrieved.extend(retrieved_docs)
                all_doc_scores.extend(doc_scores.detach().cpu())

        if self.args.model_type in ["rag-token", "rag-sequence"]:
            outputs = self.encoder_tokenizer.batch_decode(
                all_outputs,
                skip_special_tokens=self.args.skip_special_tokens,
                clean_up_tokenization_spaces=True,
            )
        elif self.args.use_multiprocessed_decoding:
            if self.args.multiprocessing_chunksize == -1:
                chunksize = max(len(all_outputs) // (self.args.process_count * 2), 500)
            else:
                chunksize = self.args.multiprocessing_chunksize

            self.model.to("cpu")
            with Pool(self.args.process_count) as p:
                outputs = list(
                    tqdm(
                        p.imap(self._decode, all_outputs, chunksize=chunksize),
                        total=len(all_outputs),
                        desc="Decoding outputs",
                        disable=self.args.silent,
                    )
                )
            self._move_model_to_device()
        else:
            outputs = [
                self.decoder_tokenizer.decode(
                    output_id,
                    skip_special_tokens=self.args.skip_special_tokens,
                    clean_up_tokenization_spaces=True,
                )
                for output_id in all_outputs
            ]

        if self.args.num_return_sequences > 1:
            if self.args.model_type in ["rag-token", "rag-sequence"]:
                return (
                    [
                        outputs[i: i + self.args.num_return_sequences]
                        for i in range(0, len(outputs), self.args.num_return_sequences)
                    ],
                    [
                        all_retrieved[i: i + self.args.num_return_sequences]
                        for i in range(0, len(outputs), self.args.num_return_sequences)
                    ],
                    [
                        all_doc_scores[i: i + self.args.num_return_sequences]
                        for i in range(0, len(outputs), self.args.num_return_sequences)
                    ],
                )
            else:
                return [
                    outputs[i: i + self.args.num_return_sequences]
                    for i in range(0, len(outputs), self.args.num_return_sequences)
                ]
        else:
            if self.args.model_type in ["rag-token", "rag-sequence"]:
                return outputs, all_retrieved, all_doc_scores
            else:
                return outputs


