import torch
import os
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from tqdm import tqdm

from src.fastsp.cs2s_train import process_s2s_data, CustomSeq2Seq, target_vocab, get_slot_expression, mean_pooling

schema = {}


def post_process(tgt_ids, input_ids, dom, tok):
    decoded = []
    for id in tgt_ids:
        if id == target_vocab.index('<START>'):
            continue
        if id == target_vocab.index('<END>'):
            break
        else:
            if id < len(target_vocab):
                dec_tok = target_vocab[id]
                if '@ptr' in dec_tok:
                    point_tok = input_ids[id - 3]
                    decoded.append(tok.convert_ids_to_tokens(point_tok.item()))
                else:
                    decoded.append(dec_tok)
            else:
                tags = schema[dom]['intents'] + schema[dom]['slots']
                decoded.append(tags[id - 67])

    return ' '.join(decoded)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training models for fast semantic parsing")

    parser.add_argument('--data_folder', type=str, default='/home/srongali/data/top/seq2seq')
    parser.add_argument('--save_folder', type=str, default='/mnt/nfs/scratch1/srongali/semparse/top')

    parser.add_argument('--eval_domain', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--enc_checkpoint', type=str, default='roberta-base')
    parser.add_argument('--model_checkpoint', type=str, required=True)
    parser.add_argument('--use_span_encoder', action='store_true')
    parser.add_argument('--span_encoder_checkpoint', type=str, default='bert-base-uncased')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.enc_checkpoint)

    data_folder = args.data_folder
    save_folder = args.save_folder
    device = "cuda:0"

    schema = pickle.load(open(os.path.join(data_folder, 'schema.p'), 'rb'))

    eval_processed = process_s2s_data(data_folder, [args.eval_domain], 'test', args.batch_size,
                                      tokenizer, schema)

    encoder = AutoModel.from_pretrained(args.enc_checkpoint).to(device)
    d_model = encoder.config.hidden_size
    decoder = TransformerDecoder(TransformerDecoderLayer(d_model=d_model, nhead=8, batch_first=True),
                                 num_layers=6).to(device)

    tag_model = None
    tag_tokenizer = None
    tag_encoder = None

    if args.use_span_encoder:
        tag_model = args.span_encoder_checkpoint
        tag_tokenizer = AutoTokenizer.from_pretrained(tag_model)
        tag_encoder = AutoModel.from_pretrained(tag_model).to(encoder.device)

    if tag_tokenizer is None:
        tag_tokenizer = AutoTokenizer.from_pretrained(args.enc_checkpoint)
        tag_encoder = AutoModel.from_pretrained(args.enc_checkpoint).to(encoder.device)

    tag_list = [get_slot_expression(a) for a in schema[args.eval_domain]['intents'] +
                schema[args.eval_domain]['slots']]
    tag_tensors = tag_tokenizer(tag_list, return_tensors="pt", padding=True,
                                add_special_tokens=False).to(device=tag_encoder.device)
    tag_encoder.eval()
    with torch.no_grad():
        tag_outs = tag_encoder(**tag_tensors)

    if tag_model[:4] == 'bert':
        tag_embeddings = tag_outs['last_hidden_state'][:, 0, :]
    else:
        tag_embeddings = mean_pooling(tag_outs, tag_tensors['attention_mask'])

    model = CustomSeq2Seq(enc=encoder, dec=decoder, schema=schema, tag_model=tag_model,
                          tag_embeddings={args.eval_domain: tag_embeddings})
    model.load_state_dict(torch.load(os.path.join(args.model_checkpoint))['model_state_dict'])
    model.eval()
    print('Model loaded. Beginning evaluation. Num batches: {}'.format(len(eval_processed)),
          flush=True)

    mname = args.model_checkpoint.split('/')[-1]
    out_file = open(os.path.join(save_folder, '{}_{}_preds.txt'.format(args.eval_domain, mname)), 'w')

    for i in tqdm(range(len(eval_processed))):
        inp, tgt, domain = eval_processed[i]
        inp = inp.to(device=device)
        tgt = tgt.to(device=device)

        with torch.no_grad():
            preds = model(inp, tgt, domain, decode=True)

        for j in range(len(preds)):
            out_file.write(post_process(preds[j][0], inp['input_ids'][j], domain, tokenizer) + '\n')

    out_file.close()
