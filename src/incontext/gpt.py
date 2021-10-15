from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModel
import argparse
import torch
from tqdm import tqdm, trange
import os


if __name__ == "__main__":
    # gen = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")
    # text_out = gen("I want one pizza with", do_sample=True, min_length=20)
    # print("Generated: {}".format(text_out["generated_text"]))

    parser = argparse.ArgumentParser(description="GPT in-context for semantic parsing")

    parser.add_argument('--data_prefix', type=str, default="/Users/subendhu/Documents/Amazon 2021"
                                                           "/jt_data/overnight/basketball/original/basketball")
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--out_folder', type=str, default="/Users/subendhu/Documents/Amazon 2021")
    parser.add_argument('--constrained_dict_path', type=str)

    args = parser.parse_args()

    out_folder = args.out_folder

    if args.gpu:
        if not torch.cuda.is_available():
            print('No GPU found. Quitting')
            exit(1)
        device = "cuda:{}".format(torch.cuda.current_device())
    else:
        device = "cpu"

    # Data Processing
    print("Loading data from files...")

    with open(args.data_prefix + '.train.utterances.txt') as inf:
        train_utt = [l.strip() for l in inf.readlines()]

    with open(args.data_prefix + '.train.canonical.txt') as inf:
        train_can = [l.strip() for l in inf.readlines()]

    train_data = list(zip(train_utt, train_can))

    with open(args.data_prefix + '.test.utterances.txt') as inf:
        test_utt = [l.strip() for l in inf.readlines()]

    with open(args.data_prefix + '.test.canonical.txt') as inf:
        test_can = [l.strip() for l in inf.readlines()]

    test_data = list(zip(test_utt, test_can))

    # Relevance scoring
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    model = AutoModel.from_pretrained('EleutherAI/gpt-neo-1.3B').to(device)
    model.eval()

    train_embeddings = []
    test_embeddings = []

    print("Computing relevance scores...")
    print("Computing embeddings for train data...")
    for (utt, can) in tqdm(train_data):
        inputs = tokenizer(utt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        train_embeddings.append(outputs.last_hidden_state[0][-1])

    print("Computing embeddings for test data...")
    for (utt, can) in tqdm(test_data):
        inputs = tokenizer(utt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        test_embeddings.append(outputs.last_hidden_state[0][-1])

    train_embeddings = torch.stack(train_embeddings, dim=0)
    test_embeddings = torch.stack(test_embeddings, dim=0)

    relevance_scores = [[torch.cosine_similarity(i.unsqueeze(0), j.unsqueeze(0)).item() for j in test_embeddings]
                        for i in train_embeddings]

    relevance_scores = torch.tensor(relevance_scores)

    print(relevance_scores.shape)

    # Test prompt creation
    k = 10
    prompts = []
    print("Creating test prompts...")
    for i in trange(len(test_data)):
        s = relevance_scores[i]
        s_e = list(zip(s, train_data))
        s_e.sort(key=lambda x: x[0])
        top_k = s_e[-k:]
        prompt = ''.join(["src: " + ex[1][0] + " tgt: " + ex[1][1] + "\n" for ex in top_k])
        prompt += "src: " + test_data[i][0] + " tgt: "
        prompts.append(prompt)

    with open(os.path.join(args.out_folder, 'prompts.txt'), 'w') as outf:
        for i in range(test_data):
            outf.write("INPUT:\n{}\n".format(test_data[i][0]))
            outf.write("PROMPT:\n{}\n\n".format(prompts[i]))

