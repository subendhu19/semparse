from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModel
import argparse
import torch
from tqdm import tqdm

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

    train_data = zip(train_utt, train_can)

    with open(args.data_prefix + '.test.utterances.txt') as inf:
        test_utt = [l.strip() for l in inf.readlines()]

    with open(args.data_prefix + '.test.canonical.txt') as inf:
        test_can = [l.strip() for l in inf.readlines()]

    test_data = zip(test_utt, test_can)

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

        train_embeddings.append(outputs.last_hidden_state)

    print("Computing embeddings for test data...")
    for (utt, can) in tqdm(test_data):
        inputs = tokenizer(utt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        test_embeddings.append(outputs.last_hidden_state)

    train_embeddings = torch.stack(train_embeddings, dim=0)
    test_embeddings = torch.stack(test_embeddings, dim=0)

    relevance_scores = torch.cosine_similarity(train_embeddings, test_embeddings, dim=1)

    print(relevance_scores.shape)
