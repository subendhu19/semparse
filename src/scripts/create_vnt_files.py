import os
from transformers import BartTokenizerFast
import pickle

data_folder = "/home/srongali/data/jt_data/overnight"

domains = ["basketball", "calendar", "blocks", "housing", "recipes", "publications", "restaurants", "socialnetwork"]

tok = BartTokenizerFast.from_pretrained("facebook/bart-large")

for domain in domains:

    vnt_dict = {}

    train_file = os.path.join(data_folder, domain, "original", "{}.train.canonical.txt".format(domain))
    test_file = os.path.join(data_folder, domain, "original", "{}.test.canonical.txt".format(domain))

    all_cfs = []

    with open(train_file) as inf:
        for line in inf:
            all_cfs.append(line.strip())

    with open(test_file) as inf:
        for line in inf:
            all_cfs.append(line.strip())

    for cf in all_cfs:
        tokens = [2] + tok(cf)["input_ids"][1:-1]

        for i in range(1, len(tokens)):
            key = " ## ".join([str(t) for t in tokens[:i]])
            val = tokens[i]
            if key not in vnt_dict:
                vnt_dict[key] = [val]
            else:
                if val not in vnt_dict[key]:
                    vnt_dict[key].append(val)

    pickle.dump(vnt_dict, open(os.path.join(data_folder, "{}.vnt_dict.p".format(domain)), "wb"))
