import os

data_folder = "/Users/subendhu/Downloads/on"
domains = ["basketball", "calendar", "blocks", "housing", "recipes", "publications", "restaurants", "socialnetwork"]

for split in ["test"]:
    for domain in domains:
        examples = []
        with open(os.path.join(data_folder, "{}.paraphrases.{}.examples".format(domain, split))) as inf:
            all_lines = inf.readlines()

        for i in range(0, len(all_lines), 8):
            item_lines = all_lines[i: i+8]
            utt = item_lines[1].strip()[12:-2]
            can = item_lines[2].strip()[11:-2]
            exr = item_lines[4].strip().replace("edu.stanford.nlp.sempre.overnight.SimpleWorld.", "")

            examples.append([utt, can, exr])

        with open(os.path.join(data_folder, "{}.{}.utterances.txt".format(domain, split)), "w") as out1:
            with open(os.path.join(data_folder, "{}.{}.canonical.txt".format(domain, split)), "w") as out2:
                with open(os.path.join(data_folder, "{}.{}.exr.txt".format(domain, split)), "w") as out3:
                    for ex in examples:
                        out1.write(ex[0] + '\n')
                        out2.write(ex[1] + '\n')
                        out3.write(ex[2] + '\n')
