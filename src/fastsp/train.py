import torch
import os
import pickle

from transformers import BertTokenizer, BertForSequenceClassification

from random import shuffle

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Model params
MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)


# Data
data_folder = '/home/srongali/data/snips'
save_folder = '/mnt/nfs/scratch1/srongali/semparse/snips'

margin_train_data = pickle.load(open(os.path.join(data_folder, 'margin_train_data.p'), 'rb'))
intents = list(margin_train_data.keys())


held_out_intent = 'AddToPlaylist'
train_intents = [i for i in intents if i != held_out_intent]

train_processed = []

for intent in train_intents:
    for ex in margin_train_data[intent]:
        train_processed.append(['[CLS] ' + ex[0] + ' [SEP] ' + ex[1], '[CLS] ' + ex[0] + ' [SEP] ' + ex[2]])


epochs = 10
batch_size = 64
log_every = 10
device = "cuda:0"

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

margin = 0.7

print('Begin Training...')
model.train()
for epoch in range(epochs):
    shuffle(train_processed)

    update = 0
    total_updates = int(len(train_processed) / batch_size)
    for i in range(0, len(train_processed), batch_size):
        mini_batch = train_processed[i:i+batch_size]

        pos_ex = [a[0] for a in mini_batch]
        neg_ex = [a[1] for a in mini_batch]

        pos_tensors = tokenizer(pos_ex, return_tensors="pt", padding=True).to(device=device)
        neg_tensors = tokenizer(neg_ex, return_tensors="pt", padding=True).to(device=device)

        pos_outputs = model(**pos_tensors)
        neg_outputs = model(**neg_tensors)

        pos_scores = torch.sigmoid(pos_outputs.logits)
        neg_scores = torch.sigmoid(neg_outputs.logits)

        margin_scores = torch.max(torch.zeros_like(pos_scores), margin - pos_scores + neg_scores)

        loss = margin_scores.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update += 1

        if update % log_every == 0:
            print("Epoch: {}/{} \t Update: {}/{} \t Loss: {}".format(epoch, epochs, update, total_updates, loss.item()))

print('Done')

state_dict = {'model_state_dict': model.state_dict()}
save_path = os.path.join(save_folder, 'bert_wo_' + held_out_intent + '.pt')
torch.save(state_dict, save_path)
print('Checkpoint saved to {}'.format(save_path))







