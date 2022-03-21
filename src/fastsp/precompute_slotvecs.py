from src.fastsp.base_train import tag_entity_name_dict
from src.fastsp.utils import slot_descriptions

from transformers import BertTokenizerFast, BertModel
import torch
import pickle

slot_vecs = {intent: {'desc': '', 'no_desc': ''} for intent in tag_entity_name_dict}

device = "cuda:0"
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)
model.eval()

with torch.no_grad():
    for intent in slot_vecs:
        # No descriptions
        slot_list = [s for s in tag_entity_name_dict[intent]]
        slot_tensors = tokenizer(slot_list, return_tensors="pt", padding=True,
                                 add_special_tokens=True).to(device=model.device)
        slot_outs = model(**slot_tensors)
        slot_vectors = slot_outs['last_hidden_state'][:, 0, :]
        slot_vecs[intent]['no_desc'] = slot_vectors.cpu().T

        # Descriptions
        slot_list = [s for s in tag_entity_name_dict[intent]]
        for i in range(len(slot_list)):
            if slot_list[i] != "none":
                slot_list[i] += ' : ' + slot_descriptions[intent][slot_list[i]]
        slot_tensors = tokenizer(slot_list, return_tensors="pt", padding=True,
                                 add_special_tokens=True).to(device=model.device)
        slot_outs = model(**slot_tensors)
        slot_vectors = slot_outs['last_hidden_state'][:, 0, :]
        slot_vecs[intent]['desc'] = slot_vectors.cpu().T

pickle.dump(slot_vecs, 'slot_vecs.p')
