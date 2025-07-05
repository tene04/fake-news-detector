import numpy as np
import torch

from torch.utils.data import DataLoader
from utils.dataset import TextDataset
from utils.functions import *


# load all the data
bert_data = np.load('../data/bert.npz', allow_pickle=True)

bert_train_encodings = {
    'input_ids': bert_data['train_input_ids'].tolist(),
    'attention_mask': bert_data['train_attention_mask'].tolist()
}
bert_val_encodings = {
    'input_ids': bert_data['val_input_ids'].tolist(),
    'attention_mask': bert_data['val_attention_mask'].tolist()
}
bert_test_encodings = {
    'input_ids': bert_data['test_input_ids'].tolist(),
    'attention_mask': bert_data['test_attention_mask'].tolist()
}

roberta_data = np.load('../data/roberta.npz', allow_pickle=True)

roberta_train_encodings = {
    'input_ids': roberta_data['train_input_ids'].tolist(),
    'attention_mask': roberta_data['train_attention_mask'].tolist()
}
roberta_val_encodings = {
    'input_ids': roberta_data['val_input_ids'].tolist(),
    'attention_mask': roberta_data['val_attention_mask'].tolist()
}
roberta_test_encodings = {
    'input_ids': roberta_data['test_input_ids'].tolist(),
    'attention_mask': roberta_data['test_attention_mask'].tolist()
}

y_train = np.load('../data/y_train.npy', allow_pickle=True)
y_test = np.load('../data/y_test.npy', allow_pickle=True)
y_validation = np.load('../data/y_validation.npy', allow_pickle=True)


# turn it into Dataset
bert_train_dataset = TextDataset(bert_train_encodings, y_train.tolist())
bert_val_dataset = TextDataset(bert_val_encodings, y_validation.tolist())
bert_test_dataset = TextDataset(bert_test_encodings, y_test.tolist())

roberta_train_dataset = TextDataset(roberta_train_encodings, y_train.tolist())
roberta_val_dataset = TextDataset(roberta_val_encodings, y_validation.tolist())
roberta_test_dataset = TextDataset(roberta_test_encodings, y_test.tolist())


# create Dataloader
train_loader_bert = DataLoader(bert_train_dataset, batch_size=32, shuffle=True)
val_loader_bert = DataLoader(bert_val_dataset, batch_size=32, shuffle=False)
test_loader_bert = DataLoader(bert_test_dataset, batch_size=32, shuffle=False)

train_loader_roberta = DataLoader(roberta_train_dataset, batch_size=32, shuffle=True)
val_loader_roberta = DataLoader(roberta_val_dataset, batch_size=32, shuffle=False)
test_loader_roberta = DataLoader(roberta_test_dataset, batch_size=32, shuffle=False)


# train pre-trained models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert_f1 = train_and_evaluate('bert', train_loader_bert, val_loader_bert, test_loader_bert, device, epochs=5, save_path='../data/model/best_bert_model.pt')
roberta_f1 = train_and_evaluate('roberta', train_loader_roberta, val_loader_roberta, test_loader_roberta, device, epochs=5, save_path='../data/model/best_roberta_model.pt')

print(f"Best BERT F1: {bert_f1:.4f}")
print(f"Best RoBERTa F1: {roberta_f1:.4f}")

if roberta_f1 > bert_f1:
    print("RoBERTa performs better! Use RoBERTa for deployment.")
else:
    print("BERT performs better! Use BERT for deployment.")
