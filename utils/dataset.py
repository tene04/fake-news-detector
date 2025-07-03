import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        """
        Constructor of the dataset

        Args:
            encodings (dict): Tokenized inputs from a HuggingFace tokenizer 
            labels ([] or tensor): Corresponding labels for each input
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Retrieves the input data and label for a given index
        
        Args:
          idx (int): Index of the data sample to retrieve
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Returns the total number of samples in the dataset
        """
        return len(self.labels)