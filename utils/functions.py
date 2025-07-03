import re
import torch
from torch.optim import AdamW
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, BertTokenizer, RobertaTokenizer
from tqdm import tqdm


def preprocess(text):
    """
    Clean text column (remove non-ASCII characters, HTML tags if present
    and replacing tabs, newlines... with a single space)

    Args:
        text (str): Input text to clean 
    """
    text = text.encode('ascii', errors='ignore').decode('ascii')
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def combine_text_columns(df, columns):
    """
    Combine specified columns of a DataFrame into a single text string per row

    Args:
        df (pandas.DataFrame): The DataFrame containing the data
        columns ([str]): List of column names to combine
    """
    return df[columns].astype(str).agg(" ".join, axis=1)


def tokenize_texts(tokenizer, texts):
    """
    Tokenizes a list of texts using the provided tokenizer

    Args:
        tokenizer (PreTrainedTokenizer): A HuggingFace tokenizer 
        texts (pd.Series or list): A list or pandas Series of raw text strings to tokenize
    """
    return tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512)


def train_epoch(model, data_loader, optimizer, device):
    """
    Performs one training epoch for a model

    Args:
        model (torch.nn.Module): PyTorch model to be trained
        data_loader (torch.utils.data.DataLoader): DataLoader providing batches of training data
        optimizer (torch.optim.Optimizer): Optimizer used to update the model parameters
        device (torch.device): Device where training is performed (CPU or GPU)
    """
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def eval_model(model, data_loader, device):
    """
    Evaluates the model on the given dataset

    Args:
        model (torch.nn.Module): PyTorch model to evaluate
        data_loader (torch.utils.data.DataLoader): DataLoader providing batches of evaluation data
        device (torch.device): Device where evaluation is performed (CPU or GPU)
    """
    model.eval()
    preds, true_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    cm = confusion_matrix(true_labels, preds)
    return avg_loss, acc, f1, cm


def train_and_evaluate(model_name, train_loader, val_loader, test_loader, device, epochs=3, save_path=None, lr=1e-4):
    """
    Trains and evaluates a transformer model (BERT or RoBERTa) for sequence classification,
    save the best model and show the training progress (train_loss, loss, acc, fq)

    Args:
        model_name (str): Name of the model to train ('bert' or 'roberta')
        train_loader (torch.utils.data.DataLoader): DataLoader for training data
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data
        test_loader (torch.utils.data.DataLoader): DataLoader for test data
        device (torch.device): Device to run training and evaluation on (CPU or GPU)
        epochs (int, optional): Number of training epochs. Default is 3
        save_path (str, optional): Path to save the best model checkpoint. If None, model is not saved
        lr (float, optional): Learning rate for the optimizer. Default is 1e-4
    """
    print(f"Training {model_name}...")

    if model_name == 'bert':
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    else:
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    best_f1 = 0
    best_model = None

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_f1, _ = eval_model(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = model.state_dict()
            if save_path:
                torch.save(best_model, save_path)
                print(f"Best model saved to {save_path}")

    model.load_state_dict(best_model)
    test_loss, test_acc, test_f1, test_cm = eval_model(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f} - Test F1: {test_f1:.4f}")
    print("Confusion Matrix:\n", test_cm)

    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs_range, history['val_acc'], label='Val Accuracy')
    plt.plot(epochs_range, history['val_f1'], label='Val F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title(f'{model_name} Validation Metrics')
    plt.legend()

    plt.show()

    return best_f1
