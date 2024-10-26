import json
import numpy as np
from minRNN import MinRNN
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


class SarcasmDataset(Dataset):
    """
    A self defined dataset class for loading the sarcasm dataset
    """
    def __init__(self, sentences, labels, tokenizer):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        encoded_sentences = self.tokenizer.encode(sentence)

        return torch.tensor(encoded_sentences.ids, dtype=torch.long), torch.tensor(label, dtype=torch.float32)
    

def collate_fn(batch):
    """
    A self-defined collate function for padding the sequences before feeding them into the model
    """
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return padded_sequences, torch.tensor(labels, dtype=torch.float32)


def create_tokenizer(text, vocab_size):
    """
    A function to create a tokenizer for this specific dataset
    """
    tokenizer = Tokenizer(WordLevel(unk_token="<OOV>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(vocab_size=vocab_size, special_tokens=["<OOV>", "<PAD>"], min_frequency=1)
    tokenizer.train_from_iterator(text, trainer)

    return tokenizer


def train(batch_size, epochs):
    """
    A function for training this super simple model
    
    Args:
        batch_size: The batch size
        epochs: The number of epochs to train for

    Returns:
        None
    """
    with open("./data/sarcasm.json", 'r') as f:
        datastore = json.load(f)

    dataset = []
    label_dataset = []

    for item in datastore:
        dataset.append(item["headline"])
        label_dataset.append(item["is_sarcastic"])


    dataset = np.array(dataset)
    label_dataset = np.array(label_dataset)

    train_size = 0.8
    size = int(len(dataset) * train_size)

    train_sentence = dataset[:size]
    test_sentence = dataset[size:]

    train_label = label_dataset[:size]
    test_label = label_dataset[size:]

    vocab_size = len(train_sentence)
    max_length = 25

    tokenizer = create_tokenizer(train_sentence, vocab_size)
     
    train_dataset = SarcasmDataset(train_sentence, train_label, tokenizer)
    test_dataset = SarcasmDataset(test_sentence, test_label, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    vocab_size = tokenizer.get_vocab_size()

    model = MinRNN(units=128, embedding_size=100, vocab_size=vocab_size, input_length=max_length)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss()   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            prediction = torch.sigmoid(outputs.squeeze())
            prediction = (prediction >= 0.5).float()
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total

        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_acc:.4f}")

        # Validation phase
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                test_loss += loss.item()
                
                predictions = torch.sigmoid(outputs.squeeze())
                predictions = (predictions >= 0.5).float()
                test_correct += (predictions == labels).sum().item()
                test_total += labels.size(0)
        
            avg_test_loss = test_loss / len(test_loader)
            avg_test_acc = test_correct / test_total
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_acc:.4f}")
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.4f}")

if __name__ == "__main__":
    print("Training...")
    train(batch_size=64, epochs=100)
