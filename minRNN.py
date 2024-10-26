import torch
import torch.nn as nn
from minLSTMcell import MinLSTMCell

class MinRNN(nn.Module):
    def __init__(self, units, embedding_size, vocab_size, input_length):
        super(MinRNN, self).__init__()
        self.input_length = input_length
        self.units = units

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = MinLSTMCell(units, embedding_size)
        self.classification_model = nn.Sequential(
            nn.Linear(units, 64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, sentence):
        """
        Args:
            sentence: (batch_size, input_length)

        output:
            (batch_size, 1)

        """
        batch_size = sentence.shape[0]

        # Initialize the hidden state, only the h needs to be initialized
        pre_h = torch.zeros(batch_size, self.units, device=sentence.device)

        # Pass the sentence through the embedding layer for the word vectors embeddings
        embedded_sentence = self.embedding(sentence)

        sequence_length = embedded_sentence.shape[1]

        # Pass the entire sequence through the LSTM + hidden_state
        for i in range(sequence_length):
            word = embedded_sentence[:, i, :]  # (batch_size, embedding_size)
            pre_h = self.lstm(pre_h, word)  # Only update h (hidden state)

        return self.classification_model(pre_h)  # Pass the final hidden state into the classification network
