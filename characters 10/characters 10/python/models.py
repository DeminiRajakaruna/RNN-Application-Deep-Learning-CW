# models.py

import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt

#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context, vocab_index):
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context, vocab_index):
        char = context[-1]
        if self.consonant_counts[char] > self.vowel_counts[char]:
            return 0
        else:
            return 1


class RNNClassifier(nn.Module, ConsonantVowelClassifier):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, output_dim=2, num_layers=2, dropout=0.5):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, hidden = self.rnn(embedded)
        output = self.fc(hidden[-1])  # Use the last layer's hidden state
        return output

    def predict(self, context, vocab_index):
        # Convert context from characters to indices
        indices = [vocab_index.index_of(char) for char in context]
        context_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # Add batch dimension
        self.eval()
        with torch.no_grad():
            predictions = self.forward(context_tensor)
            return predictions.argmax(1).item()


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    embedding_dim = 128
    hidden_dim = 256
    output_dim = 2
    batch_size = 64
    num_epochs = 15
    learning_rate = 0.001
    weight_decay = 1e-6
    context_length = 10

    def encode_sequences(examples, vocab_index):
        return [[vocab_index.index_of(char) for char in seq[:context_length]] for seq in examples]

    train_sequences = encode_sequences(train_cons_exs, vocab_index) + encode_sequences(train_vowel_exs, vocab_index)
    train_labels = [0] * len(train_cons_exs) + [1] * len(train_vowel_exs)

    dev_sequences = encode_sequences(dev_cons_exs, vocab_index) + encode_sequences(dev_vowel_exs, vocab_index)
    dev_labels = [0] * len(dev_cons_exs) + [1] * len(dev_vowel_exs)

    def pad_and_tensorify(sequences, max_length):
        return torch.tensor(
            [seq + [0] * (max_length - len(seq)) for seq in sequences], dtype=torch.long
        )

    max_length = max(len(seq) for seq in train_sequences + dev_sequences)
    train_data = TensorDataset(
        pad_and_tensorify(train_sequences, max_length),
        torch.tensor(train_labels, dtype=torch.long)
    )
    dev_data = TensorDataset(
        pad_and_tensorify(dev_sequences, max_length),
        torch.tensor(dev_labels, dtype=torch.long)
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=batch_size)

    vocab_size = len(vocab_index)
    model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)

    epoch_losses = []
    epoch_times = []
    model.train()
    for epoch in range(num_epochs):
        start_time = time.time()  # Track epoch start time
        total_loss = 0.0
        for batch in train_loader:
            seqs, labels = batch
            optimizer.zero_grad()
            predictions = model(seqs)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Track loss and time for each epoch
        epoch_losses.append(total_loss / len(train_loader))
        epoch_times.append(time.time() - start_time)

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Time: {epoch_times[-1]:.2f}s")

    # Plot Loss vs Epochs graph
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epochs (Context Length: {context_length})")
    plt.show()

    # Calculate average loss
    average_loss = np.mean(epoch_losses)
    print(f"Average Loss: {average_loss:.4f}")

    # Plot Training Time per Epoch
    plt.plot(range(1, num_epochs + 1), epoch_times, marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Time (seconds)")
    plt.title(f"Training Time per Epoch (Context Length: {context_length})")
    plt.show()

    # Evaluate the model
    model.eval()
    num_correct = 0
    with torch.no_grad():
        for batch in dev_loader:
            seqs, labels = batch
            predictions = model(seqs)
            correct_predictions = (predictions.argmax(1) == labels).sum().item()
            num_correct += correct_predictions

    accuracy = num_correct / len(dev_data)
    print(f"Accuracy on dev set: {accuracy * 100:.2f}%")

    return model






#####################
# MODELS FOR PART 2 #
#####################



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LanguageModel(object):
    def get_log_prob_single(self, next_char, context):
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Only implemented in subclasses")

class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_log_prob_single(self, next_char, context):
        return np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)

class RNNLanguageModel(LanguageModel):
    def __init__(self, model_emb, model_dec, vocab_index):
        self.model_emb = model_emb
        self.model_dec = model_dec
        self.vocab_index = vocab_index

    def get_log_prob_single(self, next_char, context):
        context_indices = [self.vocab_index.index_of(c) for c in context]
        next_char_index = self.vocab_index.index_of(next_char)
        context_tensor = torch.tensor([context_indices], dtype=torch.long)
        output_probs = self.model_dec(self.model_emb(context_tensor)).detach().numpy()
        return np.log(output_probs[0, next_char_index])

    def get_log_prob_sequence(self, next_chars, context):
        log_prob = 0
        for i, next_char in enumerate(next_chars):
            log_prob += self.get_log_prob_single(next_char, context + next_chars[:i])
        return log_prob

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        rnn_out, _ = self.rnn(x)
        logits = self.fc(rnn_out)
        return logits

def train_lm(args, train_text, dev_text, vocab_index):
    vocab_size = len(vocab_index)
    embed_size = 128
    hidden_size = 256
    batch_size = 64
    seq_length = 50
    epochs = 10

    model = RNNModel(vocab_size, embed_size, hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def chunk_data(text, seq_length):
        data = [vocab_index.index_of(c) for c in text]
        x, y = [], []
        for i in range(len(data) - seq_length):
            x.append(data[i:i + seq_length])
            y.append(data[i + 1:i + seq_length + 1])
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    train_x, train_y = chunk_data(train_text, seq_length)
    dev_x, dev_y = chunk_data(dev_text, seq_length)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i:i + batch_size]
            batch_y = train_y[i:i + batch_size]

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_x)}")

    model.eval()
    with torch.no_grad():
        dev_log_prob = 0
        for i in range(0, len(dev_x), batch_size):
            batch_x = dev_x[i:i + batch_size]
            batch_y = dev_y[i:i + batch_size]
            logits = model(batch_x)
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            for j in range(batch_x.size(0)):
                dev_log_prob += log_probs[j, torch.arange(batch_y.size(1)), batch_y[j]].sum().item()

    avg_log_prob = dev_log_prob / len(dev_text)
    perplexity = np.exp(-avg_log_prob)
    print(f"Log prob of text: {dev_log_prob}")
    print(f"Avg log prob: {avg_log_prob}")
    print(f"Perplexity: {perplexity}")

    return RNNLanguageModel(model.embedding, model, vocab_index)



