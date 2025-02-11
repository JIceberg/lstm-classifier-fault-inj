import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from time import sleep
import os
from lstm import FaultyLSTM
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(42)

top_words = 5000
max_len = 500
batch_size = 64

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

X_train = pad_sequences(X_train, maxlen=max_len, value=0.0)
X_test = pad_sequences(X_test, maxlen=max_len, value=0.0)

X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

class Classifier(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = FaultyLSTM(embedding_dim, hidden_dim)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden, compute_grad=False, inject_faults=False):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(
            embeds,
            hidden=hidden,
            compute_grad=compute_grad,
            inject_faults=inject_faults
        )
    
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(batch_size, self.hidden_dim).zero_(),
                    weight.new(batch_size, self.hidden_dim).zero_())
        
        return hidden

embed_dim = 32
hidden_dim = 100
output_size = 1
n_layers = 1

model = Classifier(top_words, output_size, embed_dim, hidden_dim)
print(model)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train_model(model, train_loader, epochs=3):
    model.train()
    for epoch in range(epochs):
        h = model.init_hidden(batch_size)

        total_loss = 0
        total_correct = 0
        total_labels = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")

                optimizer.zero_grad()
                h = tuple([each.data for each in h])
                output, h = model(data, h)
                loss = criterion(output.squeeze(), target)
                predictions = (output.squeeze() > 0.5).float()
                correct = (predictions == target).sum().item()
                accuracy = correct / batch_size

                loss.backward()  # need to include backprop
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

                total_loss += loss.item()
                total_correct += correct
                total_labels += target.size(0)

                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
                sleep(0.1)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {100. * total_correct / total_labels:.2f}")
            
if os.path.exists('saved_model.pth'):
    model.load_state_dict(torch.load("saved_model.pth", weights_only=True))
else:
    train_model(model, train_loader, epochs=3)
    torch.save(model.state_dict(), "saved_model.pth")

def evaluate_model(model, test_loader, num_batches=400, compute_grad=False, inject_faults=False):
    model.eval()
    h = model.init_hidden(batch_size)
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        batch = 0
        for inputs, labels in test_loader:
            output, h = model(inputs, hidden=h, compute_grad=compute_grad, inject_faults=inject_faults)
            total_loss += criterion(output.squeeze(), labels)
            preds = (output.squeeze() > 0.5).float()
            accuracy = (preds == labels).sum().item() / batch_size
            total_accuracy += accuracy
            print(f"Accuracy for batch {batch+1}: {accuracy * 100.:.2f}%")
            batch += 1
            if batch >= num_batches:
                break
        print(f"Average accuracy: {total_accuracy / batch * 100.:.2f}%")
        

'''
Compute statistics
'''
# evaluate_model(model, train_loader, compute_grad=True)

evaluate_model(model, test_loader, inject_faults=True)