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
import numpy as np

np.random.seed(42)

top_words = 5000
max_len = 500
batch_size = 64

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Classifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super(Classifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = FaultyLSTM(embed_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, compute_grad=False, inject_faults=False):
        x = self.embedding(x)
        lstm_out = self.lstm(x, compute_grad=compute_grad, inject_faults=inject_faults)
        out = self.fc(lstm_out)
        return self.sigmoid(out)

embed_dim = 32
hidden_dim = 50

model = Classifier(top_words, embed_dim, hidden_dim)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train_model(model, train_loader, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_labels = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")

                optimizer.zero_grad()
                outputs = model(data).squeeze()  # Remove extra dimension
                loss = criterion(outputs, target)
                predictions = (outputs > 0.5).float()
                correct = (predictions == target).sum().item()
                accuracy = correct / batch_size

                loss.backward()  # need to include backprop
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

def evaluate_model(model, test_loader, num_batches=400, compute_grad=False):
    model.eval()
    total_clean_loss = 0
    total_faulty_loss = 0
    with torch.no_grad():
        batch = 0
        for inputs, labels in test_loader:
            outputs_clean = model(inputs, compute_grad=compute_grad).squeeze()
            total_clean_loss += criterion(outputs_clean, labels)
            print(f"Average clean loss for batch {batch+1}: {total_clean_loss / (batch+1)}")
            if not compute_grad:
                outputs_faulty = model(inputs, inject_faults=True).squeeze()
                total_faulty_loss += criterion(outputs_faulty, labels)
                print(f"Average faulty loss for batch {batch+1}: {total_faulty_loss / (batch+1)}")
            batch += 1
            if batch >= num_batches:
                break
        percent_increase = (total_faulty_loss - total_clean_loss) / total_clean_loss
        print(f"Percent increase: {percent_increase * 100.}%")
        

'''
Compute statistics
'''
# evaluate_model(model, test_loader, compute_grad=True)

evaluate_model(model, test_loader)