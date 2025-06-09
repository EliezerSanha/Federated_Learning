from flwr.client import NumPyClient, ClientApp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import pandas as pd

from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():

    # Load data from CSV
    data = pd.read_csv("data/data_3.csv")

    # Split data into input (X) and output (y)
    X = data.iloc[:, :4].values  # Input features
    y = data.iloc[:, 4].values   # Output feature

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_test, y_test

# Define the neural network class
class TabularNN(nn.Module):
    def __init__(self):
        super(TabularNN, self).__init__()
        # Define layers in __init__
        self.fc1 = nn.Linear(4, 5)  
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        # First hidden layer: Apply linear transformation, then apply activation
        x = self.fc1(x)
        x = torch.sigmoid(x)

        # Second hidden layer: Apply linear transformation, then apply activation
        x = self.fc2(x)
        x = torch.sigmoid(x)

        # Output layer: Apply linear transformation, then apply activation
        x = self.fc3(x)
        x = torch.sigmoid(x)

        return x

def train(model, train_data, epochs):
    learning_rate = 0.01
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        
        for inputs, labels in train_data:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

def test(model, test_data):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0
    
    with torch.no_grad():  # Disable gradient tracking
        for inputs, labels in test_data:
            outputs = model(inputs)
            criterion = nn.BCELoss()
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item() * inputs.size(0)  # Multiply loss by batch size
            
            predicted = (outputs.squeeze() > 0.5).float()  # Convert to binary predictions
            batch_accuracy = accuracy_score(labels.numpy(), predicted.detach().numpy())
            total_accuracy += batch_accuracy * labels.size(0)  # Multiply accuracy by batch size
            
            total_samples += labels.size(0)
    
    overall_loss = total_loss / total_samples
    overall_accuracy = total_accuracy / total_samples
    
    print(f'Loss on test set: {overall_loss:.4f}, Accuracy on test set: {overall_accuracy*100:.2f}%')
    
    return overall_loss, overall_accuracy  # Return both loss and accuracy


    
# Assuming you have X_train, y_train, X_test, and y_test as tensors
X_train, y_train, X_test, y_test = load_data()

# Create a TensorDataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create a DataLoader
trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=16, shuffle=True)


# Create the model
net = TabularNN()


# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient().to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:5006",
        client=FlowerClient().to_client(),
    )
