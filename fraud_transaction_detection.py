import pandas as pd
import numpy as np

np.random.seed(42)

device_types = ['iOS', 'Android']

# Generate non-fraudulent transactions
n_non_fraud = 800
non_fraud_transactions = pd.DataFrame({
    'transaction_amount': np.random.uniform(100, 50000, n_non_fraud),
    'transaction_time': np.random.uniform(0, 24, n_non_fraud),
    'transaction_distance': np.random.uniform(0, 50, n_non_fraud),
    'latitude': np.random.uniform(-90, 90, n_non_fraud),
    'longitude': np.random.uniform(-180, 180, n_non_fraud),
    'device_type': np.random.choice(device_types, n_non_fraud),
    'is_fraud': np.zeros(n_non_fraud, dtype=int)
})

# Generate fraudulent transactions
n_fraud = 200
fraud_transactions = pd.DataFrame({
    'transaction_amount': np.random.uniform(50000, 100000, n_fraud),
    'transaction_time': np.random.uniform(1, 4, n_fraud),
    'transaction_distance': np.random.uniform(50, 100, n_fraud),
    'latitude': np.random.uniform(-90, 90, n_fraud),
    'longitude': np.random.uniform(-180, 180, n_fraud),
    'device_type': np.random.choice(device_types, n_fraud),
    'is_fraud': np.ones(n_fraud, dtype=int)
})

# Combine datasets
transactions = pd.concat([non_fraud_transactions, fraud_transactions]).sample(frac=1).reset_index(drop=True)


# Save to CSV
transactions.to_csv('transactions_data.csv', index=False)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Custom Dataset class
class TransactionsDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.label_encoder = LabelEncoder()
        self.data['device_type'] = self.label_encoder.fit_transform(self.data['device_type'])
        self.X = self.data.drop(columns=['is_fraud']).values
        self.y = self.data['is_fraud'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# Load dataset
data = pd.read_csv('transactions_data.csv')
dataset = TransactionsDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define model
class FraudDetectionModel(nn.Module):
    def __init__(self):
        super(FraudDetectionModel, self).__init__()
        self.fc1 = nn.Linear(6, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = FraudDetectionModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'transactions_fraud_detection_model.pth')

# Evaluate the model
model.eval()
all_labels = []
all_preds = []
with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model(inputs)
        preds = outputs.squeeze().round()
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.numpy())

print(classification_report(all_labels, all_preds))
print(confusion_matrix(all_labels, all_preds))

pip install coremltools

import coremltools as ct

# Load the trained PyTorch model
model = FraudDetectionModel()
model.load_state_dict(torch.load('transactions_fraud_detection_model.pth'))
model.eval()

# Example input for tracing
example_input = torch.rand(1, 6)

# Convert the model to Core ML
example_input = torch.rand(1, 6)
traced_model = torch.jit.trace(model, example_input)
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)]
)
mlmodel.save('TransactionFraudDetectionModel.mlpackage')
