# -*- coding: utf-8 -*-
"""LungCancerModel.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/138cEsarOB5B7PYgUvvW2qCz5sUsGKTba
"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.utils import resample
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import sys
# Adjust the import statement
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models.HelperFunction.helperFunctions import train_and_evaluate
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the base path relative to the script's location
base_path = Path(__file__).resolve().parent.parent

# Define the path to the dataset
csvFilePath = base_path / 'dataset' / 'survey lung cancer.csv'

# Check if the dataset file exists
if not csvFilePath.exists():
    raise FileNotFoundError(f"Dataset file not found: {csvFilePath}")

# Load the dataset
dataset = pd.read_csv(csvFilePath)

df = pd.DataFrame(dataset)
label_encoder = LabelEncoder()
df['GENDER'] = label_encoder.fit_transform(df['GENDER'])
df['LUNG_CANCER'] = label_encoder.fit_transform(df['LUNG_CANCER'])

def add_noise(df, column, noise_factor=0.05):
    noise = np.random.normal(0, noise_factor * df[column].std(), df[column].shape[0])
    df[column] = df[column] + noise
    return df

df = add_noise(df, 'AGE')
scaler = StandardScaler()
df['AGE'] = scaler.fit_transform(df[['AGE']])

scaler_save_path = base_path / 'Models' / 'SavedModels' / 'age_scaler.pkl'
scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(scaler, scaler_save_path)
print(f"Scaler saved to {scaler_save_path}")

df_majority = df[df['LUNG_CANCER'] == 1]
df_minority = df[df['LUNG_CANCER'] == 0]

df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

binary_columns = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                  'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
                  'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
                  'SWALLOWING DIFFICULTY', 'CHEST PAIN']
for column in binary_columns:
    df_balanced[column] = df_balanced[column].map({1: 0, 2: 1})

features = df_balanced.drop('LUNG_CANCER', axis=1).to_numpy()
target = df_balanced['LUNG_CANCER'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Create DataLoader for training and testing
train_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), batch_size=32, shuffle=False)

class LungCancerClassifier(nn.Module):
    def __init__(self):
        super(LungCancerClassifier, self).__init__()
        self.fc1 = nn.Linear(15, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

    def predict(self, input_data, return_confidence=False):
        self.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            output = self(input_tensor)
            confidence = output.item()
            prediction = 1 if confidence >= 0.5 else 0
            if return_confidence:
                return prediction, confidence * 100
            return prediction

def train_model():
    model = LungCancerClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trained_model, metrics = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer,device, num_epochs=150)

    model_save_path = base_path / 'Models' / 'SavedModels' / 'lung_cancer_model.pth'
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model()
