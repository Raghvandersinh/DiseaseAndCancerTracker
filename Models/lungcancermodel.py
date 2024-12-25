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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import kaggle
import joblib

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

# print(dataset.head())

df = pd.DataFrame(dataset)
label_encoder = LabelEncoder()
df['GENDER'] = label_encoder.fit_transform(df['GENDER'])
df['LUNG_CANCER'] = label_encoder.fit_transform(df['LUNG_CANCER'])
# print(df.head())

lung_cancer_counts = df['LUNG_CANCER'].value_counts()
# print(lung_cancer_counts)

def add_noise(df, column, noise_factor = 0.05):
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

df_minority_upsampled=  resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# print(df_balanced['LUNG_CANCER'].value_counts())
# print(df_balanced)
binary_columns = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                  'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
                  'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
                  'SWALLOWING DIFFICULTY', 'CHEST PAIN']
for column in binary_columns:
    df_balanced[column] = df_balanced[column].map({1:0, 2:1})
# print(df_balanced.head())

features = df_balanced.drop('LUNG_CANCER', axis=1).to_numpy()
target = df_balanced['LUNG_CANCER'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class LungCancerClassifier(nn.Module):
    def __init__(self):
        super(LungCancerClassifier, self).__init__()
        # Define your model layers here
        self.fc1 = nn.Linear(15, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

    def predict(self, input_data, return_confidence=False):
        self.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            output = self(input_tensor)
            prediction = torch.round(output).item()
            confidence = torch.sigmoid(output).item() * 100  # Convert to percentage

        if return_confidence:
            return prediction, confidence
        return prediction

model = LungCancerClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def compute_metrics(true_labels, predictions, probablities = None):
  true_labels = true_labels.cpu().numpy()
  predictions = predictions.cpu().numpy()
  if probablities is not None:
    roc_auc = roc_auc_score(true_labels, probablities)
  else:
    roc_auc = None
  accuracy = accuracy_score(true_labels, predictions)
  precision = precision_score(true_labels, predictions)
  recall = recall_score(true_labels, predictions)
  f1 = f1_score(true_labels, predictions)
  return {
      'accuracy': accuracy,
      'precision': precision,
      'recall': recall,
      'f1': f1,
      'roc_auc': roc_auc
  }

# prompt: Create a loss graph

epoch = 500
train_metrics = {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
test_metrics =  {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}

for epoch in range(epoch):
  output = model(X_train)
  loss= criterion(output, y_train.unsqueeze(1))
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  train_metrics['loss'].append(loss.item())
  train_pred = (output > 0.5).float()
  train_probs = output.detach().numpy()
  train_metrics_dict = compute_metrics(y_train, train_pred, train_probs)
  for key, value in train_metrics_dict.items():
    train_metrics[key].append(value)
  # if (epoch+1) % 100 == 0:
  #     print(f"Epoch {epoch + 1}, Train Loss: {loss.item():.4f}, "
  #             f"Accuracy: {train_metrics_dict['accuracy']:.4f}, Precision: {train_metrics_dict['precision']:.4f}, "
  #             f"Recall: {train_metrics_dict['recall']:.4f}, F1: {train_metrics_dict['f1']:.4f}, "
  #             f"ROC-AUC: {train_metrics_dict['roc_auc']:.4f}")

  model.eval()
  with torch.inference_mode():
    test_output = model(X_test)
    test_loss = criterion(test_output, y_test.unsqueeze(1))
    test_metrics['loss'].append(test_loss.item())

    test_pred = (test_output > 0.5).float()
    test_prob = test_output.detach().numpy()
    test_metrics_dict = compute_metrics(y_test, test_pred, test_prob)
    for key, value in test_metrics_dict.items():
      test_metrics[key].append(value)
    # if (epoch+1) % 100 == 0:
    #   print(print(f"Epoch {epoch + 1}, Test Loss: {test_loss.item():.4f}, "
    #               f"Accuracy: {test_metrics_dict['accuracy']:.4f}, Precision: {test_metrics_dict['precision']:.4f}, "
    #               f"Recall: {test_metrics_dict['recall']:.4f}, F1: {test_metrics_dict['f1']:.4f}, "
    #               f"ROC-AUC: {test_metrics_dict['roc_auc']:.4f}"))



def plot_metrics(metric_dict, metric_name, title):
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics[metric_name], label=f'Train {metric_name}')
    plt.plot(test_metrics[metric_name], label=f'Test {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot each metric
# plot_metrics(train_metrics, 'loss', 'Training and Test Loss')
# plot_metrics(train_metrics, 'accuracy', 'Training and Test Accuracy')
# plot_metrics(train_metrics, 'f1', 'Training and Test F1-Score')
# plot_metrics(train_metrics, 'roc_auc', 'Training and Test ROC-AUC')


# Define the path to save the scaler


model_save_path = base_path / 'Models' / 'SavedModels' / 'lung_cancer_model.pth'

# Check if the directory exists, if not, create it
model_save_path.parent.mkdir(parents=True, exist_ok=True)

# Check if the model file already exists

torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")