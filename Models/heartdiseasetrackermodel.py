import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim 
from sklearn.preprocessing import StandardScaler, LabelEncoder    
from sklearn.model_selection import train_test_split
import kaggle
from pathlib import Path
import joblib
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models.HelperFunction import helperFunctions as hp
from torch.utils.data import DataLoader as dl

basePath = Path.cwd().parent
csvFilePath = basePath / 'dataset' / 'heart.csv'

if not csvFilePath.exists():
    kaggle.api.dataset_download_files("johnsmith88/heart-disease-dataset", path=basePath / 'dataset', unzip=True)
else:
    print(f"Dataset file found: {csvFilePath}")

dataset = pd.read_csv(csvFilePath)
df = pd.DataFrame(dataset)
df.drop_duplicates(subset=None, keep='first', inplace=True)

for col in df.columns:
    print(f"{col}: {df[col].unique()}")
    print("Maximum: ", df[col].max())
    print("Minimum: ", df[col].min())
    print("-" * 50)

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

scaler = StandardScaler()
continous_col = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
df[continous_col] = scaler.fit_transform(df[continous_col])
#print(df.head())

label_encoder = LabelEncoder()
oridinal_col = ['restecg', 'slope', 'ca',]
df[oridinal_col] = df[oridinal_col].apply(label_encoder.fit_transform)
#print(df.head())

pd.get_dummies(df, columns=['cp', 'thal'], drop_first=True)
print(df.head())

label_count = df['target'].value_counts()
label_count

features = df.drop('target', axis=1).to_numpy()
target = df['target'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler_save_path = Path.cwd().parent/'Models'/'SavedModels'/'HeartDiseaseScaler.pkl'
scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(scaler, scaler_save_path)
print(f"Scaler saved at: {scaler_save_path}")

class HeartDiseaseClassification(nn.Module):
    def __init__(self):
        super(HeartDiseaseClassification, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
    def predict(self, input_data, return_confidence=False):
        self.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(input_data, dtype=torch.float32)    
            output = self(input_tensor)
            confidence = output.item()
            prediction = 1 if confidence > 0.5 else 0
            if return_confidence:
                return prediction, confidence *100
            return prediction

train_dataloader = dl(list(zip(X_train, y_train)), batch_size=32, shuffle=True)
test_dataloader = dl(list(zip(X_test, y_test)), batch_size=32, shuffle=False)
model = HeartDiseaseClassification()
loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train():
    trained_models, metrics = hp.train_and_evaluate(model,train_dataloader, test_dataloader,loss, optimizer,device, 1000, 100)
    model_save_path = Path.cwd().parent/'Models'/'SavedModels'/'HeartDiseaseModel.pth'
    torch.save(trained_models.state_dict(), model_save_path)
    print(f"Model saved at: {model_save_path}")



if __name__ == "__main__":
    #mean_score, fold_accuracies, fold_losses = hp.cross_validate(model, features, target, cv=5, scoring='accuracy', epochs=1000)
    #print("Mean Score:", mean_score)
    print(X_train.shape)
    train()
    print("Training Completed")



