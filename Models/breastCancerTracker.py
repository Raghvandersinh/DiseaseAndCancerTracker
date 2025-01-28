
import torch
from torch import nn
import kaggle
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models.HelperFunction import helperFunctions as hp
import joblib


basePath = Path(__file__).resolve().parent.parent
csv_file_path = Path.cwd().parent/'dataset'/'breastCancer'

if csv_file_path.exists():
    print(f"Folder already exists at: {csv_file_path}")
else:
    csv_file_path.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_files("uciml/breast-cancer-wisconsin-data", path=csv_file_path, unzip=True)


df = pd.read_csv(csv_file_path/'data.csv')
df.head()


len(df)


df.drop(columns=['id'], inplace=True)
df.drop(columns=['Unnamed: 32'], inplace=True)
value_counts = {col: df[col].value_counts() for col in df.columns}
# for col, counts in value_counts.items():
#     print(f"Value counts for {col}:  {counts}")
#     print()


features = df.drop(columns=['diagnosis'])
target = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# len(X_train)


encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)



# for col, counts in value_counts.items():
#     print(f"Maximum Value and Minimum Value for {col}:")
#     print(f"Maximum Value: {df[col].max()} and Minimum Value: {df[col].min()}\n")


torch.manual_seed(42)
scaler = PowerTransformer()
right_skewed_col = ['area_mean', 'texture_mean', 'perimeter_mean', 'radius_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'fractal_dimension_mean', 'area_se', 'texture_se', 'perimeter_se', 'radius_se', 'compactness_se', 'concavity_se', 'concave points_se', 'smoothness_se', 'symmetry_se', 'fractal_dimension_se', 'area_worst', 'texture_worst', 'perimeter_worst', 'radius_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
X_train[right_skewed_col]= scaler.fit_transform(X_train[right_skewed_col])
X_test[right_skewed_col]= scaler.fit_transform(X_test[right_skewed_col])

scaler_save_path = basePath/'Models'/'Transformations'/'BreastCancerScaler.pkl'
scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(scaler, scaler_save_path)



device = 'cuda' if torch.cuda.is_available() else 'cpu'


import seaborn as sns

# Loop through each column in the DataFrame
# for column in X_train.columns:
#     # Create a new figure for each column
#     plt.figure(figsize=(8, 6))
    
#     # Plot the histogram using Seaborn
#     sns.histplot(X_train[column], kde=True, bins=30)  # kde=True adds a kernel density estimate curve
    
#     # Add titles and labels
#     plt.title(f"Histogram of {column}")
#     plt.xlabel(column)
#     plt.ylabel("Frequency")
    
#     # Show the plot
#     plt.show()



train_dataset = TensorDataset(torch.tensor(X_train.values).float(), torch.tensor(y_train).float())
test_dataset = TensorDataset(torch.tensor(X_test.values).float(), torch.tensor(y_test).float())

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)



class breastCancerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(30, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

model = breastCancerModel()
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    


def train_and_eval():
    
    model_save_path = basePath/'Models'/'SavedModels'/'BreastCancerTracker.pth'
    trained_models, metrics = hp.train_and_evaluate(model,train_dataloader, test_dataloader,loss, optimizer,device, 150, 10, patience=10, save_path=model_save_path)

#125 Epoch and 0.0001 learning rate
if __name__ == "__main__":
    # train_and_eval() 
    print("Hello")



