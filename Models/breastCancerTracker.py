
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
sys.path.append(os.path.abspath(os.path.join(Path.cwd(), '..')))
from Models.HelperFunction import helperFunctions as hp



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
for col, counts in value_counts.items():
    print(f"Value counts for {col}:  {counts}")
    print()


features = df.drop(columns=['diagnosis'])
target = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
len(X_train)


encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)



for col, counts in value_counts.items():
    print(f"Maximum Value and Minimum Value for {col}:")
    print(f"Maximum Value: {df[col].max()} and Minimum Value: {df[col].min()}\n")


torch.manual_seed(42)

right_skewed_col = ['area_mean', 'texture_mean', 'perimeter_mean', 'radius_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'fractal_dimension_mean', 'area_se', 'texture_se', 'perimeter_se', 'radius_se', 'compactness_se', 'concavity_se', 'concave points_se', 'smoothness_se', 'symmetry_se', 'fractal_dimension_se', 'area_worst', 'texture_worst', 'perimeter_worst', 'radius_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
X_train[right_skewed_col]= PowerTransformer().fit_transform(X_train[right_skewed_col])
X_test


device = 'cuda' if torch.cuda.is_available() else 'cpu'


import seaborn as sns

# Loop through each column in the DataFrame
for column in X_train.columns:
    # Create a new figure for each column
    plt.figure(figsize=(8, 6))
    
    # Plot the histogram using Seaborn
    sns.histplot(X_train[column], kde=True, bins=30)  # kde=True adds a kernel density estimate curve
    
    # Add titles and labels
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    
    # Show the plot
    plt.show()



train_dataset = TensorDataset(torch.tensor(X_train.values).float(), torch.tensor(y_train).float())
test_dataset = TensorDataset(torch.tensor(X_test.values).float(), torch.tensor(y_test).float())

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)



class breastCancerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

model = breastCancerModel()
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    


def train_and_eval():
    basePath = Path(__file__).resolve().parent.parent
    trained_models, metrics = hp.train_and_evaluate(model,train_dataloader, test_dataloader,loss, optimizer,device, 275, 50)
    model_save_path = basePath/'Models'/'SavedModels'/'BreastCancerTracker.pth'
    torch.save(trained_models.state_dict(), model_save_path)
    print(f"Model saved at: {model_save_path}")


if __name__ == "__main__":
    mean_accuracy, mean_precision, mean_auc, mean_loss, fold_accuracies, fold_precisions, fold_aucs, fold_losses = hp.cross_validate(model, train_dataloader, optimizer, loss, cv=5, scoring='accuracy', epochs=100)
    print("Mean Accuracy:", mean_accuracy)
    print("Mean Precision:", mean_precision)
    print("Mean AUC:", mean_auc)    
    print("Mean Loss:", mean_loss)  
    print("Fold Accuracies:", fold_accuracies)  
    print("Fold Precisions:", fold_precisions)  
    print("Fold AUCs:", fold_aucs)  
    print("Fold Losses:", fold_losses)  



