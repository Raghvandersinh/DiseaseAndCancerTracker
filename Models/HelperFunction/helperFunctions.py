import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=150):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(torch.float32), labels.to(torch.float32).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}")
    
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(torch.float32), labels.to(torch.float32).unsqueeze(1)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    

    return model, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

def visualize_classification(X_train, y_train, X_test=None, y_test=None, model=None):
    plt.figure(figsize=(12, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                cmap='viridis', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Before Training Data')
    plt.colorbar(label='Class')
    plt.show()

    if model is not None:
        h = .02
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:,0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:,1].max() + 1  
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape) 

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.8)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolors='k')    
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('After Training Data')    
        plt.colorbar(label='Class')
        plt.show()

    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        plt.title('Confusion Matrix')   
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()


            

