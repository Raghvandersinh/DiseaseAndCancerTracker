import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer,device, num_epochs=150):
    train_losses = []  # List to store the training loss for each epoch
    test_losses = []   # List to store the test loss for each epoch
    train_accuracies = []  # List to store the training accuracy for each epoch
    test_accuracies = []   # List to store the test accuracy for each epoch

    for epoch in range(num_epochs):
        # Training Phase
        model.train().to(device)
        train_loss = 0.0
        correct_train_preds = 0
        total_train_preds = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(torch.float32).to(device), labels.to(torch.float32).unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            preds = (outputs > 0.5).float()  # Assuming binary classification
            correct_train_preds += (preds == labels).sum().item()
            total_train_preds += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = correct_train_preds / total_train_preds  # Accuracy for training set
        train_accuracies.append(train_accuracy)

        # Evaluation Phase (Test Loss and Accuracy)
        model.eval()
        test_loss = 0.0
        correct_test_preds = 0
        total_test_preds = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(torch.float32).to(device), labels.to(torch.float32).unsqueeze(1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Calculate accuracy
                preds = (outputs > 0.5).float()  # Assuming binary classification
                correct_test_preds += (preds == labels).sum().item()
                total_test_preds += labels.size(0)

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accuracy = correct_test_preds / total_test_preds  # Accuracy for test set
        test_accuracies.append(test_accuracy)

        # Print training and test loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Plot the training and test loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', color='red', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the training and test accuracy over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy', color='red', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # After training, evaluate the model on the test set
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(torch.float32), labels.to(torch.float32).unsqueeze(1)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics Calculation
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




def visualize_classification(train_loader, test_loader=None, model=None):
    # Extract training data from the train_loader
    X_train = []
    y_train = []
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for inputs, labels in train_loader:
            X_train.append(inputs.numpy())  # Convert inputs to numpy array
            y_train.append(labels.numpy())  # Convert labels to numpy array
    X_train = np.vstack(X_train)  # Stack the batches into a single array
    y_train = np.hstack(y_train)  # Stack the labels into a single array

    # Visualize training data
    plt.figure(figsize=(12, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Before Training Data')
    plt.colorbar(label='Class')
    plt.show()

    # If a model is provided, plot the decision boundary
    if model is not None:
        h = .02
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict on the meshgrid
        Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32))
        Z = Z.detach().numpy().reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.8)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolors='k')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('After Training Data')
        plt.colorbar(label='Class')
        plt.show()

    # If test_loader is provided, visualize the test data and confusion matrix
    if test_loader is not None:
        X_test = []
        y_test = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                X_test.append(inputs.numpy())
                y_test.append(labels.numpy())
        X_test = np.vstack(X_test)
        y_test = np.hstack(y_test)

        # Make predictions on the test set
        all_preds = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs, labels = inputs.to(torch.float32), labels.to(torch.float32).unsqueeze(1)
                outputs = model(inputs)
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())

        # Generate confusion matrix
        cm = confusion_matrix(y_test, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()


            

