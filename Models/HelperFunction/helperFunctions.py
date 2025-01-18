from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import  roc_curve,accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.utils.data as data
import time
import timeit
from tqdm import tqdm


def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer,device, num_epochs=150, epochs_rate=10):
    train_losses = []  # List to store the training loss for each epoch
    test_losses = []   # List to store the test loss for each epoch
    train_accuracies = []  # List to store the training accuracy for each epoch
    test_accuracies = []   # List to store the test accuracy for each epoch
    model.to(device)
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        correct_train_preds = 0
        total_train_preds = 0
        
        for inputs, labels in train_loader:
            # Ensure inputs and labels are moved to the correct device
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
                # Ensure inputs and labels are moved to the correct device
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
        if (epoch + 1) % epochs_rate == 0:
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
            # Ensure inputs and labels are moved to the correct device
            inputs, labels = inputs.to(torch.float32).to(device), labels.to(torch.float32).unsqueeze(1).to(device)
            
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())  # Move predictions back to CPU for metrics calculation
            all_labels.extend(labels.cpu().numpy())  # Move labels back to CPU

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


def cross_validate(model, X, y,optimizer, loss_fn,cv=5, scoring='accuracy',  regression=False, device=None, batch_size=32, epochs=1000  ):
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_precisions = []
    fold_aucs = []
    fold_losses = []
    
    # Loop through the splits
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Training fold {fold + 1}/{cv}")
        
        X_train, X_val = torch.tensor(X[train_idx], dtype=torch.float32).to(device), torch.tensor(X[val_idx], dtype=torch.float32).to(device)
        y_train, y_val = torch.tensor(y[train_idx], dtype=torch.float32).to(device), torch.tensor(y[val_idx], dtype=torch.float32).to(device)

        # Reset model parameters to avoid data leakage between folds
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

        # Create DataLoader for batch processing
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        val_data = torch.utils.data.TensorDataset(X_val, y_val)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

        # Training loop
        model.train()
        epoch_accuracies = []
        epoch_losses = []

        for epoch in range(epochs):
            running_loss = 0.0
            correct_preds = 0
            total_preds = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_X).squeeze()
                loss = loss_fn(predictions, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                # Calculate accuracy
                preds = (predictions > 0.5).float()
                correct_preds += (preds == batch_y).sum().item()
                total_preds += len(batch_y)
            
            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct_preds / total_preds
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

        # Store fold results
        fold_losses.append(np.mean(epoch_losses))
        fold_accuracies.append(np.mean(epoch_accuracies))

        # Evaluate the model
        # Evaluate the model
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_labels = []
            for batch_X, batch_y in val_loader:
                val_preds_batch = model(batch_X).squeeze().cpu().numpy()  # Get probabilities
                val_labels_batch = batch_y.cpu().numpy()
                
                val_preds.extend(val_preds_batch)  # Store probabilities for AUC
                val_labels.extend(val_labels_batch)

            # Calculate metrics
            accuracy = accuracy_score(val_labels, (np.array(val_preds) > 0.5).astype(int))
            precision = precision_score(val_labels, (np.array(val_preds) > 0.5).astype(int))

            # Calculate AUC
            auc = roc_auc_score(val_labels, val_preds)  # Use probabilities for AUC
            
            # Confusion matrix for correct and incorrect predictions
            tn, fp, fn, tp = confusion_matrix(val_labels, (np.array(val_preds) > 0.5).astype(int)).ravel()
            correct_predictions = tp + tn
            incorrect_predictions = fp + fn
            fpr, tpr, _ = roc_curve(val_labels, val_preds)

            print(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, AUC: {auc:.4f}")
            print(f"Correct Predictions: {correct_predictions}, Incorrect Predictions: {incorrect_predictions}")

    mean_accuracy = np.mean(fold_accuracies)
    
    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fold_accuracies, marker='o', label='Accuracy per fold')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per fold in Cross-validation')
    plt.grid(True)
    plt.xticks(np.arange(cv))

    plt.subplot(1, 2, 2)
    plt.plot(fold_losses, marker='o', label='Loss per fold')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.title('Loss per fold in Cross-validation')
    plt.grid(True)
    plt.xticks(np.arange(cv))

    plt.tight_layout()
    plt.show()
    
    cm = confusion_matrix(val_labels, (np.array(val_preds) > 0.5).astype(int))
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred: 0', 'Pred: 1'], yticklabels=['True: 0', 'True: 1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for Fold {fold + 1}')
    plt.show()
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, marker='o', label=f"Fold {fold + 1}")
    plt.plot([0, 1], [0, 1], 'k--')  # Random classifier line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Fold {fold + 1}')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()


    return mean_accuracy, fold_accuracies, fold_losses

def train_and_evaluate_2d(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=150, epochs_rate=10):
    start_time = timeit.default_timer()  # Start timing the whole training process

    train_losses = []  # List to store the training loss for each epoch
    test_losses = []   # List to store the test loss for each epoch
    train_accuracies = []  # List to store the training accuracy for each epoch
    test_accuracies = []   # List to store the test accuracy for each epoch
    model.to(device)

    for epoch in range(num_epochs):
        # Start time for each epoch
        epoch_start_time = timeit.default_timer()

        # Training Phase
        model.train()
        train_loss = 0.0
        correct_train_preds = 0
        total_train_preds = 0
        
        # Add tqdm for progress bar during training
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", ncols=100):
            # Ensure inputs and labels are moved to the correct device
            inputs, labels = inputs.to(torch.float32).to(device), labels.to(torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # If it's multi-class classification, use CrossEntropyLoss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            correct_train_preds += (preds == labels).sum().item()
            total_train_preds += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = correct_train_preds / total_train_preds
        train_accuracies.append(train_accuracy)

        # Evaluation Phase (Test Loss and Accuracy)
        model.eval()
        test_loss = 0.0
        correct_test_preds = 0
        total_test_preds = 0
        all_preds = []
        all_labels = []
        
        # Add tqdm for progress bar during evaluation
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Epoch {epoch + 1} Evaluation", ncols=100):
                # Ensure inputs and labels are moved to the correct device
                inputs, labels = inputs.to(torch.float32).to(device), labels.to(torch.long).to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Calculate accuracy
                _, preds = torch.max(outputs, 1)
                correct_test_preds += (preds == labels).sum().item()
                total_test_preds += labels.size(0)

                # Collect all predictions and labels for confusion matrix
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accuracy = correct_test_preds / total_test_preds
        test_accuracies.append(test_accuracy)

        # Print training and test loss every 'epochs_rate' epochs
        if (epoch + 1) % epochs_rate == 0:
            print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Calculate the time taken for the epoch
        epoch_end_time = timeit.default_timer()
        epoch_time = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch + 1} took {epoch_time:.2f} seconds")

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
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')  # Average for multi-class classification
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    roc_auc = roc_auc_score(all_labels, all_preds, average='macro', multi_class='ovr')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")

    # Generate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row (true label)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=np.arange(cm.shape[0]), yticklabels=np.arange(cm.shape[1]))
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # Record and print total time taken for training and evaluation
    end_time = timeit.default_timer()  # End timing the whole training process
    total_time = end_time - start_time  # Calculate the total time taken
    print(f"Total training and evaluation time: {total_time:.2f} seconds")
    
    return model, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
