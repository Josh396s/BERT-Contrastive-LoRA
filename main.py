import time
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from loss import SupConLoss
from tqdm import tqdm as progress_bar
from utils import plot_metric, set_seed, setup_gpus, check_directories, plot_losses
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params
from model import Classifier, ScenarioModel, SupConModel, CustomModel
from torch import nn
from peft import get_peft_model, LoraConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def baseline_train(args, model, datasets, tokenizer):
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    
    train_loader = get_dataloader(args, datasets['train'], split='train')
    val_loader = get_dataloader(args, datasets['validation'], split='validation')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    #Include LoRA
    if args.lora:
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=16,
            target_modules=args.lora_targets,
            lora_dropout=args.lora_dropout
        )
        print(f"Trainable Parameters Before LoRA: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        model = get_peft_model(model, config)
        print(f"Trainable Parameters After LoRA: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    
    # Setup model's optimizer_scheduler if you have
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    # Training loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()
        
        for step, batch in progress_bar(enumerate(train_loader),total=len(train_loader)):
            inputs, labels = prepare_inputs(batch)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            losses += loss.item()

        scheduler.step()
        train_losses.append(losses / len(train_loader))
        val_loss, val_acc = run_eval(args, model, datasets, tokenizer, split='validation')
        val_losses.append(val_loss)
        train_accuracies.append(evaluate_knn_accuracy(model, train_loader,device))
        val_accuracies.append(evaluate_knn_accuracy(model, val_loader,device))

        epoch_time = time.time() - start_time

        if epoch_count + 1 == 7:
            with open("results.txt", "a") as f:
                f.write(f"Epoch 7 - Training Time: {epoch_time:.2f} seconds, Val Acc: {val_acc:.2f}%\n")

        print(f'Epoch {epoch_count} | Training Loss: {train_losses[-1]} | Validation Loss: {val_loss} | Validation Accuracy: {val_acc}')
    
    plot_metric(train_accuracies, val_accuracies, "Accuracy", "base2")


def custom_train(args, model, datasets, tokenizer):
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    
    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    
    # Technique 1: layer-wise learning rate decay (LLRD)
    optimizer = model.optimizer

    train_losses, val_losses = [], []
    for epoch_count in range(args.n_epochs):
        start_time = time.time() 
        losses = 0
        model.train()
        
        for step, batch in progress_bar(enumerate(train_dataloader),total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            losses += loss.item()

        train_losses.append(losses / len(train_dataloader))
        val_loss , val_acc = run_eval(args, model, datasets, tokenizer, split='validation')
        val_losses.append(val_loss)

        epoch_time = time.time() - start_time

        if epoch_count + 1 == 7:
            with open("results.txt", "a") as f:
                f.write(f"Epoch 7 - Training Time: {epoch_time:.2f} seconds, Val Acc: {val_acc:.2f}%\n")

        print(f'Epoch {epoch_count} | Training Loss: {train_losses[-1]} | Validation Loss: {val_loss} | Validation Accuracy: {val_acc}')
    
    plot_losses(train_losses, val_losses, "custom")

def run_eval(args, model, datasets, tokenizer, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)
    
    total_loss = 0
    total = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
            inputs, labels = prepare_inputs(batch)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
          
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    print(f'{split} Loss: {avg_loss} Accuracy: {accuracy} | dataset split {split} size:', len(datasets[split]))
    return avg_loss, accuracy

# Evaluate embedding quality with k-NN classifier
def evaluate_knn_accuracy(model, loader, device):
    model.eval()
    embeddings, labels = [], []

    with torch.no_grad():
        for batch in loader:
            inputs, batch_labels = prepare_inputs(batch)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            features = model(inputs)

            embeddings.append(features.cpu())
            labels.append(batch_labels.cpu())

    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(embeddings, labels)
    preds = knn.predict(embeddings)

    return accuracy_score(labels, preds)


def supcon_train(args, model, datasets, tokenizer):
    start_time = time.time()
    train_loader = get_dataloader(args, datasets['train'], split='train')
    val_loader = get_dataloader(args, datasets['validation'], split='validation')

    criterion = SupConLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(args.n_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            start_time = time.time() 
            inputs, labels = prepare_inputs(batch)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = labels.to(device)
    
            features = model(inputs)
            n_views = 2
            batch_size = labels.size(0)
            features = features.view(batch_size, n_views, -1)

            loss = criterion(features, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        train_acc = evaluate_knn_accuracy(model, train_loader, device)
        val_acc = evaluate_knn_accuracy(model, val_loader, device)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        epoch_time = time.time() - start_time

        if epoch + 1 == 7:
            with open("results.txt", "a") as f:
                f.write(f"Epoch 7 - Training Time: {epoch_time:.2f} seconds, Val Acc: {val_acc:.2f}%\n")

        print(f'Epoch {epoch} | Training Loss: {train_losses[-1]} | Validation Loss: {val_losses} | Validation Accuracy: {val_acc}')
        print(f"Epoch [{epoch+1}/{args.n_epochs}] - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # Plotting
    plot_metric( train_accuracies, val_accuracies, "acc", "supcontrain_acc")
    print("Contrastive Pretraining Complete!")

import torch

def fine_tune_classifier(args, classifier, datasets, tokenizer, optimizer=None):
    """
    Fine-tunes the classifier on the given dataset.

    Args:
        args: Training arguments containing hyperparameters.
        classifier: The neural network classifier model.
        datasets (dict): Dictionary containing 'train' and 'validation' datasets.
        tokenizer: Tokenizer for input processing.
        optimizer: Optimizer (default: None, initializes Adam).

    Returns:
        classifier: Fine-tuned classifier.
    """
    # Get DataLoaders
    train_loader = get_dataloader(args, datasets['train'], split='train')
    val_loader = get_dataloader(args, datasets['validation'], split='validation')

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Move model to the appropriate device
    classifier.to(device)

    # If optimizer is not provided, initialize Adam optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    # Track losses and accuracy
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(args.n_epochs):
        classifier.train()
        total_train_loss, correct_train, total_train = 0, 0, 0

        for batch in train_loader:
            inputs, labels = prepare_inputs(batch)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = labels.to(device)

            # Forward pass
            logits = classifier(inputs)
            loss = criterion(logits, labels)
    
            logits = classifier(inputs)
    
            loss = criterion(logits, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Compute training accuracy
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # Validation step
        val_loss, val_acc = run_eval(args, classifier, datasets, tokenizer, split='validation')

        # Store results
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: "
              f"Train Loss = {avg_train_loss:.4f}, Train Acc = {train_accuracy:.2f}% | "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")

    # Plot Losses
    plot_metric(train_losses, val_losses, "Loss", "loss_plot")

    # Plot Accuracy
    plot_metric(train_accs, val_accs, "Accuracy", "accuracy_plot")

    print("Classifier Fine-Tuning Complete!")
    return classifier


if __name__ == "__main__":
    args = params()
    args = setup_gpus(args)
    args = check_directories(args)
    set_seed(args)

    cache_results, already_exist = check_cache(args)
    tokenizer = load_tokenizer(args)

    if already_exist:
        features = cache_results
    else:
        data = load_data()
        features = prepare_features(args, data, tokenizer, cache_results)
    datasets = process_data(args, features, tokenizer)
    for k,v in datasets.items():
        print(k, len(v))
 
    if args.task == 'baseline':
        model = ScenarioModel(args, tokenizer, target_size=18).to(device)
        run_eval(args, model, datasets, tokenizer, split='validation')
        run_eval(args, model, datasets, tokenizer, split='test')
        baseline_train(args, model, datasets, tokenizer)
        run_eval(args, model, datasets, tokenizer, split='test')
    elif args.task == 'custom':
        model = CustomModel(args, tokenizer, target_size=18).to(device)
        run_eval(args, model, datasets, tokenizer, split='validation')
        run_eval(args, model, datasets, tokenizer, split='test')
        custom_train(args, model, datasets, tokenizer)
        run_eval(args, model, datasets, tokenizer, split='test')
    elif args.task == 'supcon':
        model = SupConModel(args, tokenizer, target_size = 18, feat_dim=768).to(device)
        print(model)
        supcon_train(args, model, datasets, tokenizer)

    # Fine-Tune Classifier on Frozen Encoder
    # Freeze encoder weights
    for param in model.encoder.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = False

    # Replace projection head with classifier
    model.head = Classifier(args, target_size=18).to(device)
    
    # Ensure the classifier's parameters are trainable
    for param in model.head.parameters():
        param.requires_grad = True

    # Reset optimizer for classifier head only
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.learning_rate)

    # Train classifier (ensure data is not normalized for classifier training)
    fine_tune_classifier(args, model, datasets, tokenizer, optimizer=optimizer)

    # Evaluate final classifier
    run_eval(args, model, datasets, tokenizer, split='test')
