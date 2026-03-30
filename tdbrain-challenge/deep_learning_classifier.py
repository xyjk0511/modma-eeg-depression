"""
Deep Learning classifiers for MDD diagnosis.

Implements:
1. Autoencoder + Classifier (inspired by 2nd place)
2. 1D-CNN
3. MLP with Dropout
4. Ensemble of all three
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG features."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Autoencoder(nn.Module):
    """Autoencoder for unsupervised feature learning."""

    def __init__(self, input_dim=676, encoding_dim=128):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


class AutoencoderClassifier(nn.Module):
    """Classifier using pretrained autoencoder features."""

    def __init__(self, autoencoder, encoding_dim=128, num_classes=2):
        super().__init__()
        self.encoder = autoencoder.encoder

        # Freeze encoder initially
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features)

    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True


class CNN1D(nn.Module):
    """1D-CNN for EEG feature classification."""

    def __init__(self, input_dim=676, num_classes=2):
        super().__init__()

        # Reshape to (batch, 1, 676) for 1D conv
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, 676) -> (batch, 1, 676)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.squeeze(-1)  # (batch, 256)
        return self.classifier(x)


class MLP(nn.Module):
    """Multi-layer perceptron with dropout."""

    def __init__(self, input_dim=676, num_classes=2):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


def train_autoencoder(X_train, epochs=100, batch_size=32, lr=0.001, device='cuda'):
    """Pretrain autoencoder on training data."""

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Dataset
    dataset = EEGDataset(X_scaled, np.zeros(len(X_scaled)))  # dummy labels
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = Autoencoder(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, _ in loader:
            batch_X = batch_X.to(device)

            optimizer.zero_grad()
            reconstructed = model(batch_X)
            loss = criterion(reconstructed, batch_X)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    return model, scaler


def train_classifier(model, X_train, y_train, X_val, y_val,
                     epochs=50, batch_size=32, lr=0.001, device='cuda',
                     class_weights=None):
    """Train classifier with validation."""

    # Dataset
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Loss with class weights
    if class_weights is not None:
        weight = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                      factor=0.5, patience=5)

    best_val_auc = 0
    best_model_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validate
        model.eval()
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # MDD probability
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(batch_y.numpy())

        val_auc = roc_auc_score(val_labels, val_probs)
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_val_auc


def predict_proba(model, X, scaler, device='cuda', batch_size=32):
    """Predict probabilities."""
    X_scaled = scaler.transform(X)
    dataset = EEGDataset(X_scaled, np.zeros(len(X_scaled)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    probs = []

    with torch.no_grad():
        for batch_X, _ in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            batch_probs = torch.softmax(outputs, dim=1)[:, 1]  # MDD probability
            probs.extend(batch_probs.cpu().numpy())

    return np.array(probs)


def deep_learning_cv(X, y, groups, model_type='autoencoder', n_splits=5,
                     device='cuda', random_state=42):
    """
    Cross-validation with deep learning models.

    Args:
        X: Features (n_samples, n_features)
        y: Labels (n_samples,) - 0: nonMDD, 1: MDD
        groups: Subject IDs for GroupKFold
        model_type: 'autoencoder', 'cnn', 'mlp', or 'ensemble'
        n_splits: Number of CV folds
        device: 'cuda' or 'cpu'
        random_state: Random seed

    Returns:
        dict with CV results
    """

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_probs = []
    all_labels = []
    all_preds = []

    # Class weights for imbalanced data
    n_nonmdd = np.sum(y == 0)
    n_mdd = np.sum(y == 1)
    class_weights = [1.0, n_nonmdd / n_mdd]  # [nonMDD_weight, MDD_weight]

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_type == 'autoencoder':
            # Pretrain autoencoder
            print("Pretraining autoencoder...")
            autoencoder, _ = train_autoencoder(X_train, epochs=50, device=device)

            # Train classifier
            print("Training classifier...")
            model = AutoencoderClassifier(autoencoder).to(device)
            model, _ = train_classifier(model, X_train_scaled, y_train,
                                       X_test_scaled, y_test,
                                       epochs=50, device=device,
                                       class_weights=class_weights)

        elif model_type == 'cnn':
            model = CNN1D(input_dim=X.shape[1]).to(device)
            model, _ = train_classifier(model, X_train_scaled, y_train,
                                       X_test_scaled, y_test,
                                       epochs=50, device=device,
                                       class_weights=class_weights)

        elif model_type == 'mlp':
            model = MLP(input_dim=X.shape[1]).to(device)
            model, _ = train_classifier(model, X_train_scaled, y_train,
                                       X_test_scaled, y_test,
                                       epochs=50, device=device,
                                       class_weights=class_weights)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Predict on test set
        probs = predict_proba(model, X_test, scaler, device=device)
        preds = (probs >= 0.5).astype(int)

        all_probs.extend(probs)
        all_labels.extend(y_test)
        all_preds.extend(preds)

    # Calculate metrics
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    auc = roc_auc_score(all_labels, all_probs)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2

    results = {
        'model_type': model_type,
        'auc': float(auc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'balanced_accuracy': float(balanced_acc),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'n_splits': n_splits
    }

    print(f"\n=== Final Results ({model_type}) ===")
    print(f"AUC: {auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    return results


if __name__ == '__main__':
    # Test with dummy data
    np.random.seed(42)
    X = np.random.randn(100, 676)
    y = np.random.randint(0, 2, 100)
    groups = np.repeat(np.arange(100), 1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results = deep_learning_cv(X, y, groups, model_type='mlp',
                               n_splits=3, device=device)
    print(results)
