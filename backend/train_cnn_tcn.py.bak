"""
train_cnn_tcn.py  —  Training script for CNN+TCN ADHD classifier

Architecture pipeline (matches system diagram):
  1. Load .mat EEG files (ADHD_part1/2, Control_part1/2)
  2. Preprocess: bandpass 0.5-50Hz, notch 50Hz, artifact rejection, 2s epochs
  3. Z-score normalise per channel per epoch
  4. Subject-level train/val split (no data leakage)
  5. Train CNN+TCN model
  6. Evaluate and save best checkpoint

Usage:
    python train_cnn_tcn.py
    python train_cnn_tcn.py --adhd_root ../ADHD --epochs 50 --batch_size 32
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             confusion_matrix, classification_report)

# Add backend to path when running from project root
sys.path.insert(0, os.path.dirname(__file__))
from cnn_tcn_model import build_model, count_params
from eeg_processor import EEGProcessor

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_ADHD_ROOT  = os.path.join(os.path.dirname(__file__), '..', 'ADHD')
DEFAULT_MODEL_DIR  = os.path.join(os.path.dirname(__file__), 'models')
DATASET_FOLDERS    = {
    'ADHD_part1':    1,
    'ADHD_part2':    1,
    'Control_part1': 0,
    'Control_part2': 0,
}
EPOCH_LEN          = 2.0    # seconds
SRATE              = 128    # Hz (no srate in files)
EPOCH_SAMPLES      = int(EPOCH_LEN * SRATE)   # 256
N_CHANNELS         = 19


# ─── Dataset loading ──────────────────────────────────────────────────────────

def load_all_subjects(adhd_root: str, verbose: bool = True):
    """
    Load raw EEG epochs from all 4 dataset folders.

    Returns
    -------
    X : np.ndarray  (total_epochs, N_CHANNELS, EPOCH_SAMPLES)
    y : np.ndarray  (total_epochs,)  0=Control, 1=ADHD
    groups : np.ndarray  (total_epochs,)  subject index (for CV)
    """
    processor = EEGProcessor(default_srate=SRATE)

    X_all, y_all, groups = [], [], []
    subject_id = 0
    total_files = 0
    failed_files = 0

    for folder, label in DATASET_FOLDERS.items():
        folder_path = os.path.join(adhd_root, folder)
        if not os.path.isdir(folder_path):
            if verbose:
                print(f'  [WARN] Folder not found: {folder_path}')
            continue

        mat_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mat')])
        if verbose:
            print(f'  {folder}: {len(mat_files)} subjects  (label={label})')

        for fname in mat_files:
            fpath = os.path.join(folder_path, fname)
            total_files += 1
            try:
                epochs_arr, n_ep = processor.get_cnn_tcn_epochs(
                    fpath,
                    epoch_len=EPOCH_LEN,
                    overlap=0.5,
                    artifact_thresh=100.0,
                    target_channels=N_CHANNELS,
                    target_samples=EPOCH_SAMPLES,
                )
                if n_ep == 0:
                    failed_files += 1
                    continue

                X_all.append(epochs_arr)
                y_all.extend([label] * n_ep)
                groups.extend([subject_id] * n_ep)
                subject_id += 1

                if verbose:
                    print(f'    {fname}: {n_ep} epochs')

            except Exception as e:
                failed_files += 1
                if verbose:
                    print(f'    [FAIL] {fname}: {e}')

    if not X_all:
        raise RuntimeError('No EEG epochs loaded. Check ADHD root path.')

    X = np.concatenate(X_all, axis=0).astype(np.float32)
    y = np.array(y_all, dtype=np.int64)
    groups = np.array(groups, dtype=np.int64)

    if verbose:
        n_subj = subject_id
        n_adhd = int((y == 1).sum())
        n_ctrl = int((y == 0).sum())
        print(f'\n  Total: {n_subj} subjects | {len(y)} epochs '
              f'(ADHD={n_adhd}, Control={n_ctrl})')
        print(f'  Epoch shape: {X.shape}')
        if failed_files:
            print(f'  Failed to load: {failed_files}/{total_files} files')

    return X, y, groups


# ─── Training helpers ─────────────────────────────────────────────────────────

def augment_batch(X: torch.Tensor, noise_std: float = 0.05) -> torch.Tensor:
    """Simple online augmentation: Gaussian noise."""
    noise = torch.randn_like(X) * noise_std
    return X + noise


def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Balance classes in each mini-batch."""
    counts  = np.bincount(y)
    weights = 1.0 / counts[y]
    return WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))


def train_epoch(model, loader, criterion, optimizer, device, augment=True):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        if augment:
            Xb = augment_batch(Xb)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        n          += len(yb)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_probs, all_labels  = [], []
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        loss   = criterion(logits, yb)
        probs  = torch.softmax(logits, 1)[:, 1]
        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        n          += len(yb)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())
    acc = correct / n
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
    return total_loss / n, acc, auc


# ─── Main training function ───────────────────────────────────────────────────

def train(adhd_root: str, model_dir: str,
          n_epochs: int = 50, batch_size: int = 32,
          lr: float = 1e-3, weight_decay: float = 1e-4,
          patience: int = 12, val_split: float = 0.2,
          device_str: str = 'auto', verbose: bool = True):

    os.makedirs(model_dir, exist_ok=True)

    # ── Device ───────────────────────────────────────────
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f'\nDevice: {device}')

    # ── Load data ────────────────────────────────────────
    print('\n[1/5] Loading EEG epochs...')
    X, y, groups = load_all_subjects(adhd_root, verbose=verbose)

    # ── Subject-level split ──────────────────────────────
    print('\n[2/5] Splitting subjects into train/val...')
    unique_subjects = np.unique(groups)
    subj_labels = np.array([int(y[groups == s].mean() >= 0.5)
                            for s in unique_subjects])

    train_subjs, val_subjs = train_test_split(
        unique_subjects, test_size=val_split, random_state=42,
        stratify=subj_labels
    )

    train_mask = np.isin(groups, train_subjs)
    val_mask   = np.isin(groups, val_subjs)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    print(f'  Train epochs: {len(y_train)}  |  Val epochs: {len(y_val)}')
    print(f'  Train subjects: {len(train_subjs)}  |  Val subjects: {len(val_subjs)}')

    # ── DataLoaders ──────────────────────────────────────
    sampler = make_weighted_sampler(y_train)
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.LongTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # ── Model ────────────────────────────────────────────
    print('\n[3/5] Building CNN+TCN model...')
    model = build_model(n_channels=N_CHANNELS, epoch_samples=EPOCH_SAMPLES).to(device)
    print(f'  Parameters: {count_params(model):,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    # ── Training loop ────────────────────────────────────
    print(f'\n[4/5] Training for up to {n_epochs} epochs (early stop patience={patience})...')
    best_val_acc = 0.0
    best_epoch   = 0
    no_improve   = 0
    history      = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}

    best_model_path = os.path.join(model_dir, 'cnn_tcn_model.pth')

    for ep in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        dt = time.time() - t0
        print(f'  Ep {ep:3d}/{n_epochs} | '
              f'loss {train_loss:.4f}/{val_loss:.4f} | '
              f'acc {train_acc:.4f}/{val_acc:.4f} | '
              f'AUC {val_auc:.4f} | {dt:.1f}s')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = ep
            no_improve   = 0
            torch.save({
                'epoch':      ep,
                'model_state_dict': model.state_dict(),
                'val_acc':    val_acc,
                'val_auc':    val_auc,
                'n_channels': N_CHANNELS,
                'epoch_samples': EPOCH_SAMPLES,
            }, best_model_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  Early stopping at epoch {ep} (best epoch {best_epoch})')
                break

    # ── Final evaluation ─────────────────────────────────
    print(f'\n[5/5] Evaluating best model (epoch {best_epoch}, val_acc={best_val_acc:.4f})...')

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, final_acc, final_auc = eval_epoch(model, val_loader, criterion, device)

    # Full predictions for confusion matrix
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            preds = model(Xb.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    cm = confusion_matrix(all_labels, all_preds).tolist()
    cr = classification_report(all_labels, all_preds, target_names=['Control','ADHD'])

    print(f'\n  Val Accuracy : {final_acc:.4f}')
    print(f'  Val AUC-ROC  : {final_auc:.4f}')
    print(f'\n  Confusion Matrix:\n  {cm}')
    print(f'\n  Classification Report:\n{cr}')

    # ── Save metadata ────────────────────────────────────
    meta = {
        'model_type':    'cnn_tcn',
        'trained':       True,
        'n_channels':    N_CHANNELS,
        'epoch_samples': EPOCH_SAMPLES,
        'srate':         SRATE,
        'best_epoch':    best_epoch,
        'val_accuracy':  round(final_acc, 4),
        'val_auc':       round(final_auc, 4),
        'confusion_matrix': cm,
        'n_train_epochs': len(y_train),
        'n_val_epochs':   len(y_val),
        'history': {
            'val_acc': [round(v, 4) for v in history['val_acc']],
            'val_auc': [round(v, 4) for v in history['val_auc']],
        }
    }
    meta_path = os.path.join(model_dir, 'model_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n  Model saved   : {best_model_path}')
    print(f'  Metadata saved: {meta_path}')

    return meta


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train CNN+TCN ADHD EEG Classifier')
    p.add_argument('--adhd_root',  default=DEFAULT_ADHD_ROOT,
                   help='Root folder containing ADHD_part1/2, Control_part1/2')
    p.add_argument('--model_dir',  default=DEFAULT_MODEL_DIR)
    p.add_argument('--epochs',     type=int,   default=50)
    p.add_argument('--batch_size', type=int,   default=32)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--patience',   type=int,   default=12)
    p.add_argument('--val_split',  type=float, default=0.2)
    p.add_argument('--device',     default='auto', choices=['auto','cpu','cuda'])
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('=' * 60)
    print(' CNN+TCN ADHD EEG Classifier - Training')
    print('=' * 60)
    print(f'  ADHD root : {args.adhd_root}')
    print(f'  Model dir : {args.model_dir}')
    print(f'  Epochs    : {args.epochs}')
    print(f'  Batch     : {args.batch_size}')
    print(f'  LR        : {args.lr}')
    print('=' * 60)

    meta = train(
        adhd_root   = args.adhd_root,
        model_dir   = args.model_dir,
        n_epochs    = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        patience    = args.patience,
        val_split   = args.val_split,
        device_str  = args.device,
    )
    print('\nTraining complete.')
