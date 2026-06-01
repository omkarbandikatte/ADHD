"""
train_cnn_tcn.py  —  Enhanced training for CNN+TCN ADHD EEG classifier

Key improvements over baseline:
  1. FocalLoss  — down-weights easy samples; forces model to learn hard ones
  2. Label Smoothing (0.1)  — prevents overconfident predictions
  3. Rich augmentation per batch:
       - Gaussian noise
       - Random time shift (±10%)
       - Amplitude scaling per channel (±15%)
       - Channel dropout (p=0.1)
       - Mixup (alpha=0.2)
  4. OneCycleLR scheduler (outperforms CosineAnnealing on small datasets)
  5. Early stopping on val AUC (more reliable than accuracy for ADHD/Control)
  6. Optimal decision threshold via Youden's J statistic
  7. Full metrics report: Sensitivity, Specificity, F1, ROC-AUC, MCC

Usage:
    python train_cnn_tcn.py
    python train_cnn_tcn.py --adhd_root ../ADHD --epochs 80 --batch_size 32
"""

import os, sys, json, time, argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
)

sys.path.insert(0, os.path.dirname(__file__))
from cnn_tcn_model import build_model, count_params
from eeg_processor import EEGProcessor

# ─── Paths / constants ────────────────────────────────────────────────────────

DEFAULT_ADHD_ROOT = os.path.join(os.path.dirname(__file__), '..', 'ADHD')
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

DATASET_FOLDERS = {
    'ADHD_part1':    1,
    'ADHD_part2':    1,
    'Control_part1': 0,
    'Control_part2': 0,
}
EPOCH_LEN     = 2.0    # seconds
SRATE         = 128    # Hz
EPOCH_SAMPLES = int(EPOCH_LEN * SRATE)   # 256
N_CHANNELS    = 19


# ─── Loss functions ───────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=2 focuses training on hard misclassified examples.
    class_weights handles any residual label imbalance.
    """

    def __init__(self, gamma: float = 2.0,
                 class_weights: torch.Tensor = None,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.gamma           = gamma
        self.class_weights   = class_weights
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.size(1)

        # Label smoothing
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smoothing / (n_cls - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_p  = F.log_softmax(logits, dim=1)
        p      = torch.exp(log_p)

        # Focal weight: (1 - p_t)^gamma
        p_t    = (p * smooth_targets).sum(dim=1)
        focal  = (1.0 - p_t) ** self.gamma

        # Per-sample cross-entropy with smoothed labels
        ce     = -(smooth_targets * log_p).sum(dim=1)
        loss   = focal * ce

        if self.class_weights is not None:
            w    = self.class_weights[targets]
            loss = loss * w

        return loss.mean()


# ─── Dataset loading ──────────────────────────────────────────────────────────

def load_all_subjects(adhd_root: str, verbose: bool = True):
    """
    Load raw EEG epochs from all 4 dataset folders.

    Returns
    -------
    X      : (total_epochs, N_CHANNELS, EPOCH_SAMPLES)
    y      : (total_epochs,)  0=Control, 1=ADHD
    groups : (total_epochs,)  subject index — used to prevent data leakage
    """
    processor = EEGProcessor(default_srate=SRATE)

    X_all, y_all, groups = [], [], []
    subject_id   = 0
    total_files  = 0
    failed_files = 0

    for folder, label in DATASET_FOLDERS.items():
        folder_path = os.path.join(adhd_root, folder)
        if not os.path.isdir(folder_path):
            if verbose:
                print(f'  [WARN] Folder not found: {folder_path}')
            continue

        mat_files = sorted(f for f in os.listdir(folder_path) if f.endswith('.mat'))
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

    X      = np.concatenate(X_all, axis=0).astype(np.float32)
    y      = np.array(y_all,    dtype=np.int64)
    groups = np.array(groups,   dtype=np.int64)

    if verbose:
        n_adhd = int((y == 1).sum())
        n_ctrl = int((y == 0).sum())
        print(f'\n  Subjects loaded : {subject_id}')
        print(f'  Epochs total    : {len(y)}  (ADHD={n_adhd}, Control={n_ctrl})')
        print(f'  Epoch shape     : {X.shape}')
        if failed_files:
            print(f'  Failed          : {failed_files}/{total_files}')

    return X, y, groups


# ─── Augmentation ─────────────────────────────────────────────────────────────

def augment_batch(X: torch.Tensor,
                  noise_std:  float = 0.05,
                  time_shift: float = 0.10,
                  amp_scale:  float = 0.15,
                  ch_drop_p:  float = 0.10) -> torch.Tensor:
    """In-batch GPU/CPU augmentation."""
    # 1. Gaussian noise
    X = X + torch.randn_like(X) * noise_std

    # 2. Cyclic time shift
    if time_shift > 0:
        T         = X.shape[-1]
        max_shift = max(1, int(T * time_shift))
        shift     = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if shift != 0:
            X = torch.roll(X, int(shift), dims=-1)

    # 3. Per-channel amplitude jitter
    if amp_scale > 0:
        scale = 1.0 + torch.randn(X.shape[0], X.shape[1], 1,
                                   device=X.device) * amp_scale
        X = X * scale.clamp(0.5, 2.0)

    # 4. Channel dropout
    if ch_drop_p > 0:
        mask = (torch.rand(X.shape[0], X.shape[1], 1,
                           device=X.device) > ch_drop_p).float()
        X = X * mask

    return X


def mixup_batch(X: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Blend two random samples and their labels."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(X.shape[0], device=X.device)
    X_mix = lam * X + (1.0 - lam) * X[idx]
    return X_mix, y, y[idx], lam


# ─── Utilities ────────────────────────────────────────────────────────────────

def compute_class_weights(y: np.ndarray, device) -> torch.Tensor:
    counts  = np.bincount(y, minlength=2)
    weights = len(y) / (2.0 * counts.astype(float))
    return torch.FloatTensor(weights).to(device)


def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    counts  = np.bincount(y)
    weights = 1.0 / counts[y].astype(float)
    return WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))


# ─── Train / eval ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        Xb = augment_batch(Xb)

        # Mixup
        Xb, ya, yb_mix, lam = mixup_batch(Xb, yb)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb_mix)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        n          += len(yb)

    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for Xb, yb in loader:
        probs = torch.softmax(model(Xb.to(device)), dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(yb.numpy())
    y_true  = np.array(all_labels)
    y_probs = np.array(all_probs)
    acc     = accuracy_score(y_true, (y_probs >= 0.5).astype(int))
    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = 0.0
    return acc, auc, y_true, y_probs


# ─── Metrics ──────────────────────────────────────────────────────────────────

def find_best_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Youden's J: maximise Sensitivity + Specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    best_idx = int(np.argmax(tpr - fpr))
    return float(thresholds[best_idx])


def compute_all_metrics(y_true: np.ndarray, y_probs: np.ndarray,
                        threshold: float = 0.5) -> dict:
    y_pred = (y_probs >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    f1          = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = 0.0
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        'sensitivity':    round(float(sensitivity), 4),
        'specificity':    round(float(specificity), 4),
        'f1':             round(float(f1),          4),
        'auc':            round(float(auc),          4),
        'mcc':            round(float(mcc),          4),
        'accuracy':       round(float(accuracy_score(y_true, y_pred)), 4),
        'threshold':      round(float(threshold),    4),
        'confusion_matrix': cm.tolist(),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    }


# ─── Main training ────────────────────────────────────────────────────────────

def train(adhd_root: str, model_dir: str,
          n_epochs:     int   = 80,
          batch_size:   int   = 32,
          lr:           float = 1e-3,
          weight_decay: float = 1e-4,
          patience:     int   = 15,
          val_split:    float = 0.2,
          device_str:   str   = 'auto',
          verbose:      bool  = True):

    os.makedirs(model_dir, exist_ok=True)

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if device_str == 'auto' else torch.device(device_str))
    print(f'\nDevice: {device}')

    # Load
    print('\n[1/5] Loading EEG epochs...')
    X, y, groups = load_all_subjects(adhd_root, verbose=verbose)

    # Subject-level split
    print('\n[2/5] Subject-level train/val split...')
    unique_subjects = np.unique(groups)
    subj_labels     = np.array([int(y[groups == s].mean() >= 0.5)
                                for s in unique_subjects])
    train_subjs, val_subjs = train_test_split(
        unique_subjects, test_size=val_split, random_state=42,
        stratify=subj_labels,
    )
    train_mask = np.isin(groups, train_subjs)
    val_mask   = np.isin(groups, val_subjs)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    print(f'  Train: {len(y_train)} epochs ({(y_train==1).sum()} ADHD) | {len(train_subjs)} subjects')
    print(f'  Val  : {len(y_val)} epochs ({(y_val==1).sum()} ADHD)   | {len(val_subjs)} subjects')

    # DataLoaders
    sampler      = make_weighted_sampler(y_train)
    train_ds     = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds       = TensorDataset(torch.FloatTensor(X_val),   torch.LongTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # Model + loss + optimiser
    print('\n[3/5] Building enhanced CNN+TCN model...')
    model = build_model(n_channels=N_CHANNELS, epoch_samples=EPOCH_SAMPLES).to(device)
    print(f'  Parameters: {count_params(model):,}')

    class_weights = compute_class_weights(y_train, device)
    criterion     = FocalLoss(gamma=2.0, class_weights=class_weights, label_smoothing=0.1)
    optimizer     = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler     = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs,
        pct_start=0.1,
        anneal_strategy='cos',
    )

    # Training loop — early stop on AUC
    print(f'\n[4/5] Training (max {n_epochs} epochs, patience={patience} on AUC)...')
    best_val_auc = 0.0
    best_epoch   = 0
    no_improve   = 0
    history      = {'val_acc': [], 'val_auc': []}
    best_path    = os.path.join(model_dir, 'cnn_tcn_model.pth')

    for ep in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device)
        val_acc, val_auc, _, _ = eval_epoch(model, val_loader, device)

        history['val_acc'].append(round(val_acc, 4))
        history['val_auc'].append(round(val_auc, 4))

        flag = ''
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = ep
            no_improve   = 0
            flag         = '  ← best AUC'
            torch.save({
                'epoch': ep, 'model_state_dict': model.state_dict(),
                'val_acc': val_acc, 'val_auc': val_auc,
                'n_channels': N_CHANNELS, 'epoch_samples': EPOCH_SAMPLES,
            }, best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  Early stopping at epoch {ep}')
                break

        print(f'  Ep {ep:3d}/{n_epochs} | loss {train_loss:.4f} | '
              f'tr_acc {train_acc:.4f} | val_acc {val_acc:.4f} | '
              f'AUC {val_auc:.4f} | {time.time()-t0:.1f}s{flag}')

    # Final evaluation
    print(f'\n[5/5] Evaluating best model (epoch {best_epoch}, AUC={best_val_auc:.4f})...')
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    _, _, y_true, y_probs = eval_epoch(model, val_loader, device)

    m05      = compute_all_metrics(y_true, y_probs, threshold=0.5)
    opt_thr  = find_best_threshold(y_true, y_probs)
    mopt     = compute_all_metrics(y_true, y_probs, threshold=opt_thr)

    print('\n  ┌────────────────────────┬────────────┬──────────────────┐')
    print('  │ Metric                 │ thresh=0.5 │ Optimal thresh   │')
    print('  ├────────────────────────┼────────────┼──────────────────┤')
    for key, label in [('sensitivity','Sensitivity'), ('specificity','Specificity'),
                       ('f1','F1-Score'), ('auc','ROC-AUC'), ('mcc','MCC'),
                       ('accuracy','Accuracy')]:
        print(f'  │ {label:<22} │   {m05[key]:.4f}   │   {mopt[key]:.4f}           │')
    print(f'  │ {"Threshold":<22} │   0.5000   │   {opt_thr:.4f}           │')
    print('  └────────────────────────┴────────────┴──────────────────┘')

    cm = mopt['confusion_matrix']
    print(f'\n  Confusion Matrix (optimal threshold={opt_thr:.3f}):')
    print(f'                  Pred:Control  Pred:ADHD')
    print(f'  Actual:Control   TN={cm[0][0]:4d}       FP={cm[0][1]:4d}')
    print(f'  Actual:ADHD      FN={cm[1][0]:4d}       TP={cm[1][1]:4d}')
    y_pred_opt = (y_probs >= opt_thr).astype(int)
    print(f'\n  Classification Report:\n'
          + classification_report(y_true, y_pred_opt,
                                  target_names=['Control','ADHD'], digits=4))

    # Save metadata
    meta = {
        'model_type': 'cnn_tcn', 'version': '2.0', 'trained': True,
        'n_channels': N_CHANNELS, 'epoch_samples': EPOCH_SAMPLES, 'srate': SRATE,
        'best_epoch':        best_epoch,
        'val_accuracy':      mopt['accuracy'],
        'val_auc':           mopt['auc'],
        'val_sensitivity':   mopt['sensitivity'],
        'val_specificity':   mopt['specificity'],
        'val_f1':            mopt['f1'],
        'val_mcc':           mopt['mcc'],
        'optimal_threshold': opt_thr,
        'confusion_matrix':  cm,
        'n_subjects':        int(len(unique_subjects)),
        'dataset':           'IEEE DataPort ADHD EEG (19-channel .mat)',
        'validation':        'Subject-level holdout split',
        'history':           history,
    }
    with open(os.path.join(model_dir, 'model_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n  Model saved: {best_path}')
    return meta


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--adhd_root',    default=DEFAULT_ADHD_ROOT)
    p.add_argument('--model_dir',    default=DEFAULT_MODEL_DIR)
    p.add_argument('--epochs',       type=int,   default=80)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience',     type=int,   default=15)
    p.add_argument('--val_split',    type=float, default=0.2)
    p.add_argument('--device',       default='auto', choices=['auto','cpu','cuda'])
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('=' * 65)
    print('  CNN+TCN Enhanced Training — Sensitivity/Specificity/F1/AUC/MCC')
    print('=' * 65)
    t0   = time.time()
    meta = train(
        adhd_root=args.adhd_root, model_dir=args.model_dir,
        n_epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, val_split=args.val_split,
        device_str=args.device,
    )
    print(f'\nTotal time: {time.time()-t0:.1f}s')
    print('Next step: python export_onnx.py  →  push api/models/cnn_tcn_model.onnx')
