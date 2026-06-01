"""
train_model.py  —  Train the ADHD classifier on real EEG data

Dataset structure (auto-detected):
  <root>/ADHD_part1/*.mat  → label 1 (ADHD)
  <root>/ADHD_part2/*.mat  → label 1 (ADHD)
  <root>/Control_part1/*.mat → label 0 (Control)
  <root>/Control_part2/*.mat → label 0 (Control)

Usage:
  # Full EEG pipeline from .mat files (recommended):
  python train_model.py --adhd_root ../ADHD

  # Fast mode — use precomputed CSV features:
  python train_model.py --csv "../ADHD/CSV Files/ADHD Data Set.csv"

  # Legacy subdirectory mode:
  python train_model.py --data_dir /path --adhd_dir adhd --normal_dir normal

  # Synthetic demo:
  python train_model.py --demo
"""

import os, sys, argparse, json, time
import numpy as np

# Adjust path for direct execution
sys.path.insert(0, os.path.dirname(__file__))
from eeg_processor import EEGProcessor
from adhd_predictor import ADHDPredictor


# ─── Dataset loaders ────────────────────────────────────────────────────────

def load_dataset_from_adhd_root(adhd_root: str) -> tuple:
    """
    Load from the ADHD project root which contains:
      ADHD_part1/, ADHD_part2/  → ADHD subjects (label 1)
      Control_part1/, Control_part2/ → Control subjects (label 0)
    Returns (X, y, filenames, feature_names)
    """
    processor = EEGProcessor()
    X, y, filenames = [], [], []
    feature_names = None

    dir_label_map = [
        ('ADHD_part1',    1),
        ('ADHD_part2',    1),
        ('Control_part1', 0),
        ('Control_part2', 0),
    ]

    for subdir, label in dir_label_map:
        folder = os.path.join(adhd_root, subdir)
        if not os.path.isdir(folder):
            print(f'  ⚠ Directory not found, skipping: {folder}')
            continue
        mat_files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.mat'))
        print(f'\n  [{subdir}] → label={label} ({len(mat_files)} files)')
        for fname in mat_files:
            fpath = os.path.join(folder, fname)
            tag = 'ADHD' if label == 1 else 'Control'
            try:
                proc = processor.preprocess(fpath)
                feats = processor.extract_features(proc)
                vec = feats['feature_vector']
                X.append(vec)
                y.append(label)
                filenames.append(f'{subdir}/{fname}')
                if feature_names is None:
                    feature_names = feats['feature_names']
                print(f'    ✓ [{tag}] {fname}  ({len(vec)} features, '
                      f'{proc["n_epochs"]} epochs)')
            except Exception as e:
                print(f'    ⚠ Skipping {fname}: {e}')

    return np.array(X, dtype=float), np.array(y, dtype=int), filenames, feature_names


def load_dataset_from_csv(csv_path: str) -> tuple:
    """
    Load precomputed mean/median/mode features from the project CSV file.
    CSV structure:
      Row 0: channel group headers (FZ, CZ, PZ, …)
      Row 1: stat headers (mean, median, mode, …)
      Col 0: Sr No.
      Col -1: ADHD/healthy label (0=Control, 1=ADHD)
    Returns (X, y, filenames, feature_names)
    """
    import pandas as pd

    df = pd.read_csv(csv_path, header=[0, 1])

    # Last column is the label
    label_col = df.columns[-1]
    y = df[label_col].values.astype(int)

    # Build feature matrix: all columns except first (Sr No.) and last (label)
    feat_df = df.iloc[:, 1:-1]

    # Build meaningful feature names from multi-level columns
    channels = [
        'FZ', 'CZ', 'PZ', 'C3', 'T3', 'C4', 'T4',
        'FP1', 'FP2', 'F3', 'F4', 'F7', 'F8',
        'P3', 'P4', 'T5', 'T6', 'O1', 'O2'
    ]
    stats = ['mean', 'median', 'mode']
    feature_names = [f'{ch}_{st}' for ch in channels for st in stats]

    X = feat_df.values.astype(float)
    # Trim/pad feature_names to match actual column count
    n_features = X.shape[1]
    if len(feature_names) > n_features:
        feature_names = feature_names[:n_features]
    elif len(feature_names) < n_features:
        feature_names += [f'feat_{i}' for i in range(len(feature_names), n_features)]

    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    filenames = [f'subject_{i+1}' for i in range(len(y))]
    return X, y, filenames, feature_names


def load_dataset_from_dir(data_dir: str, label_file: str = None,
                           adhd_subdir: str = 'adhd', normal_subdir: str = 'normal'):
    """
    Legacy loader: separate subdirs or flat dir + CSV label file.
    """
    processor = EEGProcessor()
    X, y, filenames = [], [], []
    feature_names = None

    adhd_dir   = os.path.join(data_dir, adhd_subdir)
    normal_dir = os.path.join(data_dir, normal_subdir)

    if os.path.isdir(adhd_dir) and os.path.isdir(normal_dir):
        for label, folder in [(1, adhd_dir), (0, normal_dir)]:
            for fname in sorted(os.listdir(folder)):
                if not fname.lower().endswith('.mat'):
                    continue
                fpath = os.path.join(folder, fname)
                try:
                    print(f'  Processing [{("ADHD" if label else "Normal")}] {fname}…')
                    proc  = processor.preprocess(fpath)
                    feats = processor.extract_features(proc)
                    X.append(feats['feature_vector'])
                    y.append(label)
                    filenames.append(fname)
                    if feature_names is None:
                        feature_names = feats['feature_names']
                except Exception as e:
                    print(f'  ⚠ Skipping {fname}: {e}')
    elif label_file and os.path.isfile(label_file):
        import csv
        labels = {}
        with open(label_file) as f:
            for row in csv.DictReader(f):
                labels[row.get('filename', row.get('file', ''))] = int(
                    row.get('label', row.get('class', 0)))
        for fname in sorted(os.listdir(data_dir)):
            if not fname.lower().endswith('.mat'):
                continue
            fpath = os.path.join(data_dir, fname)
            label = labels.get(fname, labels.get(os.path.splitext(fname)[0], None))
            if label is None:
                print(f'  ⚠ No label for {fname}, skipping.')
                continue
            try:
                print(f'  Processing [{("ADHD" if label else "Normal")}] {fname}…')
                proc  = processor.preprocess(fpath)
                feats = processor.extract_features(proc)
                X.append(feats['feature_vector'])
                y.append(label)
                filenames.append(fname)
                if feature_names is None:
                    feature_names = feats['feature_names']
            except Exception as e:
                print(f'  ⚠ Skipping {fname}: {e}')
    else:
        raise ValueError(
            'Specify either:\n'
            '  --data_dir with adhd/ and normal/ subdirs, or\n'
            '  --data_dir + --label_file (CSV with columns: filename, label)')

    return np.array(X, dtype=float), np.array(y, dtype=int), filenames, feature_names


def generate_synthetic_dataset(n_adhd: int = 80, n_normal: int = 80):
    """Generate synthetic training data for demo/testing."""
    print(f'  Generating {n_adhd} ADHD + {n_normal} Normal synthetic samples…')
    rng = np.random.default_rng(42)

    def adhd_features():
        base = rng.standard_normal(57) * 0.5
        base[:19]  += rng.uniform(2, 4, 19)    # high mean amplitude
        base[19:38] -= rng.uniform(1, 3, 19)   # lower median
        base[38:]  += rng.uniform(1, 2, 19)    # higher mode
        return base

    def normal_features():
        base = rng.standard_normal(57) * 0.5
        base[:19]  -= rng.uniform(0.5, 1.5, 19)
        base[19:38] += rng.uniform(0.5, 1.5, 19)
        return base

    X = np.vstack([
        np.array([adhd_features() for _ in range(n_adhd)]),
        np.array([normal_features() for _ in range(n_normal)]),
    ])
    y = np.array([1] * n_adhd + [0] * n_normal)
    idx = rng.permutation(len(y))
    channels = [
        'FZ', 'CZ', 'PZ', 'C3', 'T3', 'C4', 'T4',
        'FP1', 'FP2', 'F3', 'F4', 'F7', 'F8',
        'P3', 'P4', 'T5', 'T6', 'O1', 'O2'
    ]
    feature_names = [f'{ch}_{st}' for ch in channels for st in ['mean', 'median', 'mode']]
    return X[idx], y[idx], feature_names


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train NeuroADHD classifier')

    # Primary modes
    parser.add_argument('--adhd_root', type=str, default=None,
                        help='Root folder containing ADHD_part1/, ADHD_part2/, '
                             'Control_part1/, Control_part2/ subdirectories')
    parser.add_argument('--csv',       type=str, default=None,
                        help='Path to precomputed features CSV (fast mode)')

    # Legacy modes
    parser.add_argument('--data_dir',   type=str, default=None,
                        help='(Legacy) Directory with adhd/ and normal/ subdirs')
    parser.add_argument('--label_file', type=str, default=None,
                        help='(Legacy) CSV file with filename, label columns')
    parser.add_argument('--adhd_dir',   type=str, default='adhd')
    parser.add_argument('--normal_dir', type=str, default='normal')

    parser.add_argument('--demo',  action='store_true', help='Train on synthetic demo data')
    parser.add_argument('--cv',    type=int, default=5, help='Cross-validation folds')
    args = parser.parse_args()

    print('=' * 60)
    print('  NeuroADHD — Model Training Script')
    print('=' * 60)

    # Auto-detect dataset root if not supplied
    if not any([args.adhd_root, args.csv, args.data_dir, args.demo]):
        # Try relative path from backend/ → ../ADHD
        candidate = os.path.join(os.path.dirname(__file__), '..', 'ADHD')
        if os.path.isdir(os.path.join(candidate, 'ADHD_part1')):
            args.adhd_root = os.path.abspath(candidate)
            print(f'\n[Auto-detected] ADHD root: {args.adhd_root}')
        else:
            # Try CSV fallback
            csv_candidate = os.path.join(
                os.path.dirname(__file__), '..', 'ADHD', 'CSV Files', 'ADHD Data Set.csv')
            if os.path.isfile(csv_candidate):
                args.csv = os.path.abspath(csv_candidate)
                print(f'\n[Auto-detected] CSV: {args.csv}')
            else:
                print('ERROR: Could not auto-detect dataset. Use --adhd_root, --csv, or --demo')
                sys.exit(1)

    predictor = ADHDPredictor()
    t0 = time.time()

    if args.demo:
        print('\n[Mode] Synthetic demo dataset')
        X, y, feature_names = generate_synthetic_dataset(n_adhd=80, n_normal=80)

    elif args.adhd_root:
        print(f'\n[Mode] Full EEG pipeline from .mat files')
        print(f'  Root: {args.adhd_root}')
        X, y, files, feature_names = load_dataset_from_adhd_root(args.adhd_root)
        print(f'\nLoaded {len(y)} subjects: {int(y.sum())} ADHD, {int((y==0).sum())} Control')
        print(f'Feature vector size: {X.shape[1]}')

    elif args.csv:
        print(f'\n[Mode] Precomputed CSV features (fast)')
        print(f'  File: {args.csv}')
        X, y, files, feature_names = load_dataset_from_csv(args.csv)
        print(f'\nLoaded {len(y)} subjects: {int(y.sum())} ADHD, {int((y==0).sum())} Control')
        print(f'Feature vector size: {X.shape[1]}')

    else:
        print(f'\n[Mode] Legacy directory mode')
        X, y, files, feature_names = load_dataset_from_dir(
            args.data_dir, args.label_file, args.adhd_dir, args.normal_dir)
        print(f'\nLoaded {len(y)} subjects: {int(y.sum())} ADHD, {int((y==0).sum())} Normal')

    if len(y) < 10:
        print('ERROR: Need at least 10 samples to train. Exiting.')
        sys.exit(1)

    print(f'\nTraining ensemble classifier (RF + SVM + XGBoost) with {args.cv}-fold CV…')
    meta = predictor.train(X, y, feature_names=feature_names, cv_folds=args.cv)

    elapsed = time.time() - t0
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'adhd_model.joblib')

    print('\n' + '=' * 60)
    print('  TRAINING RESULTS')
    print('=' * 60)
    print(f"  Dataset         : {meta['n_samples']} subjects "
          f"({meta['n_adhd']} ADHD, {meta['n_normal']} Control)")
    print(f"  Features        : {meta['n_features']}")
    print(f"  CV Folds        : {args.cv}")
    print(f"  CV Accuracy     : {meta['cv_accuracy_mean']:.4f} +/- {meta['cv_accuracy_std']:.4f}")
    print(f"  CV AUC-ROC      : {meta['cv_auc_mean']:.4f} +/- {meta['cv_auc_std']:.4f}")
    print(f"  Train Accuracy  : {meta['train_accuracy']:.4f}")
    print(f"  Train AUC-ROC   : {meta['train_auc']:.4f}")
    if 'classification_report' in meta:
        print(f"\n  Classification Report:")
        print('  ' + meta['classification_report'].replace('\n', '\n  '))
    if 'confusion_matrix' in meta:
        cm = meta['confusion_matrix']
        print(f"  Confusion Matrix (rows=actual, cols=predicted):")
        print(f"    Control: TN={cm[0][0]:3d}  FP={cm[0][1]:3d}")
        print(f"    ADHD   : FN={cm[1][0]:3d}  TP={cm[1][1]:3d}")
    print('-' * 60)
    print(f"  Model saved to  : {os.path.abspath(model_path)}")
    print(f"  Total time      : {elapsed:.1f}s")
    print('=' * 60)
    print('\nTraining complete! The model is ready for inference.')


if __name__ == '__main__':
    main()
