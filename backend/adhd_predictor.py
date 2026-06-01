"""
adhd_predictor.py  â€”  ADHD classification â€” CNN+TCN deep learning predictor

Primary predictor: CNN+TCN model trained on raw EEG epochs
Legacy fallback:   RF+SVM+XGBoost ensemble (if CNN+TCN not available)

Architecture pipeline:
  Raw EEG (.mat/.edf/.csv)
  -> preprocess (bandpass 0.5-50Hz, notch, artifact rejection)
  -> Z-score normalise per channel per epoch
  -> CNN Spatial Feature Learning
  -> TCN Temporal Pattern Recognition
  -> Dense + Softmax -> ADHD / Normal + confidence + Grad-CAM heatmap
"""

import os
import json
import warnings
import numpy as np
from typing import Dict, List, Optional

warnings.filterwarnings('ignore')

# â”€â”€â”€ optional imports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    from sklearn.ensemble        import RandomForestClassifier, VotingClassifier
    from sklearn.svm             import SVC
    from sklearn.preprocessing   import RobustScaler
    from sklearn.pipeline        import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.calibration     import CalibratedClassifierCV
    from sklearn.metrics         import (accuracy_score, roc_auc_score,
                                         confusion_matrix, classification_report)
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

MODEL_DIR        = os.path.join(os.path.dirname(__file__), 'models')
CNN_TCN_PATH     = os.path.join(MODEL_DIR, 'cnn_tcn_model.pth')
LEGACY_PATH      = os.path.join(MODEL_DIR, 'adhd_model.joblib')
META_PATH        = os.path.join(MODEL_DIR, 'model_meta.json')
os.makedirs(MODEL_DIR, exist_ok=True)

N_CHANNELS    = 19
EPOCH_SAMPLES = 256   # 2s @ 128Hz
SRATE         = 128


# ==============================================================================
# CNN+TCN Predictor (primary)
# ==============================================================================

class CNNTCNPredictor:
    """
    Wraps the trained CNN+TCN model for inference on raw EEG recordings.

    Usage
    -----
    predictor = CNNTCNPredictor()
    result    = predictor.predict_from_file('subject.mat')
    """

    def __init__(self):
        self.model      = None
        self.device     = None
        self.meta: Dict = {}
        self._grad_cam  = None
        self._load()

    def _load(self):
        if not HAS_TORCH:
            print('[CNNTCNPredictor] PyTorch not available.')
            return
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if os.path.exists(CNN_TCN_PATH):
            try:
                from cnn_tcn_model import build_model, GradCAMHook
                checkpoint = torch.load(CNN_TCN_PATH, map_location=self.device,
                                        weights_only=False)
                n_ch = checkpoint.get('n_channels', N_CHANNELS)
                n_t  = checkpoint.get('epoch_samples', EPOCH_SAMPLES)
                self.model = build_model(n_channels=n_ch, epoch_samples=n_t)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device).eval()
                self._grad_cam = GradCAMHook(self.model)
                print(f'[CNNTCNPredictor] Loaded CNN+TCN from {CNN_TCN_PATH}')
            except Exception as e:
                print(f'[CNNTCNPredictor] Could not load CNN+TCN model: {e}')
                self.model = None

        if os.path.exists(META_PATH):
            try:
                with open(META_PATH) as f:
                    self.meta = json.load(f)
            except Exception:
                pass

    def is_trained(self) -> bool:
        return self.model is not None

    def predict_from_file(self, filepath: str) -> Dict:
        """Full pipeline: load file -> preprocess -> predict."""
        from eeg_processor import EEGProcessor
        processor = EEGProcessor(default_srate=SRATE)
        epochs_arr, n_ep = processor.get_cnn_tcn_epochs(
            filepath,
            epoch_len=2.0, overlap=0.5,
            artifact_thresh=100.0,
            target_channels=N_CHANNELS,
            target_samples=EPOCH_SAMPLES,
        )
        if n_ep == 0:
            return self._fallback_result('Could not extract clean epochs')
        return self.predict_from_epochs(epochs_arr)

    def predict_from_epochs(self, epochs_arr: np.ndarray) -> Dict:
        """Predict from epoch array (N, C, T). Returns subject-level result."""
        if self.model is None:
            return self._fallback_result('Model not loaded')

        X = torch.FloatTensor(epochs_arr).to(self.device)

        with torch.no_grad():
            probs = F.softmax(self.model(X), dim=1)

        prob_adhd_per_epoch = probs[:, 1].cpu().numpy()
        prob_adhd  = float(prob_adhd_per_epoch.mean())
        prediction = 'ADHD' if prob_adhd > 0.5 else 'Normal'
        confidence = abs(prob_adhd - 0.5) * 2.0
        heatmap    = self._get_attention(X)

        return {
            'prediction':        prediction,
            'probability':       round(prob_adhd, 4),
            'confidence':        round(confidence, 4),
            'n_epochs':          int(len(epochs_arr)),
            'epoch_probs':       prob_adhd_per_epoch.tolist(),
            'attention_heatmap': heatmap,
            'model_type':        'CNN+TCN',
        }

    def _get_attention(self, X: 'torch.Tensor') -> List[float]:
        if self._grad_cam is None or not HAS_TORCH:
            return []
        try:
            self.model.eval()
            X_req  = X[:1].clone().detach().requires_grad_(True)
            logits = self.model(X_req)
            self.model.zero_grad()
            logits[0, 1].backward()
            cam = self._grad_cam.get_cam()
            return (cam[0].tolist() if cam.ndim > 1 else cam.tolist())
        except Exception:
            return []

    @staticmethod
    def _fallback_result(reason: str) -> Dict:
        return {
            'prediction': 'Unknown', 'probability': 0.5,
            'confidence': 0.0, 'n_epochs': 0,
            'epoch_probs': [], 'attention_heatmap': [],
            'model_type': 'CNN+TCN', 'error': reason,
        }

    def model_info(self) -> Dict:
        return {
            'algo':       'CNN + TCN (Deep Learning)',
            'accuracy':   f"{self.meta.get('val_accuracy', 0):.1%}",
            'auc':        f"{self.meta.get('val_auc', 0):.3f}",
            'validation': 'Subject-level holdout split',
            'trained':    self.is_trained(),
            'model_type': 'CNN+TCN',
        }


# ==============================================================================
# Legacy RF+SVM+XGBoost Ensemble (fallback / secondary validation)
# ==============================================================================

class ADHDPredictor:
    """Legacy feature-based ensemble predictor."""

    def __init__(self):
        self.pipeline = None
        self.meta: Dict = {}
        self._load_or_init()

    def _load_or_init(self):
        if HAS_JOBLIB and os.path.exists(LEGACY_PATH):
            try:
                self.pipeline = joblib.load(LEGACY_PATH)
                if os.path.exists(META_PATH):
                    with open(META_PATH) as f:
                        self.meta = json.load(f)
                print(f'[ADHDPredictor] Loaded legacy ensemble from {LEGACY_PATH}')
                return
            except Exception as e:
                print(f'[ADHDPredictor] Could not load legacy model: {e}')
        if HAS_SKLEARN:
            self.pipeline = self._build_pipeline()
        self.meta = {'trained': False}

    def _build_pipeline(self):
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_split=4,
            class_weight='balanced', random_state=42, n_jobs=-1,
        )
        svm_cal = CalibratedClassifierCV(
            SVC(kernel='rbf', C=10, gamma='scale', probability=False, random_state=42),
            cv=5, method='sigmoid'
        )
        estimators = [('rf', rf), ('svm', svm_cal)]
        if HAS_XGB:
            estimators.append(('xgb', XGBClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=5,
                eval_metric='logloss', random_state=42, n_jobs=-1,
            )))
        return Pipeline([
            ('scaler',   RobustScaler()),
            ('ensemble', VotingClassifier(estimators=estimators, voting='soft')),
        ])

    def train(self, X: np.ndarray, y: np.ndarray,
              feature_names=None, cv_folds: int = 5) -> Dict:
        if not HAS_SKLEARN:
            raise RuntimeError('scikit-learn not installed')
        X = np.nan_to_num(np.array(X, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.array(y, dtype=int)

        skf    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_acc = cross_val_score(self.pipeline, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        cv_auc = cross_val_score(self.pipeline, X, y, cv=skf, scoring='roc_auc',  n_jobs=-1)
        self.pipeline.fit(X, y)

        y_pred = self.pipeline.predict(X)
        y_prob = self.pipeline.predict_proba(X)[:, 1]
        cm     = confusion_matrix(y, y_pred).tolist()

        self.meta = {
            'model_type':            'ensemble',
            'trained':               True,
            'n_samples':             int(len(y)),
            'n_adhd':                int(y.sum()),
            'n_normal':              int((y == 0).sum()),
            'n_features':            int(X.shape[1]),
            'feature_names':         feature_names or [f'f{i}' for i in range(X.shape[1])],
            'cv_accuracy_mean':      float(cv_acc.mean()),
            'cv_accuracy_std':       float(cv_acc.std()),
            'cv_auc_mean':           float(cv_auc.mean()),
            'cv_auc_std':            float(cv_auc.std()),
            'train_accuracy':        float(accuracy_score(y, y_pred)),
            'train_auc':             float(roc_auc_score(y, y_prob)),
            'confusion_matrix':      cm,
            'classification_report': classification_report(y, y_pred,
                                         target_names=['Control', 'ADHD']),
        }
        if HAS_JOBLIB:
            joblib.dump(self.pipeline, LEGACY_PATH)
        with open(META_PATH, 'w') as f:
            json.dump(self.meta, f, indent=2)
        print(f'[ADHDPredictor] CV Acc={cv_acc.mean():.4f} +/- {cv_acc.std():.4f}')
        return self.meta

    def predict(self, feature_vector) -> Dict:
        X = np.nan_to_num(
            np.array(feature_vector, dtype=float).reshape(1, -1),
            nan=0.0, posinf=1e6, neginf=-1e6)
        if self.is_trained():
            prob_adhd = float(self.pipeline.predict_proba(X)[0, 1])
        else:
            prob_adhd = 0.5
        return {
            'prediction':  'ADHD' if prob_adhd > 0.5 else 'Normal',
            'probability': round(prob_adhd, 4),
            'confidence':  round(abs(prob_adhd - 0.5) * 2.0, 4),
            'model_type':  'ensemble',
        }

    def is_trained(self) -> bool:
        return (
            self.meta.get('trained', False)
            and self.pipeline is not None
            and HAS_SKLEARN
            and hasattr(self.pipeline.named_steps.get('ensemble', None), 'estimators_')
        )

    def model_info(self) -> Dict:
        algo = 'RF + SVM' + (' + XGBoost' if HAS_XGB else '') + ' Ensemble'
        return {
            'algo':       algo,
            'accuracy':   f"{self.meta.get('cv_accuracy_mean', 0):.1%}",
            'auc':        f"{self.meta.get('cv_auc_mean', 0):.3f}",
            'validation': '5-Fold CV',
            'trained':    self.is_trained(),
            'model_type': 'ensemble',
        }

    def get_feature_importances(self) -> Optional[Dict]:
        if not self.is_trained():
            return None
        try:
            rf = self.pipeline.named_steps['ensemble'].estimators_[0][1]
            if hasattr(rf, 'feature_importances_'):
                imps  = rf.feature_importances_
                names = self.meta.get('feature_names', [f'f{i}' for i in range(len(imps))])
                return dict(sorted(zip(names, imps.tolist()), key=lambda x: -x[1]))
        except Exception:
            pass
        return None


# ==============================================================================
# Unified predictor  (auto-selects best available model)
# ==============================================================================

class UnifiedPredictor:
    """
    Selects CNN+TCN (primary) when available, else falls back to ensemble.
    All external code should use this class.
    """

    def __init__(self):
        self.cnn_tcn  = CNNTCNPredictor()
        self.ensemble = ADHDPredictor()

    @property
    def active(self):
        return self.cnn_tcn if self.cnn_tcn.is_trained() else self.ensemble

    def predict_from_file(self, filepath: str) -> Dict:
        if self.cnn_tcn.is_trained():
            return self.cnn_tcn.predict_from_file(filepath)
        from eeg_processor import EEGProcessor
        proc      = EEGProcessor(default_srate=SRATE)
        processed = proc.preprocess(filepath)
        feats     = proc.extract_features(processed)
        return self.ensemble.predict(feats['feature_vector'])

    def predict_from_features(self, feature_vector) -> Dict:
        return self.ensemble.predict(feature_vector)

    def model_info(self) -> Dict:
        return self.active.model_info()

    def is_trained(self) -> bool:
        return self.cnn_tcn.is_trained() or self.ensemble.is_trained()
