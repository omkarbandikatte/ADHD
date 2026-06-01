"""
api/index.py  —  NeuroADHD Vercel Serverless API
Handles: /api/health, /api/upload, /api/process, /api/predict, /api/demo
Uses ONNX Runtime (no PyTorch) to stay within Vercel's 250MB function limit.
"""

import os, io, uuid, json, traceback, warnings
import numpy as np
import scipy.io as sio
import scipy.signal as sig
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

warnings.filterwarnings('ignore')

# NumPy 2.0 removed np.trapz — use np.trapezoid on 2.x, shim on 1.x
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

# ─── App setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# ─── Model paths (relative to this file) ─────────────────────────────────────
_DIR      = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(_DIR, 'models', 'cnn_tcn_model.onnx')
META_PATH = os.path.join(_DIR, 'models', 'model_meta.json')

# ─── Constants ───────────────────────────────────────────────────────────────
N_CHANNELS    = 19
EPOCH_SAMPLES = 256
SRATE         = 128
BANDS         = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
                 'beta': (13, 30), 'gamma': (30, 40)}
CH_NAMES      = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
                 'T3','C3','Cz','C4','T4',
                 'T5','P3','Pz','P4','T6','O1','O2']
ALLOWED_EXT   = {'.mat', '.csv', '.edf'}

# ─── Lazy-loaded ONNX session ─────────────────────────────────────────────────
_ort_session = None
_meta        = {}

def _get_session():
    global _ort_session, _meta
    if _ort_session is None:
        import onnxruntime as ort
        _ort_session = ort.InferenceSession(
            ONNX_PATH, providers=['CPUExecutionProvider']
        )
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                _meta = json.load(f)
    return _ort_session


# ═══════════════════════════════════════════════════════════════════════════════
# EEG Loading & Preprocessing (inline — no external modules)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_mat_bytes(raw_bytes: bytes):
    """Load a .mat file from bytes → (data: np.ndarray [n_ch x n_samp], srate)."""
    buf = io.BytesIO(raw_bytes)
    try:
        mat = sio.loadmat(buf, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        raise ValueError(f'Cannot parse .mat file: {e}')

    data, srate = None, None

    # EEGLAB EEG struct
    if 'EEG' in mat:
        eeg = mat['EEG']
        try:
            data  = np.array(eeg.data, dtype=float)
            srate = int(eeg.srate)
        except AttributeError:
            pass

    # Common variable names
    if data is None:
        for key in ['data','eeg','EEG','signal','Signal','eegdata','raw']:
            if key in mat and isinstance(mat[key], np.ndarray):
                data = np.array(mat[key], dtype=float)
                break

    # First non-metadata 2D array
    if data is None:
        for key, val in mat.items():
            if key.startswith('_'):
                continue
            if isinstance(val, np.ndarray) and val.ndim >= 2:
                data = np.array(val, dtype=float)
                break

    if data is None:
        raise ValueError('No EEG data array found in .mat file')

    # Sampling rate
    if srate is None:
        for key in ['srate','fs','Fs','freq','SamplingRate','sfreq']:
            if key in mat:
                try:
                    srate = int(np.squeeze(mat[key]))
                    break
                except Exception:
                    pass
    if srate is None:
        srate = 128

    # Ensure (n_ch, n_samp)
    data = data.squeeze()
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] > data.shape[1]:
        data = data.T

    return data, srate


def _bandpass(data: np.ndarray, srate: int, lo=0.5, hi=50.0) -> np.ndarray:
    # hi=50 Hz matches EEGProcessor training pipeline (was 40 Hz — mismatch fixed)
    nyq = srate / 2.0
    lo  = max(lo, 0.1)
    hi  = min(hi, nyq * 0.95)
    b, a = sig.butter(4, [lo / nyq, hi / nyq], btype='band')
    return sig.filtfilt(b, a, data, axis=-1)


def _notch(data: np.ndarray, srate: int, freq=50.0) -> np.ndarray:
    nyq = srate / 2.0
    if freq >= nyq:
        return data
    b, a = sig.iirnotch(freq / nyq, 30.0)
    return sig.filtfilt(b, a, data, axis=-1)


def _make_epochs(data: np.ndarray, srate: int,
                 epoch_sec: float = 2.0, n_ch: int = 19, ep_samples: int = 256,
                 overlap: float = 0.5, artifact_thresh: float = 100.0):
    """
    Resample → trim channels → 50% overlap epochs → artifact rejection (100 µV)
    → per-epoch per-channel z-score.  Matches EEGProcessor training pipeline exactly.
    """
    n_ch_actual, n_samp = data.shape

    # ── Trim / pad channels to N_CHANNELS ─────────────────────────────────
    if n_ch_actual >= n_ch:
        data = data[:n_ch, :]
    else:
        pad = np.zeros((n_ch - n_ch_actual, n_samp))
        data = np.vstack([data, pad])

    # ── Resample to 128 Hz ──────────────────────────────────────────────────
    if srate != 128:
        new_len = int(n_samp * 128 / srate)
        data    = sig.resample(data, new_len, axis=1)
    n_samp = data.shape[1]

    # ── Slice with 50% overlap (step=128) ──────────────────────────────────
    step     = int(ep_samples * (1.0 - overlap))        # 128 samples
    starts   = range(0, n_samp - ep_samples + 1, step)
    raw_eps  = [data[:, s:s + ep_samples] for s in starts]

    # ── Artifact rejection on raw (pre-z-score) epochs ─────────────────────
    clean = [ep for ep in raw_eps if np.abs(ep).max() <= artifact_thresh]
    if not clean:                                        # safety: keep least-noisy half
        clean = sorted(raw_eps, key=lambda e: np.abs(e).max())
        clean = clean[:max(1, len(clean) // 2)]

    # ── Per-epoch, per-channel z-score (matches training) ──────────────────
    normed = []
    for ep in clean:
        mu  = ep.mean(axis=1, keepdims=True)
        std = ep.std(axis=1,  keepdims=True) + 1e-8
        normed.append((ep - mu) / std)

    return np.stack(normed, axis=0).astype(np.float32)  # (n_epochs, N_CH, EP_SAMP)


def _band_power(ch_data: np.ndarray, srate: int) -> dict:
    """Return band powers for one channel."""
    f, pxx = sig.welch(ch_data, fs=srate, nperseg=min(256, len(ch_data)))
    powers = {}
    for band, (lo, hi) in BANDS.items():
        mask = (f >= lo) & (f <= hi)
        powers[band] = float(np.trapezoid(pxx[mask], f[mask])) if mask.any() else 0.0
    return powers


def _extract_band_powers(data: np.ndarray, srate: int, ch_names):
    """Return {channel_name: {band: power}} for all channels."""
    result = {}
    for i, ch in enumerate(ch_names):
        result[ch] = _band_power(data[i], srate)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ONNX Inference
# ═══════════════════════════════════════════════════════════════════════════════

def _predict_epochs(epochs: np.ndarray):
    """
    Run ONNX inference on (n_epochs, 19, 256) array.
    Returns: prediction, probability, confidence, epoch_probs, attention_heatmap
    """
    sess       = _get_session()
    input_name = sess.get_inputs()[0].name

    # Run batch inference
    logits = sess.run(['logits'], {input_name: epochs})[0]  # (n_epochs, 2)

    # Softmax → probabilities
    exp   = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)           # (n_epochs, 2)

    # Training labels: 0=Control, 1=ADHD  →  probs[:,0]=P(Control), probs[:,1]=P(ADHD)
    mean_prob = probs.mean(axis=0)
    adhd_prob = float(mean_prob[1])   # P(ADHD)    — index 1
    ctrl_prob = float(mean_prob[0])   # P(Control) — index 0

    # Use optimal threshold from model metadata (Youden's J), fall back to 0.5
    threshold  = float(_meta.get('optimal_threshold', 0.5))
    prediction = 'ADHD' if adhd_prob >= threshold else 'Normal'
    confidence = adhd_prob if prediction == 'ADHD' else ctrl_prob

    # Per-epoch ADHD probabilities (p[:,1] = P(ADHD))
    epoch_probs = [float(p[1]) for p in probs]

    # Channel saliency heatmap (mean absolute amplitude per channel)
    ch_salience = np.abs(epochs).mean(axis=(0, 2))   # (19,)
    mn, mx      = ch_salience.min(), ch_salience.max()
    attention   = ((ch_salience - mn) / (mx - mn + 1e-8)).tolist()

    return {
        'prediction':        prediction,
        'probability':       adhd_prob,   # always P(ADHD) for consistent display
        'confidence':        confidence,
        'threshold_used':    threshold,
        'n_epochs':          len(epochs),
        'epoch_probs':       epoch_probs,
        'attention_heatmap': attention,
        'model_type':        'CNN+TCN (ONNX)',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# In-memory session store  (keyed by session_id)
# ═══════════════════════════════════════════════════════════════════════════════
_sessions: dict = {}


def _sess_or_404(session_id):
    if not session_id or session_id not in _sessions:
        return None, jsonify({'error': 'Session not found'}), 404
    return _sessions[session_id], None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health():
    try:
        _get_session()
        loaded = True
    except Exception:
        loaded = False
    return jsonify({
        'status':       'online',
        'model_loaded': loaded,
        'model_type':   _meta.get('model_type', 'CNN+TCN'),
        'version':      '2.1.0',
        'runtime':      'onnx',
    })


@app.route('/api/upload', methods=['POST'])
def upload():
    """Accept .mat / .csv file, store bytes in memory, return session_id."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    f        = request.files['file']
    filename = secure_filename(f.filename or '')
    if not filename:
        return jsonify({'error': 'Empty filename'}), 400

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({'error': f'Unsupported type. Allowed: {sorted(ALLOWED_EXT)}'}), 400

    raw = f.read()
    if len(raw) > 50 * 1024 * 1024:
        return jsonify({'error': 'File too large (max 50 MB)'}), 413

    try:
        if ext == '.mat':
            data, srate = _load_mat_bytes(raw)
        elif ext == '.csv':
            arr   = np.loadtxt(io.StringIO(raw.decode('utf-8', errors='replace')),
                               delimiter=',', ndmin=2)
            data  = arr.T if arr.shape[0] > arr.shape[1] else arr
            srate = 128
        else:
            return jsonify({'error': 'EDF upload not supported in serverless mode'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to load file: {e}'}), 422

    n_ch, n_samp = data.shape
    duration     = n_samp / srate

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        'data':        data,
        'srate':       srate,
        'filename':    filename,
        'n_channels':  n_ch,
        'duration':    duration,
        'processed':   None,
        'result':      None,
    }

    return jsonify({
        'session_id':    session_id,
        'filename':      filename,
        'channels':      n_ch,
        'srate':         srate,
        'duration':      round(duration, 2),
        'channel_names': CH_NAMES[:n_ch] if n_ch <= len(CH_NAMES) else [f'Ch{i+1}' for i in range(n_ch)],
    })


@app.route('/api/process', methods=['POST'])
def process():
    """Preprocess EEG: filter → epoch → extract band powers."""
    body       = request.get_json(force=True) or {}
    session_id = body.get('session_id')
    sess, err, code = _sess_or_404(session_id)
    if err:
        return err, code

    try:
        data  = sess['data']
        srate = sess['srate']

        # Common Average Reference → Bandpass → Notch  (matches training pipeline)
        data_car  = data.copy() - data.mean(axis=0, keepdims=True)  # CAR first
        data_filt = _bandpass(data_car, srate)                      # 0.5–50 Hz
        data_filt = _notch(data_filt, srate)                        # 50 Hz notch

        n_ch_actual = min(data_filt.shape[0], N_CHANNELS)
        ch_names    = CH_NAMES[:n_ch_actual]

        # Band powers
        band_powers = _extract_band_powers(data_filt[:n_ch_actual], srate, ch_names)

        # Epochs: 50% overlap, artifact rejection at 100 µV, per-epoch z-score
        n_before = len(range(0, data_filt.shape[1] - EPOCH_SAMPLES + 1,
                             EPOCH_SAMPLES // 2))
        epochs   = _make_epochs(data_filt, srate)  # artifact rejection built-in
        removed  = max(0, n_before - len(epochs))

        sess['processed'] = {
            'epochs':      epochs,
            'band_powers': band_powers,
            'ch_names':    ch_names,
            'srate':       srate,
        }

        return jsonify({
            'session_id':     session_id,
            'n_epochs':       len(epochs),
            'n_features':     N_CHANNELS * 5,
            'epochs_removed': removed,
            'band_powers':    band_powers,
            'channel_names':  ch_names,
            'srate':          srate,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Run CNN+TCN ONNX inference on preprocessed epochs."""
    body       = request.get_json(force=True) or {}
    session_id = body.get('session_id')
    sess, err, code = _sess_or_404(session_id)
    if err:
        return err, code

    if sess.get('processed') is None:
        # Auto-process if not done yet
        sess['processed'] = {'epochs': None}
        try:
            data   = sess['data']
            srate  = sess['srate']
            data_car = data.copy() - data.mean(axis=0, keepdims=True)  # CAR
            df       = _bandpass(data_car, srate)                      # 0.5–50 Hz
            df       = _notch(df, srate)                               # 50 Hz
            epochs   = _make_epochs(df, srate)                         # overlap + z-score
            sess['processed']['epochs'] = epochs
        except Exception as e:
            return jsonify({'error': f'Auto-process failed: {e}'}), 500

    try:
        epochs = sess['processed']['epochs']
        if epochs is None or len(epochs) == 0:
            return jsonify({'error': 'No valid epochs to classify'}), 422

        result = _predict_epochs(epochs)
        sess['result'] = result

        heatmap = result['attention_heatmap']
        ch_att  = list(heatmap) if len(heatmap) == N_CHANNELS else [0.05] * N_CHANNELS

        attn_shap = sorted(
            [{'feature': ch, 'shap': round(v, 4), 'value': 1.0}
             for ch, v in zip(CH_NAMES, ch_att)],
            key=lambda x: abs(x['shap']), reverse=True
        )

        band_powers = sess['processed'].get('band_powers', {}) if sess.get('processed') else {}
        features    = {}
        if band_powers:
            frontal_chs = [c for c in ['Fp1','Fp2','Fz','F3','F4'] if c in band_powers]
            central_chs = [c for c in ['Cz','C3','C4'] if c in band_powers]
            if frontal_chs:
                features['frontal_theta'] = round(
                    np.mean([band_powers[c]['theta'] for c in frontal_chs]), 4)
                features['frontal_beta']  = round(
                    np.mean([band_powers[c]['beta']  for c in frontal_chs]), 4)
                t = features['frontal_theta']
                b = features['frontal_beta']
                features['avg_tbr'] = round(t / (b + 1e-8), 4)
            if central_chs:
                features['central_beta'] = round(
                    np.mean([band_powers[c]['beta'] for c in central_chs]), 4)

        return jsonify({
            'session_id':        session_id,
            'prediction':        result['prediction'],
            'probability':       result['probability'],
            'confidence':        result['confidence'],
            'n_epochs':          result['n_epochs'],
            'epoch_probs':       result['epoch_probs'],
            'attention_heatmap': heatmap,
            'base_value':        0.5,
            'shap_values':       attn_shap,
            'model_info':        _meta,
            'features':          features,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/demo', methods=['POST'])
def demo():
    """Return pre-computed demo results (no file needed)."""
    body    = request.get_json(force=True) or {}
    is_adhd = body.get('adhd', True)

    rng  = np.random.default_rng(42 if is_adhd else 99)
    prob = float(rng.uniform(0.72, 0.94)) if is_adhd else float(rng.uniform(0.06, 0.28))
    pred = 'ADHD' if prob > 0.5 else 'Normal'
    conf = max(prob, 1 - prob)

    sign       = 1 if is_adhd else -1
    demo_shap  = [
        {'feature': 'Frontal Theta Power (Fz)',  'shap': round( 0.42 * sign, 4), 'value': 8.5},
        {'feature': 'Theta/Beta Ratio (Fp1)',     'shap': round( 0.36 * sign, 4), 'value': 4.2},
        {'feature': 'Theta/Beta Ratio (Fp2)',     'shap': round( 0.31 * sign, 4), 'value': 3.9},
        {'feature': 'Central Beta Power (Cz)',    'shap': round(-0.24 * sign, 4), 'value': 1.8},
        {'feature': 'Frontal Asymmetry Index',    'shap': round( 0.21 * sign, 4), 'value': 0.35},
        {'feature': 'Parietal Alpha (P3)',         'shap': round(-0.17 * sign, 4), 'value': 4.1},
        {'feature': 'Sample Entropy (F3)',         'shap': round( 0.14 * sign, 4), 'value': 1.2},
        {'feature': 'Delta Power (O1)',             'shap': round( 0.09 * sign, 4), 'value': 2.1},
        {'feature': 'Gamma Power (Cz)',             'shap': round( 0.08 * sign, 4), 'value': 0.9},
        {'feature': 'Theta Coherence (Fp1-Fp2)',   'shap': round( 0.12 * sign, 4), 'value': 0.67},
        {'feature': 'Beta Power (C3)',              'shap': round(-0.15 * sign, 4), 'value': 2.2},
        {'feature': 'Alpha Power (Fz)',             'shap': round(-0.11 * sign, 4), 'value': 3.8},
    ]

    demo_bands = {
        'delta': 2.1, 'theta': 8.5 if is_adhd else 3.2,
        'alpha': 4.0, 'beta':  2.0 if is_adhd else 5.5, 'gamma': 0.9
    }
    band_powers = {ch: dict(demo_bands) for ch in CH_NAMES}
    epoch_probs = (rng.uniform(0.55, 0.95, 30) if is_adhd
                   else rng.uniform(0.05, 0.45, 30)).tolist()

    info = dict(_meta) if _meta else {
        'model_type': 'CNN+TCN', 'val_accuracy': 0.7110,
        'val_auc': 0.7101, 'n_subjects': 121,
    }

    return jsonify({
        'prediction':    pred,
        'probability':   round(prob, 4),
        'confidence':    round(conf, 4),
        'n_epochs':      30,
        'epoch_probs':   [round(p, 4) for p in epoch_probs],
        'base_value':    0.5,
        'shap_values':   demo_shap,
        'band_powers':   band_powers,
        'channel_names': CH_NAMES,
        'srate':         SRATE,
        'model_info':    info,
        'features': {
            'avg_tbr':       4.25 if is_adhd else 0.58,
            'frontal_theta': 8.50 if is_adhd else 3.20,
            'central_beta':  2.00 if is_adhd else 5.50,
        },
    })


@app.route('/api/explain', methods=['POST'])
def explain():
    body       = request.get_json(force=True) or {}
    session_id = body.get('session_id')
    sess, err, code = _sess_or_404(session_id)
    if err:
        return err, code
    if not sess.get('result'):
        return jsonify({'error': 'Run /api/predict first'}), 400
    return jsonify({
        'attention_heatmap': sess['result'].get('attention_heatmap', []),
        'model_type':        'CNN+TCN',
    })


# ─── Vercel WSGI entry point ──────────────────────────────────────────────────
# Vercel calls `handler` (the Flask WSGI app) for Python serverless functions.
handler = app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
