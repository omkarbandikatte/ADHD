"""
eeg_processor.py  —  Full EEG signal processing pipeline
Supports: .mat (MATLAB/EEGLAB), .edf, .csv
Steps: Load → Bandpass filter → Notch filter → Artifact rejection →
       Re-referencing → Epoching → Feature extraction (PSD, TBR, coherence,
       Hjorth parameters, sample entropy, asymmetry indices)
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import scipy.io as sio
import scipy.signal as sig
import scipy.stats as stats


class EEGProcessor:
    """
    Handles loading, preprocessing, and feature extraction for EEG data.
    Flexible enough to handle different .mat file structures from various
    EEG systems (Emotiv, Neuroscan, EEGLAB, custom).
    """

    # Standard 10-20 channel names
    STANDARD_CHANNELS = [
        'Fp1','Fp2','F3','F4','F7','F8','Fz',
        'C3','C4','Cz','P3','P4','Pz',
        'O1','O2','T3','T4','T5','T6',
    ]

    # Frequency band definitions (Hz)
    BANDS = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta':  (13.0, 30.0),
        'gamma': (30.0, 40.0),
    }

    # Frontal channel pairs for asymmetry
    ASYMMETRY_PAIRS = [('Fp1','Fp2'), ('F3','F4'), ('F7','F8')]

    def __init__(self, default_srate: int = 256):
        self.default_srate = default_srate

    # ─── File Loading ─────────────────────────────────────────────────────────

    def load_file(self, filepath: str) -> Dict:
        """Load EEG file and return basic info dict."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.mat':
            return self._load_mat(filepath)
        elif ext == '.edf':
            return self._load_edf(filepath)
        elif ext == '.csv':
            return self._load_csv(filepath)
        else:
            raise ValueError(f'Unsupported format: {ext}')

    def _load_mat(self, filepath: str) -> Dict:
        """
        Load MATLAB .mat file.
        Handles multiple common structures:
          - EEG.data (EEGLAB format)
          - data / eeg / signal (generic)
          - <filename> key  (custom: shape = (n_samples, n_channels))
          - Direct matrix
        """
        try:
            mat = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        except Exception as e:
            raise ValueError(f'Cannot read .mat file: {e}')

        data, srate, ch_names = None, None, None

        # ── Try EEGLAB structure ──
        if 'EEG' in mat:
            eeg = mat['EEG']
            try:
                data    = np.array(eeg.data)
                srate   = int(eeg.srate)
                chanlocs = eeg.chanlocs
                if hasattr(chanlocs, '__iter__'):
                    ch_names = [str(c.labels) for c in chanlocs]
            except AttributeError:
                pass

        # ── Try common variable names ──
        if data is None:
            for key in ['data','eeg','EEG','signal','Signal','eegdata','raw']:
                if key in mat and isinstance(mat[key], np.ndarray):
                    data = np.array(mat[key], dtype=float)
                    break

        # ── Try filename-based key (e.g. v1p.mat → key 'v1p') ──
        if data is None:
            basename = os.path.splitext(os.path.basename(filepath))[0]
            if basename in mat and isinstance(mat[basename], np.ndarray):
                data = np.array(mat[basename], dtype=float)

        # ── Fall back to first non-metadata array ──
        if data is None:
            for key, val in mat.items():
                if key.startswith('_'):
                    continue
                if isinstance(val, np.ndarray) and val.ndim >= 2:
                    data = np.array(val, dtype=float)
                    break

        # ── Sampling rate ──
        if srate is None:
            for key in ['srate','fs','Fs','freq','SamplingRate','sfreq']:
                if key in mat:
                    try:
                        srate = int(np.squeeze(mat[key]))
                        break
                    except Exception:
                        pass
        if srate is None:
            srate = self.default_srate

        # ── Channel names ──
        if ch_names is None:
            for key in ['chanlocs','labels','channels','ch_names']:
                if key in mat:
                    try:
                        val = mat[key]
                        if isinstance(val, np.ndarray):
                            ch_names = [str(v).strip() for v in val.flat]
                            break
                    except Exception:
                        pass

        if data is None:
            raise ValueError('Could not find EEG data array in .mat file. Expected variables: EEG.data, data, eeg, signal')

        # Ensure shape: (n_channels, n_samples)
        data = np.array(data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # If rows > cols it's likely (n_samples, n_channels) — transpose
        if data.shape[0] > data.shape[1]:
            data = data.T

        n_ch, n_samp = data.shape
        if ch_names is None or len(ch_names) != n_ch:
            ch_names = self.STANDARD_CHANNELS[:n_ch] if n_ch <= len(self.STANDARD_CHANNELS) \
                       else [f'Ch{i+1}' for i in range(n_ch)]

        return {
            'data': data,
            'srate': srate,
            'n_channels': n_ch,
            'n_samples': n_samp,
            'duration': n_samp / srate,
            'channel_names': ch_names,
        }

    def _load_edf(self, filepath: str) -> Dict:
        """Load EDF/EDF+ file using pyEDFlib (optional) or raw header parsing."""
        try:
            import pyedflib
            f = pyedflib.EdfReader(filepath)
            n_ch  = f.signals_in_file
            srate = int(f.getSampleFrequency(0))
            ch_names = [f.getLabel(i).strip() for i in range(n_ch)]
            n_samp = f.getNSamples()[0]
            data   = np.array([f.readSignal(i) for i in range(n_ch)])
            f._close()
            return {
                'data': data, 'srate': srate, 'n_channels': n_ch,
                'n_samples': n_samp, 'duration': n_samp / srate,
                'channel_names': ch_names,
            }
        except ImportError:
            raise ValueError('pyedflib not installed. Run: pip install pyEDFlib')

    def _load_csv(self, filepath: str) -> Dict:
        """Load CSV file: rows=samples, cols=channels (first col may be timestamp)."""
        arr = np.genfromtxt(filepath, delimiter=',', names=True, dtype=float)
        if arr.dtype.names:
            ch_names = list(arr.dtype.names)
            data = np.column_stack([arr[n] for n in ch_names]).T
        else:
            data = np.loadtxt(filepath, delimiter=',').T
            ch_names = self.STANDARD_CHANNELS[:data.shape[0]]
        srate = self.default_srate
        n_ch, n_samp = data.shape
        return {
            'data': data, 'srate': srate, 'n_channels': n_ch,
            'n_samples': n_samp, 'duration': n_samp / srate,
            'channel_names': ch_names,
        }

    # ─── Preprocessing ────────────────────────────────────────────────────────

    def preprocess(self, filepath: str,
                   bandpass: Tuple[float,float] = (0.5, 50.0),
                   notch: float = 50.0,
                   epoch_len: float = 2.0,
                   overlap: float = 0.5,
                   artifact_thresh: float = 100.0) -> Dict:
        """
        Complete preprocessing pipeline.
        Returns processed dict including epoched data and metadata.
        """
        info = self.load_file(filepath)
        data     = info['data']
        srate    = info['srate']
        ch_names = info['channel_names']

        # 1. Common average reference
        data = data - data.mean(axis=0, keepdims=True)

        # 2. Bandpass filter
        data = self._bandpass(data, bandpass[0], bandpass[1], srate)

        # 3. Notch filter (power line)
        data = self._notch(data, notch, srate)

        # 4. Epoch
        epochs, epoch_times = self._epoch(data, srate, epoch_len, overlap)

        # 5. Artifact rejection
        clean_epochs, removed = self._reject_artifacts(epochs, artifact_thresh)

        # Z-score normalise each epoch per channel (architecture requirement)
        norm_epochs = self._zscore_epochs(clean_epochs)

        return {
            'data':          data,
            'epochs':        clean_epochs,
            'norm_epochs':   norm_epochs,    # Z-score normalised — used by CNN+TCN
            'epoch_times':   epoch_times,
            'srate':         srate,
            'n_epochs':      len(clean_epochs),
            'epochs_removed': removed,
            'channel_names': ch_names,
            'n_channels':    data.shape[0],
        }

    def _bandpass(self, data: np.ndarray, lo: float, hi: float, fs: int) -> np.ndarray:
        nyq = fs / 2.0
        lo_n, hi_n = lo / nyq, min(hi / nyq, 0.99)
        b, a = sig.butter(4, [lo_n, hi_n], btype='band')
        return sig.filtfilt(b, a, data, axis=1)

    def _notch(self, data: np.ndarray, freq: float, fs: int, Q: float = 30.0) -> np.ndarray:
        nyq = fs / 2.0
        if freq >= nyq:
            return data
        b, a = sig.iirnotch(freq / nyq, Q)
        return sig.filtfilt(b, a, data, axis=1)

    def _epoch(self, data: np.ndarray, fs: int, epoch_len: float, overlap: float
               ) -> Tuple[List[np.ndarray], List[float]]:
        n_samp_epoch = int(epoch_len * fs)
        step         = int(n_samp_epoch * (1 - overlap))
        n_samp_total = data.shape[1]
        epochs, times = [], []
        start = 0
        while start + n_samp_epoch <= n_samp_total:
            epochs.append(data[:, start:start + n_samp_epoch])
            times.append(start / fs)
            start += step
        return epochs, times

    def _reject_artifacts(self, epochs: List[np.ndarray], thresh: float
                           ) -> Tuple[List[np.ndarray], int]:
        clean, removed = [], 0
        for ep in epochs:
            if np.max(np.abs(ep)) <= thresh:
                clean.append(ep)
            else:
                removed += 1
        return clean if clean else epochs[:max(1, len(epochs)//2)], removed

    @staticmethod
    def _zscore_epochs(epochs: List[np.ndarray]) -> List[np.ndarray]:
        """Z-score normalise each epoch per channel (mean=0, std=1)."""
        normed = []
        for ep in epochs:
            mu  = ep.mean(axis=1, keepdims=True)
            std = ep.std(axis=1,  keepdims=True) + 1e-8
            normed.append((ep - mu) / std)
        return normed

    def get_cnn_tcn_epochs(self, filepath: str,
                           epoch_len: float = 2.0,
                           overlap: float = 0.5,
                           artifact_thresh: float = 100.0,
                           target_channels: int = 19,
                           target_samples: int = 256) -> Tuple[np.ndarray, int]:
        """
        Full preprocessing pipeline returning fixed-size numpy arrays
        ready for the CNN+TCN model.

        Returns
        -------
        epochs_arr : np.ndarray  shape (N_epochs, target_channels, target_samples)
        n_epochs   : int
        """
        proc = self.preprocess(
            filepath,
            bandpass=(0.5, 50.0),
            notch=50.0,
            epoch_len=epoch_len,
            overlap=overlap,
            artifact_thresh=artifact_thresh,
        )
        norm_epochs = proc['norm_epochs']

        out = []
        for ep in norm_epochs:
            n_ch, n_t = ep.shape
            # Pad or trim channels
            if n_ch < target_channels:
                pad = np.zeros((target_channels - n_ch, n_t))
                ep  = np.vstack([ep, pad])
            else:
                ep = ep[:target_channels]

            # Pad or trim time
            if n_t < target_samples:
                pad = np.zeros((target_channels, target_samples - n_t))
                ep  = np.hstack([ep, pad])
            else:
                ep = ep[:, :target_samples]

            out.append(ep.astype(np.float32))

        if not out:
            return np.zeros((0, target_channels, target_samples), dtype=np.float32), 0

        return np.stack(out, axis=0), len(out)

    # ─── Feature Extraction ───────────────────────────────────────────────────

    def extract_features(self, proc: Dict) -> Dict:
        """
        Extract 42+ neuroscientifically relevant features:
        - Band powers (absolute + relative) per channel × 5 bands
        - Theta/Beta ratio per channel
        - Frontal asymmetry indices (alpha, theta, beta)
        - Hjorth parameters (activity, mobility, complexity) per channel
        - Sample entropy per channel
        - Inter-channel coherence (frontal pair)
        """
        epochs       = proc['epochs']
        srate        = proc['srate']
        ch_names     = proc['channel_names']
        n_ch         = proc['n_channels']

        # Average features across epochs
        band_powers_per_ep = [self._band_powers(ep, srate) for ep in epochs]  # list of (n_ch, 5)
        avg_bp = np.mean(band_powers_per_ep, axis=0)  # (n_ch, 5)

        band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']

        # Per-channel band power dict (for topographic map)
        band_powers_dict = {
            ch_names[i]: {b: float(avg_bp[i, j]) for j, b in enumerate(band_names)}
            for i in range(n_ch)
        }

        # Relative band powers
        total_power = avg_bp.sum(axis=1, keepdims=True) + 1e-12
        rel_bp = avg_bp / total_power  # (n_ch, 5)

        # Theta/Beta ratios
        theta_idx, beta_idx = band_names.index('theta'), band_names.index('beta')
        tbr = avg_bp[:, theta_idx] / (avg_bp[:, beta_idx] + 1e-12)
        avg_tbr = float(tbr.mean())

        # Frontal theta (Fz), central beta (Cz)
        frontal_theta, central_beta = self._frontal_central_features(ch_names, avg_bp, band_names)

        # Asymmetry indices
        asym_features = self._asymmetry(ch_names, avg_bp, band_names)

        # Hjorth parameters
        hjorth_features = self._hjorth_all(epochs, n_ch)

        # Sample entropy
        entropy_features = self._sample_entropy_all(epochs, n_ch)

        # Coherence (Fp1-Fp2)
        coh_features = self._coherence(epochs, ch_names, srate)

        # ── Build flat feature vector ──
        feature_dict = {}
        for i, ch in enumerate(ch_names):
            for j, b in enumerate(band_names):
                feature_dict[f'{ch}_{b}_abs'] = float(avg_bp[i, j])
                feature_dict[f'{ch}_{b}_rel'] = float(rel_bp[i, j])
            feature_dict[f'{ch}_tbr'] = float(tbr[i])

        feature_dict.update(asym_features)
        feature_dict.update(hjorth_features)
        feature_dict.update(entropy_features)
        feature_dict.update(coh_features)

        feature_names  = list(feature_dict.keys())
        feature_vector = np.array(list(feature_dict.values()), dtype=float)

        # Replace NaN/Inf
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)

        return {
            'feature_vector':  feature_vector.tolist(),
            'feature_names':   feature_names,
            'band_powers':     band_powers_dict,
            'avg_tbr':         avg_tbr,
            'frontal_theta':   frontal_theta,
            'central_beta':    central_beta,
            'tbr_per_channel': {ch_names[i]: float(tbr[i]) for i in range(n_ch)},
        }

    def _band_powers(self, epoch: np.ndarray, fs: int) -> np.ndarray:
        """Welch PSD → integrate power in each band. Returns (n_ch, n_bands)."""
        n_ch = epoch.shape[0]
        powers = np.zeros((n_ch, len(self.BANDS)))
        nperseg = min(fs * 2, epoch.shape[1])
        for i in range(n_ch):
            f, psd = sig.welch(epoch[i], fs=fs, nperseg=nperseg, scaling='density')
            for j, (bname, (lo, hi)) in enumerate(self.BANDS.items()):
                mask = (f >= lo) & (f < hi)
                powers[i, j] = np.trapezoid(psd[mask], f[mask]) if mask.sum() > 0 else 0.0
        return powers

    def _frontal_central_features(self, ch_names, avg_bp, band_names):
        theta_idx = band_names.index('theta')
        beta_idx  = band_names.index('beta')
        try:
            fz_idx = ch_names.index('Fz') if 'Fz' in ch_names else 0
            cz_idx = ch_names.index('Cz') if 'Cz' in ch_names else 0
        except ValueError:
            fz_idx, cz_idx = 0, 0
        return float(avg_bp[fz_idx, theta_idx]), float(avg_bp[cz_idx, beta_idx])

    def _asymmetry(self, ch_names, avg_bp, band_names) -> Dict:
        asym = {}
        for lch, rch in self.ASYMMETRY_PAIRS:
            if lch in ch_names and rch in ch_names:
                li = ch_names.index(lch)
                ri = ch_names.index(rch)
                for j, b in enumerate(band_names):
                    l_p, r_p = avg_bp[li, j], avg_bp[ri, j]
                    asym[f'asym_{lch}_{rch}_{b}'] = float(
                        (r_p - l_p) / (r_p + l_p + 1e-12))
        return asym

    def _hjorth_all(self, epochs, n_ch) -> Dict:
        acts, mobs, comps = [], [], []
        for ep in epochs:
            a, m, c = self._hjorth(ep)
            acts.append(a); mobs.append(m); comps.append(c)
        avg_a = np.mean(acts, axis=0)
        avg_m = np.mean(mobs, axis=0)
        avg_c = np.mean(comps, axis=0)
        return {f'hjorth_activity_{i}': float(avg_a[i]) for i in range(n_ch)} | \
               {f'hjorth_mobility_{i}': float(avg_m[i]) for i in range(n_ch)} | \
               {f'hjorth_complexity_{i}': float(avg_c[i]) for i in range(n_ch)}

    @staticmethod
    def _hjorth(epoch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        activity   = np.var(epoch, axis=1)
        d1 = np.diff(epoch, axis=1)
        mobility   = np.sqrt(np.var(d1, axis=1) / (activity + 1e-12))
        d2 = np.diff(d1, axis=1)
        complexity = np.sqrt(np.var(d2, axis=1) / (np.var(d1, axis=1) + 1e-12)) / (mobility + 1e-12)
        return activity, mobility, complexity

    def _sample_entropy_all(self, epochs, n_ch, m: int = 2, r_frac: float = 0.2) -> Dict:
        # Use only first 3 epochs to keep computation fast
        sample_epochs = epochs[:3] if len(epochs) > 3 else epochs
        ents = []
        for ep in sample_epochs:
            ep_ent = []
            for i in range(n_ch):
                ep_ent.append(self._sample_entropy(ep[i], m, r_frac))
            ents.append(ep_ent)
        avg_ent = np.mean(ents, axis=0)
        return {f'sampent_{i}': float(avg_ent[i]) for i in range(n_ch)}

    @staticmethod
    def _sample_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
        """Vectorised sample entropy — O(N) per template length using numpy broadcast."""
        x = x[:256]  # limit for speed
        N = len(x)
        r = r_frac * float(np.std(x) + 1e-12)
        if r == 0 or N < m + 2:
            return 0.0

        def _count_matches(template_len: int) -> int:
            # Build matrix of all template windows: shape (N-template_len, template_len)
            idx = np.arange(N - template_len)
            # Use numpy stride tricks for efficient windowing
            windows = np.array([x[i:i + template_len] for i in idx])
            # Pairwise Chebyshev distance via broadcast: (n, n)
            diff = np.abs(windows[:, None, :] - windows[None, :, :]).max(axis=2)
            # Count upper-triangle matches (i < j), excluding diagonal
            mask = np.triu(diff < r, k=1)
            return int(mask.sum())

        B = _count_matches(m)
        A = _count_matches(m + 1)
        if B == 0:
            return 0.0
        return float(-np.log(A / (B + 1e-12) + 1e-12))

    def _coherence(self, epochs, ch_names, srate) -> Dict:
        coh = {}
        pairs = [('Fp1','Fp2'), ('F3','F4'), ('C3','C4')]
        for lch, rch in pairs:
            if lch in ch_names and rch in ch_names:
                li = ch_names.index(lch)
                ri = ch_names.index(rch)
                coh_vals = []
                for ep in epochs:
                    f, Cxy = sig.coherence(ep[li], ep[ri], fs=srate, nperseg=min(srate, ep.shape[1]))
                    mask_theta = (f >= 4) & (f < 8)
                    mask_beta  = (f >= 13) & (f < 30)
                    coh_vals.append({
                        f'coh_{lch}_{rch}_theta': float(Cxy[mask_theta].mean()) if mask_theta.sum() else 0.0,
                        f'coh_{lch}_{rch}_beta':  float(Cxy[mask_beta].mean())  if mask_beta.sum()  else 0.0,
                    })
                # Average across epochs
                for key in coh_vals[0]:
                    coh[key] = float(np.mean([c[key] for c in coh_vals]))
        return coh
