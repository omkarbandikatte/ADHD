"""
explainer.py  —  SHAP-based explainability for ADHD predictions
Generates SHAP values using TreeExplainer (RF) or KernelExplainer (SVM/fallback).
Returns structured data for waterfall, bar, and force plots in the frontend.
"""

import numpy as np
from typing import Dict, List, Optional


class SHAPExplainer:
    """
    Wraps SHAP library to explain individual ADHD predictions.
    Falls back to permutation-based importance if SHAP is not installed.
    """

    # Clinically named priority features (for readable explanations)
    CLINICAL_MAP = {
        '_theta_abs':    'Theta Band Power',
        '_beta_abs':     'Beta Band Power',
        '_alpha_abs':    'Alpha Band Power',
        '_delta_abs':    'Delta Band Power',
        '_gamma_abs':    'Gamma Band Power',
        '_tbr':          'Theta/Beta Ratio',
        'asym_':         'Frontal Asymmetry',
        'hjorth_activity': 'Hjorth Activity',
        'hjorth_mobility': 'Hjorth Mobility',
        'hjorth_complexity': 'Hjorth Complexity',
        'sampent_':      'Sample Entropy',
        'coh_':          'EEG Coherence',
    }

    def __init__(self, predictor):
        self.predictor = predictor
        self._shap_explainer = None
        self._background: Optional[np.ndarray] = None

    def _get_background(self, n: int = 50) -> np.ndarray:
        """Generate a synthetic background distribution for SHAP KernelExplainer."""
        rng = np.random.default_rng(42)
        n_feat = self.predictor.meta.get('n_features', 42)
        return rng.standard_normal((n, n_feat)) * 2.0

    def _build_explainer(self):
        if not self.predictor.is_trained():
            return
        try:
            import shap
            ensemble = self.predictor.pipeline.named_steps['ensemble']
            scaler   = self.predictor.pipeline.named_steps['scaler']

            # Try TreeExplainer on the Random Forest estimator first (fastest)
            try:
                rf = None
                for name, est in ensemble.estimators_:
                    if hasattr(est, 'feature_importances_'):
                        rf = est
                        break
                if rf is not None:
                    self._shap_explainer = shap.TreeExplainer(rf)
                    self._shap_type = 'tree'
                    return
            except Exception:
                pass

            # Fallback: KernelExplainer on the full pipeline
            bg = self._get_background()
            bg_scaled = scaler.transform(bg)
            self._shap_explainer = shap.KernelExplainer(
                lambda x: ensemble.predict_proba(x)[:, 1],
                bg_scaled,
            )
            self._shap_type = 'kernel'
        except ImportError:
            self._shap_explainer = None
            self._shap_type = 'fallback'

    def explain(self, feature_vector: List[float],
                feature_names: Optional[List[str]] = None) -> Dict:
        """
        Compute SHAP values for a single prediction.
        Returns structured dict with base_value and per-feature SHAP values.
        """
        fv = np.array(feature_vector, dtype=float)
        fv = np.nan_to_num(fv, nan=0.0, posinf=1e6, neginf=-1e6)
        names = feature_names or [f'Feature_{i}' for i in range(len(fv))]

        if self.predictor.is_trained() and self._shap_explainer is None:
            self._build_explainer()

        shap_vals = None

        if self._shap_explainer is not None:
            try:
                import shap
                scaler = self.predictor.pipeline.named_steps['scaler']
                fv_scaled = scaler.transform(fv.reshape(1, -1))[0]

                if self._shap_type == 'tree':
                    sv = self._shap_explainer.shap_values(fv_scaled.reshape(1, -1))
                    # For binary RF: sv is list [class0, class1]
                    if isinstance(sv, list):
                        sv = sv[1]  # ADHD class
                    shap_vals = sv.flatten()
                    base_value = float(self._shap_explainer.expected_value[1]
                                       if isinstance(self._shap_explainer.expected_value, (list, np.ndarray))
                                       else self._shap_explainer.expected_value)
                else:
                    sv = self._shap_explainer.shap_values(fv_scaled.reshape(1, -1), nsamples=100)
                    shap_vals  = sv.flatten()
                    base_value = float(self._shap_explainer.expected_value)

            except Exception as e:
                print(f"[SHAPExplainer] SHAP computation failed: {e}. Using fallback.")

        if shap_vals is None:
            # Permutation-based fallback approximation
            shap_vals, base_value = self._permutation_importance(fv, feature_names)

        # Format output
        result = [
            {
                'feature': self._clean_name(names[i]),
                'shap':    float(round(shap_vals[i], 6)),
                'value':   float(fv[i]),
            }
            for i in range(min(len(names), len(shap_vals)))
        ]
        # Sort by absolute SHAP descending
        result.sort(key=lambda x: abs(x['shap']), reverse=True)

        return {
            'base_value': round(base_value, 4),
            'values':     result[:20],  # top 20
        }

    def _permutation_importance(self, fv: np.ndarray,
                                 feature_names: Optional[List[str]]) -> tuple:
        """
        Approximate SHAP via sensitivity analysis (perturb each feature, measure Δp).
        """
        base_prob = self.predictor.predict(fv.tolist())['probability']
        shap_vals = np.zeros(len(fv))
        rng = np.random.default_rng(0)

        for i in range(len(fv)):
            perturbed = fv.copy()
            perturbed[i] = fv[i] + rng.standard_normal() * (abs(fv[i]) * 0.2 + 1e-6)
            p_prob = self.predictor.predict(perturbed.tolist())['probability']
            shap_vals[i] = p_prob - base_prob

        return shap_vals, base_prob - shap_vals.sum() / 2

    def _clean_name(self, name: str) -> str:
        """Convert raw feature key to human-readable label."""
        for key, human in self.CLINICAL_MAP.items():
            if key.lower() in name.lower():
                # Extract channel name if possible
                parts = name.split('_')
                ch = parts[0].upper() if parts[0] else ''
                if ch and len(ch) <= 4:
                    return f'{human} ({ch})'
                return human
        # Generic cleanup
        return name.replace('_', ' ').title()

    def global_importance(self) -> Optional[List[Dict]]:
        """Return global feature importance from Random Forest if available."""
        imp = self.predictor.get_feature_importances()
        if imp is None:
            return None
        return [{'feature': self._clean_name(k), 'importance': v}
                for k, v in list(imp.items())[:20]]
