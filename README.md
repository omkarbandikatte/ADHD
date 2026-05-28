# NeuroADHD — EEG-Based ADHD Detection with Explainable AI

A complete full-stack research system that processes raw EEG signals from `.mat` files,
extracts clinically validated biomarkers, classifies ADHD with an ML ensemble,
and explains every prediction with SHAP values — designed for professor demonstrations.

---

## Project Structure

```
ADHD/
├── index.html              ← Landing page (animations, biomarkers, pipeline overview)
├── dashboard.html          ← Full analysis dashboard
├── css/
│   ├── style.css           ← Global styles (landing page)
│   └── dashboard.css       ← Dashboard styles
├── js/
│   ├── main.js             ← Landing page interactions + neural network animation
│   ├── eeg-viz.js          ← EEG signal charts, band power, SHAP visualizations
│   ├── brain-map.js        ← D3 topographic brain map
│   └── dashboard.js        ← Dashboard controller, file upload, pipeline animation
└── backend/
    ├── app.py              ← Flask REST API
    ├── eeg_processor.py    ← EEG signal processing pipeline
    ├── adhd_predictor.py   ← ML ensemble (Random Forest + SVM + XGBoost)
    ├── explainer.py        ← SHAP explainability
    ├── train_model.py      ← Model training script
    └── requirements.txt    ← Python dependencies
```

---

## Quick Start — Demo Mode (No Backend Required)

1. Open `index.html` in any modern browser (Chrome / Edge / Firefox)
2. Click **"Open Full Dashboard"**
3. Click the **"Demo"** button in the top bar
4. Watch the full pipeline run: EEG loading → filtering → feature extraction → AI prediction → SHAP explanation

> Demo mode runs entirely in the browser using synthetic ADHD EEG data. No Python needed.

---

## Full Setup — With Real .mat Data

### 1. Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the Flask backend

```bash
cd backend
python app.py
```

The API will be available at `http://localhost:5000/api`

### 3. Open the dashboard

Open `index.html` (or serve via any HTTP server) and click **"Open Full Dashboard"**.

### 4. Upload your .mat EEG file

- Click the upload zone or drag & drop your `.mat` file
- The system auto-detects: channels, sampling rate, signal structure
- Supports EEGLAB format (`EEG.data`), generic MATLAB arrays (`data`, `eeg`, `signal`)
- Also accepts `.edf` and `.csv` formats

---

## Training the Model on Your Own Dataset

### Option A: Subdirectory structure

```
your_data/
├── adhd/
│   ├── subject01.mat
│   ├── subject02.mat
│   └── ...
└── normal/
    ├── subject10.mat
    └── ...
```

```bash
cd backend
python train_model.py --data_dir your_data/ --cv 5
```

### Option B: Flat directory + CSV labels

```csv
filename,label
subject01.mat,1
subject02.mat,0
```

```bash
python train_model.py --data_dir your_data/ --label_file labels.csv
```

### Option C: Synthetic demo training

```bash
python train_model.py --demo
```

---

## .mat File Format Support

The system handles these common structures automatically:

| Format | Variable Names |
|--------|---------------|
| EEGLAB | `EEG.data`, `EEG.srate`, `EEG.chanlocs` |
| Generic | `data`, `eeg`, `signal`, `raw` |
| Custom | Any 2D matrix (channels × samples or samples × channels) |

**Sampling rate** detected from: `srate`, `fs`, `Fs`, `freq`, `SamplingRate`

**Channel names** detected from: `chanlocs`, `labels`, `channels`, `ch_names`

If variables cannot be found automatically, the system falls back to 256 Hz and
standard 10-20 channel names.

---

## Dashboard Features

| Section | What It Shows |
|---------|--------------|
| **Upload** | Drag & drop .mat file, progress pipeline animation |
| **EEG Signals** | Multi-channel raw EEG waveforms, scrollable time window |
| **Band Power** | Absolute/relative power bar charts, PSD (Welch), theta/beta ratio per channel |
| **Brain Map** | D3 topographic map (IDW interpolation), switchable by band and metric |
| **Pipeline** | Step-by-step processing log with timing |
| **Prediction** | Semicircle probability gauge, verdict card, confidence breakdown |
| **Explainability** | SHAP waterfall chart, importance bar chart, force plot |
| **Report** | Auto-generated clinical summary, printable / downloadable JSON |

---

## ADHD EEG Biomarkers (What the Model Looks For)

| Biomarker | ADHD Pattern | Band |
|-----------|-------------|------|
| Theta power | ↑ Elevated (frontal) | 4–8 Hz |
| Beta power | ↓ Reduced | 13–30 Hz |
| **Theta/Beta Ratio** | **> 3.0 (key marker)** | — |
| Alpha power | ↓ Reduced task modulation | 8–13 Hz |
| Frontal asymmetry | Altered L/R balance | — |
| Sample entropy | ↓ Reduced complexity | — |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Backend status |
| POST | `/api/upload` | Upload .mat file (multipart/form-data) |
| POST | `/api/process` | Run preprocessing pipeline |
| POST | `/api/predict` | Run ADHD ML prediction |
| POST | `/api/explain` | Generate SHAP explanations |
| POST | `/api/demo` | Return demo results (no file needed) |

---

## Tech Stack

**Frontend:** HTML5, CSS3, JavaScript ES6+, Chart.js 4, D3.js 7  
**Backend:** Python 3.10+, Flask 3, Flask-CORS  
**Signal Processing:** SciPy (signal, io), NumPy  
**Machine Learning:** scikit-learn (RF + SVM ensemble), XGBoost (optional)  
**Explainability:** SHAP (TreeExplainer / KernelExplainer)  
**EEG Data:** .mat (EEGLAB/MATLAB), .edf, .csv  

---

## Citation / Disclaimer

This system is a **research prototype** intended for academic demonstration.
It does **not** constitute a clinical diagnostic tool. Results must be reviewed
by a qualified neurologist alongside standardised clinical assessments (DSM-5/ICD-11).
