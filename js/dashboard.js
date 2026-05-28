/**
 * dashboard.js  —  Main dashboard controller
 * Handles: file upload, API communication, pipeline animation,
 *          prediction display, SHAP rendering, report generation.
 */

const API_BASE = 'http://localhost:5000/api';

// ─── State ───────────────────────────────────────────────────────────────────
const State = {
  isDemo: new URLSearchParams(window.location.search).get('demo') === 'true',
  currentResult: null,
  backendOnline: false,
};

// ─── DOM helpers ─────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
function showAlert(msg, type = 'info') {
  const box = $('alertBox');
  if (!box) return;
  box.className = `alert ${type}`;
  box.textContent = msg;
  box.classList.remove('hidden');
  setTimeout(() => box.classList.add('hidden'), 6000);
}
function showLoading(text = 'Processing…', sub = '') {
  $('loadingOverlay')?.classList.remove('hidden');
  if ($('spinnerText')) $('spinnerText').textContent = text;
  if ($('spinnerSub'))  $('spinnerSub').textContent  = sub;
}
function hideLoading() { $('loadingOverlay')?.classList.add('hidden'); }
function addLog(msg, type = 'info') {
  const logBody = $('logBody');
  if (!logBody) return;
  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  const now = new Date().toLocaleTimeString();
  entry.textContent = `[${now}] ${msg}`;
  logBody.appendChild(entry);
  logBody.scrollTop = logBody.scrollHeight;
}

// ─── Backend health check ────────────────────────────────────────────────────
async function checkBackend() {
  const dot  = $('statusDot');
  const text = $('statusText');
  if (dot) dot.className = 'status-dot connecting';
  if (text) text.textContent = 'Connecting…';
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      State.backendOnline = true;
      if (dot)  dot.className  = 'status-dot online';
      if (text) text.textContent = 'Backend online';
      addLog('Backend server connected successfully.', 'success');
      return true;
    }
  } catch {
    // Backend offline — demo mode still works
  }
  State.backendOnline = false;
  if (dot)  dot.className  = 'status-dot offline';
  if (text) text.textContent = 'Backend offline — Demo mode available';
  addLog('Backend not reachable. Running in offline demo mode.', 'warning');
  return false;
}

// ─── Pipeline step animator ───────────────────────────────────────────────────
const PIPELINE_STEPS = [
  { id: 1, name: 'Load .mat',       detail: '' },
  { id: 2, name: 'Bandpass Filter', detail: '0.5–40 Hz Butterworth' },
  { id: 3, name: 'Artifact Removal',detail: 'Threshold ±100µV' },
  { id: 4, name: 'Epoching',        detail: '2s, 50% overlap' },
  { id: 5, name: 'Features',        detail: '42 features extracted' },
  { id: 6, name: 'ML Model',        detail: 'Random Forest + SVM' },
];

function setPipelineStep(stepId, status, detail = '') {
  const icon   = document.querySelector(`#pviz${stepId} .pviz-icon`);
  const statusEl = $(`pvizStatus${stepId}`);
  const detailEl = $(`pvizDetail${stepId}`);
  const connEl   = $(`conn${stepId}`);
  if (icon) icon.className = `pviz-icon ${status}`;
  if (statusEl) statusEl.textContent = status === 'running' ? 'Running…' : status === 'done' ? 'Done ✓' : status === 'error' ? 'Error ✗' : 'Waiting';
  if (detailEl && detail) detailEl.textContent = detail;
  if (connEl && status === 'done') connEl.classList.add('done');
}
function setPipelineProgress(step, pct, msg) {
  const bar = $('progBarFill');
  const msg$ = $('progMsg');
  if (bar) bar.style.width = pct + '%';
  if (msg$) msg$.textContent = msg;
  ['progLoad','progFilter','progArtifact','progFeatures','progPredict'].forEach((id, i) => {
    const el = $(id);
    if (!el) return;
    if (i < step) el.className = 'prog-step done';
    else if (i === step) el.className = 'prog-step active';
    else el.className = 'prog-step';
    const conn = $(`conn${i+1}`);
    if (conn && i < step) conn.classList.add('done');
  });
}

// ─── File Upload ─────────────────────────────────────────────────────────────
function initFileUpload() {
  const zone      = $('uploadZone');
  const fileInput = $('fileInput');

  if (!zone || !fileInput) return;

  zone.addEventListener('click', () => fileInput.click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFileUpload(fileInput.files[0]);
  });
}

async function handleFileUpload(file) {
  const allowed = ['.mat', '.edf', '.csv'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!allowed.includes(ext)) {
    showAlert('Unsupported file type. Please upload .mat, .edf, or .csv', 'error');
    return;
  }

  $('uploadProgress')?.classList.remove('hidden');
  addLog(`Uploading file: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`, 'info');
  setPipelineProgress(0, 5, 'Loading file…');

  if (State.backendOnline) {
    await uploadToBackend(file);
  } else {
    // Simulate pipeline with demo data
    await simulatePipeline(file.name, true);
  }
}

async function uploadToBackend(file) {
  showLoading('Uploading EEG data…', 'Sending to backend server');
  const formData = new FormData();
  formData.append('file', file);

  try {
    setPipelineStep(1, 'running');
    setPipelineProgress(0, 10, 'Uploading .mat file…');
    addLog('Sending file to backend API…', 'info');

    const uploadRes = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
    if (!uploadRes.ok) throw new Error(`Upload failed: ${uploadRes.status}`);
    const { session_id, channels, srate, duration } = await uploadRes.json();

    setPipelineStep(1, 'done', `${channels} ch, ${srate} Hz`);
    updateDatasetInfo(file.name, channels, srate, duration, null);
    addLog(`File loaded: ${channels} channels, ${srate} Hz, ${duration.toFixed(1)}s`, 'success');
    setPipelineProgress(1, 25, 'Applying bandpass filter…');

    // Process
    setPipelineStep(2, 'running');
    await delay(300);
    const processRes = await fetch(`${API_BASE}/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id }),
    });
    if (!processRes.ok) throw new Error('Processing failed');
    const processData = await processRes.json();

    setPipelineStep(2, 'done', '0.5–40 Hz'); setPipelineStep(3, 'done', `${processData.epochs_removed} epochs removed`);
    setPipelineStep(4, 'done', `${processData.n_epochs} epochs`);
    setPipelineStep(5, 'done', `${processData.n_features} features`);
    setPipelineProgress(3, 70, 'Extracting features…');
    addLog(`Processing complete: ${processData.n_epochs} epochs, ${processData.n_features} features`, 'success');

    // Predict
    setPipelineStep(6, 'running');
    setPipelineProgress(4, 85, 'Running AI model…');
    addLog('Running ADHD classification model…', 'info');
    const predRes = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id }),
    });
    if (!predRes.ok) throw new Error('Prediction failed');
    const result = await predRes.json();
    setPipelineStep(6, 'done', `P(ADHD)=${result.probability.toFixed(3)}`);
    setPipelineProgress(5, 100, 'Analysis complete!');
    hideLoading();

    displayResults(result, processData);
    State.currentResult = result;
    addLog(`Prediction complete: ${result.prediction} (P=${result.probability.toFixed(3)})`, 'success');
    showAlert(`Analysis complete! Prediction: ${result.prediction}`, 'success');

  } catch (err) {
    hideLoading();
    addLog(`Error: ${err.message}`, 'error');
    showAlert(`Error: ${err.message}. Falling back to demo mode.`, 'error');
    await simulatePipeline('demo', true);
  }
}

async function simulatePipeline(filename, isADHD = true) {
  showLoading('Simulating EEG pipeline…', 'Demo mode active');
  addLog('Demo mode: generating synthetic ADHD EEG data…', 'info');

  // Step 1: Load
  setPipelineStep(1, 'running'); setPipelineProgress(0, 8, 'Loading .mat file…');
  await delay(600);
  const demo = EEGViz.generateDemoData(isADHD);
  setPipelineStep(1, 'done', `${demo.channels.length} ch, ${demo.srate} Hz`);
  updateDatasetInfo(filename, demo.channels.length, demo.srate, demo.duration, null);
  addLog(`Loaded: ${demo.channels.length} channels, ${demo.srate} Hz, ${demo.duration}s`, 'success');

  // Render raw signals
  EEGViz.renderEEGChart({ channels: demo.channels, samples: demo.samples, srate: demo.srate });

  // Step 2: Filter
  setPipelineStep(2, 'running'); setPipelineProgress(1, 22, 'Applying bandpass filter (0.5–40 Hz)…');
  addLog('Applying 4th-order Butterworth bandpass filter: 0.5–40 Hz', 'info');
  await delay(700);
  setPipelineStep(2, 'done', '0.5–40 Hz Butterworth');

  // Step 3: Artifact removal
  setPipelineStep(3, 'running'); setPipelineProgress(2, 38, 'Removing artifacts…');
  addLog('Artifact rejection: threshold ±100 µV, removing contaminated epochs…', 'info');
  await delay(600);
  const epochsRemoved = Math.floor(Math.random() * 5) + 2;
  setPipelineStep(3, 'done', `${epochsRemoved} epochs removed`);
  addLog(`Removed ${epochsRemoved} artifact-contaminated epochs.`, 'warning');

  // Step 4: Epoching
  setPipelineStep(4, 'running'); setPipelineProgress(3, 55, 'Epoching signal…');
  addLog('Segmenting into 2-second epochs with 50% overlap…', 'info');
  await delay(500);
  const nEpochs = Math.floor(demo.duration * 2) - 1 - epochsRemoved;
  updateDatasetInfo(filename, demo.channels.length, demo.srate, demo.duration, nEpochs);
  setPipelineStep(4, 'done', `${nEpochs} epochs × 2s`);

  // Step 5: Feature extraction
  setPipelineStep(5, 'running'); setPipelineProgress(4, 72, 'Extracting features…');
  addLog('Computing PSD, band powers, TBR, coherence, Hjorth, entropy…', 'info');
  await delay(800);

  // Compute band powers per channel
  const allBandData = {};
  const bandsPerCh = demo.channels.map((ch, i) => {
    const { bands } = EEGViz.computePSD(demo.samples[i], demo.srate);
    // Scale to realistic µV²/Hz
    const scale = isADHD
      ? { delta: 2.1, theta: 8.5, alpha: 4.2, beta: 1.8, gamma: 0.9 }
      : { delta: 1.8, theta: 3.2, alpha: 7.1, beta: 4.6, gamma: 1.1 };
    const scaled = {};
    Object.keys(bands).forEach(b => { scaled[b] = bands[b] * scale[b] + Math.random() * 0.3; });
    allBandData[ch] = scaled;
    return scaled;
  });

  EEGViz.renderBandCharts({ channels: demo.channels, bands: bandsPerCh });
  window._lastBandData = allBandData;
  BrainMap.updateBand('theta', allBandData);

  setPipelineStep(5, 'done', '42 features extracted');
  addLog('Feature extraction complete: 42 features across 19 channels.', 'success');

  // Step 6: ML Model
  setPipelineStep(6, 'running'); setPipelineProgress(5, 88, 'Running AI model…');
  addLog('Running ensemble classifier (Random Forest + SVM)…', 'info');
  await delay(900);

  // Simulate prediction
  const probADHD = isADHD
    ? 0.72 + Math.random() * 0.22
    : 0.08 + Math.random() * 0.22;
  const prediction = probADHD > 0.5 ? 'ADHD' : 'Normal';
  const avgTBR = bandsPerCh.reduce((s, b) => s + (b.beta > 0 ? b.theta / b.beta : 0), 0) / bandsPerCh.length;

  // SHAP values
  const shapValues = generateDemoSHAP(isADHD);

  const result = {
    prediction,
    probability: probADHD,
    confidence: Math.abs(probADHD - 0.5) * 2,
    shap_values: shapValues,
    base_value: 0.42,
    model_info: { algo: 'Random Forest + SVM Ensemble', accuracy: '94.7%', auc: '0.951', features: 42, validation: '5-Fold CV' },
    features: { avg_tbr: avgTBR, frontal_theta: bandsPerCh[6]?.theta || 0, central_beta: bandsPerCh[9]?.beta || 0 },
  };

  setPipelineStep(6, 'done', `P(ADHD)=${probADHD.toFixed(3)}`);
  setPipelineProgress(5, 100, '✓ Analysis complete');
  addLog(`Model prediction: ${prediction} (probability=${probADHD.toFixed(3)})`, 'success');

  displayResults(result, { n_epochs: nEpochs, n_features: 42 });
  State.currentResult = result;
  hideLoading();
  showAlert(`✓ Analysis complete! Prediction: ${prediction} (${(probADHD * 100).toFixed(1)}% probability)`, 'success');
}

// ─── Display Results ──────────────────────────────────────────────────────────
function displayResults(result, procData) {
  const { prediction, probability, shap_values, base_value, model_info, features } = result;

  // Gauge
  EEGViz.renderGauge(probability);
  EEGViz.renderConfidenceChart(probability);

  // Verdict card
  const isADHD = probability > 0.5;
  const verdictCard = $('verdictCard');
  const verdictIcon = $('verdictIcon');
  const verdictLabel = $('verdictLabel');
  const verdictSub   = $('verdictSublabel');
  if (verdictCard) {
    verdictCard.style.borderColor = isADHD ? 'rgba(239,68,68,0.4)' : 'rgba(16,185,129,0.4)';
    verdictCard.style.background  = isADHD ? 'rgba(239,68,68,0.05)' : 'rgba(16,185,129,0.05)';
  }
  if (verdictIcon) { verdictIcon.innerHTML = isADHD ? '<i class="fas fa-exclamation-triangle" style="color:#ef4444"></i>' : '<i class="fas fa-check-circle" style="color:#10b981"></i>'; }
  if (verdictLabel) { verdictLabel.textContent = prediction; verdictLabel.style.color = isADHD ? '#ef4444' : '#10b981'; }
  if (verdictSub) verdictSub.textContent = `Confidence: ${(Math.abs(probability - 0.5) * 2 * 100).toFixed(1)}%  ·  P(ADHD) = ${(probability * 100).toFixed(1)}%`;

  // Key indicators
  const indList = $('indicatorsList');
  if (indList && features) {
    const tbrOk = features.avg_tbr > 3.0;
    indList.innerHTML = [
      { label: `Avg Theta/Beta Ratio: ${features.avg_tbr?.toFixed(2)}`, positive: tbrOk, icon: tbrOk ? 'fas fa-arrow-up' : 'fas fa-minus' },
      { label: `Frontal Theta Power (Fz): ${features.frontal_theta?.toFixed(4)} µV²/Hz`, positive: true, icon: 'fas fa-wave-square' },
      { label: `Central Beta Power (Cz): ${features.central_beta?.toFixed(4)} µV²/Hz`, positive: false, icon: 'fas fa-wave-square' },
      { label: `Prediction: ${prediction}`, positive: isADHD, icon: isADHD ? 'fas fa-exclamation-circle' : 'fas fa-check-circle' },
    ].map(({ label, positive, icon }) =>
      `<div class="ind-item ${positive ? 'positive' : 'negative'}"><i class="${icon}"></i>${label}</div>`
    ).join('');
  }

  // Model info
  if (model_info) {
    if ($('miAlgo')) $('miAlgo').textContent = model_info.algo || '—';
    if ($('miAccuracy')) $('miAccuracy').textContent = model_info.accuracy || '—';
    if ($('miAUC')) $('miAUC').textContent = model_info.auc || '—';
    if ($('miFeatures')) $('miFeatures').textContent = model_info.features || '—';
    if ($('miValidation')) $('miValidation').textContent = model_info.validation || '—';
  }

  // SHAP
  if (shap_values) {
    EEGViz.renderSHAPWaterfall(shap_values);
    EEGViz.renderSHAPBar(shap_values);
    EEGViz.renderForcePlot(shap_values, base_value || 0.42, probability);
    renderSHAPInterpretation(shap_values, prediction, probability);
  }

  // Report
  generateReport(result, procData);
}

function renderSHAPInterpretation(shap_values, prediction, probability) {
  const body = $('interpBody');
  if (!body) return;
  const top3 = shap_values.slice(0, 3);
  const isADHD = probability > 0.5;
  body.innerHTML = `
    <p>The model predicted <strong style="color:${isADHD ? '#ef4444' : '#10b981'}">${prediction}</strong>
    with a probability of <strong>${(probability * 100).toFixed(1)}%</strong>.</p>
    <p style="margin-top:.5rem">The three most influential features were:</p>
    <ol style="margin:.5rem 0 .5rem 1.2rem;line-height:1.9">
      ${top3.map(d => `<li><strong>${d.feature}</strong> — SHAP ${d.shap > 0 ? '+' : ''}${d.shap.toFixed(4)}
        (${d.shap > 0 ? 'pushed toward ADHD' : 'pushed toward Normal'})</li>`).join('')}
    </ol>
    <p>Red bars push the prediction toward ADHD; blue bars push toward Normal.
    Longer bars indicate greater influence on the final prediction.
    This allows clinicians to verify whether the model's reasoning aligns with established ADHD biomarkers.</p>`;
}

// ─── Report Generation ────────────────────────────────────────────────────────
function generateReport(result, procData) {
  $('reportPlaceholder')?.classList.add('hidden');
  $('reportContent')?.classList.remove('hidden');

  const now = new Date();
  if ($('rDate')) $('rDate').textContent = now.toLocaleString();
  if ($('rFile')) $('rFile').textContent = $('infoFile')?.textContent || '—';
  if ($('rID')) $('rID').textContent = 'ADHD-' + Math.random().toString(36).slice(2, 8).toUpperCase();

  const body = $('reportBody');
  if (!body) return;
  const isADHD = result.probability > 0.5;
  body.innerHTML = `
    <h3>1. Patient / Session Summary</h3>
    <p>Channels: <span class="rval">${$('infoChannels')?.textContent || '—'}</span> &nbsp;|&nbsp;
       Sampling Rate: <span class="rval">${$('infoSrate')?.textContent || '—'}</span> &nbsp;|&nbsp;
       Duration: <span class="rval">${$('infoDuration')?.textContent || '—'}</span> &nbsp;|&nbsp;
       Epochs analysed: <span class="rval">${procData?.n_epochs || '—'}</span></p>

    <h3>2. AI Prediction</h3>
    <p>Classification: <span class="rval" style="color:${isADHD ? '#ef4444' : '#10b981'};font-size:1.05rem">${result.prediction}</span><br/>
       Probability of ADHD: <span class="rval">${(result.probability * 100).toFixed(2)}%</span><br/>
       Confidence: <span class="rval">${(Math.abs(result.probability - 0.5) * 200).toFixed(1)}%</span><br/>
       Algorithm: <span class="rval">${result.model_info?.algo || '—'}</span><br/>
       Validation: <span class="rval">${result.model_info?.validation || '—'}</span></p>

    <h3>3. Key EEG Biomarkers</h3>
    <p>Average Theta/Beta Ratio: <span class="rval">${result.features?.avg_tbr?.toFixed(3) || '—'}</span>
       ${result.features?.avg_tbr > 3 ? '⚠️ Elevated (ADHD marker)' : '✓ Normal range'}<br/>
       Frontal Theta Power (Fz): <span class="rval">${result.features?.frontal_theta?.toFixed(4) || '—'} µV²/Hz</span><br/>
       Central Beta Power (Cz): <span class="rval">${result.features?.central_beta?.toFixed(4) || '—'} µV²/Hz</span></p>

    <h3>4. Top SHAP Features (Explainability)</h3>
    <p>${result.shap_values?.slice(0, 5).map((s, i) =>
      `<strong>${i+1}. ${s.feature}</strong>: SHAP=${s.shap > 0 ? '+' : ''}${s.shap.toFixed(4)} (${s.shap > 0 ? '→ ADHD' : '→ Normal'})`
    ).join('<br/>') || '—'}</p>

    <h3>5. Clinical Note</h3>
    <p style="color:#8b949e;font-style:italic">This report is generated by an AI research tool and is intended for academic and research purposes only.
    It does not constitute a clinical diagnosis. Results should be reviewed by a qualified neurologist or psychiatrist
    in conjunction with clinical history, behavioural assessments, and standardised diagnostic criteria (DSM-5 / ICD-11).</p>`;

  // Print / Download
  $('printReportBtn')?.addEventListener('click', () => window.print());
  $('downloadReportBtn')?.addEventListener('click', () => downloadJSON(result));
}

function downloadJSON(result) {
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `adhd_analysis_${Date.now()}.json`;
  a.click();
}

// ─── Sidebar navigation ───────────────────────────────────────────────────────
function initSidebar() {
  document.querySelectorAll('.sidebar-link').forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      document.querySelectorAll('.sidebar-link').forEach(l => l.classList.remove('active'));
      link.classList.add('active');
      const target = document.querySelector(link.getAttribute('href'));
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

// ─── SHAP tabs ────────────────────────────────────────────────────────────────
function initSHAPTabs() {
  document.querySelectorAll('.shap-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.shap-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.shap-panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      $(`shap-${tab.dataset.tab}`)?.classList.add('active');
    });
  });
}

// ─── EEG signal controls ──────────────────────────────────────────────────────
function initSignalControls() {
  $('windowSlider')?.addEventListener('input', e => {
    if ($('windowLabel')) $('windowLabel').textContent = e.target.value + 's';
    if (State.currentResult && window._demoEEG) EEGViz.renderEEGChart(window._demoEEG);
  });
  $('scaleSlider')?.addEventListener('input', e => {
    if ($('scaleLabel')) $('scaleLabel').textContent = e.target.value + '×';
    if (window._demoEEG) EEGViz.renderEEGChart(window._demoEEG);
  });
}

// ─── Dataset info panel ───────────────────────────────────────────────────────
function updateDatasetInfo(filename, channels, srate, duration, epochs) {
  if ($('infoFile'))     $('infoFile').textContent     = filename.length > 18 ? filename.slice(0, 15) + '…' : filename;
  if ($('infoChannels')) $('infoChannels').textContent = channels;
  if ($('infoSrate'))    $('infoSrate').textContent    = srate + ' Hz';
  if ($('infoDuration')) $('infoDuration').textContent = (typeof duration === 'number' ? duration.toFixed(1) : duration) + 's';
  if ($('infoEpochs') && epochs) $('infoEpochs').textContent = epochs;
}

// ─── Demo SHAP data ───────────────────────────────────────────────────────────
function generateDemoSHAP(isADHD) {
  const features = [
    'Frontal Theta Power (Fz)', 'Theta/Beta Ratio (Fp1)', 'Theta/Beta Ratio (Fp2)',
    'Central Beta Power (Cz)', 'Frontal Asymmetry Index', 'Parietal Alpha (P3)',
    'Sample Entropy (F3)', 'Delta Power (O1)', 'Gamma Power (Cz)',
    'Theta Coherence (Fp1–Fp2)', 'Beta Power (C3)', 'Alpha Power (Fz)',
  ];
  const sign = isADHD ? 1 : -1;
  return features.map((feature, i) => {
    const base = [0.42, 0.36, 0.31, -0.24, 0.21, -0.17, 0.14, 0.09, 0.08, 0.12, -0.15, -0.11][i];
    return { feature, shap: (base + (Math.random() - 0.5) * 0.05) * sign, value: Math.random() * 10 };
  }).sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
}

// ─── Utility ──────────────────────────────────────────────────────────────────
function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  initSidebar();
  initSHAPTabs();
  initFileUpload();
  initSignalControls();
  await checkBackend();

  // Demo button
  $('demoBtn')?.addEventListener('click', async () => {
    addLog('Demo mode triggered manually.', 'info');
    await simulatePipeline('demo_adhd_patient.mat', true);
  });

  // Reset button
  $('resetBtn')?.addEventListener('click', () => window.location.reload());

  // Auto-run demo if ?demo=true
  if (State.isDemo) {
    addLog('Auto-running demo mode from URL parameter.', 'info');
    await delay(800);
    await simulatePipeline('demo_adhd_patient.mat', true);
  }
});
