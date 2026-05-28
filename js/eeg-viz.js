/**
 * eeg-viz.js  —  EEG signal rendering & frequency band charts
 * Uses Chart.js for all plots.
 */

const EEGViz = (() => {
  // ── colour palette for up to 19 channels ──
  const CH_COLORS = [
    '#00d4ff','#7c3aed','#10b981','#f59e0b','#ef4444',
    '#06b6d4','#a78bfa','#34d399','#fbbf24','#f87171',
    '#22d3ee','#818cf8','#6ee7b7','#fcd34d','#fca5a5',
    '#67e8f9','#c4b5fd','#86efac','#fde68a',
  ];
  const BAND_COLORS = {
    delta: '#818cf8', theta: '#f59e0b', alpha: '#10b981',
    beta:  '#00d4ff', gamma: '#a78bfa',
  };
  const BAND_RANGES = {
    delta: '0.5–4 Hz', theta: '4–8 Hz', alpha: '8–13 Hz',
    beta: '13–30 Hz', gamma: '30–40 Hz',
  };
  const BAND_ADHD = {
    delta: '→ Neutral',
    theta: '↑ Elevated (ADHD)',
    alpha: '↓ Reduced (ADHD)',
    beta:  '↓ Reduced (ADHD)',
    gamma: '→ Variable',
  };

  let eegChart       = null;
  let bandPowerChart = null;
  let relBandChart   = null;
  let tbrChart       = null;
  let psdChart       = null;
  let animFrame      = null;
  let signalData     = null; // { channels: string[], data: Float32Array[], srate: number }

  // ─── Helpers ────────────────────────────────────────────────────────────────

  /** Generate synthetic EEG-like signal for demo mode */
  function syntheticEEG(channelIdx, nSamples, srate, isADHD = true) {
    const signal = new Float32Array(nSamples);
    const dt = 1 / srate;
    // ADHD: higher theta, lower beta
    const thetaAmp = isADHD ? 28 : 14;
    const betaAmp  = isADHD ? 8  : 18;
    const alphaAmp = isADHD ? 12 : 20;
    for (let i = 0; i < nSamples; i++) {
      const t = i * dt + channelIdx * 0.3;
      signal[i] =
        thetaAmp * Math.sin(2 * Math.PI * 6.0 * t + channelIdx) +
        alphaAmp * Math.sin(2 * Math.PI * 10.5 * t + channelIdx * 0.5) +
        betaAmp  * Math.sin(2 * Math.PI * 20.0 * t + channelIdx * 0.8) +
        5 * Math.sin(2 * Math.PI * 2.0 * t) +
        (Math.random() * 2 - 1) * 8; // noise
    }
    return signal;
  }

  /** Compute Welch PSD (simplified — averaged periodogram per band) */
  function computePSD(signal, srate) {
    const n = Math.min(signal.length, 512);
    const bands = { delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0 };
    const freqRes = srate / n;
    const psd = new Array(n / 2).fill(0);

    // Simple power via DFT magnitude (for demonstration)
    for (let k = 0; k < n / 2; k++) {
      let re = 0, im = 0;
      for (let t = 0; t < n; t++) {
        const angle = 2 * Math.PI * k * t / n;
        re += (signal[t] || 0) * Math.cos(angle);
        im -= (signal[t] || 0) * Math.sin(angle);
      }
      psd[k] = (re * re + im * im) / (n * n);
    }

    const freqAxis = psd.map((_, k) => k * freqRes);
    psd.forEach((power, k) => {
      const f = freqAxis[k];
      if      (f >= 0.5 && f < 4)  bands.delta += power;
      else if (f >= 4   && f < 8)  bands.theta += power;
      else if (f >= 8   && f < 13) bands.alpha += power;
      else if (f >= 13  && f < 30) bands.beta  += power;
      else if (f >= 30  && f < 40) bands.gamma += power;
    });
    return { bands, psd, freqAxis };
  }

  // ─── EEG Multichannel Plot ──────────────────────────────────────────────────

  function renderEEGChart(data) {
    signalData = data;
    const canvas = document.getElementById('eegChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (eegChart) { eegChart.destroy(); eegChart = null; }

    const { channels, samples, srate } = data;
    const windowSec = parseInt(document.getElementById('windowSlider')?.value || 4);
    const scaleFactor = parseInt(document.getElementById('scaleSlider')?.value || 3);
    const wSamples = Math.min(windowSec * srate, samples[0].length);
    const timeLabels = Array.from({ length: wSamples }, (_, i) => (i / srate).toFixed(2));
    const OFFSET_UV = 120 * scaleFactor; // vertical separation between channels

    const datasets = channels.map((ch, i) => ({
      label: ch,
      data: Array.from(samples[i].slice(0, wSamples)).map((v, idx) => ({
        x: idx / srate,
        y: v + i * OFFSET_UV,
      })),
      borderColor: CH_COLORS[i % CH_COLORS.length],
      backgroundColor: 'transparent',
      borderWidth: 1.2,
      pointRadius: 0,
      tension: 0.1,
    }));

    eegChart = new Chart(ctx, {
      type: 'line',
      data: { datasets },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false },
        },
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'Time (s)', color: '#8b949e', font: { size: 11 } },
            ticks: { color: '#8b949e', maxTicksLimit: 10, font: { size: 10 } },
            grid: { color: 'rgba(255,255,255,0.04)' },
          },
          y: {
            title: { display: true, text: 'Amplitude (µV)', color: '#8b949e', font: { size: 11 } },
            ticks: {
              color: '#8b949e',
              font: { size: 10 },
              callback(val) {
                const chIdx = Math.round(val / OFFSET_UV);
                return channels[chIdx] ? `${channels[chIdx]}` : '';
              },
            },
            grid: { color: 'rgba(255,255,255,0.04)' },
          },
        },
      },
    });

    // Channel legend
    const legend = document.getElementById('channelLegend');
    if (legend) {
      legend.innerHTML = channels.map((ch, i) => `
        <div class="leg-item">
          <div class="leg-dot" style="background:${CH_COLORS[i % CH_COLORS.length]}"></div>
          <span>${ch}</span>
        </div>`).join('');
    }

    // Populate channel select
    const sel = document.getElementById('channelSelect');
    if (sel) {
      sel.innerHTML = channels.map((ch, i) =>
        `<option value="${i}" selected>${ch}</option>`).join('');
    }
  }

  // ─── Band Power Charts ──────────────────────────────────────────────────────

  function renderBandCharts(bandData) {
    /**
     * bandData: {
     *   channels: string[],
     *   bands: { delta, theta, alpha, beta, gamma }[]  — per channel
     * }
     */
    const { channels, bands } = bandData;
    const bandNames = ['delta', 'theta', 'alpha', 'beta', 'gamma'];

    // Average across channels
    const avgBands = {};
    bandNames.forEach(b => {
      avgBands[b] = bands.reduce((s, ch) => s + ch[b], 0) / bands.length;
    });
    const total = Object.values(avgBands).reduce((a, b) => a + b, 0);
    const relBands = {};
    bandNames.forEach(b => relBands[b] = avgBands[b] / total * 100);

    // 1. Absolute Band Power (bar)
    const absCtx = document.getElementById('bandPowerChart')?.getContext('2d');
    if (absCtx) {
      if (bandPowerChart) bandPowerChart.destroy();
      bandPowerChart = new Chart(absCtx, {
        type: 'bar',
        data: {
          labels: bandNames.map(n => n.charAt(0).toUpperCase() + n.slice(1)),
          datasets: [{
            label: 'Avg Power (µV²/Hz)',
            data: bandNames.map(b => avgBands[b].toFixed(3)),
            backgroundColor: bandNames.map(b => BAND_COLORS[b] + 'aa'),
            borderColor: bandNames.map(b => BAND_COLORS[b]),
            borderWidth: 2,
            borderRadius: 6,
          }],
        },
        options: chartOptions('Avg Power (µV²/Hz)'),
      });
    }

    // 2. Relative Band Power (doughnut)
    const relCtx = document.getElementById('relBandChart')?.getContext('2d');
    if (relCtx) {
      if (relBandChart) relBandChart.destroy();
      relBandChart = new Chart(relCtx, {
        type: 'doughnut',
        data: {
          labels: bandNames.map(n => n.charAt(0).toUpperCase() + n.slice(1)),
          datasets: [{
            data: bandNames.map(b => relBands[b].toFixed(1)),
            backgroundColor: bandNames.map(b => BAND_COLORS[b] + 'cc'),
            borderColor: '#1c2333',
            borderWidth: 3,
            hoverOffset: 6,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          plugins: {
            legend: {
              position: 'right',
              labels: { color: '#8b949e', boxWidth: 12, font: { size: 11 } },
            },
            tooltip: { callbacks: { label: ctx => `${ctx.label}: ${ctx.raw}%` } },
          },
        },
      });
    }

    // 3. Theta/Beta Ratio per channel
    const tbrCtx = document.getElementById('tbrChart')?.getContext('2d');
    if (tbrCtx) {
      if (tbrChart) tbrChart.destroy();
      const tbrs = bands.map(b => b.beta > 0 ? (b.theta / b.beta).toFixed(2) : 0);
      const tbrColors = tbrs.map(v => v >= 3.0
        ? 'rgba(239,68,68,0.7)'
        : 'rgba(16,185,129,0.7)');
      tbrChart = new Chart(tbrCtx, {
        type: 'bar',
        data: {
          labels: channels,
          datasets: [{
            label: 'Theta/Beta Ratio',
            data: tbrs,
            backgroundColor: tbrColors,
            borderColor: tbrColors.map(c => c.replace('0.7', '1')),
            borderWidth: 1.5,
            borderRadius: 4,
          }],
        },
        options: {
          ...chartOptions('Theta/Beta Ratio'),
          plugins: {
            ...chartOptions('').plugins,
            annotation: {
              annotations: {
                threshold: {
                  type: 'line',
                  yMin: 3.0, yMax: 3.0,
                  borderColor: '#ef4444',
                  borderWidth: 1.5,
                  borderDash: [6, 4],
                  label: { content: 'ADHD Threshold (3.0)', enabled: true, color: '#ef4444', font: { size: 10 } },
                },
              },
            },
          },
        },
      });
    }

    // 4. PSD (line) — averaged across all channels
    const psdCtx = document.getElementById('psdChart')?.getContext('2d');
    if (psdCtx) {
      if (psdChart) psdChart.destroy();
      // Create frequency axis 0.5–40 Hz with 0.5 Hz resolution
      const freqs = [];
      const powers = [];
      for (let f = 0.5; f <= 40; f += 0.5) {
        freqs.push(f.toFixed(1));
        // Interpolate from band values
        let p = 0;
        if (f < 4)  p = avgBands.delta / 7;
        else if (f < 8)  p = avgBands.theta / 8;
        else if (f < 13) p = avgBands.alpha / 10 * (1 + .3 * Math.sin(f));
        else if (f < 30) p = avgBands.beta  / 34 * (1 + .2 * Math.cos(f));
        else             p = avgBands.gamma / 20;
        powers.push((p * (0.8 + Math.random() * 0.4)).toFixed(4));
      }
      psdChart = new Chart(psdCtx, {
        type: 'line',
        data: {
          labels: freqs,
          datasets: [{
            label: 'PSD (µV²/Hz)',
            data: powers,
            borderColor: '#00d4ff',
            backgroundColor: 'rgba(0,212,255,0.06)',
            borderWidth: 1.5,
            pointRadius: 0,
            fill: true,
            tension: 0.4,
          }],
        },
        options: {
          ...chartOptions('Power (µV²/Hz)'),
          scales: {
            x: { ...xScale('Frequency (Hz)'), ticks: { ...xScale('').ticks, maxTicksLimit: 10 } },
            y: yScale('Power (µV²/Hz)'),
          },
        },
      });
    }

    // 5. TBR summary card
    const avgTBR = (avgBands.theta / avgBands.beta).toFixed(2);
    const tbrEl = document.getElementById('tbrValue');
    const tbrNote = document.getElementById('tbrNote');
    const tbrCard = document.getElementById('tbrCard');
    if (tbrEl) tbrEl.textContent = avgTBR;
    if (tbrNote && tbrCard) {
      if (parseFloat(avgTBR) >= 3.0) {
        tbrNote.textContent = 'Elevated — ADHD pattern detected';
        tbrNote.style.color = '#ef4444';
        tbrCard.style.borderColor = 'rgba(239,68,68,0.4)';
      } else {
        tbrNote.textContent = 'Within normal range (< 3.0)';
        tbrNote.style.color = '#10b981';
        tbrCard.style.borderColor = 'rgba(16,185,129,0.4)';
      }
    }

    // 6. Band table
    const tbody = document.getElementById('bandTableBody');
    if (tbody) {
      tbody.innerHTML = bandNames.map(b => `
        <tr>
          <td>${b.charAt(0).toUpperCase() + b.slice(1)}</td>
          <td>${BAND_RANGES[b]}</td>
          <td>${avgBands[b].toFixed(4)} µV²/Hz</td>
          <td>${relBands[b].toFixed(1)}%</td>
          <td style="color:${BAND_COLORS[b]}">${BAND_ADHD[b]}</td>
        </tr>`).join('');
    }
  }

  // ─── Gauge Chart (prediction) ───────────────────────────────────────────────

  function renderGauge(probability) {
    const canvas = document.getElementById('gaugeChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const pct = Math.round(probability * 100);
    const angle = Math.PI + Math.PI * probability; // half circle
    const isADHD = probability > 0.5;
    const color = probability > 0.75 ? '#ef4444'
                : probability > 0.5  ? '#f59e0b'
                : '#10b981';

    ctx.clearRect(0, 0, 280, 280);
    const cx = 140, cy = 160, r = 110;

    // Background arc
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Value arc
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI, angle);
    ctx.strokeStyle = color;
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.shadowColor = color;
    ctx.shadowBlur = 20;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Needle
    const needleAngle = Math.PI + Math.PI * probability;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + (r - 20) * Math.cos(needleAngle), cy + (r - 20) * Math.sin(needleAngle));
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, 7, 0, 2 * Math.PI);
    ctx.fillStyle = '#fff';
    ctx.fill();

    // Labels
    ctx.fillStyle = '#484f58';
    ctx.font = 'bold 11px Inter';
    ctx.fillText('Normal', 28, cy + 28);
    ctx.fillText('ADHD',   220, cy + 28);

    // Center label
    const gaugePct  = document.getElementById('gaugePct');
    const gaugeVerd = document.getElementById('gaugeVerdict');
    if (gaugePct) { gaugePct.textContent = pct + '%'; gaugePct.style.color = color; }
    if (gaugeVerd) { gaugeVerd.textContent = isADHD ? 'ADHD Likely' : 'Normal Likely'; gaugeVerd.style.color = color; }
  }

  // ─── Confidence Bar Chart (prediction) ─────────────────────────────────────

  function renderConfidenceChart(probADHD) {
    const ctx = document.getElementById('confidenceChart')?.getContext('2d');
    if (!ctx) return;
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Normal', 'ADHD'],
        datasets: [{
          data: [(1 - probADHD).toFixed(3), probADHD.toFixed(3)],
          backgroundColor: ['rgba(16,185,129,0.7)', 'rgba(239,68,68,0.7)'],
          borderColor: ['#10b981', '#ef4444'],
          borderWidth: 2,
          borderRadius: 6,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: true,
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => ` ${(c.raw * 100).toFixed(1)}%` } } },
        scales: {
          x: { min: 0, max: 1, ticks: { color: '#8b949e', callback: v => `${(v*100).toFixed(0)}%`, font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
          y: { ticks: { color: '#e6edf3', font: { size: 11, weight: '600' } }, grid: { display: false } },
        },
      },
    });
  }

  // ─── SHAP Charts ────────────────────────────────────────────────────────────

  function renderSHAPWaterfall(shapValues) {
    /** shapValues: [{feature, value, shap}] sorted by |shap| desc */
    const ctx = document.getElementById('shapWaterfallChart')?.getContext('2d');
    if (!ctx) return;
    const top = shapValues.slice(0, 12);
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: top.map(d => d.feature),
        datasets: [{
          label: 'SHAP Value',
          data: top.map(d => d.shap),
          backgroundColor: top.map(d => d.shap > 0 ? 'rgba(239,68,68,0.75)' : 'rgba(59,130,246,0.75)'),
          borderColor: top.map(d => d.shap > 0 ? '#ef4444' : '#3b82f6'),
          borderWidth: 1.5,
          borderRadius: 4,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: c => ` SHAP: ${c.raw > 0 ? '+' : ''}${c.raw.toFixed(4)}`,
              afterLabel: c => ` → ${c.raw > 0 ? 'Pushes toward ADHD' : 'Pushes toward Normal'}`,
            },
          },
        },
        scales: {
          x: {
            ticks: { color: '#8b949e', font: { size: 10 } },
            grid: { color: 'rgba(255,255,255,0.05)' },
            title: { display: true, text: 'SHAP Value (impact on prediction)', color: '#8b949e', font: { size: 10 } },
          },
          y: { ticks: { color: '#e6edf3', font: { size: 10 } }, grid: { display: false } },
        },
      },
    });
  }

  function renderSHAPBar(shapValues) {
    const ctx = document.getElementById('shapBarChart')?.getContext('2d');
    if (!ctx) return;
    const top = shapValues.slice(0, 12).sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: top.map(d => d.feature),
        datasets: [{
          label: '|SHAP| Importance',
          data: top.map(d => Math.abs(d.shap).toFixed(4)),
          backgroundColor: 'rgba(0,212,255,0.65)',
          borderColor: '#00d4ff',
          borderWidth: 1.5,
          borderRadius: 4,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#8b949e', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: { ticks: { color: '#e6edf3', font: { size: 10 } }, grid: { display: false } },
        },
      },
    });
  }

  function renderForcePlot(shapValues, baseValue, prediction) {
    const container = document.getElementById('shapForcePlot');
    if (!container) return;
    const top = shapValues.slice(0, 8);
    const totalW = 600;
    const positives = top.filter(d => d.shap > 0);
    const negatives = top.filter(d => d.shap < 0);
    const sumPos = positives.reduce((s, d) => s + d.shap, 0);
    const sumNeg = Math.abs(negatives.reduce((s, d) => s + d.shap, 0));

    container.innerHTML = `
      <div style="width:100%;font-size:.8rem;color:#8b949e;margin-bottom:.5rem;">
        Base value: <strong style="color:#e6edf3">${baseValue.toFixed(3)}</strong>
        &nbsp;→&nbsp; Prediction: <strong style="color:${prediction > 0.5 ? '#ef4444' : '#10b981'}">${prediction.toFixed(3)}</strong>
      </div>
      <div style="display:flex;width:100%;height:50px;border-radius:8px;overflow:hidden;border:1px solid #30363d">
        ${positives.map(d => {
          const w = (d.shap / (sumPos + sumNeg) * 100).toFixed(1);
          return `<div title="${d.feature}: +${d.shap.toFixed(4)}" style="width:${w}%;background:rgba(239,68,68,0.7);display:flex;align-items:center;justify-content:center;font-size:.65rem;color:#fff;overflow:hidden;white-space:nowrap;border-right:1px solid rgba(239,68,68,.3)">${d.feature.split(' ')[0]}</div>`;
        }).join('')}
        ${negatives.map(d => {
          const w = (Math.abs(d.shap) / (sumPos + sumNeg) * 100).toFixed(1);
          return `<div title="${d.feature}: ${d.shap.toFixed(4)}" style="width:${w}%;background:rgba(59,130,246,0.7);display:flex;align-items:center;justify-content:center;font-size:.65rem;color:#fff;overflow:hidden;white-space:nowrap;border-right:1px solid rgba(59,130,246,.3)">${d.feature.split(' ')[0]}</div>`;
        }).join('')}
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:.4rem;font-size:.7rem;color:#8b949e">
        <span style="color:#ef4444">← Pushes toward ADHD</span>
        <span style="color:#3b82f6">Pushes toward Normal →</span>
      </div>`;
  }

  // ─── Chart option helpers ────────────────────────────────────────────────────

  function xScale(label) {
    return {
      ticks: { color: '#8b949e', font: { size: 10 } },
      grid: { color: 'rgba(255,255,255,0.04)' },
      title: label ? { display: true, text: label, color: '#8b949e', font: { size: 10 } } : {},
    };
  }
  function yScale(label) {
    return {
      ticks: { color: '#8b949e', font: { size: 10 } },
      grid: { color: 'rgba(255,255,255,0.04)' },
      title: label ? { display: true, text: label, color: '#8b949e', font: { size: 10 } } : {},
    };
  }
  function chartOptions(yLabel) {
    return {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: '#161b22', borderColor: '#30363d', borderWidth: 1, titleColor: '#e6edf3', bodyColor: '#8b949e' },
      },
      scales: {
        x: xScale(''),
        y: yScale(yLabel),
      },
    };
  }

  // ─── Demo data generator ────────────────────────────────────────────────────

  function generateDemoData(isADHD = true) {
    const channels = ['Fp1','Fp2','F3','F4','F7','F8','Fz','C3','C4','Cz','P3','P4','Pz','O1','O2','T3','T4','T5','T6'];
    const srate = 256;
    const duration = 10; // seconds
    const nSamples = srate * duration;
    const samples = channels.map((_, i) => syntheticEEG(i, nSamples, srate, isADHD));

    const bandsPerCh = channels.map((_, i) => {
      const { bands } = computePSD(samples[i], srate);
      return bands;
    });

    return {
      channels,
      samples,
      srate,
      duration,
      bands: bandsPerCh,
    };
  }

  // ─── Public API ─────────────────────────────────────────────────────────────

  return {
    generateDemoData,
    computePSD,
    renderEEGChart,
    renderBandCharts,
    renderGauge,
    renderConfidenceChart,
    renderSHAPWaterfall,
    renderSHAPBar,
    renderForcePlot,
    CH_COLORS,
    BAND_COLORS,
  };
})();
