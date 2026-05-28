/**
 * brain-map.js  —  D3-based EEG topographic brain map
 * Renders a 2D scalp projection with colour-coded channel power.
 */

const BrainMap = (() => {
  // Standard 10-20 electrode positions (normalised to unit circle)
  const ELECTRODE_POSITIONS = {
    'Fp1':{ x: -0.17, y:  0.92 }, 'Fp2':{ x:  0.17, y:  0.92 },
    'F7': { x: -0.64, y:  0.64 }, 'F3': { x: -0.33, y:  0.59 },
    'Fz': { x:  0.00, y:  0.60 }, 'F4': { x:  0.33, y:  0.59 },
    'F8': { x:  0.64, y:  0.64 },
    'T3': { x: -0.95, y:  0.00 }, 'C3': { x: -0.45, y:  0.00 },
    'Cz': { x:  0.00, y:  0.00 }, 'C4': { x:  0.45, y:  0.00 },
    'T4': { x:  0.95, y:  0.00 },
    'T5': { x: -0.64, y: -0.64 }, 'P3': { x: -0.33, y: -0.55 },
    'Pz': { x:  0.00, y: -0.55 }, 'P4': { x:  0.33, y: -0.55 },
    'T6': { x:  0.64, y: -0.64 },
    'O1': { x: -0.17, y: -0.90 }, 'O2': { x:  0.17, y: -0.90 },
  };

  let currentData = null;
  let svg = null;
  const SIZE = 300;
  const R = 130;
  const CX = SIZE / 2, CY = SIZE / 2;

  function init(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    svg = d3.select(container)
      .append('svg')
      .attr('viewBox', `0 0 ${SIZE} ${SIZE}`)
      .attr('width', SIZE)
      .attr('height', SIZE);

    // Scalp outline
    svg.append('ellipse')
      .attr('cx', CX).attr('cy', CY)
      .attr('rx', R).attr('ry', R)
      .attr('fill', 'rgba(28,35,51,0.9)')
      .attr('stroke', '#30363d')
      .attr('stroke-width', 2);

    // Nose indicator
    svg.append('path')
      .attr('d', `M ${CX-10} ${CY - R + 5} Q ${CX} ${CY - R - 18} ${CX+10} ${CY - R + 5}`)
      .attr('fill', 'none')
      .attr('stroke', '#484f58')
      .attr('stroke-width', 2);

    // Left/right ear indicators
    [[CX - R - 2, CY], [CX + R + 2, CY]].forEach(([ex, ey]) => {
      svg.append('ellipse')
        .attr('cx', ex).attr('cy', ey)
        .attr('rx', 6).attr('ry', 10)
        .attr('fill', 'rgba(28,35,51,0.9)')
        .attr('stroke', '#484f58')
        .attr('stroke-width', 2);
    });
  }

  function project(nx, ny) {
    // nx, ny in [-1, 1]; map to SVG space (y is flipped)
    return { x: CX + nx * R * 0.88, y: CY - ny * R * 0.88 };
  }

  function colorScale(value, min, max) {
    const t = (value - min) / (max - min + 1e-9);
    // Blue → Green → Yellow → Red
    if (t < 0.25) return d3.interpolateRgb('#3b82f6', '#10b981')(t * 4);
    if (t < 0.5)  return d3.interpolateRgb('#10b981', '#f59e0b')((t - 0.25) * 4);
    if (t < 0.75) return d3.interpolateRgb('#f59e0b', '#ef4444')((t - 0.5) * 4);
    return d3.interpolateRgb('#ef4444', '#dc2626')((t - 0.75) * 4);
  }

  function render(channelValues, bandName) {
    /**
     * channelValues: { [channelName]: number }
     * bandName: string (for display)
     */
    if (!svg) init('topoMap');
    currentData = channelValues;

    const values = Object.values(channelValues);
    const minV = Math.min(...values);
    const maxV = Math.max(...values);

    // Remove old channel elements
    svg.selectAll('.ch-circle, .ch-label, .interp-patch').remove();

    // Interpolated background patches (simplified grid)
    const gridSize = 18;
    for (let gx = 0; gx < gridSize; gx++) {
      for (let gy = 0; gy < gridSize; gy++) {
        const nx = (gx / (gridSize - 1)) * 2 - 1;
        const ny = (gy / (gridSize - 1)) * 2 - 1;
        if (nx * nx + ny * ny > 0.95) continue; // outside head
        const { x, y } = project(nx, ny);

        // IDW interpolation from nearby channels
        let wSum = 0, vSum = 0;
        Object.entries(ELECTRODE_POSITIONS).forEach(([ch, pos]) => {
          const val = channelValues[ch];
          if (val === undefined) return;
          const d2 = (nx - pos.x) ** 2 + (ny - pos.y) ** 2;
          const w = 1 / (d2 + 0.001);
          wSum += w; vSum += w * val;
        });
        const interpVal = wSum > 0 ? vSum / wSum : minV;
        const cellW = SIZE / gridSize;

        svg.append('rect')
          .attr('class', 'interp-patch')
          .attr('x', x - cellW / 2)
          .attr('y', y - cellW / 2)
          .attr('width', cellW)
          .attr('height', cellW)
          .attr('fill', colorScale(interpVal, minV, maxV))
          .attr('opacity', 0.55)
          .attr('clip-path', 'url(#headClip)');
      }
    }

    // Clip to head circle
    if (!svg.select('#headClip').node()) {
      svg.append('defs').append('clipPath').attr('id', 'headClip')
        .append('ellipse')
        .attr('cx', CX).attr('cy', CY).attr('rx', R - 2).attr('ry', R - 2);
    }

    // Draw electrode circles
    Object.entries(ELECTRODE_POSITIONS).forEach(([ch, pos]) => {
      const val = channelValues[ch];
      const { x, y } = project(pos.x, pos.y);
      const color = val !== undefined ? colorScale(val, minV, maxV) : '#484f58';
      const r = val !== undefined ? 9 : 6;

      svg.append('circle')
        .attr('class', 'ch-circle')
        .attr('cx', x).attr('cy', y)
        .attr('r', r)
        .attr('fill', color)
        .attr('stroke', '#0d1117')
        .attr('stroke-width', 1.5)
        .attr('opacity', val !== undefined ? 0.95 : 0.4)
        .append('title')
        .text(`${ch}: ${val !== undefined ? val.toFixed(4) + ' µV²/Hz' : 'N/A'}`);

      svg.append('text')
        .attr('class', 'ch-label')
        .attr('x', x).attr('y', y + 1)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '6.5px')
        .attr('font-weight', '700')
        .attr('fill', '#fff')
        .attr('pointer-events', 'none')
        .text(ch.length <= 3 ? ch : ch.slice(0, 2));
    });

    // Update colorbar labels
    const fmt = v => v < 0.001 ? v.toExponential(2) : v.toFixed(4);
    const cbMin = document.getElementById('cbMin');
    const cbMid = document.getElementById('cbMid');
    const cbMax = document.getElementById('cbMax');
    if (cbMin) cbMin.textContent = fmt(minV);
    if (cbMid) cbMid.textContent = fmt((minV + maxV) / 2);
    if (cbMax) cbMax.textContent = fmt(maxV);

    // Update channel list
    const chList = document.getElementById('channelList');
    if (chList) {
      chList.innerHTML = Object.entries(channelValues)
        .sort(([, a], [, b]) => b - a)
        .map(([ch, val]) => {
          const color = colorScale(val, minV, maxV);
          return `<div class="ch-row">
            <div class="ch-color" style="background:${color}"></div>
            <span class="ch-name">${ch}</span>
            <span class="ch-value">${fmt(val)}</span>
          </div>`;
        }).join('');
    }
  }

  function updateBand(bandName, allBandData) {
    /**
     * allBandData: { [channel]: { delta, theta, alpha, beta, gamma } }
     */
    const channelValues = {};
    Object.entries(allBandData).forEach(([ch, bands]) => {
      channelValues[ch] = bands[bandName] || 0;
    });
    render(channelValues, bandName);
  }

  return { init, render, updateBand, ELECTRODE_POSITIONS };
})();

// Auto-init on DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
  BrainMap.init('topoMap');

  // Band selector buttons
  document.querySelectorAll('.band-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.band-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const band = btn.dataset.band;
      if (window._lastBandData) BrainMap.updateBand(band, window._lastBandData);
    });
  });

  // Map metric selector
  document.getElementById('mapMetric')?.addEventListener('change', e => {
    const metric = e.target.value;
    if (!window._lastBandData) return;
    const activeBand = document.querySelector('.band-btn.active')?.dataset.band || 'theta';
    // For TBR metric
    if (metric === 'tbr' && window._lastBandData) {
      const tbrValues = {};
      Object.entries(window._lastBandData).forEach(([ch, b]) => {
        tbrValues[ch] = b.beta > 0 ? b.theta / b.beta : 0;
      });
      BrainMap.render(tbrValues, 'tbr');
    } else if (metric === 'relative' && window._lastBandData) {
      const relValues = {};
      Object.entries(window._lastBandData).forEach(([ch, b]) => {
        const total = Object.values(b).reduce((s, v) => s + v, 0);
        relValues[ch] = total > 0 ? (b[activeBand] / total) : 0;
      });
      BrainMap.render(relValues, activeBand);
    } else {
      BrainMap.updateBand(activeBand, window._lastBandData);
    }
  });
});
