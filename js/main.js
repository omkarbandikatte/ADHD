/* ── Landing page JS: counter animations, hero EEG wave, navbar scroll ── */

// ═══ Navbar scroll behaviour ═══
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 40);
});

// ═══ Animated counter (hero stats) ═══
function animateCounters() {
  document.querySelectorAll('.stat-number[data-target]').forEach(el => {
    const target = parseFloat(el.dataset.target);
    const duration = 1800;
    const step = target / (duration / 16);
    let current = 0;
    const timer = setInterval(() => {
      current = Math.min(current + step, target);
      el.textContent = Number.isInteger(target) ? Math.floor(current) : current.toFixed(1);
      if (current >= target) clearInterval(timer);
    }, 16);
  });
}

// Observe hero stats section
const statsObserver = new IntersectionObserver(entries => {
  entries.forEach(e => { if (e.isIntersecting) { animateCounters(); statsObserver.disconnect(); } });
}, { threshold: 0.4 });
const statsEl = document.querySelector('.hero-stats');
if (statsEl) statsObserver.observe(statsEl);

// ═══ Hero animated EEG canvas ═══
(function initHeroEEG() {
  const canvas = document.getElementById('heroEEG');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  const CHANNELS = [
    { label: 'Fp1', color: '#00d4ff', freq: 7.2,  amp: 22, phase: 0.0,   noiseAmp: 4 },
    { label: 'Fp2', color: '#7c3aed', freq: 6.8,  amp: 20, phase: 0.8,   noiseAmp: 4 },
    { label: 'Fz',  color: '#10b981', freq: 9.1,  amp: 18, phase: 1.6,   noiseAmp: 3 },
    { label: 'Cz',  color: '#f59e0b', freq: 10.5, amp: 15, phase: 2.4,   noiseAmp: 3 },
    { label: 'Pz',  color: '#ef4444', freq: 8.3,  amp: 17, phase: 3.2,   noiseAmp: 4 },
  ];
  const NPTS = W;
  let offset = 0;

  function generateSignal(ch, t) {
    let v = 0;
    v += ch.amp * Math.sin(2 * Math.PI * ch.freq * t + ch.phase);
    v += ch.amp * 0.35 * Math.sin(2 * Math.PI * (ch.freq * 1.8) * t + ch.phase * 0.7);
    v += ch.noiseAmp * (Math.random() * 2 - 1);
    return v;
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    const trackH = H / CHANNELS.length;

    CHANNELS.forEach((ch, i) => {
      const baseline = trackH * i + trackH / 2;
      ctx.beginPath();
      ctx.strokeStyle = ch.color;
      ctx.lineWidth = 1.4;
      ctx.shadowColor = ch.color;
      ctx.shadowBlur = 6;
      for (let x = 0; x < NPTS; x++) {
        const t = (x + offset) / 200;
        const y = baseline + generateSignal(ch, t);
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Channel label
      ctx.fillStyle = ch.color;
      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.fillText(ch.label, 6, baseline - 2);
    });

    offset += 1.5;
    requestAnimationFrame(draw);
  }
  draw();
})();

// ═══ Animated neural network background ═══
(function initBrainCanvas() {
  const canvas = document.getElementById('brainCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  const NODES = Array.from({ length: 60 }, () => ({
    x: Math.random() * W,
    y: Math.random() * H,
    vx: (Math.random() - .5) * .3,
    vy: (Math.random() - .5) * .3,
    r: Math.random() * 2 + 1.5,
  }));
  const DIST = 160;

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Connections
    for (let i = 0; i < NODES.length; i++) {
      for (let j = i + 1; j < NODES.length; j++) {
        const dx = NODES[i].x - NODES[j].x;
        const dy = NODES[i].y - NODES[j].y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < DIST) {
          const alpha = (1 - d / DIST) * 0.4;
          ctx.beginPath();
          ctx.strokeStyle = `rgba(0,212,255,${alpha})`;
          ctx.lineWidth = .5;
          ctx.moveTo(NODES[i].x, NODES[i].y);
          ctx.lineTo(NODES[j].x, NODES[j].y);
          ctx.stroke();
        }
      }
    }

    // Nodes
    NODES.forEach(n => {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0,212,255,0.6)';
      ctx.shadowColor = '#00d4ff';
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowBlur = 0;

      n.x += n.vx; n.y += n.vy;
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
    });
    requestAnimationFrame(draw);
  }
  draw();
})();

// ═══ Pipeline steps scroll-in ═══
const pipeObs = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) e.target.classList.add('visible');
  });
}, { threshold: 0.15 });
document.querySelectorAll('.pipeline-step').forEach(el => pipeObs.observe(el));

// ═══ Band wave SVG paths (landing page biomarkers) ═══
function generateSVGWave(freq, amp, points, width, height) {
  const midY = height / 2;
  let d = `M 0 ${midY}`;
  for (let i = 1; i <= points; i++) {
    const x = (i / points) * width;
    const y = midY + Math.sin(i * freq * 0.2) * amp;
    d += ` L ${x} ${y}`;
  }
  return d;
}

document.querySelectorAll('.wave-path.theta').forEach(p => p.setAttribute('d', generateSVGWave(7,  14, 60, 120, 50)));
document.querySelectorAll('.wave-path.beta').forEach(p  => p.setAttribute('d', generateSVGWave(18, 8,  60, 120, 50)));
document.querySelectorAll('.wave-path.alpha').forEach(p => p.setAttribute('d', generateSVGWave(10, 12, 60, 120, 50)));
document.querySelectorAll('.wave-path').forEach(p => {
  p.setAttribute('stroke', '#00d4ff');
  p.setAttribute('stroke-width', '2');
  p.setAttribute('fill', 'none');
  p.setAttribute('opacity', '0.8');
});

// ═══ Smooth-scroll nav links ═══
document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    const target = document.querySelector(link.getAttribute('href'));
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
});

// ═══ Check for demo param ═══
if (new URLSearchParams(window.location.search).get('demo') === 'true') {
  window.location.href = 'dashboard.html?demo=true';
}
