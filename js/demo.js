/* ════════════════════════════════════════════════════════════
   NeuroADHD — Demo Video Engine  (demo.js)
   Click Play → animates through all 9 steps like a video
   ════════════════════════════════════════════════════════════ */
'use strict';

const Demo = (() => {

  /* ── Config ──────────────────────────────────────────── */
  const STEPS = [
    { n: 0, label: 'Title',        short: '',          dur: 0     },
    { n: 1, label: 'EEG Acquire',  short: '1\nAcquire', dur: 12000 },
    { n: 2, label: 'Preprocess',   short: '2\nPrep',    dur: 14000 },
    { n: 3, label: 'Frequency',    short: '3\nFreq',    dur: 13000 },
    { n: 4, label: 'Features',     short: '4\nFeat',    dur: 12000 },
    { n: 5, label: 'CNN + TCN',    short: '5\nDL',      dur: 16000 },
    { n: 6, label: 'Prediction',   short: '6\nPred',    dur: 11000 },
    { n: 7, label: 'Attn Heatmap', short: '7\nAttn',    dur: 12000 },
    { n: 8, label: 'Brain Map',    short: '8\nMap',     dur: 9000  },
    { n: 9, label: 'Summary',      short: '9\nSum',     dur: 8000  },
  ];

  /* ── State ───────────────────────────────────────────── */
  let cur       = 0;
  let playing   = false;
  let timers    = [];
  let raf       = null;
  let charts    = {};

  /* ── Chart registry ─────────────────────────────────── */
  function killChart(id) { if (charts[id]) { charts[id].destroy(); delete charts[id]; } }
  function regChart(id, c) { charts[id] = c; }

  /* ══════════════════════════════════════════════════════
     PUBLIC
  ══════════════════════════════════════════════════════ */
  function goTo(idx) {
    if (idx < 0 || idx >= STEPS.length) return;
    clearAll();

    /* transition out */
    const old = document.querySelector('.step.active');
    if (old) {
      if (idx > cur) old.classList.add('exit-l');
      setTimeout(() => old.classList.remove('active','exit-l'), 450);
    }

    cur = idx;
    const el = document.getElementById(`step-${cur}`);
    if (el) el.classList.add('active');

    updateHeader();
    updateTimeline();
    runScene(cur);

    /* auto-advance */
    if (playing && cur > 0 && cur < STEPS.length - 1) {
      const t = setTimeout(() => goTo(cur + 1), STEPS[cur].dur);
      timers.push(t);
      animateStepFill(STEPS[cur].dur);
    }
  }

  function startAutoPlay() {
    playing = true;
    updatePlayBtn();
    goTo(1);
  }

  function togglePlay() {
    if (playing) {
      playing = false;
      clearAll();
      updatePlayBtn();
    } else {
      playing = true;
      updatePlayBtn();
      if (cur === 0) goTo(1);
      else {
        const remaining = STEPS[cur]?.dur;
        if (remaining) {
          const t = setTimeout(() => goTo(cur + 1), remaining);
          timers.push(t);
          animateStepFill(remaining);
        }
      }
    }
  }

  function switchTopoMap(band) {
    document.querySelectorAll('.bsw').forEach(b => b.classList.toggle('active', b.dataset.band === band));
    renderTopo(band);
    const lbl = document.getElementById('topoBl');
    const names = { theta:'θ Theta Power (4–8 Hz)', alpha:'α Alpha Power (8–13 Hz)', beta:'β Beta Power (13–30 Hz)' };
    if (lbl) lbl.textContent = names[band] || band;
  }

  /* ══════════════════════════════════════════════════════
     INTERNALS
  ══════════════════════════════════════════════════════ */
  function clearAll() {
    timers.forEach(clearTimeout);
    timers = [];
    cancelAnimationFrame(raf);
    raf = null;
    document.getElementById('stepFill').style.width = '0%';
  }

  function updatePlayBtn() {
    const icon = document.getElementById('playIcon');
    const btn  = document.getElementById('playBtn');
    if (icon) icon.className = playing ? 'fas fa-pause' : 'fas fa-play';
    if (btn)  btn.classList.toggle('active', playing);
  }

  function updateHeader() {
    const s = STEPS[cur];
    document.getElementById('tbNum').textContent  = cur > 0 ? `Step ${cur} / 9` : '';
    document.getElementById('tbName').textContent = cur > 0 ? s.label : 'Press Play to begin';
  }

  function updateTimeline() {
    /* overall fill */
    const pct = cur === 0 ? 0 : ((cur - 1) / (STEPS.length - 2)) * 100;
    document.getElementById('vtFill').style.width = pct + '%';

    /* dots & labels */
    document.querySelectorAll('.vt-dot').forEach((d, i) => {
      d.classList.toggle('active',  i + 1 === cur);
      d.classList.toggle('visited', i + 1 < cur);
    });
    document.querySelectorAll('.vt-lbl').forEach((l, i) => {
      l.classList.toggle('active',  i + 1 === cur);
      l.classList.toggle('visited', i + 1 < cur);
    });
  }

  /* Step-level progress fill */
  function animateStepFill(dur) {
    const fill  = document.getElementById('stepFill');
    const start = performance.now();
    fill.style.width = '0%';

    function frame(now) {
      const t = Math.min((now - start) / dur, 1);
      fill.style.width = (t * 100) + '%';
      if (t < 1) raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);
  }

  /* ══════════════════════════════════════════════════════
     SCENE DISPATCHER
  ══════════════════════════════════════════════════════ */
  function runScene(idx) {
    const fns = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9];
    if (fns[idx]) fns[idx]();
  }

  /* ── HELPER: delay timer ────────────────────────────── */
  function at(ms, fn) { const t = setTimeout(fn, ms); timers.push(t); return t; }

  /* ── HELPER: counter animation ──────────────────────── */
  function counter(id, from, to, dur, dec, prefix='', suffix='', cb) {
    const el = document.getElementById(id);
    if (!el) return;
    const s0 = performance.now();
    function f(now) {
      const t  = Math.min((now - s0) / dur, 1);
      const e  = t < .5 ? 2*t*t : -1+(4-2*t)*t;
      el.textContent = prefix + (from + (to - from)*e).toFixed(dec) + suffix;
      if (t < 1) { const r = requestAnimationFrame(f); timers.push(r); }
      else { el.textContent = prefix + to.toFixed(dec) + suffix; if (cb) cb(); }
    }
    const r = requestAnimationFrame(f);
    timers.push(r);
  }

  /* ══════════════════════════════════════════════════════
     S0 — TITLE
  ══════════════════════════════════════════════════════ */
  function s0() {
    const cvs = document.getElementById('titleCvs');
    if (!cvs || cvs._run) return;
    cvs._run = true;
    const ctx = cvs.getContext('2d');
    function resize() { cvs.width = cvs.offsetWidth; cvs.height = cvs.offsetHeight; }
    resize(); window.addEventListener('resize', resize);
    const nodes = Array.from({length:60}, () => ({
      x:Math.random()*cvs.width, y:Math.random()*cvs.height,
      vx:(Math.random()-.5)*.5, vy:(Math.random()-.5)*.5, r:1.5+Math.random()*1.5
    }));
    function draw() {
      if (!cvs._run) return;
      ctx.clearRect(0,0,cvs.width,cvs.height);
      nodes.forEach((a,i) => {
        nodes.slice(i+1).forEach(b => {
          const d = Math.hypot(a.x-b.x,a.y-b.y);
          if (d < 130) {
            ctx.strokeStyle = `rgba(0,212,255,${.05*(1-d/130)})`;
            ctx.lineWidth = .5;
            ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
          }
        });
        ctx.beginPath(); ctx.arc(a.x,a.y,a.r,0,Math.PI*2);
        ctx.fillStyle='rgba(0,212,255,.45)'; ctx.fill();
        a.x+=a.vx; a.y+=a.vy;
        if(a.x<0||a.x>cvs.width)  a.vx*=-1;
        if(a.y<0||a.y>cvs.height)  a.vy*=-1;
      });
      requestAnimationFrame(draw);
    }
    draw();
  }

  /* ══════════════════════════════════════════════════════
     S1 — EEG RECORDING SCENE
  ══════════════════════════════════════════════════════ */

  /* 19-channel 10-20 positions on the SVG head (cx=200, cy=168, rx=38, ry=44) */
  const ELEC_POS = [
    {n:'Fp1',x:183,y:126},{n:'Fp2',x:217,y:126},
    {n:'F7', x:166,y:145},{n:'F3', x:182,y:140},{n:'Fz', x:200,y:136},{n:'F4', x:218,y:140},{n:'F8', x:234,y:145},
    {n:'T3', x:163,y:168},{n:'C3', x:181,y:164},{n:'Cz', x:200,y:161},{n:'C4', x:219,y:164},{n:'T4', x:237,y:168},
    {n:'T5', x:166,y:190},{n:'P3', x:182,y:187},{n:'Pz', x:200,y:185},{n:'P4', x:218,y:187},{n:'T6', x:234,y:190},
    {n:'O1', x:184,y:205},{n:'O2', x:216,y:205},
  ];

  function s1() {
    /* ── 1. fade in doctor ── */
    at(200, () => {
      const g = document.getElementById('doctorG');
      if(g) { g.style.transition='opacity .6s'; g.style.opacity='1'; }
    });

    /* ── 2. fade in patient ── */
    at(700, () => {
      const g = document.getElementById('patientG');
      if(g) { g.style.transition='opacity .6s'; g.style.opacity='1'; }
    });

    /* ── 3. doctor arm extends toward patient ── */
    at(1200, () => {
      const arm = document.getElementById('docArm');
      if(!arm) return;
      let prog = 0;
      const t = setInterval(() => {
        prog = Math.min(prog + 4, 100);
        const x2 = 90 + (160 - 90) * prog / 100;
        const y2 = 178 + (170 - 178) * prog / 100;
        arm.setAttribute('x2', x2); arm.setAttribute('y2', y2);
        if(prog >= 100) clearInterval(t);
      }, 16);
      timers.push(t);
    });

    /* ── 4. step 1 active — electrode placement ── */
    at(1000, () => {
      document.getElementById('rs1')?.classList.add('active');
    });

    /* ── 5. electrodes appear one by one ── */
    at(1800, () => {
      const g = document.getElementById('electrodeG');
      if(!g || g._b) return; g._b = true;
      ELEC_POS.forEach((ep, i) => {
        at(1800 + i * 140, () => {
          const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
          c.setAttribute('cx', ep.x); c.setAttribute('cy', ep.y); c.setAttribute('r', '4');
          c.setAttribute('fill', '#00d4ff'); c.setAttribute('stroke', '#0d1117'); c.setAttribute('stroke-width', '1.2');
          c.setAttribute('opacity', '0');
          c.style.transition = 'opacity .3s';
          g.appendChild(c);
          requestAnimationFrame(() => requestAnimationFrame(() => c.setAttribute('opacity','0.85')));

          /* tiny label for a few key channels */
          if(['Fz','Cz','Pz','T3','T4'].includes(ep.n)) {
            const t = document.createElementNS('http://www.w3.org/2000/svg','text');
            t.setAttribute('x', ep.x + 5); t.setAttribute('y', ep.y - 5);
            t.setAttribute('font-size','6'); t.setAttribute('fill','#8b949e');
            t.setAttribute('font-family','Inter,sans-serif');
            t.textContent = ep.n; g.appendChild(t);
          }
        });
      });
    });

    /* ── 6. step 1 done, step 2 active — amplifier connected ── */
    at(4800, () => {
      document.getElementById('rs1')?.classList.replace('active','done')||document.getElementById('rs1')?.classList.add('done');
      document.getElementById('rs2')?.classList.add('active');
      const mg = document.getElementById('machineG');
      if(mg) { mg.style.transition='opacity .6s'; mg.style.opacity='1'; }
    });

    /* ── 7. wires appear ── */
    at(5200, () => {
      const wg = document.getElementById('wiresG');
      if(wg) { wg.style.transition='opacity .7s'; wg.style.opacity='1'; }
    });

    /* ── 8. step 2 done, step 3 active — recording starts ── */
    at(6000, () => {
      document.getElementById('rs2')?.classList.replace('active','done')||document.getElementById('rs2')?.classList.add('done');
      document.getElementById('rs3')?.classList.add('active');
      document.getElementById('rst3').style.display = 'flex';
      /* red LED blinks */
      const led = document.getElementById('recLED');
      if(led) {
        led.setAttribute('fill','#ef4444');
        led.style.animation = 'ledPulse 1s ease-in-out infinite';
      }
      /* signal pulse travels wire */
      const pulse = document.getElementById('sigPulse');
      const anim  = document.getElementById('pulseAnim');
      if(pulse && anim) {
        pulse.setAttribute('opacity','1');
        anim.beginElement?.();
      }
      /* animate mini EEG on machine screen */
      animateMachineEEG();
    });

    /* ── 9. live EEG canvas starts streaming ── */
    at(6400, () => {
      const lbl = document.getElementById('eegCvsLbl');
      if(lbl) lbl.textContent = 'Live EEG — 5 channels (raw, unfiltered) — 256 Hz';
      streamRawEEG();
    });

    /* ── 10. step 3 done, step 4 active — data exported ── */
    at(8500, () => {
      document.getElementById('rs3')?.classList.replace('active','done')||document.getElementById('rs3')?.classList.add('done');
      document.getElementById('rst3').innerHTML = '✓ Done';
      document.getElementById('rst3').className = 'rs-tag rs-done';
      document.getElementById('rs4')?.classList.add('active');
    });

    /* ── 11. step 4 done + file card ready + info tiles ── */
    at(9800, () => {
      document.getElementById('rs4')?.classList.replace('active','done')||document.getElementById('rs4')?.classList.add('done');
      const s = document.getElementById('fcStat');
      if(s){ s.innerHTML='<i class="fas fa-check-circle"></i> Loaded'; s.className='fc-status done'; }
    });
    [['itCh','19'],['itSr','256 Hz'],['itDur','600 s'],['itPts','153,600']].forEach(([id,val],i) => {
      at(9800 + i * 380, () => {
        const e = document.getElementById(id); if(e) e.textContent = val;
        document.querySelectorAll('.it')[i]?.classList.add('glow');
      });
    });
  }

  /* tiny scrolling EEG lines on the machine screen */
  function animateMachineEEG() {
    const lines = [
      {id:'mEEG1', baseY:178, amp:5,  freq:1.2, col:'#00d4ff'},
      {id:'mEEG2', baseY:185, amp:3.5,freq:1.8, col:'#7c3aed'},
      {id:'mEEG3', baseY:192, amp:2.5,freq:2.5, col:'#10b981'},
    ];
    let phase = 0, running = true;
    function frame() {
      if(!running) return;
      phase += 0.06;
      lines.forEach(l => {
        const el = document.getElementById(l.id); if(!el) return;
        let pts = '';
        for(let x = 308; x <= 382; x += 2) {
          const t = (x - 308) / 74;
          const y = l.baseY + Math.sin(t * Math.PI * 4 * l.freq + phase) * l.amp
                    + Math.sin(t * Math.PI * 8 * l.freq + phase * .7) * l.amp * .4
                    + (Math.random() - .5) * 1.2;
          pts += `${x},${y.toFixed(1)} `;
        }
        el.setAttribute('points', pts.trim());
      });
      const r = requestAnimationFrame(frame); timers.push(r);
    }
    frame();
  }

  /* full-width live streaming EEG canvas */
  function streamRawEEG() {
    const cvs = document.getElementById('rawEEG'); if(!cvs) return;
    const ctx  = cvs.getContext('2d');
    cvs.width  = cvs.offsetWidth || 400;
    cvs.height = cvs.offsetHeight || 130;
    const W = cvs.width, H = cvs.height;
    const cols = ['#00d4ff','#7c3aed','#f59e0b','#10b981','#ef4444'];
    const labels = ['Fz','Cz','Pz','T3','T4'];
    const sp = H / 5;
    let offset = 0, running = true;

    function draw() {
      if(!running) return;
      offset += 1.5;
      ctx.fillStyle = '#0d1117'; ctx.fillRect(0,0,W,H);
      cols.forEach((col, c) => {
        ctx.strokeStyle = col; ctx.lineWidth = 0.9; ctx.globalAlpha = 0.82;
        ctx.beginPath();
        for(let x = 0; x < W; x++) {
          const t = (x + offset) / W * Math.PI * 14;
          const y = sp*(c+.5) + Math.sin(t*.8+c)*16 + Math.sin(t*1.6+c)*8
                  + Math.sin(t*5+c*.7)*4 + (Math.random()-.5)*10;
          x === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
        }
        ctx.stroke();
        ctx.globalAlpha = 1; ctx.fillStyle = col; ctx.font = '8px JetBrains Mono,monospace';
        ctx.fillText(labels[c], 3, sp*(c+.5)-2);
      });
      ctx.globalAlpha = 1;
      const r = requestAnimationFrame(draw); timers.push(r);
    }
    draw();
  }

  /* ══════════════════════════════════════════════════════
     S2 — PREPROCESSING
  ══════════════════════════════════════════════════════ */
  function s2() {
    /* 4 pipeline steps to match architecture diagram */
    const IDS=['ps1','ps2','ps3','ps4'];
    IDS.forEach(id=>document.getElementById(id)?.classList.remove('active','done'));

    IDS.forEach((id,i) => {
      at(300+i*2600, () => {
        IDS.forEach(j=>document.getElementById(j)?.classList.remove('active'));
        if(i>0) document.getElementById(IDS[i-1])?.classList.replace('active','done')||
                document.getElementById(IDS[i-1])?.classList.add('done');
        document.getElementById(id)?.classList.add('active');
      });
      at(300+i*2600+2000, () => {
        document.getElementById(id)?.classList.replace('active','done')||
        document.getElementById(id)?.classList.add('done');
      });
    });

    at(400,  () => drawFilter('beforeCvs', true));
    at(2800, () => drawFilter('afterCvs',  false));
    at(6000, () => buildEpochs());
  }

  function drawFilter(id, noisy) {
    const cvs = document.getElementById(id); if(!cvs) return;
    cvs.width = cvs.offsetWidth||440; cvs.height = 80;
    const ctx=cvs.getContext('2d'), W=cvs.width, H=80, mid=H/2;
    ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);
    ctx.strokeStyle=noisy?'#ef4444':'#10b981';
    ctx.lineWidth=1.2; ctx.globalAlpha=.9; ctx.beginPath();
    for(let x=0;x<W;x++){
      const t=x/W*Math.PI*10;
      let y=mid+Math.sin(t)*22+Math.sin(t*2.1)*10;
      if(noisy){ y+=Math.sin(t*50)*10+(Math.random()-.5)*18; }
      x===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.stroke(); ctx.globalAlpha=1;
  }

  function buildEpochs() {
    const tr=document.getElementById('epochTrack'); if(!tr||tr._b) return; tr._b=true;
    const rej=[3,8,12,21];
    for(let i=0;i<37;i++){
      const b=document.createElement('div');
      b.className='eb'+(rej.includes(i)?' rej':'');
      b.style.width='20px'; b.title=`Epoch ${i+1}${rej.includes(i)?' — REJECTED':''}`;
      b.textContent=rej.includes(i)?'✕':i+1;
      tr.appendChild(b);
      at(50+i*55, ()=>b.classList.add('show'));
    }
  }

  /* ══════════════════════════════════════════════════════
     S3 — FREQUENCY
  ══════════════════════════════════════════════════════ */
  const BANDS_DATA={
    delta:{pct:18,val:'12.4 µV²',id:'delta'},
    theta:{pct:84,val:'8.5 µV²', id:'theta'},
    alpha:{pct:32,val:'4.1 µV²', id:'alpha'},
    beta: {pct:20,val:'1.8 µV²', id:'beta'},
    gamma:{pct:27,val:'2.3 µV²', id:'gamma'},
  };

  function s3() {
    Object.entries(BANDS_DATA).forEach(([key,d],i) => {
      at(300+i*1200, () => {
        document.getElementById(`b-${key}`)?.classList.add('show');
        const f=document.getElementById(`bdf-${key}`);
        const v=document.getElementById(`bdv-${key}`);
        if(f) f.style.width=d.pct+'%';
        if(v) v.textContent=d.val;
      });
    });
    at(1500, () => renderPSD());
    at(6500, () => {
      counter('tbrNum',0,4.72,1800,2,'','',() => {
        const e=document.getElementById('tbrNote');
        if(e) e.innerHTML='⚠ <strong style="color:#ef4444">TBR = 4.72 exceeds the 3.0 threshold</strong> — frontal theta hyperactivation is the primary ADHD biomarker (Loo & Makeig, 2012).';
        document.getElementById('tbrNum').style.color='var(--r)';
      });
    });
  }

  function renderPSD() {
    killChart('psdChart');
    const ctx=document.getElementById('psdChart'); if(!ctx) return;
    const freq=Array.from({length:80},(_,i)=>(i+1)*.5);
    const adhd=freq.map(f=>{
      if(f<4)  return 3+Math.random()*2;
      if(f<8)  return 4.5+Math.random()*1.5;
      if(f<13) return 2.1+Math.random();
      if(f<30) return 1.2+Math.random()*.5;
      return .4+Math.random()*.3;
    });
    const norm=freq.map(f=>{
      if(f<4)  return 3+Math.random()*2;
      if(f<8)  return 1.2+Math.random();
      if(f<13) return 3.5+Math.random()*1.5;
      if(f<30) return 2.5+Math.random();
      return .5+Math.random()*.3;
    });
    regChart('psdChart', new Chart(ctx,{
      type:'line',
      data:{
        labels:freq.map(f=>f%5===0?f+'Hz':''),
        datasets:[
          {label:'ADHD — Fz',data:adhd,borderColor:'#ef4444',borderWidth:1.8,fill:true,backgroundColor:'rgba(239,68,68,.07)',tension:.4,pointRadius:0},
          {label:'Normal — Fz',data:norm,borderColor:'#3b82f6',borderWidth:1.5,fill:false,tension:.4,pointRadius:0,borderDash:[4,3]},
        ],
      },
      options:{responsive:true,animation:{duration:1000},
        plugins:{legend:{labels:{color:'#8b949e',font:{size:10}}}},
        scales:{
          x:{ticks:{color:'#484f58',font:{size:9}},grid:{color:'rgba(255,255,255,.04)'}},
          y:{ticks:{color:'#484f58',font:{size:9}},grid:{color:'rgba(255,255,255,.04)'},
            title:{display:true,text:'Power (µV²/Hz)',color:'#8b949e',font:{size:9}}},
        },
      },
    }));
  }

  /* ══════════════════════════════════════════════════════
     S4 — FEATURES
  ══════════════════════════════════════════════════════ */
  const FG=[
    {id:1,chips:['Delta mean','Theta mean','Alpha mean','Beta mean','Gamma mean','Delta std','Theta std','Alpha std','Beta std','Gamma std']},
    {id:2,chips:['TBR mean','TBR std']},
    {id:3,chips:['Fp1-Fp2 α','F3-F4 α','F7-F8 α','Fp1-Fp2 θ','F3-F4 θ','F7-F8 θ','Fp1-Fp2 β','F3-F4 β','F7-F8 β']},
    {id:4,chips:['Activity Fz','Mobility Fz','Complexity Fz','Activity Cz','Mobility Cz','Complexity Cz','SampEn Fz','SampEn Cz','SampEn Pz','SampEn T3','SampEn T4','SampEn O1']},
    {id:5,chips:['Fp1-Fp2 θ coh','F3-F4 θ coh','C3-C4 θ coh','Fp1-Fp2 β coh','F3-F4 β coh','C3-C4 β coh','T3-T4 θ coh','T3-T4 β coh','P3-P4 θ coh']},
  ];

  function s4() {
    let tot=0;
    FG.forEach((g,gi) => {
      const delay=200+gi*1800;
      at(delay, () => {
        const fg=document.getElementById(`fg${g.id}`); if(!fg) return;
        fg.classList.add('show');
        const chipsEl=document.getElementById(`fgc${g.id}`);
        if(chipsEl && !chipsEl._b){
          chipsEl._b=true;
          g.chips.forEach((name,ci) => {
            const c=document.createElement('span'); c.className='chip'; c.textContent=name;
            chipsEl.appendChild(c);
            at(delay+100+ci*70, ()=>c.classList.add('show'));
          });
        }
      });
      at(delay+g.chips.length*70+200, () => {
        tot+=g.chips.length;
        const cur_tot=tot;
        counter('ftNum', cur_tot-g.chips.length, cur_tot, 500, 0);
      });
    });
    at(500, () => buildHeatmap());
    at(5000, () => renderFeatChart());
  }

  function buildHeatmap() {
    const el=document.getElementById('fvHeatmap'); if(!el||el._b) return; el._b=true;
    const vals=Array.from({length:42},()=>Math.random());
    [0,1,10,11].forEach(i=>vals[i]=.75+Math.random()*.25);
    [3,13].forEach(i=>vals[i]=.1+Math.random()*.15);
    vals.forEach((v,i) => {
      const c=document.createElement('div'); c.className='fv-cell';
      const r=Math.round(59+(239-59)*v), g2=Math.round(130+(68-130)*v), b=Math.round(246+(68-246)*v);
      c.style.background=`rgb(${r},${g2},${b})`; c.title=`Feature ${i+1}: ${v.toFixed(3)}`;
      el.appendChild(c);
      at(50+i*25, ()=>c.classList.add('show'));
    });
  }

  function renderFeatChart() {
    killChart('featChart');
    const ctx=document.getElementById('featChart'); if(!ctx) return;
    const names=['Frontal θ Fz','TBR mean','θ/β asym F3-F4','SampEn Fz','Frontal α','β Fz','T3-T4 θ coh','Hjorth Mob'];
    const vals=[.42,.38,.29,.24,.21,.18,.15,.12];
    regChart('featChart', new Chart(ctx,{
      type:'bar',
      data:{labels:names,datasets:[{label:'Magnitude',data:vals,backgroundColor:vals.map((_,i)=>i<4?'rgba(239,68,68,.7)':'rgba(59,130,246,.7)'),borderRadius:4}]},
      options:{indexAxis:'y',responsive:true,animation:{duration:800},
        plugins:{legend:{display:false}},
        scales:{
          x:{ticks:{color:'#484f58',font:{size:9}},grid:{color:'rgba(255,255,255,.04)'}},
          y:{ticks:{color:'#8b949e',font:{size:9}},grid:{display:false}},
        },
      },
    }));
  }

  /* ══════════════════════════════════════════════════════
     S5 — CNN + TCN DEEP LEARNING ENGINE
  ══════════════════════════════════════════════════════ */
  function s5() {
    /* sequential layer reveal: Input → Feature Extraction → CNN → TCN → Dense → Output */
    const seq=[
      {id:'mfInp',   delay:0},
      {id:'mfScl',   delay:1400},
      {id:'mfCnn',   delay:2800, info:{el:'mfPCnn', txt:'Processing spatial features…'}},
      {id:'mfTcn',   delay:4400, info:{el:'mfPTcn', txt:'Learning temporal patterns…'}},
      {id:'mfDense', delay:6000, prob:{el:'mfPDense',v:0.873}},
      {id:'mfOut',   delay:7800},
    ];
    seq.forEach(s => {
      at(s.delay, () => {
        document.getElementById(s.id)?.classList.add('show');
        if(s.prob) counter(s.prob.el, 0, s.prob.v, 900, 3, 'P = ');
        if(s.info) {
          const el=document.getElementById(s.info.el);
          if(el) el.textContent=s.info.txt;
        }
      });
    });
    at(7800, () => {
      const l=document.getElementById('mfLbl'), c=document.getElementById('mfConf');
      const o=document.getElementById('mfOut');
      if(l) l.textContent='ADHD'; if(c) c.textContent='87.3% confidence';
      if(o){o.style.borderColor='rgba(239,68,68,.45)';o.querySelector('i').style.color='#ef4444';}
    });

    at(2800, ()=>drawCNNFeatureMaps());
    at(4500, ()=>renderROC());

    /* CV results */
    [['cvAcc','94.7 ± 2.1 %'],['cvAUC','0.971'],['cvSens','93.8 %'],['cvSpec','95.4 %']].forEach(([id,v],i)=>{
      at(8500+i*350, () => { const r=document.getElementById(id); if(r) r.querySelector('span:last-child').textContent=v; });
    });
  }

  function drawCNNFeatureMaps() {
    const cvs=document.getElementById('treeCvs'); if(!cvs) return;
    const ctx=cvs.getContext('2d');
    const W=cvs.width=cvs.offsetWidth||440, H=200;
    ctx.clearRect(0,0,W,H);

    /* draw 3 simulated CNN feature maps side by side */
    const channels=['Fz','Cz','Pz'], mapW=(W-60)/3, mapH=H-40;
    const mapColors=[
      {hi:'#ef4444',lo:'#1e1b4b'},  /* Fz — frontal: high activation (ADHD theta) */
      {hi:'#f59e0b',lo:'#1a2e1a'},  /* Cz — central: medium */
      {hi:'#3b82f6',lo:'#0f172a'},  /* Pz — parietal: low */
    ];

    channels.forEach((ch,ci) => {
      const x0=30+ci*(mapW+15), y0=30;
      /* feature map heatmap */
      const rows=12, cols=16, cw=mapW/cols, ch2=mapH/rows;
      const peakRow=ci===0?3:ci===1?5:7; /* channel-specific peak row */
      for(let r=0;r<rows;r++) for(let c=0;c<cols;c++){
        const dist=Math.abs(r-peakRow)/rows;
        const v=Math.max(0, 1-dist*1.8-(Math.random()*.25));
        const mc=mapColors[ci];
        const lo=[parseInt(mc.lo.slice(1,3),16),parseInt(mc.lo.slice(3,5),16),parseInt(mc.lo.slice(5,7),16)];
        const hi=[parseInt(mc.hi.slice(1,3),16),parseInt(mc.hi.slice(3,5),16),parseInt(mc.hi.slice(5,7),16)];
        const rgb=lo.map((l,i)=>Math.round(l+(hi[i]-l)*v));
        ctx.fillStyle=`rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
        ctx.fillRect(x0+c*cw, y0+r*ch2, cw-.5, ch2-.5);
      }
      /* border */
      ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
      ctx.strokeRect(x0, y0, mapW, mapH);
      /* label */
      ctx.fillStyle='#8b949e'; ctx.font='10px Inter,sans-serif'; ctx.textAlign='center';
      ctx.fillText(`${ch} channel`, x0+mapW/2, y0+mapH+14);
      ctx.fillStyle=ci===0?'#ef4444':ci===1?'#f59e0b':'#3b82f6';
      ctx.font='bold 9px Inter,sans-serif';
      ctx.fillText(ci===0?'HIGH':ci===1?'MED':'LOW', x0+mapW/2, y0+mapH+26);
    });
    /* title */
    ctx.fillStyle='#484f58'; ctx.font='8px Inter,sans-serif'; ctx.textAlign='left';
    ctx.fillText('Conv1D feature maps — activation intensity × time steps', 8, 16);
    ctx.textAlign='left';
  }

  function renderROC() {
    killChart('rocChart');
    const ctx=document.getElementById('rocChart'); if(!ctx) return;
    const fpr=[0,.01,.02,.04,.07,.1,.15,.2,.3,.4,.55,.7,.85,1];
    const tpr=[0,.55,.7,.82,.88,.92,.95,.96,.97,.975,.98,.99,.995,1];
    regChart('rocChart', new Chart(ctx,{
      type:'line',
      data:{labels:fpr,datasets:[
        {label:'ROC (AUC=0.971)',data:tpr,borderColor:'#00d4ff',borderWidth:2,fill:true,backgroundColor:'rgba(0,212,255,.07)',tension:.35,pointRadius:0},
        {label:'Chance',data:fpr,borderColor:'#484f58',borderWidth:1,borderDash:[4,3],pointRadius:0,fill:false},
      ]},
      options:{responsive:true,animation:{duration:1000},
        plugins:{legend:{labels:{color:'#8b949e',font:{size:9}}}},
        scales:{
          x:{type:'linear',min:0,max:1,title:{display:true,text:'FPR',color:'#8b949e',font:{size:9}},ticks:{color:'#484f58',font:{size:8}},grid:{color:'rgba(255,255,255,.04)'}},
          y:{min:0,max:1,title:{display:true,text:'TPR',color:'#8b949e',font:{size:9}},ticks:{color:'#484f58',font:{size:8}},grid:{color:'rgba(255,255,255,.04)'}},
        },
      },
    }));
  }

  /* ══════════════════════════════════════════════════════
     S6 — PREDICTION
  ══════════════════════════════════════════════════════ */
  function s6() {
    animateGauge(.873, () => {
      at(200, () => {
        const v=document.getElementById('verdict');
        const l=document.getElementById('vdLbl'), s=document.getElementById('vdSub'), ic=document.getElementById('vdIcon');
        if(v) v.classList.add('adhd');
        if(ic) ic.innerHTML='<i class="fas fa-exclamation-circle" style="color:#ef4444"></i>';
        if(l) l.textContent='ADHD Predicted';
        if(s) s.textContent='87.3% probability — high confidence';
      });
      at(400, () => {
        document.getElementById('cbrN').style.width='12.7%';
        document.getElementById('cbrA').style.width='87.3%';
        document.getElementById('cvrN').textContent='12.7%';
        document.getElementById('cvrA').textContent='87.3%';
      });
    });
    ['bm1','bm2','bm3','bm4'].forEach((id,i)=>at(3500+i*300, ()=>document.getElementById(id)?.classList.add('show')));
  }

  function animateGauge(target, cb) {
    const cvs=document.getElementById('gaugeCvs'); if(!cvs) return;
    const ctx=cvs.getContext('2d');
    const W=cvs.width=300, H=cvs.height=280;
    const cx=W/2, cy=H*.6, R=Math.min(W,H)*.42;

    function draw(v) {
      ctx.clearRect(0,0,W,H);
      /* track arcs */
      [[0,.33,'#10b981'],[.33,.66,'#f59e0b'],[.66,1,'#ef4444']].forEach(([s,e,col])=>{
        ctx.beginPath(); ctx.arc(cx,cy,R,Math.PI+s*Math.PI,Math.PI+e*Math.PI);
        ctx.strokeStyle=col+'44'; ctx.lineWidth=18; ctx.lineCap='butt'; ctx.stroke();
      });
      /* fill */
      const grd=ctx.createLinearGradient(cx-R,cy,cx+R,cy);
      grd.addColorStop(0,'#10b981'); grd.addColorStop(.5,'#f59e0b'); grd.addColorStop(1,'#ef4444');
      ctx.beginPath(); ctx.arc(cx,cy,R,Math.PI,Math.PI+v*Math.PI);
      ctx.strokeStyle=grd; ctx.lineWidth=18; ctx.lineCap='round'; ctx.stroke();
      /* needle */
      const a=Math.PI+v*Math.PI;
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(cx+(R-8)*Math.cos(a),cy+(R-8)*Math.sin(a));
      ctx.strokeStyle='#fff'; ctx.lineWidth=2.5; ctx.lineCap='round'; ctx.stroke();
      ctx.beginPath(); ctx.arc(cx,cy,6,0,Math.PI*2); ctx.fillStyle='#fff'; ctx.fill();
      /* pct */
      const el=document.getElementById('gcPct'); if(el) el.textContent=`${Math.round(v*100)}%`;
    }

    const dur=2200, s0=performance.now();
    function frame(now) {
      const t=Math.min((now-s0)/dur,1), e=t<.5?2*t*t:-1+(4-2*t)*t;
      draw(e*target);
      if(t<1){ const r=requestAnimationFrame(frame); timers.push(r); }
      else { draw(target); if(cb) cb(); }
    }
    requestAnimationFrame(frame);
  }

  /* ══════════════════════════════════════════════════════
     S7 — ATTENTION HEATMAP
  ══════════════════════════════════════════════════════ */

  /* TCN attention weights per channel — higher = more diagnostically relevant */
  const ATTN_CH=[
    {name:'Fz  (Frontal midline)',  v:.91, col:'#ef4444'},
    {name:'F3  (Left frontal)',     v:.87, col:'#ef4444'},
    {name:'F4  (Right frontal)',    v:.84, col:'#ef4444'},
    {name:'Fp1 (Left prefrontal)', v:.79, col:'#f59e0b'},
    {name:'Fp2 (Right prefrontal)',v:.76, col:'#f59e0b'},
    {name:'Cz  (Central midline)', v:.62, col:'#f59e0b'},
    {name:'C3  (Left central)',     v:.54, col:'#f59e0b'},
    {name:'C4  (Right central)',    v:.51, col:'#f59e0b'},
    {name:'T3  (Left temporal)',    v:.38, col:'#3b82f6'},
    {name:'Pz  (Parietal midline)', v:.31, col:'#3b82f6'},
    {name:'P3  (Left parietal)',    v:.27, col:'#3b82f6'},
    {name:'O1  (Left occipital)',   v:.18, col:'#3b82f6'},
  ];

  function s7() {
    /* channel attention list */
    const list=document.getElementById('shapList');
    if(list && !list._b){
      list._b=true;
      ATTN_CH.forEach((d,i)=>{
        const row=document.createElement('div'); row.className='sl-row';
        const dir=d.v>=.7?'pos':d.v>=.4?'mid':'neg';
        row.innerHTML=`<div class="sl-name">${d.name}</div>
          <div class="sl-bar"><div class="sl-fill ${dir}" id="sf${i}" style="width:0%;background:${d.col}"></div></div>
          <div class="sl-val" style="color:${d.col}">${d.v.toFixed(2)}</div>`;
        list.appendChild(row);
        at(300+i*120, ()=>{
          row.classList.add('show');
          const f=document.getElementById(`sf${i}`);
          if(f) f.style.width=(d.v*100)+'%';
        });
      });
    }
    at(600,  ()=>renderAttnHeatmap());
    at(4500, ()=>renderAttnBar());
  }

  /* channel × time heatmap — shows where TCN focuses */
  function renderAttnHeatmap() {
    killChart('shapChart');
    const cvs=document.getElementById('shapChart'); if(!cvs) return;
    const ctx=cvs.getContext('2d');
    const channels=['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','Pz','O1','O2'];
    const attnPeak={Fz:.92,F3:.87,F4:.84,Fp1:.78,Fp2:.75,Cz:.62,C3:.54,C4:.51,F7:.6,F8:.57,T3:.38,T4:.35,Pz:.31,O1:.18,O2:.2};
    const nT=60; /* time steps */
    const W=cvs.offsetWidth||440, H=260;
    cvs.width=W; cvs.height=H;
    const cellW=W/nT, cellH=H/channels.length;

    channels.forEach((ch,ci)=>{
      const peak=attnPeak[ch]||.3;
      for(let t=0;t<nT;t++){
        /* gaussian bump centred at a channel-specific time */
        const tPeak=ci<7?20+ci*2:35+ci*2; /* frontal channels peak earlier */
        const tDist=Math.abs(t-tPeak)/nT;
        const v=Math.max(0, peak*(1-tDist*4)+Math.random()*.08);
        const r=Math.round(59+(239-59)*v),
              g=Math.round(130+(68-130)*v),
              b=Math.round(246+(68-246)*v);
        ctx.fillStyle=`rgb(${r},${g},${b})`;
        ctx.fillRect(t*cellW, ci*cellH, cellW-.5, cellH-.5);
      }
      /* channel label */
      ctx.fillStyle=peak>.6?'#ef4444':peak>.4?'#f59e0b':'#8b949e';
      ctx.font='bold 8px JetBrains Mono,monospace'; ctx.textAlign='left';
      ctx.fillText(ch, 3, ci*cellH+cellH*.65);
    });
    /* axes */
    ctx.fillStyle='#484f58'; ctx.font='8px Inter,sans-serif'; ctx.textAlign='center';
    ctx.fillText('Time steps →', W/2, H-4);
    /* colour legend */
    const grd=ctx.createLinearGradient(W-70,0,W,0);
    grd.addColorStop(0,'rgb(59,130,246)'); grd.addColorStop(.5,'rgb(16,185,129)'); grd.addColorStop(1,'rgb(239,68,68)');
    ctx.fillStyle=grd; ctx.fillRect(W-68,4,60,7);
    ctx.fillStyle='#484f58'; ctx.font='7px Inter,sans-serif';
    ctx.fillText('Low',W-72,11); ctx.fillText('High',W-2,11);
  }

  function renderAttnBar() {
    killChart('attnBarChart');
    const ctx=document.getElementById('attnBarChart'); if(!ctx) return;
    const top=ATTN_CH.slice(0,8);
    regChart('attnBarChart', new Chart(ctx,{
      type:'bar',
      data:{labels:top.map(d=>d.name.split(' ')[0]),datasets:[{
        label:'Mean Attention',data:top.map(d=>d.v),
        backgroundColor:top.map(d=>d.col+'bb'),borderRadius:4,
      }]},
      options:{responsive:true,animation:{duration:800},
        plugins:{legend:{display:false}},
        scales:{
          x:{ticks:{color:'#8b949e',font:{size:9}},grid:{color:'rgba(255,255,255,.04)'}},
          y:{min:0,max:1,ticks:{color:'#484f58',font:{size:9}},grid:{color:'rgba(255,255,255,.04)'},
            title:{display:true,text:'Attention weight',color:'#8b949e',font:{size:9}}},
        },
      },
    }));
  }

  /* ══════════════════════════════════════════════════════
     S8 — BRAIN MAP
  ══════════════════════════════════════════════════════ */
  const EP={
    Fp1:[-.18,-.92],Fp2:[.18,-.92],F7:[-.55,-.6],F3:[-.3,-.52],Fz:[0,-.54],F4:[.3,-.52],F8:[.55,-.6],
    T3:[-.82,0],C3:[-.45,0],Cz:[0,0],C4:[.45,0],T4:[.82,0],
    T5:[-.55,.6],P3:[-.3,.52],Pz:[0,.54],P4:[.3,.52],T6:[.55,.6],
    O1:[-.18,.92],O2:[.18,.92],
  };
  const BVALS={
    theta:{Fp1:.9,Fp2:.85,F3:.82,F4:.78,Fz:.95,F7:.71,F8:.68,C3:.52,C4:.5,Cz:.6,T3:.42,T4:.4,T5:.38,T6:.35,P3:.28,P4:.25,Pz:.22,O1:.18,O2:.2},
    alpha:{Fp1:.25,Fp2:.28,F3:.42,F4:.4,Fz:.35,F7:.45,F8:.43,C3:.6,C4:.58,Cz:.65,T3:.55,T4:.52,T5:.7,T6:.72,P3:.82,P4:.8,Pz:.88,O1:.9,O2:.85},
    beta: {Fp1:.28,Fp2:.3,F3:.35,F4:.32,Fz:.3,F7:.38,F8:.36,C3:.55,C4:.52,Cz:.5,T3:.6,T4:.62,T5:.48,T6:.45,P3:.42,P4:.4,Pz:.38,O1:.3,O2:.32},
  };

  function s8() {
    renderTopo('theta');
    document.querySelectorAll('.bsw').forEach(b=>b.classList.toggle('active',b.dataset.band==='theta'));
    /* auto-cycle through bands */
    at(3000, ()=>switchTopoMap('alpha'));
    at(6000, ()=>switchTopoMap('beta'));
    at(9000 - 500, ()=>switchTopoMap('theta'));
  }

  function renderTopo(band) {
    const svg=document.getElementById('topoSVG'); if(!svg) return;
    const vals=BVALS[band]||BVALS.theta;
    const W=400,H=400,cx=200,cy=210,R=168;
    svg.innerHTML='';
    /* bg */
    const sc=el('circle',{cx,cy,r:R,fill:'#161b22',stroke:'#30363d','stroke-width':'2'});
    svg.appendChild(sc);
    /* IDW grid */
    const GRID=50, step=(R*2)/GRID;
    const eps=Object.entries(EP);
    for(let gy=0;gy<=GRID;gy++) for(let gx=0;gx<=GRID;gx++){
      const px=(cx-R)+gx*step, py=(cy-R)+gy*step;
      if((px-cx)**2+(py-cy)**2>R*R) continue;
      let ws=0,vs=0;
      eps.forEach(([ch,pos])=>{
        const ex=cx+pos[0]*R, ey=cy+pos[1]*R;
        const d=Math.hypot(px-ex,py-ey)+1e-6, w=1/(d*d);
        ws+=w; vs+=w*(vals[ch]||.5);
      });
      const rect=el('rect',{x:px,y:py,width:step+.5,height:step+.5,fill:v2col(vs/ws),opacity:'0.88'});
      svg.appendChild(rect);
    }
    /* re-clip */
    const bc=el('circle',{cx,cy,r:R,fill:'none',stroke:'#484f58','stroke-width':'2'});
    svg.appendChild(bc);
    /* nose */
    const nose=el('polygon',{points:`${cx-9},${cy-R+2} ${cx},${cy-R-16} ${cx+9},${cy-R+2}`,fill:'#161b22',stroke:'#484f58','stroke-width':'1.5'});
    svg.appendChild(nose);
    /* electrodes */
    eps.forEach(([ch,pos])=>{
      const ex=cx+pos[0]*R, ey=cy+pos[1]*R, v=vals[ch]||.5;
      svg.appendChild(el('circle',{cx:ex,cy:ey,r:'7',fill:'#0d1117',stroke:v2col(v),'stroke-width':'2'}));
      const t=el('text',{x:ex,y:ey+3,'text-anchor':'middle','font-size':'6.5','font-family':'Inter,sans-serif',fill:'#e6edf3'});
      t.textContent=ch; svg.appendChild(t);
    });
  }

  function el(tag,attrs){
    const e=document.createElementNS('http://www.w3.org/2000/svg',tag);
    Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));
    return e;
  }
  function lerp(a,b,t){return [Math.round(a[0]+(b[0]-a[0])*t),Math.round(a[1]+(b[1]-a[1])*t),Math.round(a[2]+(b[2]-a[2])*t)];}
  function v2col(v){
    const stops=[[59,130,246],[0,212,255],[16,185,129],[245,158,11],[239,68,68]];
    const s=Math.min(v,1)*(stops.length-1), i=Math.floor(s), f=s-i;
    const [r,g,b]=lerp(stops[i],stops[Math.min(i+1,stops.length-1)],f);
    return `rgb(${r},${g},${b})`;
  }

  /* ══════════════════════════════════════════════════════
     S9 — SUMMARY
  ══════════════════════════════════════════════════════ */
  function s9() {
    ['sc1','sc2','sc3','sc4','sc5','sc6'].forEach((id,i)=>at(150+i*200,()=>document.getElementById(id)?.classList.add('show')));
  }

  /* ══════════════════════════════════════════════════════
     TIMELINE & HEADER BUILD
  ══════════════════════════════════════════════════════ */
  function buildTimeline() {
    const dotsEl  = document.getElementById('vtDots');
    const labsEl  = document.getElementById('vtLabels');
    if (!dotsEl || !labsEl) return;

    STEPS.slice(1).forEach((s, i) => {
      const pct = (i / (STEPS.length - 2)) * 100;

      /* dot */
      const dot = document.createElement('div');
      dot.className = 'vt-dot';
      dot.style.left = pct + '%';
      dot.title = s.label;
      dot.addEventListener('click', () => goTo(s.n));
      dotsEl.appendChild(dot);

      /* label */
      const lbl = document.createElement('div');
      lbl.className = 'vt-lbl';
      lbl.innerHTML = `<span class="vl-n">${s.n}</span>${s.label}`;
      lbl.addEventListener('click', () => goTo(s.n));
      labsEl.appendChild(lbl);
    });
  }

  /* ══════════════════════════════════════════════════════
     KEYBOARD
  ══════════════════════════════════════════════════════ */
  function bindKeys() {
    document.addEventListener('keydown', e => {
      if (e.key === 'ArrowRight' || e.key === 'PageDown') { e.preventDefault(); goTo(cur + 1); }
      if (e.key === 'ArrowLeft'  || e.key === 'PageUp')   { e.preventDefault(); goTo(cur - 1); }
      if (e.key === ' ')                                   { e.preventDefault(); togglePlay(); }
      if (e.key === 'f' || e.key === 'F') {
        if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
        else document.exitFullscreen?.();
      }
    });
    document.querySelectorAll('.bsw').forEach(b =>
      b.addEventListener('click', () => switchTopoMap(b.dataset.band))
    );
  }

  /* ══════════════════════════════════════════════════════
     INIT
  ══════════════════════════════════════════════════════ */
  function init() {
    buildTimeline();
    updateHeader();
    updateTimeline();
    runScene(0);

    document.getElementById('startBtn')?.addEventListener('click', startAutoPlay);
    document.getElementById('playBtn' )?.addEventListener('click', togglePlay);
    document.getElementById('prevBtn' )?.addEventListener('click', () => goTo(cur - 1));
    document.getElementById('nextBtn' )?.addEventListener('click', () => goTo(cur + 1));
    document.getElementById('fsBtn'   )?.addEventListener('click', () => {
      if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
      else document.exitFullscreen?.();
    });
    bindKeys();
  }

  document.addEventListener('DOMContentLoaded', init);

  return { goTo, togglePlay, startAutoPlay, switchTopoMap };

})();
