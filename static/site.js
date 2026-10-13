// Menu drawer, ambient forest sound and the cursor ring.
(function () {
  const menuBtn = document.querySelector('.fp-menu-btn');
  const drawer = document.getElementById('fp-drawer');
  if (menuBtn && drawer) {
    menuBtn.addEventListener('click', () => {
      const open = drawer.hidden;
      drawer.hidden = !open;
      menuBtn.setAttribute('aria-expanded', String(open));
      menuBtn.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    });
  }

  // Ambient sound: wind (filtered noise) plus synthesised birdsong. Created only after a tap.
  function makeAudio() {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    const ctx = new AC();
    const master = ctx.createGain(); master.gain.value = 0; master.connect(ctx.destination);
    const len = ctx.sampleRate * 4, buf = ctx.createBuffer(1, len, ctx.sampleRate), d = buf.getChannelData(0);
    let b = 0;
    for (let i = 0; i < len; i++) { b = 0.985 * b + 0.015 * (Math.random() * 2 - 1) * 3.2; d[i] = b; }
    const src = ctx.createBufferSource(); src.buffer = buf; src.loop = true;
    const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 520;
    const wind = ctx.createGain(); wind.gain.value = 0.3;
    const lfo = ctx.createOscillator(); lfo.frequency.value = 0.07;
    const lfoAmt = ctx.createGain(); lfoAmt.gain.value = 0.15;
    lfo.connect(lfoAmt); lfoAmt.connect(wind.gain);
    src.connect(lp); lp.connect(wind); wind.connect(master); src.start(); lfo.start();
    const chirp = (when, f0, f1, dur, vol, pan) => {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.frequency.setValueAtTime(f0, when); o.frequency.exponentialRampToValueAtTime(f1, when + dur);
      g.gain.setValueAtTime(0.0001, when); g.gain.exponentialRampToValueAtTime(vol, when + dur * 0.25); g.gain.exponentialRampToValueAtTime(0.0001, when + dur);
      o.connect(g);
      if (ctx.createStereoPanner) { const p = ctx.createStereoPanner(); p.pan.value = pan; g.connect(p); p.connect(master); } else g.connect(master);
      o.start(when); o.stop(when + dur + 0.02);
    };
    let timer = null, on = false;
    const phrase = () => {
      if (!on) return;
      let t = ctx.currentTime + 0.05;
      const pan = Math.random() * 1.4 - 0.7, base = 2400 + Math.random() * 2200, vol = 0.025 + Math.random() * 0.035, kind = Math.random();
      if (kind < 0.4) { const n = 3 + Math.floor(Math.random() * 5); for (let i = 0; i < n; i++) { chirp(t, base * (1 + Math.random() * 0.15), base * (0.75 + Math.random() * 0.5), 0.07 + Math.random() * 0.06, vol, pan); t += 0.09 + Math.random() * 0.07; } }
      else if (kind < 0.7) { for (let i = 0; i < 10; i++) { chirp(t, base * 1.1, base * 0.9, 0.035, vol * 0.8, pan); t += 0.045; } }
      else { chirp(t, base * 0.8, base * 1.3, 0.22, vol, pan); chirp(t + 0.28, base * 1.3, base * 0.85, 0.26, vol, pan); }
      timer = setTimeout(phrase, 1400 + Math.random() * 4200);
    };
    return {
      start: () => { on = true; ctx.resume(); master.gain.cancelScheduledValues(ctx.currentTime); master.gain.setTargetAtTime(0.85, ctx.currentTime, 0.5); clearTimeout(timer); timer = setTimeout(phrase, 600); },
      stop: () => { on = false; clearTimeout(timer); master.gain.setTargetAtTime(0, ctx.currentTime, 0.25); }
    };
  }
  const soundBtn = document.querySelector('.fp-sound-btn');
  let audio = null;
  if (soundBtn) {
    soundBtn.addEventListener('click', () => {
      const on = !soundBtn.classList.contains('is-on');
      if (on && !audio) audio = makeAudio();
      if (!audio) return;
      if (on) audio.start(); else audio.stop();
      soundBtn.classList.toggle('is-on', on);
      soundBtn.setAttribute('aria-pressed', String(on));
      soundBtn.setAttribute('aria-label', on ? 'Turn ambient sound off' : 'Turn ambient sound on');
    });
  }

  // Cursor ring that grows over links, buttons and bird photos.
  const cursor = document.querySelector('.fp-cursor');
  if (cursor && window.matchMedia('(pointer: fine)').matches) {
    window.addEventListener('pointermove', (e) => {
      cursor.style.transform = 'translate(' + e.clientX + 'px,' + e.clientY + 'px)';
      cursor.classList.add('is-on');
      cursor.classList.toggle('is-hot', !!(e.target.closest && e.target.closest('a,button,input,.fp-photo')));
    }, { passive: true });
    document.addEventListener('pointerout', (e) => { if (!e.relatedTarget) cursor.classList.remove('is-on'); });
  }
})();
