/* ============================================================
   Candle-vLLM Documentation — Main JavaScript
   ============================================================ */

const THEMES = ['light', 'dark', 'enflame'];

function getPreferredTheme() {
  const stored = localStorage.getItem('theme');
  if (stored && THEMES.includes(stored)) return stored;
  return 'light';
}

function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('theme', theme);
  const sel = document.querySelector('.theme-select');
  if (sel) sel.value = theme;
}

setTheme(getPreferredTheme());

/* ---- Language ---- */
function getLang() {
  return localStorage.getItem('lang') || 'zh';
}
function setLang(lang) {
  localStorage.setItem('lang', lang);
  document.querySelectorAll('[data-zh][data-en]').forEach(el => {
    el.textContent = el.getAttribute('data-' + lang);
  });
  const btn = document.querySelector('.lang-toggle');
  if (btn) btn.textContent = lang === 'zh' ? 'EN' : '中文';
}

/* ---- DOMContentLoaded ---- */
document.addEventListener('DOMContentLoaded', () => {
  const sel = document.querySelector('.theme-select');
  if (sel) {
    sel.value = getPreferredTheme();
    sel.addEventListener('change', () => setTheme(sel.value));
  }

  const langBtn = document.querySelector('.lang-toggle');
  if (langBtn) {
    langBtn.addEventListener('click', () => {
      setLang(getLang() === 'zh' ? 'en' : 'zh');
    });
  }
  setLang(getLang());

  initScrollProgress();
  initHeaderScroll();
  initReveal();
  initCounters();
  initTabs();
  initCopyButtons();
  initSidebarTracking();
  initSearch();
  initParticles();
  initCardTracking();
  initTypingAnimation();
  initActiveNav();
  initMobileNav();
});

/* ---- Scroll Progress ---- */
function initScrollProgress() {
  const bar = document.querySelector('.scroll-progress');
  if (!bar) return;
  window.addEventListener('scroll', () => {
    const h = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.width = h > 0 ? (window.scrollY / h * 100) + '%' : '0%';
  }, { passive: true });
}

/* ---- Header Scroll ---- */
function initHeaderScroll() {
  const header = document.querySelector('.site-header');
  if (!header) return;
  window.addEventListener('scroll', () => {
    header.classList.toggle('scrolled', window.scrollY > 10);
  }, { passive: true });
}

/* ---- Reveal on Scroll ---- */
function initReveal() {
  const els = document.querySelectorAll('.reveal');
  if (!els.length) return;
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('visible'); observer.unobserve(e.target); } });
  }, { threshold: 0.08 });
  els.forEach(el => observer.observe(el));
}

/* ---- Counters ---- */
function initCounters() {
  const cards = document.querySelectorAll('[data-counter]');
  if (!cards.length) return;
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        animateCounter(e.target);
        observer.unobserve(e.target);
      }
    });
  }, { threshold: 0.3 });
  cards.forEach(c => observer.observe(c));
}

function animateCounter(el) {
  const target = parseInt(el.getAttribute('data-target'), 10);
  const suffix = el.getAttribute('data-suffix') || '';
  const duration = 2000;
  const start = performance.now();
  function update(now) {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(ease * target) + suffix;
    if (t < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

/* ---- Tabs ---- */
function initTabs() {
  document.querySelectorAll('.tab-btn[data-tab]').forEach(btn => {
    btn.addEventListener('click', () => {
      const group = btn.closest('.tabs') || btn.parentElement;
      const contentParent = group.parentElement;
      group.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      contentParent.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
      const target = contentParent.querySelector('#' + btn.getAttribute('data-tab'));
      if (target) target.classList.add('active');
    });
  });
}

/* ---- Copy Code ---- */
function initCopyButtons() {
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const text = btn.getAttribute('data-copy-text') || btn.closest('pre')?.querySelector('code')?.textContent || '';
      navigator.clipboard.writeText(text).then(() => {
        const orig = btn.textContent;
        btn.textContent = '✓';
        setTimeout(() => { btn.textContent = orig; }, 1500);
      });
    });
  });
}

/* ---- Sidebar Active Tracking ---- */
function initSidebarTracking() {
  const links = document.querySelectorAll('.doc-sidebar a[href^="#"]');
  if (!links.length) return;
  const sections = Array.from(links).map(a => document.getElementById(a.getAttribute('href').slice(1))).filter(Boolean);
  window.addEventListener('scroll', () => {
    let current = '';
    for (const sec of sections) {
      if (sec.getBoundingClientRect().top <= 120) current = sec.id;
    }
    links.forEach(a => {
      a.classList.toggle('active', a.getAttribute('href') === '#' + current);
    });
  }, { passive: true });
}

/* ---- Search ---- */
const SEARCH_INDEX = [
  { title: 'Home', url: 'index.html', desc: 'Candle-vLLM overview, features, and quick start', keywords: 'home overview features performance' },
  { title: 'Getting Started', url: 'getting-started.html', desc: 'Installation, prerequisites, and first run', keywords: 'install setup prerequisites cuda metal docker build' },
  { title: 'User Guide', url: 'user-guide.html', desc: 'Model loading, quantization, KV cache, multi-GPU', keywords: 'model gguf fp8 quantization kv cache multi-gpu turbo' },
  { title: 'Developer Guide', url: 'developer-guide.html', desc: 'Architecture, contributing, extending models', keywords: 'architecture scheduler pipeline contributing extend model' },
  { title: 'API Reference', url: 'api-reference.html', desc: 'CLI flags, REST endpoints, environment variables', keywords: 'cli api rest openai endpoint flag env' },
  { title: 'Downloads', url: 'downloads.html', desc: 'Pre-built packages, one-line install, DEB and tarball downloads', keywords: 'download install deb tarball binary package gpu sm cuda metal' },
];

function initSearch() {
  const overlay = document.querySelector('.search-overlay');
  if (!overlay) return;
  const input = overlay.querySelector('.search-input');
  const results = overlay.querySelector('.search-results');
  const triggers = document.querySelectorAll('[data-search-trigger]');

  function open() { overlay.classList.add('active'); setTimeout(() => input.focus(), 100); }
  function close() { overlay.classList.remove('active'); input.value = ''; results.innerHTML = '<div class="search-hint">Type to search documentation...</div>'; }

  triggers.forEach(t => t.addEventListener('click', open));
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') { e.preventDefault(); open(); }
    if (e.key === 'Escape') close();
  });

  input.addEventListener('input', () => {
    const q = input.value.toLowerCase().trim();
    if (!q) { results.innerHTML = '<div class="search-hint">Type to search documentation...</div>'; return; }
    const matches = SEARCH_INDEX.filter(item =>
      item.title.toLowerCase().includes(q) || item.desc.toLowerCase().includes(q) || item.keywords.includes(q)
    );
    if (!matches.length) { results.innerHTML = '<div class="search-hint">No results found</div>'; return; }
    results.innerHTML = matches.map(m =>
      `<a class="search-result-item" href="${m.url}"><div class="search-result-title">${m.title}</div><div class="search-result-desc">${m.desc}</div></a>`
    ).join('');
  });
}

/* ---- Particle System ---- */
function initParticles() {
  const canvas = document.getElementById('particles');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let w, h, particles = [], mouse = { x: -999, y: -999 };

  function resize() { w = canvas.width = canvas.offsetWidth; h = canvas.height = canvas.offsetHeight; }
  resize();
  window.addEventListener('resize', resize);
  canvas.addEventListener('mousemove', e => { mouse.x = e.offsetX; mouse.y = e.offsetY; });
  canvas.addEventListener('mouseleave', () => { mouse.x = -999; mouse.y = -999; });

  function getColors() {
    const theme = document.documentElement.getAttribute('data-theme');
    if (theme === 'dark') return { r: 129, g: 140, b: 248 };
    if (theme === 'enflame') return { r: 245, g: 158, b: 11 };
    return { r: 99, g: 102, b: 241 };
  }

  for (let i = 0; i < 50; i++) {
    particles.push({
      x: Math.random() * 2000, y: Math.random() * 1000,
      vx: (Math.random() - 0.5) * 0.5, vy: (Math.random() - 0.5) * 0.5,
      size: Math.random() * 2 + 1
    });
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);
    const c = getColors();
    particles.forEach(p => {
      const dx = mouse.x - p.x, dy = mouse.y - p.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 150) { p.vx += dx * 0.0003; p.vy += dy * 0.0003; }
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;
      p.x = Math.max(0, Math.min(w, p.x));
      p.y = Math.max(0, Math.min(h, p.y));
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${c.r},${c.g},${c.b},0.4)`;
      ctx.fill();
    });
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < 130) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(${c.r},${c.g},${c.b},${0.15 * (1 - d / 130)})`;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }
  draw();
}

/* ---- Card Mouse Tracking ---- */
function initCardTracking() {
  document.querySelectorAll('.feature-card, .community-card').forEach(card => {
    card.addEventListener('mousemove', e => {
      const rect = card.getBoundingClientRect();
      card.style.setProperty('--mouse-x', ((e.clientX - rect.left) / rect.width * 100) + '%');
      card.style.setProperty('--mouse-y', ((e.clientY - rect.top) / rect.height * 100) + '%');
    });
  });
}

/* ---- Hero Typing Animation ---- */
function initTypingAnimation() {
  const container = document.getElementById('hero-typing-code');
  if (!container) return;
  const codeEl = container.querySelector('code');
  if (!codeEl) return;

  const lines = [
    { text: '$ candle-vllm --m Qwen/Qwen3.6-27B-FP8 --ui-server', html: '<span class="token-comment">$</span> <span class="token-keyword">candle-vllm</span> <span class="token-flag">--m</span> <span class="token-string">Qwen/Qwen3.6-27B-FP8</span> <span class="token-flag">--ui-server</span>' },
    { text: '', html: '' },
    { text: '  Loading model weights... ████████████ 100%', html: '  Loading model weights... <span class="token-value">████████████</span> <span class="token-number">100%</span>' },
    { text: '  KV cache: 24576 blocks (turbo4, ~3.7x compression)', html: '  KV cache: <span class="token-number">24576</span> blocks (<span class="token-value">turbo4</span>, ~<span class="token-number">3.7x</span> compression)' },
    { text: '  CUDA graphs captured for batch 1..64', html: '  CUDA graphs captured for batch <span class="token-number">1</span>..<span class="token-number">64</span>' },
    { text: '', html: '' },
    { text: '  ✓ API server ready at http://0.0.0.0:2000', html: '  <span class="token-value">✓</span> API server ready at <span class="token-string">http://0.0.0.0:2000</span>' },
    { text: '  ✓ Web UI ready at http://0.0.0.0:1999', html: '  <span class="token-value">✓</span> Web UI ready at <span class="token-string">http://0.0.0.0:1999</span>' },
  ];

  const CHAR_DELAY = 14;
  const LINE_PAUSE = 60;

  let lineIdx = 0;
  let charIdx = 0;
  let currentLines = [];

  function renderCurrent() {
    const html = currentLines.map((l, i) => {
      if (i < lineIdx) return l.html;
      if (i === lineIdx) return l.text.substring(0, charIdx) + '<span class="typing-cursor"></span>';
      return '';
    }).join('\n');
    codeEl.innerHTML = html;
  }

  function typeNext() {
    if (lineIdx >= lines.length) {
      codeEl.innerHTML = currentLines.map(l => l.html).join('\n');
      return;
    }
    if (charIdx < lines[lineIdx].text.length) {
      charIdx++;
      renderCurrent();
      setTimeout(typeNext, CHAR_DELAY);
    } else {
      currentLines[lineIdx].typed = true;
      codeEl.innerHTML = currentLines.map((l, i) => i <= lineIdx ? l.html : '').join('\n');
      lineIdx++;
      charIdx = 0;
      if (lineIdx < lines.length) {
        currentLines.push({ ...lines[lineIdx] });
        setTimeout(typeNext, LINE_PAUSE);
      } else {
        codeEl.innerHTML = currentLines.map(l => l.html).join('\n');
      }
    }
  }

  currentLines.push({ ...lines[0] });
  setTimeout(typeNext, 500);
}

/* ---- Active Nav ---- */
function initActiveNav() {
  const path = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-links a').forEach(a => {
    const href = a.getAttribute('href');
    a.classList.toggle('active', href === path);
  });
}

/* ---- Mobile Nav ---- */
function initMobileNav() {
  const toggle = document.querySelector('.mobile-toggle');
  const nav = document.querySelector('.nav-links');
  if (!toggle || !nav) return;
  toggle.addEventListener('click', () => nav.classList.toggle('mobile-open'));
}
