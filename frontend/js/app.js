/* ==========================================================================
   SkinIntelli — single-page app: hash router + auth shell + all screens.
   Talks to the real backend via api.js. No fabricated AI results anywhere:
   conditions are shown as "AI model result", "computer-vision estimate" or
   "not currently supported", exactly as the backend reports them.
   ========================================================================== */
(function () {
  'use strict';
  const CFG = window.SI_CONFIG;
  const root = () => document.getElementById('root');

  // ---------------------------------------------------------------- helpers
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const h = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; };
  const qs = (sel, ctx = document) => ctx.querySelector(sel);
  const qsa = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];
  const cap = (s) => (s || '').charAt(0).toUpperCase() + (s || '').slice(1);

  function toast(msg, kind = '') {
    const host = document.getElementById('toast-host');
    const t = h(`<div class="toast ${kind}">${esc(msg)}</div>`);
    host.appendChild(t);
    setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity .3s'; setTimeout(() => t.remove(), 300); }, 5200);
  }

  function fmtDateTime(iso) {
    const d = new Date(iso);
    return { date: d.toLocaleDateString(undefined, { day: '2-digit', month: 'short', year: 'numeric' }),
             time: d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }) };
  }

  // ---- condition presentation -------------------------------------------
  const TIER_BADGE = {
    model: { cls: 'ai', text: 'AI model result' },
    cv: { cls: 'cv', text: 'Computer-vision analysis' },
    estimate: { cls: 'estimate', text: 'Estimated (computer vision)' },
    unavailable: { cls: 'na', text: 'Not currently supported' },
  };
  function statusPill(entry, tier) {
    if (!entry) return `<span class="pill na">No data</span>`;
    if (entry.status === 'not_available' || tier === 'unavailable')
      return `<span class="pill na">Not currently supported</span>`;
    if (entry.status === 'error') return `<span class="pill attention">Model error</span>`;
    if (tier === 'cv') return `<span class="pill cv">CV scan</span>`;
    if (entry.status === 'estimated') return `<span class="pill estimate">Estimated</span>`;
    // detected
    return `<span class="pill ai">Detected</span>`;
  }
  function severityPill(entry) {
    if (!entry || entry.score == null) return '';
    const s = entry.score;
    const [lbl, cls] = s >= 0.67 ? ['High', 'attention'] : s >= 0.34 ? ['Moderate', 'moderate'] : ['Low', 'good'];
    return `<span class="pill ${cls}">${lbl}</span>`;
  }

  function shiOf(results) {
    const s = results && results.skin_health_index;
    if (!s || s.value == null) return { value: null, band: (s && s.band) || 'not_enough_data' };
    return s;
  }
  function mainConcern(results) {
    const conds = (results && results.conditions) || {};
    let best = null;
    for (const { key, label, tier } of CFG.CONDITIONS) {
      if (tier === 'unavailable') continue;
      const c = conds[key];
      if (!c) continue;
      const sc = key === 'acne'
        ? ((c.detail && c.detail.severity_index) || 0) / 4
        : (typeof c.score === 'number' ? c.score : 0);
      if ((c.status === 'detected' || c.status === 'estimated') && sc > (best ? best.sc : 0.2))
        best = { label, sc, key };
    }
    return best ? best.label : 'No prominent concern';
  }

  // ================================================================ STATE
  const App = {
    state: { user: null, weather: null, prefs: loadPrefs(), lastAnalysis: null, lastRecommendation: null, historyCache: null },
    blobUrls: [],

    async boot() {
      window.addEventListener('hashchange', () => App.route());
      window.addEventListener('api:unauthorized', () => { App.state.user = null; toast('Session expired — please log in again.', 'warn'); location.hash = '#/'; App.route(); });
      window.addEventListener('resize', () => { if (App._redraw) App._redraw(); });

      if (API.isLoggedIn()) {
        try {
          App.state.user = await API.me();
          App.loadWeather();     // fire and forget
        } catch (e) { API.clearToken(); }
      }
      App.route();
    },

    _nav: 0,
    isCurrent(n) { return App._nav === n; },

    route() {
      const authed = !!App.state.user;
      let hash = location.hash.replace(/^#/, '') || '/';
      if (!authed) { renderAuth(hash.startsWith('/signup') ? 'signup' : 'login'); return; }
      if (hash === '/' || hash.startsWith('/login') || hash.startsWith('/signup')) { location.hash = '#/dashboard'; return; }

      const nav = ++App._nav;                       // supersedes any in-flight view render
      App._redraw = null;
      const [, seg, arg] = hash.match(/^\/([^/]+)(?:\/([^/]+))?/) || [];
      const views = {
        dashboard: viewDashboard, scan: viewScan, history: viewHistory,
        result: (n) => viewResult(arg, n), plan: viewPlan, products: viewProducts,
        profile: viewProfile, settings: viewSettings,
      };
      const fn = views[seg] || viewDashboard;
      renderShell(seg === 'result' ? 'history' : seg);
      Promise.resolve(fn(nav)).catch((e) => {
        if (!App.isCurrent(nav)) return;
        qs('#view').innerHTML = `<div class="card card-pad empty"><i class="fa-solid fa-triangle-exclamation"></i>${esc(e.message || 'Something went wrong.')}</div>`;
      });
    },

    async loadWeather() {
      const saved = loadEnv();
      try {
        if (saved && saved.env_mode === 'manual') { App.state.weather = { source: 'manual', ...saved }; App.paintWeatherChip(); return; }
        if (saved && saved.latitude != null && saved.longitude != null) {
          App.state.weather = await API.weatherByCoords(saved.latitude, saved.longitude);
        } else if (saved && saved.place_name) {
          App.state.weather = await API.weatherByPlace(saved.place_name);
        }
      } catch (_) { /* stays null / unknown */ }
      App.paintWeatherChip();
    },
    paintWeatherChip() {
      const el = qs('#weather-chip'); if (!el) return;
      const w = App.state.weather;
      if (!w || w.temperature_c == null) {
        el.innerHTML = `<i class="fa-solid fa-cloud"></i><span class="wmeta">Environment not set<br><button class="linklike" onclick="location.hash='#/settings'">Add it</button></span>`;
        return;
      }
      el.innerHTML = `<i class="fa-solid fa-cloud-sun"></i><span class="temp">${Math.round(w.temperature_c)}&deg;C</span>
        <span class="wmeta">${w.humidity_pct != null ? 'Humidity ' + Math.round(w.humidity_pct) + '%' : ''}
        ${w.place_name ? '<br>' + esc(w.place_name.split(',').slice(0, 2).join(',')) : ''}</span>`;
    },

    revokeBlobs() { App.blobUrls.forEach((u) => URL.revokeObjectURL(u)); App.blobUrls = []; },
    trackBlob(u) { App.blobUrls.push(u); return u; },
  };
  window.App = App;

  function loadPrefs() { try { return JSON.parse(localStorage.getItem(CFG.PREFS_KEY)) || {}; } catch (_) { return {}; } }
  function savePrefs(p) { localStorage.setItem(CFG.PREFS_KEY, JSON.stringify(p)); }
  function loadEnv() { try { return JSON.parse(localStorage.getItem(CFG.ENV_KEY)) || null; } catch (_) { return null; } }
  function saveEnv(e) { localStorage.setItem(CFG.ENV_KEY, JSON.stringify(e || {})); }
  function planProgress() { try { return JSON.parse(localStorage.getItem(CFG.PLAN_PROGRESS_KEY)) || {}; } catch (_) { return {}; } }
  function savePlanProgress(p) { localStorage.setItem(CFG.PLAN_PROGRESS_KEY, JSON.stringify(p)); }

  // ============================================================ AUTH SCREEN
  function renderAuth(mode) {
    const isSignup = mode === 'signup';
    root().innerHTML = `
    <div class="auth-wrap">
      <div class="auth-left">
        <div class="brand"><span class="brand-mark"><i class="fa-solid fa-leaf"></i></span>
          <div><div class="brand-name">SkinIntelli</div><div class="brand-sub">Context-Aware Skin Intelligence</div></div></div>
        <h1 class="auth-hero-h1">Understand Today.<span class="accent">Protect Tomorrow.</span></h1>
        <p class="auth-hero-p">AI-powered skin analysis with context-aware insights and personalised care for healthier skin. An academic research project &mdash; not a medical device.</p>
        <div class="auth-features">
          <div class="auth-feature"><div class="fi"><i class="fa-solid fa-microscope"></i></div><b>AI Skin Analysis</b><span>Pretrained vision models for acne, skin type & more</span></div>
          <div class="auth-feature"><div class="fi"><i class="fa-solid fa-cloud-sun"></i></div><b>Context-Aware</b><span>Weather, lifestyle & environmental data</span></div>
          <div class="auth-feature"><div class="fi"><i class="fa-solid fa-shield-heart"></i></div><b>Personalised Care</b><span>Guidance tuned to your skin and surroundings</span></div>
        </div>
        <div class="auth-cta-row">
          <button class="btn btn-primary btn-lg" onclick="location.hash='#/login'">Login</button>
          <button class="btn btn-ghost btn-lg" onclick="location.hash='#/signup'">Sign Up</button>
        </div>
      </div>
      <div class="auth-right">
        <div class="auth-card">
          <div class="brand" style="margin-bottom:16px"><span class="brand-mark"><i class="fa-solid fa-leaf"></i></span>
            <div><div class="brand-name">SkinIntelli</div></div></div>
          <h2>${isSignup ? 'Create your account' : 'Welcome back!'}</h2>
          <p class="sub">${isSignup ? 'Start your personalised skin journey.' : 'Login to continue your personalised skin journey.'}</p>
          <div id="auth-error" class="form-error hidden"></div>
          <form id="auth-form">
            ${isSignup ? `<div class="field"><label>Full name</label><input type="text" id="a-name" autocomplete="name" placeholder="Your name"></div>` : ''}
            <div class="field"><label>Email</label>
              <div class="input-wrap"><input type="email" id="a-email" required autocomplete="email" placeholder="Enter your email"></div></div>
            <div class="field"><label>Password</label>
              <div class="input-wrap">
                <input type="password" id="a-pass" required minlength="8" autocomplete="${isSignup ? 'new-password' : 'current-password'}" placeholder="Enter your password">
                <button type="button" class="toggle-eye" aria-label="Show password"><i class="fa-regular fa-eye"></i></button>
              </div></div>
            ${isSignup ? '' : `<div class="form-row"><span></span><button type="button" class="linklike" id="forgot-link">Forgot password?</button></div>`}
            <button class="btn btn-primary btn-block btn-lg" type="submit">${isSignup ? 'Create account' : 'Login'}</button>
          </form>
          <div class="divider-or">or continue with</div>
          <div class="oauth-row">
            <button class="btn btn-ghost" id="oauth-google"><i class="fa-brands fa-google"></i> Google</button>
            <button class="btn btn-ghost" id="oauth-ms"><i class="fa-brands fa-microsoft"></i> Microsoft</button>
          </div>
          <p class="form-note">${isSignup
            ? `Already have an account? <button class="linklike" onclick="location.hash='#/login'">Login</button>`
            : `Don't have an account? <button class="linklike" onclick="location.hash='#/signup'">Sign Up</button>`}</p>
        </div>
      </div>
    </div>`;

    const err = qs('#auth-error');
    const showErr = (m) => { err.textContent = m; err.classList.remove('hidden'); };
    qs('.toggle-eye').onclick = (e) => {
      const inp = qs('#a-pass'); inp.type = inp.type === 'password' ? 'text' : 'password';
      e.currentTarget.innerHTML = `<i class="fa-regular ${inp.type === 'password' ? 'fa-eye' : 'fa-eye-slash'}"></i>`;
    };
    if (qs('#forgot-link')) qs('#forgot-link').onclick = () =>
      toast('Password reset isn’t enabled in this academic build. Create a new account if needed.', 'warn');
    qs('#oauth-google').onclick = qs('#oauth-ms').onclick = () =>
      toast('Social sign-in is a planned feature — use email/password for now.', 'warn');

    qs('#auth-form').onsubmit = async (e) => {
      e.preventDefault();
      err.classList.add('hidden');
      const email = qs('#a-email').value.trim();
      const pass = qs('#a-pass').value;
      if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) return showErr('Please enter a valid email address.');
      if (pass.length < 8) return showErr('Password must be at least 8 characters.');
      const btn = qs('#auth-form button[type=submit]');
      btn.disabled = true; btn.textContent = 'Please wait…';
      try {
        if (isSignup) { await API.register(email, pass, qs('#a-name').value.trim()); }
        await API.login(email, pass);
        localStorage.setItem(CFG.EMAIL_KEY, email);
        App.state.user = await API.me();
        App.loadWeather();
        location.hash = '#/dashboard';
        App.route();
      } catch (ex) {
        showErr(ex.status === 401 ? 'Incorrect email or password.'
              : ex.status === 400 ? 'An account with this email already exists.'
              : ex.message);
        btn.disabled = false; btn.textContent = isSignup ? 'Create account' : 'Login';
      }
    };
  }

  // ============================================================= APP SHELL
  const NAV = [
    ['dashboard', 'Dashboard', 'fa-house'],
    ['scan', 'Scan Skin', 'fa-camera'],
    ['history', 'Scan History', 'fa-clock-rotate-left'],
    ['plan', 'Current Plan', 'fa-clipboard-list'],
    ['products', 'Product Recommendations', 'fa-bag-shopping'],
  ];
  const NAV2 = [
    ['profile', 'Profile', 'fa-user'],
    ['settings', 'Settings', 'fa-gear'],
  ];

  function renderShell(active) {
    const u = App.state.user || {};
    const name = (u.full_name || u.email || 'there').split(/[ @]/)[0];
    const initials = (u.full_name || u.email || 'S').split(/[ @]/).map((s) => s[0]).join('').slice(0, 2).toUpperCase();
    if (qs('#app-shell')) { // just switch active nav + rerender view container
      qsa('.nav-item').forEach((n) => n.classList.toggle('active', n.dataset.seg === active));
      return;
    }
    root().innerHTML = `
    <div class="app-shell" id="app-shell">
      <aside class="sidebar" id="sidebar">
        <div class="brand"><span class="brand-mark"><i class="fa-solid fa-leaf"></i></span>
          <div><div class="brand-name">SkinIntelli</div></div></div>
        ${NAV.map(([s, l, i]) => navBtn(s, l, i, active)).join('')}
        <div class="nav-sep"></div>
        ${NAV2.map(([s, l, i]) => navBtn(s, l, i, active)).join('')}
        <div class="nav-spacer"></div>
        <button class="nav-item" id="logout-btn"><i class="fa-solid fa-arrow-right-from-bracket"></i> Logout</button>
      </aside>
      <div class="app-main">
        <header class="topbar">
          <div style="display:flex;align-items:center;gap:12px">
            <button class="menu-btn" id="menu-btn"><i class="fa-solid fa-bars"></i></button>
            <div class="topbar-greeting"><b>Welcome back, ${esc(cap(name))} <span aria-hidden="true">👋</span></b>
              <span>Here's your skin overview and personalised insights.</span></div>
          </div>
          <div class="topbar-right">
            <div class="weather-chip" id="weather-chip"><i class="fa-solid fa-cloud"></i></div>
            <div class="user-chip"><span class="avatar">${esc(initials)}</span>${esc(cap(name))}</div>
          </div>
        </header>
        <main class="view" id="view"></main>
      </div>
    </div>`;
    qsa('.nav-item[data-seg]').forEach((b) => b.onclick = () => { location.hash = '#/' + b.dataset.seg; closeSidebar(); });
    qs('#logout-btn').onclick = () => { API.logout(); App.state.user = null; App.revokeBlobs(); location.hash = '#/'; App.route(); };
    qs('#menu-btn').onclick = () => { qs('#sidebar').classList.add('open'); const s = h('<div class="scrim"></div>'); s.onclick = closeSidebar; qs('#app-shell').appendChild(s); };
    App.paintWeatherChip();
  }
  function navBtn(seg, label, icon, active) {
    return `<button class="nav-item ${seg === active ? 'active' : ''}" data-seg="${seg}"><i class="fa-solid ${icon}"></i> ${label}</button>`;
  }
  function closeSidebar() { const sb = qs('#sidebar'); if (sb) sb.classList.remove('open'); const sc = qs('.scrim'); if (sc) sc.remove(); }

  // =========================================================== DASHBOARD
  async function viewDashboard(nav) {
    const view = qs('#view');
    view.innerHTML = `<div class="empty"><i class="fa-solid fa-spinner fa-spin"></i>Loading your dashboard…</div>`;
    let hist = App.state.historyCache;
    try { hist = await API.history(); App.state.historyCache = hist; } catch (e) { /* show empty */ }
    if (nav != null && !App.isCurrent(nav)) return;
    hist = hist || [];
    const latest = hist.find((r) => r.results && r.results.conditions) || null;
    const results = latest ? latest.results : null;
    const shi = shiOf(results);
    const w = App.state.weather;

    const gauge = (() => {
      if (shi.value == null) return `<div class="shi-gauge"><div class="val"><b>—</b><small>/100</small></div></div>
        <div class="band ${shi.band}">Not enough data</div>
        <div class="hint">Run a scan to generate your Skin Health Index.</div>`;
      const c = 2 * Math.PI * 76, off = c - (shi.value / 100) * c;
      const col = shi.value >= 65 ? 'var(--good)' : shi.value >= 50 ? 'var(--moderate)' : 'var(--attention)';
      return `<div class="shi-gauge">
        <svg viewBox="0 0 168 168"><circle cx="84" cy="84" r="76" stroke="var(--bg-tint)" stroke-width="12" fill="none"></circle>
        <circle cx="84" cy="84" r="76" stroke="${col}" stroke-width="12" fill="none" stroke-linecap="round"
          stroke-dasharray="${c}" stroke-dashoffset="${off}"></circle></svg>
        <div class="val"><b>${shi.value}</b><small>/100</small></div></div>
        <div class="band ${shi.band}">${cap(shi.band.replace('_', ' '))}</div>
        <div class="hint">${shi.value >= 65 ? "You're doing well — keep following your routine." : 'A few areas could use attention. See your plan.'}</div>`;
    })();

    const scoreChip = (c) => {
      const s = (c && typeof c.score === 'number') ? c.score : null;
      if (s == null) return '';
      const [, cls] = s >= 0.67 ? ['', 'attention'] : s >= 0.34 ? ['', 'moderate'] : ['', 'good'];
      return `<span class="pill ${cls} score-chip" title="Severity 0–1">${s.toFixed(2)}</span>`;
    };
    const concernRows = CFG.CONDITIONS.map(({ key, label, tier, icon }) => {
      const c = results && results.conditions ? results.conditions[key] : null;
      let right;
      if (tier === 'unavailable' || (c && c.status === 'not_available')) right = `<span class="pill na">Not supported</span>`;
      else if (!c || !results) right = `<span class="pill neutral">—</span>`;
      else if (c.status === 'estimated') right = `${scoreChip(c)}${severityPill(c)}<span class="pill estimate">Est.</span>`;
      else if (c.status === 'error') right = `<span class="pill attention">Error</span>`;
      else if (key === 'skin_type') right = `<span class="pill good">${esc(cap(c.label || '—'))}</span>`;
      else right = `${scoreChip(c)}${severityPill(c) || '<span class="pill ai">Detected</span>'}`;
      return `<div class="crow"><span class="cname"><i class="fa-solid ${icon}"></i>${label}</span><span class="cright">${right}</span></div>`;
    }).join('');

    const envRow = (label, icon, val, unit) => `
      <div class="env-row"><span class="el"><i class="fa-solid ${icon}"></i>${label}</span>
      <span class="ev ${val == null ? 'unknown' : ''}">${val == null ? 'Unknown' : esc(val) + (unit || '')}</span></div>`;

    const tip = latest && latest.recommendation && latest.recommendation.summary && latest.recommendation.summary[0]
      ? latest.recommendation.summary[0]
      : (w && w.uv_index != null && w.uv_index >= 3
          ? `UV index is ${w.uv_index >= 6 ? 'high' : 'moderate'} today. Don't skip sunscreen when going out.`
          : 'Consistency matters more than any single product — a simple routine you actually follow wins.');

    view.innerHTML = `
      <div class="grid-4">
        <div class="card card-pad shi">
          <div class="card-title-row"><h3>Skin Health Index (SHI)</h3></div>
          ${gauge}
          <button class="btn btn-soft btn-sm mt-18" onclick="location.hash='#/${latest ? 'result/' + latest.id : 'scan'}'">${latest ? 'View details' : 'Run a scan'}</button>
        </div>
        <div class="card card-pad">
          <div class="card-title-row"><h3>Key Skin Concerns</h3></div>
          ${results ? concernRows : `<div class="empty" style="padding:20px"><i class="fa-solid fa-clipboard-question"></i>No scan yet.<br><button class="linklike" onclick="location.hash='#/scan'">Run your first scan</button></div>`}
          ${results ? `<button class="btn btn-soft btn-sm mt-18" onclick="location.hash='#/result/${latest.id}'">View full analysis</button>` : ''}
        </div>
        <div class="card card-pad">
          <div class="card-title-row"><h3>Environmental Overview</h3></div>
          ${envRow('Temperature', 'fa-temperature-half', w && w.temperature_c != null ? Math.round(w.temperature_c) : null, '°C')}
          ${envRow('Humidity', 'fa-droplet', w && w.humidity_pct != null ? Math.round(w.humidity_pct) : null, '%')}
          ${envRow('UV Index', 'fa-sun', w && w.uv_index != null ? w.uv_index : null, '')}
          ${envRow('Air Quality', 'fa-wind', w && (w.air_quality_label || w.air_quality_aqi != null)
            ? (w.air_quality_label ? w.air_quality_label + ' ' : 'AQI ') + (w.air_quality_aqi != null ? '(' + Math.round(w.air_quality_aqi) + ')' : '') : null, '')}
          <button class="btn btn-soft btn-sm mt-18" onclick="location.hash='#/settings'">${w && w.temperature_c != null ? 'Update location' : 'Set your location'}</button>
        </div>
        <div class="card card-pad tip-card">
          <div class="card-title-row"><h3>Daily Skin Tip</h3></div>
          <div class="tip-ico">💡</div>
          <p>${esc(tip)}</p>
        </div>
      </div>

      <div class="grid-2 mt-18">
        <div class="card card-pad">
          <div class="card-title-row"><h3>Quick Actions</h3></div>
          <div class="qa-grid">
            <button class="qa" onclick="location.hash='#/scan'"><i class="fa-solid fa-camera"></i><b>Scan Skin</b><span>Analyse your skin</span></button>
            <button class="qa" onclick="location.hash='#/history'"><i class="fa-solid fa-clock-rotate-left"></i><b>Scan History</b><span>Previous scans</span></button>
            <button class="qa" onclick="location.hash='#/plan'"><i class="fa-solid fa-clipboard-list"></i><b>Current Plan</b><span>Your routine & steps</span></button>
            <button class="qa" onclick="location.hash='#/products'"><i class="fa-solid fa-bag-shopping"></i><b>Products</b><span>Recommendations</span></button>
          </div>
        </div>
        <div class="card card-pad">
          <div class="card-title-row"><h3>Recent Scans</h3><button class="linklike" onclick="location.hash='#/history'">View all</button></div>
          ${hist.length ? `<div class="scan-strip" id="recent-strip"></div>` : `<div class="empty" style="padding:20px"><i class="fa-solid fa-image"></i>No scans yet.</div>`}
        </div>
      </div>`;

    if (hist.length) fillScanStrip(qs('#recent-strip'), hist.slice(0, 4));
  }

  async function fillScanStrip(host, rows) {
    host.innerHTML = rows.map((r) => {
      const { date, time } = fmtDateTime(r.created_at);
      const shi = shiOf(r.results);
      const band = shi.value == null ? '' : shi.value >= 65 ? 'good' : shi.value >= 50 ? 'moderate' : 'attention';
      return `<button class="scan-thumb" onclick="location.hash='#/result/${r.id}'">
        <div class="imgbox"><img data-img="${r.image_id}" alt="scan ${esc(date)}"></div>
        <div class="meta"><div class="d">${esc(date)}</div><div class="t">${esc(time)}</div>
        ${shi.value != null ? `<span class="pill ${band}" style="margin-top:6px">${shi.value}/100</span>` : ''}</div>
      </button>`;
    }).join('');
    hydrateImages(host);
  }

  async function hydrateImages(host) {
    for (const img of qsa('img[data-img]', host)) {
      try { img.src = App.trackBlob(await API.imageBlobUrl(Number(img.dataset.img))); }
      catch (_) { img.style.background = 'var(--sage-100)'; }
    }
  }

  // ============================================================== SCAN
  App._scan = { file: null, stream: null, dataUrl: null };

  // ---- shared modal + analysis-flow helpers (Scan screen) --------------
  function closeModal() { const m = qs('#si-modal'); if (m) m.remove(); }
  function openModal(inner, opts) {
    closeModal();
    const size = opts && opts.xl ? 'xl' : opts && opts.wide ? 'wide' : '';
    const wrap = h(`<div class="si-modal-scrim" id="si-modal"><div class="si-modal ${size}" role="dialog" aria-modal="true"></div></div>`);
    wrap.querySelector('.si-modal').appendChild(inner);
    wrap.addEventListener('click', (e) => { if (e.target === wrap && !(opts && opts.sticky)) closeModal(); });
    const onEsc = (e) => { if (e.key === 'Escape') { closeModal(); document.removeEventListener('keydown', onEsc); } };
    document.addEventListener('keydown', onEsc);
    document.body.appendChild(wrap);
    return wrap;
  }

  function analyzeOverlay() {
    const o = h(`<div class="analyzing"><div class="box"><div class="ring2"></div><b>Analysing your skin…</b><div class="muted" style="margin-top:6px">Running the vision models — up to ~30s on the first scan.</div></div></div>`);
    document.body.appendChild(o);
    return o;
  }
  function finishScan(pred) {
    App.state.lastAnalysis = pred;
    App.state.lastRecommendation = pred.recommendation || null;
    App.state.historyCache = null;
    location.hash = '#/result/' + pred.id;
  }

  // Run analysis for a freshly uploaded file OR an existing image id.
  // Honours the advisory quality gate: a <=6/10 photo opens the advisory modal.
  async function analyzeFlow({ file, imageId, getCtx, previewUrl, onError }) {
    const ctx = getCtx ? getCtx() : null;
    const overlay = analyzeOverlay();
    try {
      const pred = file
        ? await API.scan(file, ctx)
        : await API.predict(imageId, ctx, false);
      overlay.remove();
      if (pred && pred.status === 'advisory') {
        let pv = previewUrl || App._scan.dataUrl || null;
        if (!pv && pred.image_id != null) {
          try { pv = App.trackBlob(await API.imageBlobUrl(Number(pred.image_id))); } catch (_) { /* no preview */ }
        }
        openAdvisoryModal(pred, {
          getCtx,
          previewUrl: pv,
          imageId: pred.temp_image_id || pred.image_id || pred._imageId,
        });
        return;
      }
      finishScan(pred);
    } catch (ex) {
      overlay.remove();
      if (onError) onError(ex); else toast(ex.message, 'err');
    }
  }

  async function proceedAnyway(imageId, getCtx) {
    const overlay = analyzeOverlay();
    try {
      const pred = await API.predict(imageId, getCtx ? getCtx() : null, true);
      overlay.remove();
      finishScan(pred);
    } catch (ex) {
      overlay.remove();
      toast(ex.message, 'err');
    }
  }

  function openAdvisoryModal(pred, { getCtx, previewUrl, imageId }) {
    const q = pred.quality || {};
    const score = q.score != null ? q.score : '?';
    const badgeCls = q.score >= 6 ? 'good' : q.score >= 4 ? 'moderate' : 'attention';
    const inner = h(`<div class="si-advisory">
      <div class="si-modal-head"><h3><i class="fa-solid fa-triangle-exclamation"></i> Check your photo quality</h3>
        <button class="si-x" aria-label="Close">&times;</button></div>
      <div class="si-adv-body">
        <div class="si-adv-preview">${previewUrl ? `<img src="${esc(previewUrl)}" alt="Captured photo">` : `<div class="si-adv-noimg"><i class="fa-solid fa-image"></i></div>`}</div>
        <div class="si-adv-info">
          <div class="si-qbadge ${badgeCls}">Photo Quality: ${esc(String(score))}/10 ${q.label ? '· ' + esc(q.label) : ''}</div>
          <p class="si-adv-note">Photo quality directly influences analysis accuracy. Built-in webcams often blur fine pores and micro-textures.</p>
          <p class="si-adv-msg">${esc(q.message || q.advisory || '')}</p>
        </div>
      </div>
      <div class="si-modal-foot">
        <button class="btn btn-ghost btn-sm" data-act="retake"><i class="fa-solid fa-rotate-left"></i> Retake photo</button>
        <button class="btn btn-primary btn-sm" data-act="proceed">Proceed with ${esc(String(score))}/10 photo</button>
      </div>
    </div>`);
    inner.querySelector('.si-x').onclick = closeModal;
    inner.querySelector('[data-act="retake"]').onclick = () => {
      closeModal();
      App._scan.file = null; App._scan.dataUrl = null;
      const rb = qs('#run-btn'); if (rb) rb.disabled = true;
      const pv = qs('#preview'); if (pv) pv.classList.add('hidden');
      const ph = qs('#ph'); if (ph) ph.classList.remove('hidden');
      const st = qs('#scan-status'); if (st) st.textContent = 'Retake your photo — steady camera, even lighting.';
      const cb = qs('#cam-btn'); if (cb) cb.scrollIntoView({ behavior: 'smooth', block: 'center' });
    };
    inner.querySelector('[data-act="proceed"]').onclick = () => {
      closeModal();
      proceedAnyway(imageId, getCtx);
    };
    openModal(inner, { sticky: true });
  }

  function openPhoneCameraModal(getCtx) {
    const inner = h(`<div class="si-phone">
      <div class="si-modal-head"><h3><i class="fa-solid fa-mobile-screen-button"></i> Use your phone camera</h3>
        <button class="si-x" aria-label="Close">&times;</button></div>
      <div class="si-tabs">
        <button class="si-tab active" data-tab="qr">QR code capture</button>
        <button class="si-tab" data-tab="webcam">Phone as wireless webcam</button>
      </div>
      <div class="si-tab-body" data-body="qr">
        <p class="si-adv-note">Phone cameras are far sharper than a built-in laptop webcam — better for pores, fine wrinkles and blackheads.</p>
        <div class="si-warn" id="si-warn" hidden></div>
        <div class="si-qr" id="si-qr"><div class="empty" style="padding:24px"><i class="fa-solid fa-spinner fa-spin"></i> Generating a secure link…</div></div>
        <ol class="si-steps">
          <li>Make sure your phone and this laptop are on the <b>exact same Wi-Fi</b> network.</li>
          <li>Scan the QR code, or type the URL above into your phone's browser.</li>
          <li>Snap a high-res selfie — it appears here automatically and analysis begins.</li>
          <li>If it won't connect, allow <b>Python</b> through Windows Defender Firewall on <b>Private networks</b>.</li>
        </ol>
        <div class="si-ipfix">
          <label for="si-ip">Laptop IP</label>
          <input id="si-ip" type="text" inputmode="decimal" placeholder="192.168.x.x" autocomplete="off">
          <button class="btn btn-ghost btn-sm" id="si-ip-go"><i class="fa-solid fa-rotate"></i> Regenerate QR</button>
        </div>
        <div class="si-poll muted" id="si-poll">Waiting for your phone…</div>
        <button class="btn btn-ghost btn-sm" id="si-regen" hidden><i class="fa-solid fa-rotate"></i> Generate a new code</button>
      </div>
      <div class="si-tab-body" data-body="webcam" hidden>
        <p class="si-adv-note">Turn your phone into a normal webcam your browser can pick from the camera dropdown.</p>
        <ol class="si-steps">
          <li>Install a virtual-camera app: <b>DroidCam</b>, <b>Iriun Webcam</b>, or Windows 11 <b>Phone Link</b> (mobile camera).</li>
          <li>Connect the phone and laptop (same Wi-Fi or USB) and open the companion app on both.</li>
          <li>Back here, click <b>Take webcam photo</b> and choose the phone camera from the browser's video-input dropdown for an HD feed.</li>
        </ol>
      </div>
    </div>`);
    inner.querySelector('.si-x').onclick = () => { stopPhonePoll(); closeModal(); };
    inner.querySelectorAll('.si-tab').forEach((t) => t.onclick = () => {
      inner.querySelectorAll('.si-tab').forEach((x) => x.classList.toggle('active', x === t));
      inner.querySelectorAll('.si-tab-body').forEach((b) => b.hidden = b.dataset.body !== t.dataset.tab);
    });
    const wrap = openModal(inner, { sticky: true, wide: true });
    wrap.addEventListener('click', (e) => { if (e.target === wrap) { stopPhonePoll(); closeModal(); } });
    startPhoneSession(inner, getCtx);
  }

  App._phonePoll = null;
  function stopPhonePoll() { if (App._phonePoll) { clearInterval(App._phonePoll); App._phonePoll = null; } }

  const PHONE_POLL_MS = 1500;
  const PHONE_TIMEOUT_MS = 150000;   // stop waiting after 2.5 min

  async function startPhoneSession(modalEl, getCtx, ipOverride) {
    stopPhonePoll();
    const box = qs('#si-qr', modalEl);
    const warnEl = qs('#si-warn', modalEl);
    const pollEl = qs('#si-poll', modalEl);
    const regenBtn = qs('#si-regen', modalEl);
    const ipInput = qs('#si-ip', modalEl);
    const ipGo = qs('#si-ip-go', modalEl);
    const rerun = (ov) => startPhoneSession(modalEl, getCtx, ov);
    if (ipGo) ipGo.onclick = () => rerun((ipInput && ipInput.value.trim()) || undefined);
    if (ipInput) ipInput.onkeydown = (e) => { if (e.key === 'Enter') ipGo.click(); };
    if (regenBtn) regenBtn.hidden = true;
    if (warnEl) warnEl.hidden = true;
    if (box) box.innerHTML = `<div class="empty" style="padding:24px"><i class="fa-solid fa-spinner fa-spin"></i> Generating a secure link…</div>`;
    if (pollEl) { pollEl.textContent = 'Waiting for your phone…'; pollEl.className = 'si-poll muted'; }

    let sess;
    try { sess = await API.createQrSession(ipOverride); }
    catch (ex) {
      if (box) box.innerHTML = `<div class="empty" style="padding:20px">${esc(ex.message)}</div>`;
      if (regenBtn) { regenBtn.hidden = false; regenBtn.onclick = () => rerun(ipOverride); }
      return;
    }

    if (ipInput && sess.local_ip && !ipInput.value.trim()) ipInput.value = sess.local_ip;

    // surface backend warnings (bad IP / bound to localhost / virtual adapter)
    const warns = Array.isArray(sess.warnings) ? sess.warnings : [];
    if (sess.ip_confident === false) warns.unshift('This laptop has no Wi-Fi / LAN address right now — the phone cannot connect. Connect to Wi-Fi and regenerate, set the IP manually below, or use the "Phone as wireless webcam" tab.');
    if (warnEl && warns.length) {
      warnEl.hidden = false;
      warnEl.innerHTML = warns.map((w) => `<div><i class="fa-solid fa-triangle-exclamation"></i> ${esc(w)}</div>`).join('');
    }

    if (box) {
      const src = sess.qr_png || sess.qr_svg;
      const u = esc(sess.capture_url);
      box.innerHTML = src
        ? `<div class="si-qr-img"><img src="${esc(src)}" alt="QR code — scan with your phone" width="260" height="260"></div>
           <div class="si-qr-url">On your phone, open:<br><a href="${u}" target="_blank" rel="noopener">${u}</a>
             <button class="si-copy" type="button" title="Copy URL"><i class="fa-regular fa-copy"></i></button></div>`
        : `<div class="si-qr-fallback"><b>Open this on your phone:</b><br><a href="${u}" target="_blank" rel="noopener">${u}</a></div>`;
      const cp = qs('.si-copy', box);
      if (cp) cp.onclick = () => {
        (navigator.clipboard ? navigator.clipboard.writeText(sess.capture_url) : Promise.reject())
          .then(() => toast('URL copied', '')).catch(() => toast('Copy failed — long-press the link', 'warn'));
      };
    }

    const started = Date.now();
    let failStreak = 0;
    App._phonePoll = setInterval(async () => {
      if (Date.now() - started > PHONE_TIMEOUT_MS) {
        stopPhonePoll();
        if (pollEl) { pollEl.textContent = 'No photo received yet. If the link would not open on your phone, check you are on the same Wi-Fi.'; pollEl.className = 'si-poll err'; }
        if (regenBtn) { regenBtn.hidden = false; regenBtn.onclick = () => rerun(ipOverride); }
        return;
      }
      let st;
      try { st = await API.sessionStatus(sess.session_id); failStreak = 0; }
      catch (ex) {
        failStreak += 1;
        if (ex.status === 404) {
          stopPhonePoll();
          if (pollEl) { pollEl.textContent = 'This code expired. Generate a new one.'; pollEl.className = 'si-poll err'; }
          if (regenBtn) { regenBtn.hidden = false; regenBtn.onclick = () => rerun(ipOverride); }
        } else if (failStreak >= 3 && pollEl) {
          pollEl.textContent = 'Trouble reaching the server — retrying…';
          pollEl.className = 'si-poll err';
        }
        return;
      }
      if (st && st.status === 'ready' && st.image_id) {
        stopPhonePoll();
        if (pollEl) { pollEl.textContent = 'Photo received — starting analysis…'; pollEl.className = 'si-poll ok'; }
        setTimeout(() => { closeModal(); analyzeFlow({ imageId: st.image_id, getCtx }); }, 500);
      } else if (pollEl && failStreak === 0) {
        pollEl.textContent = 'Waiting for your phone…'; pollEl.className = 'si-poll muted';
      }
    }, PHONE_POLL_MS);
  }

  function viewScan() {
    const s = App._scan;
    const env = loadEnv() || { env_mode: 'current_location' };
    qs('#view').innerHTML = `
      <div class="view-head"><h1>Scan Your Skin</h1><p>Position your face inside the frame for a clear analysis. Good, even lighting helps.</p></div>
      <div class="scan-layout">
        <div class="card card-pad">
          <div class="capture-stage" id="stage">
            <video id="cam" autoplay playsinline muted class="hidden"></video>
            <img id="preview" class="hidden" alt="Selected photo">
            <div class="placeholder" id="ph"><i class="fa-solid fa-camera fa-2x"></i><div style="margin-top:10px">Start your camera or upload a photo</div></div>
            <div class="face-guide" id="guide" style="display:none"></div>
          </div>
          <div class="quality-bar hidden" id="quality"></div>
          <div class="scan-controls">
            <button class="btn btn-ghost btn-sm" id="cam-btn"><i class="fa-solid fa-video"></i> Take webcam photo</button>
            <button class="btn btn-ghost btn-sm" id="phone-btn"><i class="fa-solid fa-mobile-screen-button"></i> Use phone camera</button>
            <label class="btn btn-ghost btn-sm" style="cursor:pointer"><i class="fa-solid fa-upload"></i> Upload photo
              <input type="file" id="file-in" accept="image/jpeg,image/png,image/webp" class="hidden"></label>
            <button class="btn btn-ghost btn-sm hidden" id="cap-btn"><i class="fa-solid fa-circle"></i> Capture</button>
            <button class="btn btn-primary btn-sm" id="run-btn" disabled><i class="fa-solid fa-wand-magic-sparkles"></i> Run analysis</button>
          </div>
          <div class="scan-status" id="scan-status"></div>
        </div>

        <div class="card card-pad">
          <div class="card-title-row"><h3>Environmental context</h3></div>
          <p class="muted" style="font-size:.85rem;margin-bottom:10px">Optional. Feeds the recommendations only — never the vision models. Missing values stay unknown.</p>
          <div class="env-modes" id="env-modes">
            <button class="env-mode-btn" data-m="current_location">Use current location</button>
            <button class="env-mode-btn" data-m="select_location">Select location</button>
            <button class="env-mode-btn" data-m="manual">Manual</button>
          </div>
          <div id="env-body"></div>
          <div class="card-title-row" style="margin-top:18px"><h3>About you</h3></div>
          <div class="grid-2" style="gap:10px">
            <div class="field"><label>Age</label><input type="number" id="q-age" min="1" max="120" placeholder="e.g. 28"></div>
            <div class="field"><label>Skin sensitivity / stress</label>
              <select id="q-stress"><option value="">—</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></div>
          </div>
          <div class="grid-2" style="gap:10px">
            <div class="field"><label>Sleep (hrs/night)</label><input type="number" id="q-sleep" min="0" max="24" step="0.5" placeholder="e.g. 7"></div>
            <div class="field"><label>Sun exposure</label>
              <select id="q-sun"><option value="">—</option><option value="low">Low</option><option value="moderate">Moderate</option><option value="high">High</option></select></div>
          </div>
        </div>
      </div>`;

    bindEnvPicker(env);
    // ---- capture wiring ----
    const stage = qs('#stage'), cam = qs('#cam'), preview = qs('#preview'), ph = qs('#ph'), guide = qs('#guide');
    const runBtn = qs('#run-btn'), status = qs('#scan-status'), quality = qs('#quality');

    function setQuality(cls, txt) { quality.className = 'quality-bar ' + cls; quality.textContent = ''; quality.classList.remove('hidden');
      quality.appendChild(h(`<span><i class="fa-solid ${cls === 'good' ? 'fa-circle-check' : 'fa-lightbulb'}"></i> ${esc(txt)}</span>`)); }

    function checkLighting(source, w, hh) {
      try {
        const cv = document.createElement('canvas'); cv.width = 80; cv.height = 80;
        const cx = cv.getContext('2d'); cx.drawImage(source, 0, 0, 80, 80);
        const d = cx.getImageData(0, 0, 80, 80).data; let sum = 0;
        for (let i = 0; i < d.length; i += 4) sum += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
        const avg = sum / (d.length / 4);
        if (avg < 45) setQuality('bad', 'Too dark — move to brighter, even light.');
        else if (avg > 225) setQuality('warn', 'Overexposed — reduce direct glare.');
        else setQuality('good', 'Lighting looks good. Ready to analyse.');
      } catch (_) {}
    }

    qs('#cam-btn').onclick = async () => {
      if (App._scan.stream) { stopCam(); return; }
      try {
        App._scan.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 960 }, facingMode: 'user' } });
        cam.srcObject = App._scan.stream; cam.classList.remove('hidden'); preview.classList.add('hidden'); ph.classList.add('hidden');
        guide.style.display = 'block'; qs('#cap-btn').classList.remove('hidden');
        qs('#cam-btn').innerHTML = '<i class="fa-solid fa-video-slash"></i> Stop camera';
        const loop = () => { if (!App._scan.stream) return; if (cam.videoWidth) checkLighting(cam, cam.videoWidth, cam.videoHeight); requestAnimationFrame(loop); };
        loop();
      } catch (e) { status.textContent = 'Camera unavailable (' + e.message + '). Upload a photo instead.'; }
    };
    function stopCam() {
      if (App._scan.stream) { App._scan.stream.getTracks().forEach((t) => t.stop()); App._scan.stream = null; }
      cam.classList.add('hidden'); guide.style.display = 'none'; qs('#cap-btn').classList.add('hidden');
      qs('#cam-btn').innerHTML = '<i class="fa-solid fa-video"></i> Take webcam photo';
      if (!App._scan.file) { ph.classList.remove('hidden'); runBtn.disabled = true; }
    }
    qs('#cap-btn').onclick = () => {
      const cv = document.createElement('canvas'); cv.width = cam.videoWidth; cv.height = cam.videoHeight;
      cv.getContext('2d').drawImage(cam, 0, 0);
      cv.toBlob((blob) => {
        App._scan.file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
        App._scan.dataUrl = cv.toDataURL('image/jpeg', 0.92);
        preview.src = App._scan.dataUrl; preview.classList.remove('hidden'); stopCam();
        runBtn.disabled = false; status.textContent = 'Photo captured.';
      }, 'image/jpeg', 0.92);
    };
    qs('#file-in').onchange = (e) => {
      const f = e.target.files[0]; if (!f) return;
      stopCam(); App._scan.file = f;
      const r = new FileReader();
      r.onload = () => {
        App._scan.dataUrl = r.result;
        preview.src = r.result; preview.classList.remove('hidden'); ph.classList.add('hidden');
        runBtn.disabled = false; status.textContent = 'Photo ready.';
        const im = new Image(); im.onload = () => checkLighting(im, im.width, im.height); im.src = r.result;
      };
      r.readAsDataURL(f);
    };

    qs('#phone-btn').onclick = () => { stopCam(); openPhoneCameraModal(collectContext); };

    runBtn.onclick = async () => {
      if (!App._scan.file) return;
      runBtn.disabled = true; runBtn.innerHTML = '<span class="spinner"></span> Analysing…';
      const resetBtn = () => { runBtn.disabled = false; runBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Run analysis'; };
      await analyzeFlow({
        file: App._scan.file,
        getCtx: collectContext,
        previewUrl: App._scan.dataUrl,
        onError: (ex) => {
          resetBtn();
          status.textContent = ex.status === 401 ? 'Session expired — please log in again.' : ex.message;
          toast(ex.message, 'err');
        },
      });
      resetBtn();
    };

    function collectContext() {
      const num = (id) => { const v = qs('#' + id).value; return v === '' ? null : Number(v); };
      const str = (id) => qs('#' + id).value || null;
      const env = loadEnv() || {};
      const ctx = {
        env_mode: env.env_mode || null,
        temperature_c: env.temperature_c ?? null, humidity_pct: env.humidity_pct ?? null,
        uv_index: env.uv_index ?? null, air_quality_aqi: env.air_quality_aqi ?? null,
        place_name: env.place_name || null, latitude: env.latitude ?? null, longitude: env.longitude ?? null,
        age: num('q-age'), stress_level: str('q-stress'), sleep_hours: num('q-sleep'), sun_exposure: str('q-sun'),
      };
      Object.keys(ctx).forEach((k) => { if (ctx[k] === null || ctx[k] === undefined) delete ctx[k]; });
      return Object.keys(ctx).length ? ctx : null;
    }
  }

  // ---- environment picker (shared by Scan + Settings) ----
  function bindEnvPicker(current, onChange) {
    const body = qs('#env-body'); const modes = qs('#env-modes');
    let mode = (current && current.env_mode) || 'current_location';
    const paint = () => {
      qsa('.env-mode-btn', modes).forEach((b) => b.classList.toggle('active', b.dataset.m === mode));
      if (mode === 'current_location') {
        body.innerHTML = `<button class="btn btn-soft btn-sm" id="geo-btn"><i class="fa-solid fa-location-crosshairs"></i> Detect my location</button>
          <div class="muted" id="geo-out" style="font-size:.84rem;margin-top:8px">${current && current.place_name ? 'Using: ' + esc(current.place_name) : 'We’ll use your browser location to fetch live weather (Open-Meteo).'}</div>`;
        qs('#geo-btn').onclick = () => {
          const out = qs('#geo-out'); out.textContent = 'Requesting location…';
          navigator.geolocation.getCurrentPosition(async (pos) => {
            try {
              const w = await API.weatherByCoords(pos.coords.latitude, pos.coords.longitude);
              const env = { env_mode: 'current_location', latitude: pos.coords.latitude, longitude: pos.coords.longitude,
                place_name: w.place_name, temperature_c: w.temperature_c, humidity_pct: w.humidity_pct,
                uv_index: w.uv_index, air_quality_aqi: w.air_quality_aqi };
              saveEnv(env); App.state.weather = w; App.paintWeatherChip();
              out.textContent = w.source === 'open-meteo'
                ? `Live: ${w.place_name || 'your location'} — ${w.temperature_c ?? '?'}°C, ${w.humidity_pct ?? '?'}% humidity`
                : (w.note || 'Weather service unavailable.');
              if (onChange) onChange(env);
            } catch (e) { out.textContent = e.message; }
          }, (err) => { out.textContent = 'Location denied or unavailable: ' + err.message; }, { timeout: 10000 });
        };
      } else if (mode === 'select_location') {
        body.innerHTML = `<div class="field"><label>Search city / place</label>
            <input type="text" id="place-q" placeholder="e.g. Mysuru, India" value="${esc((current && current.place_name) || '')}"></div>
          <div id="place-results" class="muted" style="font-size:.84rem"></div>`;
        let tmr;
        qs('#place-q').oninput = (e) => {
          clearTimeout(tmr); const q = e.target.value.trim(); if (q.length < 2) return;
          tmr = setTimeout(async () => {
            const box = qs('#place-results'); box.textContent = 'Searching…';
            try {
              const { results } = await API.geocode(q);
              box.innerHTML = results.length ? results.map((r, i) =>
                `<button class="btn btn-ghost btn-sm" style="margin:4px 4px 0 0" data-i="${i}">${esc(r.label)}</button>`).join('')
                : 'No matches.';
              qsa('button[data-i]', box).forEach((btn) => btn.onclick = async () => {
                const r = results[Number(btn.dataset.i)];
                box.textContent = 'Fetching weather…';
                try {
                  const w = await API.weatherByCoords(r.latitude, r.longitude);
                  const env = { env_mode: 'select_location', latitude: r.latitude, longitude: r.longitude, place_name: r.label,
                    temperature_c: w.temperature_c, humidity_pct: w.humidity_pct, uv_index: w.uv_index, air_quality_aqi: w.air_quality_aqi };
                  saveEnv(env); App.state.weather = w; App.paintWeatherChip();
                  box.textContent = `Set: ${r.label} — ${w.temperature_c ?? '?'}°C`;
                  if (onChange) onChange(env);
                } catch (e) { box.textContent = e.message; }
              });
            } catch (e) { box.textContent = e.message; }
          }, 400);
        };
      } else {
        const c = current || {};
        body.innerHTML = `<div class="grid-2" style="gap:10px">
          <div class="field"><label>Temperature (°C)</label><input type="number" id="m-temp" value="${c.temperature_c ?? ''}" placeholder="unknown"></div>
          <div class="field"><label>Humidity (%)</label><input type="number" id="m-hum" min="0" max="100" value="${c.humidity_pct ?? ''}" placeholder="unknown"></div>
          <div class="field"><label>UV index</label><input type="number" id="m-uv" min="0" max="15" step="0.1" value="${c.uv_index ?? ''}" placeholder="unknown"></div>
          <div class="field"><label>Air quality (AQI)</label><input type="number" id="m-aqi" min="0" value="${c.air_quality_aqi ?? ''}" placeholder="unknown"></div>
        </div>
        <button class="btn btn-soft btn-sm" id="m-save">Save manual values</button>`;
        qs('#m-save').onclick = () => {
          const n = (id) => { const v = qs('#' + id).value; return v === '' ? null : Number(v); };
          const env = { env_mode: 'manual', temperature_c: n('m-temp'), humidity_pct: n('m-hum'), uv_index: n('m-uv'), air_quality_aqi: n('m-aqi'), place_name: null, latitude: null, longitude: null };
          saveEnv(env); App.state.weather = { source: 'manual', ...env }; App.paintWeatherChip();
          toast('Environmental values saved.'); if (onChange) onChange(env);
        };
      }
    };
    qsa('.env-mode-btn', modes).forEach((b) => b.onclick = () => { mode = b.dataset.m; paint(); });
    paint();
  }

  // ============================================================ RESULT
  async function viewResult(id, nav) {
    const view = qs('#view');
    view.innerHTML = `<div class="empty"><i class="fa-solid fa-spinner fa-spin"></i>Loading analysis…</div>`;
    let rec = null, results = null, imageId = null, createdAt = null, modelVersion = null;

    if (App.state.lastAnalysis && String(App.state.lastAnalysis.id) === String(id)) {
      const a = App.state.lastAnalysis;
      results = a.results; rec = a.recommendation; imageId = a.image_id; createdAt = a.created_at; modelVersion = a.model_version;
    } else {
      const hist = App.state.historyCache || await API.history();
      if (nav != null && !App.isCurrent(nav)) return;
      App.state.historyCache = hist;
      const row = hist.find((r) => String(r.id) === String(id));
      if (!row) { view.innerHTML = `<div class="card card-pad empty"><i class="fa-solid fa-triangle-exclamation"></i>That scan was not found.</div>`; return; }
      results = row.results; rec = row.recommendation; imageId = row.image_id; createdAt = row.created_at; modelVersion = row.model_version;
    }

    // Keep the Plan / Products screens in sync with whatever scan is on screen.
    if (rec) App.state.lastRecommendation = rec;

    const shi = shiOf(results);
    const conds = (results && results.conditions) || {};
    const { date, time } = fmtDateTime(createdAt);
    const legacy = !results || !results.conditions;

    if (legacy) {
      view.innerHTML = `<div class="view-head"><h1>Analysis · ${esc(date)}</h1><p class="muted">Legacy record (${esc(modelVersion || 'older model')}).</p></div>
        <div class="card card-pad"><pre style="white-space:pre-wrap;font-size:.82rem">${esc(JSON.stringify(results, null, 2))}</pre></div>`;
      return;
    }

    const shiCol = shi.value == null ? 'var(--text-muted)' : shi.value >= 65 ? 'var(--good)' : shi.value >= 50 ? 'var(--moderate)' : 'var(--attention)';
    const q = results.quality || null;
    const qBadgeCls = !q ? '' : q.score >= 6 ? 'good' : q.score >= 4 ? 'moderate' : 'attention';
    const qBadge = q ? `<span class="pill ${qBadgeCls}" title="${esc(q.message || '')}">Photo quality ${esc(String(q.score))}/10${q.label ? ' · ' + esc(q.label) : ''}</span>` : '';
    view.innerHTML = `
      <div class="view-head"><h1>Analysis Results</h1><p class="muted">${esc(date)} · ${esc(time)} · model set <code>${esc(results.analysis_version || modelVersion)}</code> ${qBadge}</p>
      ${q && q.is_suboptimal ? `<p class="muted" style="font-size:.82rem;margin-top:4px"><i class="fa-solid fa-circle-info"></i> ${esc(q.message || '')}</p>` : ''}</div>

      <div class="grid-3">
        <div class="card card-pad">
          <div class="card-title-row"><h3>Image inspector</h3>
            <span style="display:flex;align-items:center;gap:8px">
              ${results.face_detected ? '<span class="pill good">Face detected</span>' : '<span class="pill neutral">No face box</span>'}
              <button class="insp-expand" id="insp-expand" title="Enlarge inspector" aria-label="Enlarge inspector"><i class="fa-solid fa-up-right-and-down-left-from-center"></i></button>
            </span>
          </div>
          ${inspectorMarkup()}
          <div class="legend">
            <span class="li"><span class="pill ai" style="padding:2px 8px">AI</span> model result</span>
            <span class="li"><span class="pill estimate" style="padding:2px 8px">Est.</span> computer-vision estimate</span>
            <span class="li"><span class="pill na" style="padding:2px 8px">—</span> not supported</span>
          </div>
        </div>
        <div class="card card-pad shi">
          <div class="card-title-row"><h3>Skin Health Index</h3></div>
          ${shi.value == null
            ? `<div class="shi-gauge"><div class="val"><b>—</b></div></div><div class="band not_enough_data">Not enough data</div>
               <div class="hint">No model-backed condition produced a usable result for this image.</div>`
            : `<div class="shi-gauge">
                 <svg viewBox="0 0 168 168"><circle cx="84" cy="84" r="76" stroke="var(--bg-tint)" stroke-width="12" fill="none"></circle>
                 <circle cx="84" cy="84" r="76" stroke="${shiCol}" stroke-width="12" fill="none" stroke-linecap="round"
                   stroke-dasharray="${2 * Math.PI * 76}" stroke-dashoffset="${2 * Math.PI * 76 - (shi.value / 100) * 2 * Math.PI * 76}"></circle></svg>
                 <div class="val"><b>${shi.value}</b><small>/100</small></div></div>
               <div class="band ${shi.band}">${cap(shi.band.replace('_', ' '))}</div>
               <div class="hint" style="text-align:left"><span style="font-size:.8rem">Contributing: ${esc((shi.basis || []).join(', ') || 'none')}</span></div>
               ${!results.xai_insights ? `<div class="hint" style="text-align:left;margin-top:10px"><b>How this is calculated</b><br><span style="font-size:.8rem">${esc(shi.formula || 'Deterministic formula from the model-backed and estimated conditions.')}</span></div>` : `<div class="hint" style="text-align:left;margin-top:6px"><span style="font-size:.8rem">See <b>Why your score</b> below for the plain-language breakdown.</span></div>`}`}
        </div>
        <div class="card card-pad">
          <div class="card-title-row"><h3>Summary</h3></div>
          ${rec && rec.summary && rec.summary.length
            ? `<ul style="padding-left:18px;font-size:.9rem;color:var(--text-2)">${rec.summary.map((s) => `<li style="margin-bottom:6px">${esc(s)}</li>`).join('')}</ul>`
            : `<p class="muted">Open a live scan for a full written summary and routine — history records store the analysis but not the generated text.</p>`}
          ${rec && rec.disclaimer ? `<p class="muted" style="font-size:.78rem;margin-top:10px">${esc(rec.disclaimer)}</p>` : ''}
        </div>
      </div>

      ${xaiInsightsBlock(results.xai_insights, shi)}

      <div class="card card-pad mt-18">
        <div class="card-title-row"><h3>Condition breakdown</h3></div>
        <div class="cond-grid">${CFG.CONDITIONS.map((c) => condCard(c, conds[c.key])).join('')}</div>
      </div>

      ${rec ? recommendationBlock(rec) : ''}

      <div style="margin-top:18px;display:flex;gap:10px;flex-wrap:wrap">
        <button class="btn btn-ghost btn-sm" onclick="location.hash='#/history'">Back to history</button>
        <button class="btn btn-primary btn-sm" onclick="location.hash='#/scan'">New scan</button>
        ${rec ? `<button class="btn btn-soft btn-sm" onclick="location.hash='#/plan'">See my plan</button>` : ''}
      </div>`;

    buildInspector(qs('.inspector'), imageId, results);
    const exp = qs('#insp-expand');
    if (exp) exp.onclick = () => openInspectorModal(imageId, results);
  }

  // Plain-language "why this score" card from the backend's xai_insights.
  function xaiInsightsBlock(xai, shi) {
    if (!xai || !xai.condition_rationales) return '';
    const drv = xai.shi_drivers || {};
    const drivers = Array.isArray(drv.primary_drivers) ? drv.primary_drivers : [];
    const rats = xai.condition_rationales || {};
    const CN = (k) => (CFG.CONDITIONS.find((c) => c.key === k) || { label: cap(k.replace('_', ' ')) }).label;

    const driverList = drivers.length
      ? `<ul class="xai-drivers">${drivers.map((x) => `
          <li>
            <div class="xai-dh"><b>${esc(CN(x.condition))}</b>
              <span class="pill ${x.model_backed ? 'ai' : 'estimate'}" style="padding:1px 7px">${x.model_backed ? 'model' : 'estimate'}</span>
              <span class="xai-share">−${esc(String(x.penalty_points))} pts${x.share_pct != null ? ' · ' + esc(String(x.share_pct)) + '% of the drop' : ''}</span>
            </div>
            <div class="xai-rat">${esc(x.rationale || '')}</div>
          </li>`).join('')}</ul>`
      : `<p class="muted" style="font-size:.88rem">Nothing is currently pulling your Skin Health Index down.</p>`;

    const notable = Object.entries(rats)
      .filter(([, v]) => v && (v.status === 'detected' || v.status === 'estimated') && v.observation)
      .map(([k, v]) => `<div class="xai-obs"><b>${esc(CN(k))}${v.severity ? ' — ' + esc(String(v.severity)) : ''}</b><span>${esc(v.observation)}</span></div>`)
      .join('');

    return `<div class="card card-pad mt-18 xai-card">
      <div class="card-title-row"><h3>Why your score${shi && shi.value != null ? ' is ' + shi.value : ''}</h3>
        <span class="pill neutral">Explainable AI</span></div>
      <p class="xai-lead">${esc(drv.note || 'Plain-language notes on what the models and the computer-vision engine measured.')}</p>
      ${driverList}
      ${notable ? `<div class="card-title-row" style="margin-top:16px"><h3 style="font-size:.95rem">What each check found</h3></div>
        <div class="xai-obs-list">${notable}</div>` : ''}
      ${shi && shi.formula ? `<details class="xai-formula"><summary>How the score is computed</summary>
        <p class="muted" style="font-size:.8rem;margin:8px 0 0">${esc(shi.formula)}</p></details>` : ''}
      ${xai.method_note ? `<p class="muted" style="font-size:.76rem;margin-top:10px">${esc(xai.method_note)}</p>` : ''}
    </div>`;
  }

  // A 0-1 severity number for a condition entry (null when it has no numeric score).
  function condScore(entry) {
    return entry && typeof entry.score === 'number' ? entry.score : null;
  }
  function scoreBand(s) {
    return s >= 0.67 ? ['High', 'attention'] : s >= 0.34 ? ['Moderate', 'moderate'] : ['Low', 'good'];
  }

  function condCard({ key, label, tier, icon }, entry) {
    const badge = TIER_BADGE[tier];
    const na = tier === 'unavailable' || !entry || entry.status === 'not_available';
    if (na) return `<div class="cond-card na">
      <div class="ch"><span class="cn"><i class="fa-solid ${icon}"></i> ${label}</span><span class="pill na">Not supported</span></div>
      <div class="cl">Not currently supported — no validated model or dataset for this condition on facial photos.</div></div>`;
    if (entry.status === 'error') return `<div class="cond-card">
      <div class="ch"><span class="cn"><i class="fa-solid ${icon}"></i> ${label}</span><span class="pill attention">Error</span></div>
      <div class="cl">The model for this condition failed on this image.</div></div>`;
    const isCv = tier === 'cv';
    const est = !isCv && entry.status === 'estimated';
    const s = condScore(entry);
    const pct = s != null ? Math.round(s * 100) : null;
    const pillCls = isCv ? 'cv' : est ? 'estimate' : 'ai';
    const pillTxt = isCv ? 'CV scan' : est ? 'Estimate' : 'AI';
    const confWord = isCv ? 'analysis' : 'model';
    const [bandLbl, bandCls] = s != null ? scoreBand(s) : ['', ''];
    return `<div class="cond-card ${isCv ? 'cv' : est ? 'est' : ''}">
      <div class="ch"><span class="cn"><i class="fa-solid ${icon}"></i> ${label}</span>
        <span class="pill ${pillCls}">${pillTxt}</span></div>
      <div class="cl">${esc(entry.label || '—')}</div>
      ${pct != null ? `<div class="cbar-wrap" title="Severity 0.00 (none) to 1.00 (most severe)">
          <div class="cbar"><span style="width:${pct}%"></span></div>
          <span class="cbar-num ${bandCls}">${s.toFixed(2)}<small>/1</small></span>
        </div>
        <div class="cmeta">Severity ${s.toFixed(2)} · ${bandLbl}${entry.confidence != null ? ` · ${confWord} confidence ${Math.round(entry.confidence * 100)}%` : ''}</div>`
        : entry.confidence != null ? `<div class="cmeta">${confWord} confidence ${Math.round(entry.confidence * 100)}%</div>`
        : est ? `<div class="cmeta">${esc(badge.text)} — no confidence score</div>` : ''}
    </div>`;
  }

  function recommendationBlock(rec) {
    const list = (arr, icon) => arr && arr.length
      ? arr.map((s) => `<div class="step"><span class="sx"><i class="fa-solid ${icon}"></i></span><div class="sbody"><b>${esc(s)}</b></div></div>`).join('')
      : `<p class="muted">Nothing specific for this result.</p>`;
    const per = rec.per_condition || {};
    return `
    ${barrierRepairCard(rec)}
    ${safetyBanner(rec)}
    <div class="grid-2 mt-18">
      <div class="card card-pad">
        <div class="card-title-row"><h3>Suggested routine</h3></div>${list(rec.routine, 'fa-check')}
        <div class="card-title-row" style="margin-top:16px"><h3>Ingredients to look for</h3></div>${list(rec.ingredients, 'fa-flask')}
      </div>
      <div class="card card-pad">
        <div class="card-title-row"><h3>Per-condition notes</h3></div>
        ${Object.keys(per).length ? Object.entries(per).map(([k, v]) =>
          `<div class="step"><div class="sbody"><b>${esc((CFG.CONDITIONS.find((c) => c.key === k) || { label: k }).label)}</b><span>${esc((v || []).join(' '))}</span></div></div>`).join('') : '<p class="muted">—</p>'}
        ${(rec.context_notes || []).length ? `<div class="card-title-row" style="margin-top:16px"><h3>Environment &amp; lifestyle</h3></div>
          ${rec.context_notes.map((s) => `<div class="step"><span class="sx"><i class="fa-solid fa-cloud-sun"></i></span><div class="sbody"><span>${esc(s)}</span></div></div>`).join('')}` : ''}
      </div>
    </div>`;
  }

  // Multi-layer image inspector: Original / AI Focus (Grad-CAM) / Analysis Zones.
  // Class-scoped so it can render both inline and inside the "enlarge" modal.
  function inspectorMarkup(opts) {
    const big = opts && opts.big ? ' inspector-big' : '';
    return `<div class="inspector${big}">
      <div class="insp-stage">
        <img class="insp-base" alt="Analysed photo">
        <img class="insp-layer" alt="" hidden>
        <div class="insp-status" hidden></div>
      </div>
      <div class="layer-btns">
        <button class="lb active" data-layer="original"><i class="fa-solid fa-image"></i> Original</button>
        <button class="lb" data-layer="gradcam"><i class="fa-solid fa-fire"></i> AI Focus</button>
        <button class="lb" data-layer="zones"><i class="fa-solid fa-draw-polygon"></i> Analysis Zones</button>
      </div>
      <div class="insp-sub insp-sub-gradcam" hidden>
        <div class="insp-heads">
          <button class="hpill active" data-head="acne">Acne</button>
          <button class="hpill" data-head="wrinkles">Wrinkles</button>
        </div>
        <label class="insp-blend">Blend
          <input type="range" class="gc-opacity" min="0" max="100" value="60">
          <span class="gc-opacity-val">60%</span>
        </label>
        <p class="insp-tip"><i class="fa-solid fa-circle-info"></i> Visualizes the high-activation skin patterns influencing the neural network's decision.</p>
      </div>
      <div class="insp-sub insp-sub-zones" hidden>
        <div class="zones-legend">
          <span><i class="zdot" style="background:#eb5a5a"></i> Redness / erythema &mdash; full skin region</span>
          <span><i class="zdot" style="background:#3c96eb"></i> Dark spots &amp; pigmentation &mdash; cheeks + forehead</span>
          <span><i class="zdot" style="background:#3ec89a"></i> Pores &amp; blackheads &mdash; T-zone, nasal wings, chin</span>
        </div>
      </div>
    </div>`;
  }

  async function buildInspector(root, imageId, results) {
    if (!root) return;
    const base = qs('.insp-base', root), layer = qs('.insp-layer', root), statusEl = qs('.insp-status', root);
    const opac = qs('.gc-opacity', root), opacVal = qs('.gc-opacity-val', root);
    const cache = {};                 // "mode[:head]" -> objectURL
    let curLayer = 'original', curHead = 'acne';

    const setStatus = (msg, err) => {
      if (!statusEl) return;
      if (!msg) { statusEl.hidden = true; return; }
      statusEl.hidden = false; statusEl.textContent = msg;
      statusEl.classList.toggle('err', !!err);
    };

    try { base.src = App.trackBlob(await API.imageBlobUrl(imageId)); }
    catch (_) { setStatus('Could not load the analysed photo.', true); return; }

    async function overlayUrl(mode, head) {
      const key = mode + (mode === 'gradcam' ? ':' + head : '');
      if (cache[key]) return cache[key];
      setStatus(mode === 'gradcam' ? 'Computing Grad-CAM heatmap…' : 'Building the measurement-zone overlay…');
      layer.classList.add('loading');
      try {
        const url = App.trackBlob(await API.overlayBlobUrl(imageId, mode, head));
        cache[key] = url; setStatus(null); return url;
      } finally { layer.classList.remove('loading'); }
    }

    function applyOpacity() {
      layer.style.opacity = curLayer === 'gradcam' ? String(Number(opac.value) / 100)
        : curLayer === 'zones' ? '0.9' : '0';
    }

    async function showOverlay(mode) {
      let url;
      try { url = await overlayUrl(mode, curHead); }
      catch (ex) { layer.hidden = true; setStatus(ex.message || 'Overlay unavailable.', true); return; }
      layer.src = url; layer.hidden = false; applyOpacity();
    }

    async function setLayer(l) {
      curLayer = l;
      qsa('.lb', root).forEach((b) => b.classList.toggle('active', b.dataset.layer === l));
      qs('.insp-sub-gradcam', root).hidden = l !== 'gradcam';
      qs('.insp-sub-zones', root).hidden = l !== 'zones';
      if (l === 'original') { layer.hidden = true; setStatus(null); return; }
      await showOverlay(l === 'gradcam' ? 'gradcam' : 'zones');
    }

    qsa('.lb', root).forEach((b) => b.onclick = () => setLayer(b.dataset.layer));
    qsa('.hpill', root).forEach((p) => p.onclick = async () => {
      curHead = p.dataset.head;
      qsa('.hpill', root).forEach((x) => x.classList.toggle('active', x === p));
      if (curLayer === 'gradcam') await showOverlay('gradcam');
    });
    if (opac) opac.oninput = () => { opacVal.textContent = opac.value + '%'; applyOpacity(); };
  }

  function openInspectorModal(imageId, results) {
    const faceTag = results && results.face_detected
      ? '<span class="pill good">Face detected</span>' : '<span class="pill neutral">No face box</span>';
    const inner = h(`<div class="si-imod">
      <div class="si-modal-head"><h3><i class="fa-solid fa-magnifying-glass-plus"></i> Image inspector ${faceTag}</h3>
        <button class="si-x" aria-label="Close">&times;</button></div>
      <div class="si-imod-body">${inspectorMarkup({ big: true })}</div>
      <div class="si-modal-foot">
        <span class="muted" style="font-size:.78rem;margin-right:auto">Toggle a layer, then adjust the Grad-CAM blend or read the zone legend.</span>
        <button class="btn btn-ghost btn-sm si-x2">Close</button>
      </div>
    </div>`);
    inner.querySelectorAll('.si-x, .si-x2').forEach((b) => b.onclick = closeModal);
    openModal(inner, { xl: true });
    buildInspector(inner.querySelector('.inspector'), imageId, results);
  }

  // ============================================================ HISTORY
  // SVG trend line of the Skin Health Index across every past scan that produced one.
  function shiTrendChart(hist) {
    const pts = (hist || [])
      .filter((r) => r.results && r.results.skin_health_index && r.results.skin_health_index.value != null)
      .map((r) => ({ t: new Date(r.created_at).getTime(), v: r.results.skin_health_index.value }))
      .sort((a, b) => a.t - b.t);
    if (!pts.length) return '';
    const w = 720, h = 210, pl = 40, pr = 16, ptop = 16, pbot = 34;
    const iw = w - pl - pr, ih = h - ptop - pbot;
    const t0 = pts[0].t, t1 = pts[pts.length - 1].t;
    const X = (t) => pl + (t1 === t0 ? iw / 2 : ((t - t0) / (t1 - t0)) * iw);
    const Y = (v) => ptop + ih - (Math.max(0, Math.min(100, v)) / 100) * ih;
    const grid = [0, 50, 65, 80, 100].map((g) =>
      `<line x1="${pl}" y1="${Y(g).toFixed(1)}" x2="${w - pr}" y2="${Y(g).toFixed(1)}" class="tg-grid"></line>
       <text x="${pl - 7}" y="${(Y(g) + 3.5).toFixed(1)}" class="tg-ylab">${g}</text>`).join('');
    const last = pts[pts.length - 1].v;
    const col = last >= 65 ? 'var(--good)' : last >= 50 ? 'var(--moderate)' : 'var(--attention)';
    const poly = pts.length > 1
      ? `<polyline points="${pts.map((p) => X(p.t).toFixed(1) + ',' + Y(p.v).toFixed(1)).join(' ')}" class="tg-line" style="stroke:${col}"></polyline>` : '';
    const dots = pts.map((p) =>
      `<circle cx="${X(p.t).toFixed(1)}" cy="${Y(p.v).toFixed(1)}" r="4.5" class="tg-dot" style="fill:${col}">
        <title>${esc(new Date(p.t).toLocaleDateString())} — SHI ${p.v}/100</title></circle>`).join('');
    const fmt = (t) => new Date(t).toLocaleDateString(undefined, { day: '2-digit', month: 'short' });
    const xlab = `<text x="${pl}" y="${h - 12}" class="tg-xlab">${esc(fmt(t0))}</text>`
      + (t1 !== t0 ? `<text x="${w - pr}" y="${h - 12}" class="tg-xlab" text-anchor="end">${esc(fmt(t1))}</text>` : '');
    return `<div class="card card-pad trend-card">
      <div class="card-title-row"><h3>Skin Health Index over time</h3>
        <span class="pill ${last >= 65 ? 'good' : last >= 50 ? 'moderate' : 'attention'}">now ${last}/100</span></div>
      <svg class="trend-svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Skin Health Index trend">
        ${grid}${poly}${dots}${xlab}
      </svg>
      <p class="muted" style="font-size:.78rem;margin-top:4px">${pts.length} scan${pts.length === 1 ? '' : 's'} with a Skin Health Index. Hover a point for its date and score.</p>
    </div>`;
  }

  async function viewHistory(nav) {
    const view = qs('#view');
    view.innerHTML = `<div class="view-head"><h1>Scan History</h1><p>Every scan you run while logged in is saved here.</p></div>
      <div id="trend-slot"></div>
      <div class="card card-pad"><div id="hlist"><div class="empty"><i class="fa-solid fa-spinner fa-spin"></i>Loading…</div></div></div>`;
    let hist;
    try { hist = await API.history(); App.state.historyCache = hist; }
    catch (e) { if (nav != null && !App.isCurrent(nav)) return; qs('#hlist').innerHTML = `<div class="empty"><i class="fa-solid fa-triangle-exclamation"></i>${esc(e.message)}</div>`; return; }
    if (nav != null && !App.isCurrent(nav)) return;
    if (!hist.length) { qs('#hlist').innerHTML = `<div class="empty"><i class="fa-solid fa-image"></i>No scans yet.<br><button class="linklike" onclick="location.hash='#/scan'">Run your first scan</button></div>`; return; }

    qs('#trend-slot').innerHTML = shiTrendChart(hist);
    qs('#hlist').className = 'hist-list';
    qs('#hlist').innerHTML = hist.map((r) => {
      const { date, time } = fmtDateTime(r.created_at);
      const shi = shiOf(r.results);
      const concern = r.results && r.results.conditions ? mainConcern(r.results) : 'Legacy record';
      return `<div class="hist-row">
        <img class="hthumb" data-img="${r.image_id}" alt="">
        <div class="hmain"><b>${esc(concern)}</b><span>${esc(date)} · ${esc(time)} · <code>${esc(r.model_version || '')}</code></span></div>
        <div class="hscore"><b>${shi.value == null ? '—' : shi.value}</b><span>SHI</span></div>
        <div class="hact"><button class="btn btn-ghost btn-sm" onclick="location.hash='#/result/${r.id}'">View</button></div>
      </div>`;
    }).join('');
    hydrateImages(qs('#hlist'));
  }

  // ============================================================ PLAN
  // Short human labels for the backend's contraindication rule ids.
  const SAFETY_LABEL = {
    retinoid_x_exfoliant_acid: 'Alternate nights',
    vitamin_c_x_benzoyl_peroxide: 'Separate AM / PM',
    photosensitisers_pm_only: 'PM only',
    spf50_mandatory_with_photosensitisers: 'SPF 50 required',
    separate_bpo_and_retinoid: 'Separate benzoyl peroxide & retinoid',
    introduce_actives_gradually: 'Introduce gradually',
    avoid_high_strength_acids_with_erythema: 'Reactive skin',
    barrier_repair_protocol: 'Barrier repair',
  };

  // Amber safety banner from recommendation.contraindications (+ photosensitiser list).
  function safetyBanner(rec) {
    if (!rec) return '';
    const flags = (rec.contraindications || []).filter((f) => f && f.message);
    const ps = rec.photosensitisers_pm_only || [];
    if (!flags.length && !ps.length) return '';
    const rows = flags.map((f) => {
      const tag = SAFETY_LABEL[f.rule] || SAFETY_LABEL[f.action] || (f.severity === 'high' ? 'Important' : 'Note');
      return `<li class="sf-${esc(f.severity || 'info')}"><b>${esc(tag)}:</b> ${esc(f.message)}</li>`;
    });
    if (ps.length && !flags.some((f) => f.rule === 'photosensitisers_pm_only')) {
      rows.push(`<li class="sf-info"><b>Photosensitive → PM:</b> ${esc(ps.join(', '))} scheduled for the evening routine only.</li>`);
    }
    return `<div class="card card-pad safety-banner mt-18">
      <div class="card-title-row"><h3><i class="fa-solid fa-triangle-exclamation"></i> Safety &amp; contraindications</h3></div>
      <ul class="safety-list">${rows.join('')}</ul>
    </div>`;
  }

  // Dedicated Barrier Recovery card when recommendation.barrier_repair fired.
  function barrierRepairCard(rec) {
    const br = rec && rec.barrier_repair;
    if (!br) return '';
    const ings = (br.ingredients || []).map((i) => `<span class="pill good" style="margin:3px 4px 0 0">${esc(i)}</span>`).join('');
    const steps = (br.protocol || []).map((s, i) => `<div class="step"><span class="sx">${i + 1}</span><div class="sbody"><span>${esc(s)}</span></div></div>`).join('');
    return `<div class="card card-pad barrier-card mt-18">
      <div class="card-title-row"><h3><i class="fa-solid fa-shield-heart"></i> Barrier Recovery protocol</h3><span class="pill attention">Priority</span></div>
      <p class="muted" style="font-size:.88rem;margin:2px 0 12px">${esc(br.reason || 'Your skin barrier looks compromised — pause active ingredients and focus on repair.')}</p>
      ${steps}
      ${ings ? `<div style="margin-top:12px"><b style="font-size:.9rem">Prioritise:</b><br>${ings}</div>` : ''}
    </div>`;
  }

  function routineColumn(title, icon, steps, prefix, dp) {
    if (!steps || !steps.length) {
      return `<div class="routine-col"><h4><i class="fa-solid ${icon}"></i> ${title}</h4>
        <p class="muted" style="font-size:.85rem">Structured steps aren't stored for this record — see the routine list on the analysis page.</p></div>`;
    }
    return `<div class="routine-col"><h4><i class="fa-solid ${icon}"></i> ${title}</h4>
      ${steps.map((s, i) => {
        const id = prefix + i; const isDone = !!dp[id];
        const actives = (s.actives || []).map((a) => `<span class="ractive">${esc(a)}</span>`).join('');
        return `<div class="step ${isDone ? 'done' : ''}"><span class="sx">${i + 1}</span>
          <div class="sbody"><b>${esc(s.step || 'Step')}</b>
            ${s.detail ? `<span>${esc(s.detail)}</span>` : ''}
            ${actives ? `<div class="ractives">${actives}</div>` : ''}</div>
          <label class="schk"><input type="checkbox" data-step="${id}" ${isDone ? 'checked' : ''}></label></div>`;
      }).join('')}</div>`;
  }

  async function viewPlan(nav) {
    const view = qs('#view');
    let rec = App.state.lastRecommendation;
    if (!rec) {
      let hist;
      try { hist = App.state.historyCache || await API.history(); } catch (e) { hist = []; }
      if (nav != null && !App.isCurrent(nav)) return;
      App.state.historyCache = hist;
      const withRec = hist.find((r) => r.recommendation);
      rec = withRec ? withRec.recommendation : null;
    }
    // Consume the backend's structured routines directly — no regex splitting.
    const am = (rec && rec.am_routine) || [];
    const pm = (rec && rec.pm_routine) || [];
    const progress = planProgress();
    const dayKey = new Date().toISOString().slice(0, 10);
    const dp = progress[dayKey] || {};
    const targeted = rec && rec.per_condition ? Object.keys(rec.per_condition)
      .filter((k) => k !== 'barrier_repair')
      .map((k) => (CFG.CONDITIONS.find((c) => c.key === k) || { label: cap(k.replace('_', ' ')) }).label) : [];

    const total = am.length + pm.length;
    const done = am.map((_, i) => 'am' + i).concat(pm.map((_, i) => 'pm' + i)).filter((id) => dp[id]).length;

    view.innerHTML = `
      <div class="view-head"><h1>Current Plan &amp; Procedure</h1><p>Skincare guidance generated from your latest analysis and context — not medical advice.</p></div>
      ${!rec ? `<div class="card card-pad empty"><i class="fa-solid fa-clipboard-question"></i>No plan yet. <button class="linklike" onclick="location.hash='#/scan'">Run a scan</button> to generate one.</div>` : `
      ${barrierRepairCard(rec)}
      ${safetyBanner(rec)}
      <div class="card card-pad mt-18">
        <div class="card-title-row"><h3>Today's progress</h3><span class="pill ${total && done === total ? 'good' : 'neutral'}">${done}/${total} steps</span></div>
        <div class="progress-track"><span style="width:${total ? (done / total) * 100 : 0}%"></span></div>
      </div>
      <div class="grid-2 mt-18">
        <div class="card card-pad">${routineColumn('Morning', 'fa-sun', am, 'am', dp)}</div>
        <div class="card card-pad">${routineColumn('Evening', 'fa-moon', pm, 'pm', dp)}</div>
      </div>
      ${rec.spf && rec.spf.reason ? `<div class="card card-pad mt-18 spf-note"><div class="card-title-row"><h3><i class="fa-solid fa-umbrella-beach"></i> Sunscreen</h3><span class="pill ${rec.spf.required ? 'attention' : 'good'}">${esc(rec.spf.product || 'SPF 30+')}</span></div><p class="muted" style="font-size:.88rem;margin:2px 0 0">${esc(rec.spf.reason)}</p></div>` : ''}
      <div class="grid-2 mt-18">
        <div class="card card-pad"><div class="card-title-row"><h3>Concerns being targeted</h3></div>
          ${targeted.length ? targeted.map((t) => `<span class="pill" style="margin:3px 4px 0 0">${esc(t)}</span>`).join('') : '<p class="muted">General maintenance.</p>'}</div>
        <div class="card card-pad"><div class="card-title-row"><h3>Personalised instructions</h3></div>
          ${(rec.summary || []).map((s) => `<div class="step"><div class="sbody"><span>${esc(s)}</span></div></div>`).join('') || '<p class="muted">—</p>'}
          ${rec.disclaimer ? `<p class="muted" style="font-size:.78rem;margin-top:8px">${esc(rec.disclaimer)}</p>` : ''}</div>
      </div>`}`;

    qsa('input[data-step]').forEach((cb) => cb.onchange = () => {
      const p = planProgress(); p[dayKey] = p[dayKey] || {}; p[dayKey][cb.dataset.step] = cb.checked;
      savePlanProgress(p); viewPlan();
    });
  }

  // ============================================================ PRODUCTS
  const INGREDIENT_PRODUCTS = {
    'Salicylic acid': ['BHA exfoliant / cleanser', 'fa-vial', 'Unclogs pores; helps acne & texture'],
    'Salicylic acid (BHA)': ['BHA exfoliant', 'fa-vial', 'Unclogs pores; helps acne & texture'],
    'Benzoyl peroxide (2.5%)': ['Benzoyl peroxide spot gel', 'fa-tube', 'Targets acne-causing bacteria'],
    'Niacinamide': ['Niacinamide serum', 'fa-droplet', 'Oil balance, pores, redness'],
    'Adapalene (OTC retinoid)': ['Adapalene 0.1% gel', 'fa-prescription-bottle', 'OTC retinoid for acne & texture'],
    'Adapalene': ['Adapalene 0.1% gel', 'fa-prescription-bottle', 'OTC retinoid for acne & texture'],
    'Azelaic acid': ['Azelaic acid 10%', 'fa-flask', 'Redness, acne, uneven tone'],
    'Retinol': ['Low-strength retinol serum', 'fa-prescription-bottle-medical', 'Fine lines & texture (build up slowly)'],
    'Retinol / retinaldehyde': ['Retinaldehyde serum', 'fa-prescription-bottle-medical', 'Fine lines & texture'],
    'Peptides': ['Peptide moisturiser', 'fa-layer-group', 'Supports firmness'],
    'Hyaluronic acid': ['Hyaluronic acid serum', 'fa-droplet', 'Lightweight hydration'],
    'Ceramides': ['Ceramide moisturiser', 'fa-shield-halved', 'Barrier repair, dryness'],
    'Squalane': ['Squalane oil', 'fa-oil-can', 'Seals in moisture'],
    'Glycerin': ['Glycerin humectant', 'fa-droplet', 'Draws water into skin'],
    'Vitamin C (AM)': ['Vitamin C serum (AM)', 'fa-lemon', 'Antioxidant, brightening'],
    'Vitamin C': ['Vitamin C serum (AM)', 'fa-lemon', 'Antioxidant, brightening'],
    'Vitamin E': ['Vitamin E (with C)', 'fa-seedling', 'Antioxidant support'],
    'Centella asiatica': ['Centella (cica) serum', 'fa-spa', 'Calms redness & irritation'],
  };
  async function viewProducts(nav) {
    const view = qs('#view');
    let rec = App.state.lastRecommendation;
    if (!rec) {
      let hist;
      try { hist = App.state.historyCache || await API.history(); } catch (e) { hist = []; }
      if (nav != null && !App.isCurrent(nav)) return;
      App.state.historyCache = hist;
      const w = hist.find((r) => r.recommendation); rec = w ? w.recommendation : null;
    }
    const ings = (rec && rec.ingredients) || [];
    view.innerHTML = `
      <div class="view-head"><h1>Product Recommendations</h1><p>Derived from your skin type, detected conditions and environmental context. Ingredient guidance — not clinical product endorsements.</p></div>
      ${!ings.length ? `<div class="card card-pad empty"><i class="fa-solid fa-bag-shopping"></i>No recommendations yet. <button class="linklike" onclick="location.hash='#/scan'">Run a scan</button>.</div>` : `
      <div class="prod-grid">
        ${ings.map((ing) => {
          const p = INGREDIENT_PRODUCTS[ing] || [ing, 'fa-flask', 'Suggested for your result'];
          return `<div class="prod-card"><span class="pico"><i class="fa-solid ${p[1]}"></i></span>
            <b>${esc(p[0])}</b><div class="ptype">Key ingredient: ${esc(ing)}</div>
            <div class="pwhy">${esc(p[2])}</div></div>`;
        }).join('')}
      </div>
      <div class="card card-pad mt-18">
        <div class="card-title-row"><h3>How these were chosen</h3></div>
        <ul style="padding-left:18px;font-size:.9rem;color:var(--text-2)">
          ${(rec.summary || []).map((s) => `<li style="margin-bottom:6px">${esc(s)}</li>`).join('')}
        </ul>
        <p class="muted" style="font-size:.82rem;margin-top:10px">A branded product catalogue is a future-ready component — this build recommends ingredients, not specific SKUs, so nothing is fabricated. ${esc(rec.disclaimer || '')}</p>
      </div>`}`;
  }

  // ============================================================ PROFILE
  async function viewProfile(nav) {
    const view = qs('#view');
    view.innerHTML = `<div class="empty"><i class="fa-solid fa-spinner fa-spin"></i>Loading…</div>`;
    let me;
    try { me = await API.me(); App.state.user = me; } catch (e) { if (nav != null && !App.isCurrent(nav)) return; view.innerHTML = `<div class="card card-pad empty">${esc(e.message)}</div>`; return; }
    if (nav != null && !App.isCurrent(nav)) return;
    const ctx = me.latest_context || {};
    const kv = (k, v) => `<dt>${k}</dt><dd>${v == null || v === '' ? '<span class="muted">—</span>' : esc(v)}</dd>`;
    const created = me.created_at ? fmtDateTime(me.created_at) : null;
    view.innerHTML = `
      <div class="view-head"><h1>Profile</h1><p>Your account and the context stored with your most recent scan.</p></div>
      <div class="grid-2">
        <div class="card card-pad">
          <div class="card-title-row"><h3>Account</h3></div>
          <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px">
            <span class="avatar" style="width:52px;height:52px;font-size:1.1rem">${esc((me.full_name || me.email || 'S').split(/[ @]/).map((s) => s[0]).join('').slice(0, 2).toUpperCase())}</span>
            <div><b style="font-size:1.05rem">${esc(me.full_name || '—')}</b><div class="muted">${esc(me.email)}</div></div>
          </div>
          <dl class="kv">
            ${kv('Member since', created ? created.date : null)}
            ${kv('Scans run', me.scan_count)}
          </dl>
        </div>
        <div class="card card-pad">
          <div class="card-title-row"><h3>Latest context</h3></div>
          <dl class="kv">
            ${kv('Age', ctx.age)}
            ${kv('Sun exposure', ctx.sun_exposure)}
            ${kv('Sleep (hrs)', ctx.sleep_hours)}
            ${kv('Stress level', ctx.stress_level)}
            ${kv('Environment mode', ctx.env_mode)}
            ${kv('Location', ctx.place_name)}
            ${kv('Temperature (°C)', ctx.temperature_c != null ? Math.round(ctx.temperature_c) : null)}
            ${kv('Humidity (%)', ctx.humidity_pct != null ? Math.round(ctx.humidity_pct) : null)}
            ${kv('UV index', ctx.uv_index)}
            ${kv('Air quality (AQI)', ctx.air_quality_aqi != null ? Math.round(ctx.air_quality_aqi) : null)}
          </dl>
          <p class="muted" style="font-size:.8rem;margin-top:10px">Context is collected per-scan on the Scan screen. Editing a stored profile isn't supported by the backend in this build.</p>
        </div>
      </div>
      <div style="margin-top:18px"><button class="btn btn-ghost btn-sm" id="p-logout"><i class="fa-solid fa-arrow-right-from-bracket"></i> Log out</button></div>`;
    qs('#p-logout').onclick = () => { API.logout(); App.state.user = null; location.hash = '#/'; App.route(); };
  }

  // ============================================================ SETTINGS
  function viewSettings() {
    const view = qs('#view');
    const prefs = App.state.prefs;
    const env = loadEnv() || {};
    view.innerHTML = `
      <div class="view-head"><h1>Settings</h1><p>Environment source and local preferences. Preferences marked <em>local</em> are stored only in this browser.</p></div>
      <div class="card card-pad">
        <div class="card-title-row"><h3>Environmental data source</h3></div>
        <p class="muted" style="font-size:.85rem;margin-bottom:10px">Choose how the dashboard gets temperature, humidity, UV and air quality. Missing values stay unknown — never invented.</p>
        <div class="env-modes" id="env-modes"></div>
        <div id="env-body"></div>
      </div>
      <div class="card card-pad mt-18">
        <div class="card-title-row"><h3>Preferences <span class="pill neutral">local</span></h3></div>
        <div class="setting-row"><div><b>Show environment tips in results</b><div class="muted" style="font-size:.82rem">Climate/lifestyle notes in the recommendation block</div></div>
          <label class="switch"><input type="checkbox" id="pref-envtips" ${prefs.envTips !== false ? 'checked' : ''}><span class="track"></span></label></div>
        <div class="setting-row"><div><b>Scan reminders</b><div class="muted" style="font-size:.82rem">A gentle nudge if it's been 30+ days (shown on next visit)</div></div>
          <label class="switch"><input type="checkbox" id="pref-remind" ${prefs.remind ? 'checked' : ''}><span class="track"></span></label></div>
      </div>
      <div class="card card-pad mt-18">
        <div class="card-title-row"><h3>Account</h3></div>
        <div class="setting-row"><div><b>API endpoint</b><div class="muted" style="font-size:.82rem"><code>${esc(CFG.API_BASE_URL)}</code></div></div>
          <button class="btn btn-ghost btn-sm" id="s-check">Test connection</button></div>
        <div class="setting-row"><div><b>Log out</b></div><button class="btn btn-ghost btn-sm" id="s-logout"><i class="fa-solid fa-arrow-right-from-bracket"></i> Log out</button></div>
      </div>`;

    // env picker needs the #env-modes buttons injected
    qs('#env-modes').innerHTML = `
      <button class="env-mode-btn" data-m="current_location">Use current location</button>
      <button class="env-mode-btn" data-m="select_location">Select location</button>
      <button class="env-mode-btn" data-m="manual">Manual</button>`;
    bindEnvPicker(env, () => { App.loadWeather(); });

    qs('#pref-envtips').onchange = (e) => { prefs.envTips = e.target.checked; savePrefs(prefs); };
    qs('#pref-remind').onchange = (e) => { prefs.remind = e.target.checked; savePrefs(prefs); };
    qs('#s-check').onclick = async () => {
      try { await API.health(); toast('Backend reachable ✓', ''); }
      catch (e) { toast(e.message, 'err'); }
    };
    qs('#s-logout').onclick = () => { API.logout(); App.state.user = null; location.hash = '#/'; App.route(); };
  }

  // ---------------------------------------------------------------- start
  document.addEventListener('DOMContentLoaded', () => App.boot());
})();
