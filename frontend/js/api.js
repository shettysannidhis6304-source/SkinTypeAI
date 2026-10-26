/* ==========================================================================
   SkinIntelli API client — the only module that talks to the FastAPI backend.
   Central config in config.js. Real JWT auth, no fake authentication.
   ========================================================================== */
const API = {
  get BASE() { return window.SI_CONFIG.API_BASE_URL; },
  get TOKEN_KEY() { return window.SI_CONFIG.TOKEN_KEY; },

  getToken() { return localStorage.getItem(this.TOKEN_KEY); },
  setToken(t) { localStorage.setItem(this.TOKEN_KEY, t); },
  clearToken() { localStorage.removeItem(this.TOKEN_KEY); },
  isLoggedIn() { return !!this.getToken(); },

  async _request(path, { method = 'GET', body, isForm = false, timeout } = {}) {
    const headers = {};
    let payload = body;
    if (body && !isForm) { headers['Content-Type'] = 'application/json'; payload = JSON.stringify(body); }
    const token = this.getToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const ctrl = new AbortController();
    const to = setTimeout(() => ctrl.abort(), timeout || window.SI_CONFIG.REQUEST_TIMEOUT_MS);

    let res;
    try {
      res = await fetch(`${this.BASE}${path}`, { method, headers, body: payload, signal: ctrl.signal });
    } catch (err) {
      clearTimeout(to);
      const e = new Error(
        err.name === 'AbortError'
          ? 'The request timed out. The analysis can take up to a minute on the first scan — please try again.'
          : `Unable to connect to the SkinTypeAI server at ${this.BASE}. Please make sure the backend is running.`
      );
      e.kind = err.name === 'AbortError' ? 'timeout' : 'network';
      throw e;
    }
    clearTimeout(to);

    if (res.status === 204) return null;

    let data = null;
    const ct = res.headers.get('content-type') || '';
    if (ct.includes('application/json')) { try { data = await res.json(); } catch (_) {} }

    if (!res.ok) {
      let detail = (data && data.detail) || `Request failed (${res.status})`;
      if (typeof detail !== 'string') detail = JSON.stringify(detail);
      if (res.status === 401 && token && path !== '/login' && path !== '/register') {
        this.clearToken();
        window.dispatchEvent(new CustomEvent('api:unauthorized'));
        detail = 'Your session expired — please log in again.';
      }
      const e = new Error(this._friendly(res.status, detail));
      e.status = res.status;
      e.raw = detail;
      throw e;
    }
    return data;
  },

  _friendly(status, detail) {
    switch (status) {
      case 400: return detail || 'That request could not be processed.';
      case 401: return 'Your session expired — please log in again.';
      case 403: return 'You do not have permission to do that.';
      case 404: return detail || 'Not found.';
      case 415: return 'That file type is not supported. Use a JPEG, PNG or WebP image.';
      case 422: return detail || 'Some of the information provided was invalid.';
      case 500: return 'The server hit an unexpected error. Please try again.';
      case 503: return detail || 'The analysis models are not available on the server right now.';
      default:  return detail || `Request failed (${status}).`;
    }
  },

  // ---- auth ----
  register: (email, password, full_name) =>
    API._request('/register', { method: 'POST', body: { email, password, full_name: full_name || null } }),

  async login(email, password) {
    const d = await API._request('/login', { method: 'POST', body: { email, password } });
    API.setToken(d.access_token);
    return d;
  },
  logout() { API.clearToken(); },

  me: () => API._request('/me'),

  // ---- scan ----
  async upload(file) {
    const fd = new FormData();
    fd.append('file', file);
    return API._request('/upload', { method: 'POST', body: fd, isForm: true });
  },
  // predict(): when forceProceed is false and the photo rates <= 6/10 the backend
  // returns { status:'advisory', quality, temp_image_id } instead of running.
  predict: (imageId, context, forceProceed = false) =>
    API._request(`/predict/${imageId}?force_proceed=${forceProceed ? 'true' : 'false'}`,
      { method: 'POST', body: context || null }),

  history: () => API._request('/history'),

  async scan(file, context) {
    const { image_id } = await API.upload(file);
    const res = await API.predict(image_id, context, false);
    return { ...res, _imageId: image_id };
  },

  // ---- "Use Phone Camera" — QR capture session ----
  createQrSession: (ip) =>
    API._request(`/api/session/create-qr-session${ip ? '?ip=' + encodeURIComponent(ip) : ''}`, { timeout: 12000 }),
  sessionStatus: (sessionId) =>
    API._request(`/api/session/${encodeURIComponent(sessionId)}/status`, { timeout: 8000 }),

  // absolute URL for an <img src> (needs the auth token → we fetch as blob)
  async imageBlobUrl(pathOrId) {
    const path = typeof pathOrId === 'number' ? `/images/${pathOrId}` : pathOrId;
    const res = await fetch(`${API.BASE}${path}`, { headers: { Authorization: `Bearer ${API.getToken()}` } });
    if (!res.ok) throw new Error('image fetch failed');
    return URL.createObjectURL(await res.blob());
  },

  // XAI overlay PNG (Grad-CAM heatmap or MediaPipe measurement zones) as a blob URL
  async overlayBlobUrl(imageId, mode, head) {
    const q = mode === 'gradcam'
      ? `?mode=gradcam&head=${encodeURIComponent(head || 'acne')}`
      : '?mode=zones';
    const res = await fetch(`${API.BASE}/api/images/${imageId}/overlay${q}`,
      { headers: { Authorization: `Bearer ${API.getToken()}` } });
    if (!res.ok) {
      let msg = `Overlay failed (${res.status})`;
      try { const j = await res.json(); if (j && typeof j.detail === 'string') msg = j.detail; } catch (_) {}
      const e = new Error(msg); e.status = res.status; throw e;
    }
    return URL.createObjectURL(await res.blob());
  },

  // ---- environment ----
  weatherByCoords: (lat, lon) =>
    API._request(`/environment/weather?lat=${lat}&lon=${lon}`, { timeout: 12000 }),
  weatherByPlace: (place) =>
    API._request(`/environment/weather?place=${encodeURIComponent(place)}`, { timeout: 12000 }),
  geocode: (q) =>
    API._request(`/environment/geocode?q=${encodeURIComponent(q)}`, { timeout: 12000 }),

  health: () => API._request('/health', { timeout: 4000 }),
};
window.API = API;
