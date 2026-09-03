/* Central configuration for the SkinIntelli frontend. One source of truth. */
window.SI_CONFIG = {
  // The backend origin. Change this one line to point at a deployed backend.
  API_BASE_URL: (location.hostname === '127.0.0.1' || location.hostname === 'localhost')
    ? 'http://127.0.0.1:8000'
    : location.origin.replace(/:\d+$/, ':8000'),

  TOKEN_KEY: 'skinintelli_token',
  EMAIL_KEY: 'skinintelli_email',
  PREFS_KEY: 'skinintelli_prefs',
  ENV_KEY: 'skinintelli_env',        // last-used environment context
  PLAN_PROGRESS_KEY: 'skinintelli_plan_progress',

  REQUEST_TIMEOUT_MS: 120000,        // ML inference on CPU can take ~30s cold

  // The 10 project conditions, in display order, with their coverage tier:
  //   'model' = the project's own trained model     (score bar + confidence, "AI" badge)
  //   'cv'    = the deterministic CV analysis engine (score bar + confidence, "CV scan" badge)
  CONDITIONS: [
    { key: 'acne',           label: 'Acne',            tier: 'model', icon: 'fa-circle-dot' },
    { key: 'skin_type',      label: 'Skin type',       tier: 'model', icon: 'fa-droplet' },
    { key: 'dryness',        label: 'Dryness',         tier: 'model', icon: 'fa-hand-dots' },
    { key: 'oiliness',       label: 'Oiliness',        tier: 'model', icon: 'fa-oil-can' },
    { key: 'wrinkles',       label: 'Wrinkles',        tier: 'model', icon: 'fa-wave-square' },
    { key: 'pigmentation',   label: 'Pigmentation',    tier: 'model', icon: 'fa-circle-half-stroke' },
    { key: 'redness',        label: 'Redness',         tier: 'cv',    icon: 'fa-fire' },
    { key: 'enlarged_pores', label: 'Enlarged pores',  tier: 'cv',    icon: 'fa-grip' },
    { key: 'blackheads',     label: 'Blackheads',      tier: 'cv',    icon: 'fa-braille' },
    { key: 'dark_spots',     label: 'Dark spots',      tier: 'cv',    icon: 'fa-splotch' },
  ],
};
