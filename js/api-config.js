/**
 * api-config.js  �  Backend API URL configuration
 *
 * Deployed  : uses /api (same-origin Vercel serverless functions)
 * Local dev : tries localhost:5000 first; if unreachable within 2 s,
 *             falls back to the deployed Vercel API so the UI always works.
 */

(function () {
  const VERCEL_API   = 'https://neuro-adhd-demo.vercel.app/api';
  const LOCAL_API    = 'http://localhost:5000/api';
  const isLocal      = window.location.hostname === 'localhost' ||
                       window.location.hostname === '127.0.0.1' ||
                       window.location.protocol === 'file:';

  if (!isLocal) {
    // Deployed — always use same-origin /api
    window.API_BASE = '/api';
    return;
  }

  // Local: probe localhost; fall back to Vercel if not running
  window.API_BASE = LOCAL_API;   // optimistic default

  fetch(LOCAL_API + '/health', { signal: AbortSignal.timeout(2000) })
    .then(r => {
      if (!r.ok) throw new Error('not ok');
      // localhost is up — keep LOCAL_API (already set)
    })
    .catch(() => {
      console.info('[api-config] localhost:5000 unreachable — using Vercel API');
      window.API_BASE = VERCEL_API;
      // Re-run health check now that base URL is updated
      if (typeof window.checkBackendHealth === 'function') {
        window.checkBackendHealth();
      }
    });
})();
