/**
 * api-config.js  —  Backend API URL configuration
 *
 * After deploying the backend to Render.com, replace RENDER_BACKEND_URL
 * with your actual Render service URL (e.g. https://neuro-adhd-api.onrender.com)
 * then commit and push — Vercel will redeploy the frontend automatically.
 */

(function () {
  const RENDER_BACKEND_URL = 'https://neuro-adhd-api.onrender.com';
  const LOCAL_BACKEND_URL  = 'http://localhost:5000';

  // Use Render URL on deployed site, localhost in local development
  const isDeployed = window.location.hostname !== 'localhost' &&
                     window.location.hostname !== '127.0.0.1';

  window.API_BASE = (isDeployed ? RENDER_BACKEND_URL : LOCAL_BACKEND_URL) + '/api';
})();
