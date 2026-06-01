/**
 * api-config.js  —  Backend API URL configuration
 *
 * The API now runs as a Vercel serverless function at /api on the same origin.
 * In local development it falls back to the Flask dev server on port 5000.
 */

(function () {
  const isLocal = window.location.hostname === 'localhost' ||
                  window.location.hostname === '127.0.0.1';
  window.API_BASE = isLocal ? 'http://localhost:5000/api' : '/api';
})();
