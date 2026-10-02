/**
 * main.js — HeartCare AI
 * Global utilities shared across all pages.
 */

/* ── Fade-in on scroll ──────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  const observer = new IntersectionObserver(
    (entries) => entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.style.opacity  = '1';
        e.target.style.transform = 'translateY(0)';
      }
    }),
    { threshold: 0.08 }
  );

  document.querySelectorAll('.fade-in').forEach(el => {
    el.style.opacity   = '0';
    el.style.transform = 'translateY(18px)';
    el.style.transition = 'opacity .45s ease, transform .45s ease';
    observer.observe(el);
  });
});

/* ── Active nav link highlight ──────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  const path  = window.location.pathname;
  document.querySelectorAll('.sidebar-nav a').forEach(a => {
    if (a.getAttribute('href') === path) {
      a.classList.add('active');
    }
  });
});

/* ── Utility: format percent ─────────────────────────────────────────────── */
window.fmtPct = (v, decimals = 1) => parseFloat(v).toFixed(decimals) + '%';

/* ── Utility: show toast notification ───────────────────────────────────── */
window.showToast = (msg, type = 'info') => {
  const colors = {
    info:    'var(--teal)',
    success: 'var(--green)',
    error:   'var(--red)',
  };
  const toast = document.createElement('div');
  toast.textContent = msg;
  Object.assign(toast.style, {
    position:     'fixed',
    bottom:       '28px',
    right:        '28px',
    background:   'var(--navy-card)',
    color:        colors[type] || colors.info,
    border:       `1px solid ${colors[type] || colors.info}`,
    borderRadius: '10px',
    padding:      '12px 20px',
    fontSize:     '.84rem',
    fontFamily:   'var(--font-body)',
    zIndex:       9999,
    boxShadow:    '0 4px 24px rgba(0,0,0,.4)',
    opacity:      '0',
    transition:   'opacity .3s ease',
  });
  document.body.appendChild(toast);
  requestAnimationFrame(() => { toast.style.opacity = '1'; });
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 400);
  }, 3200);
};
