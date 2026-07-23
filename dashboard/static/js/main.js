/* ============================================================
   ExamGuard AI — Main JavaScript
   ============================================================ */

/* ---- Live clock in topbar ---- */
(function initClock() {
  const el = document.getElementById('topbarTime');
  if (!el) return;
  const tick = () => {
    el.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };
  tick();
  setInterval(tick, 1000);
})();

/* ---- Mobile sidebar toggle ---- */
(function initSidebar() {
  const btn = document.getElementById('sidebarToggle');
  const sidebar = document.getElementById('sidebar');
  if (!btn || !sidebar) return;

  btn.addEventListener('click', () => {
    sidebar.classList.toggle('open');
  });

  // Close sidebar when clicking outside
  document.addEventListener('click', (e) => {
    if (sidebar.classList.contains('open') &&
        !sidebar.contains(e.target) &&
        e.target !== btn) {
      sidebar.classList.remove('open');
    }
  });
})();

/* ---- Auto-dismiss flash messages after 5 seconds ---- */
(function initFlash() {
  const messages = document.querySelectorAll('.flash');
  messages.forEach((msg, i) => {
    setTimeout(() => {
      msg.style.transition = 'opacity .4s ease, transform .4s ease';
      msg.style.opacity = '0';
      msg.style.transform = 'translateY(-8px)';
      setTimeout(() => msg.remove(), 400);
    }, 4000 + i * 200);
  });
})();

/* ---- Confirm-delete helper for data-confirm attrs ---- */
document.addEventListener('click', (e) => {
  const btn = e.target.closest('[data-confirm]');
  if (btn && !confirm(btn.dataset.confirm)) {
    e.preventDefault();
    e.stopPropagation();
  }
});

/* ---- Login form loading state ---- */
(function initLoginForm() {
  const form = document.getElementById('loginForm');
  const btn = document.getElementById('loginBtn');
  if (!form || !btn) return;
  form.addEventListener('submit', () => {
    btn.disabled = true;
    btn.querySelector('.btn-text').textContent = 'Signing in…';
  });
})();

/* ---- Form input enhancement: add .has-value class for animations ---- */
document.querySelectorAll('input, select, textarea').forEach(el => {
  const check = () => el.classList.toggle('has-value', !!el.value);
  check();
  el.addEventListener('input', check);
  el.addEventListener('change', check);
});

/* ---- Table row click-to-expand (for future detail panels) ---- */
// Placeholder — can be extended per-page

/* ---- Keyboard shortcut: Escape closes modals ---- */
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    document.querySelectorAll('.modal-overlay').forEach(m => {
      m.style.display = 'none';
    });
  }
});

/* ---- Theme Toggle ---- */
(function initThemeToggle() {
  const btn = document.getElementById('themeToggle');
  if (!btn) return;
  const iconSun = btn.querySelector('.icon-sun');
  const iconMoon = btn.querySelector('.icon-moon');

  function updateIcons(theme) {
    if (theme === 'light') {
      iconSun.style.display = 'none';
      iconMoon.style.display = 'block';
    } else {
      iconSun.style.display = 'block';
      iconMoon.style.display = 'none';
    }
  }

  // Initial state setup based on <html data-theme="">
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
  updateIcons(currentTheme);

  btn.addEventListener('click', () => {
    const oldTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const newTheme = oldTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateIcons(newTheme);
  });
})();
