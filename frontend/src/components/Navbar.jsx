import React from 'react';
import { Globe, Menu, X, Sparkles, BookOpen, Info, Home } from 'lucide-react';
import { translations } from '../utils/translations';

export default function Navbar({ 
  currentLang, 
  onToggleLang, 
  user, 
  onLogout, 
  activeView, 
  onNavigate,
  onToggleSidebar,
  sidebarOpen
}) {
  const t = translations[currentLang] || translations.en;

  const handleNav = (view) => {
    if (onNavigate) onNavigate(view);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <header className="site-header">
      {/* Top sacred tagline bar */}
      <div className="top-micro-bar">
        <div className="container micro-container">
          <span className="micro-shloka">{t.tagline}</span>
          
          {/* Top Language Selector */}
          <div className="language-selector-wrap">
            <Globe size={14} className="globe-icon" />
            <span className="lang-label">Language / భాష:</span>
            <button 
              onClick={() => onToggleLang('en')}
              className={`lang-btn ${currentLang === 'en' ? 'active' : ''}`}
              aria-label="Switch to English"
            >
              English
            </button>
            <span className="lang-divider">|</span>
            <button 
              onClick={() => onToggleLang('te')}
              className={`lang-btn ${currentLang === 'te' ? 'active' : ''}`}
              aria-label="Switch to Telugu"
            >
              తెలుగు
            </button>
          </div>
        </div>
      </div>

      {/* Main Minimal Navbar */}
      <nav className="main-navbar">
        <div className="container nav-inner">
          {/* Left: Mobile Sidebar Toggle + Brand Logo */}
          <div className="brand-group-wrap">
            <button 
              onClick={onToggleSidebar}
              className={`sidebar-hamburger-btn ${sidebarOpen ? 'is-active' : ''}`}
              aria-label={sidebarOpen ? "Close navigation menu" : "Open navigation menu"}
              aria-expanded={Boolean(sidebarOpen)}
              title={sidebarOpen ? "Close Menu" : "Open Menu"}
            >
              {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
              <span className="hamburger-text">{sidebarOpen ? "Close" : "Menu"}</span>
            </button>

            <div className="brand-group" onClick={() => handleNav('home')}>
              <div className="brand-icon-box">
                <svg viewBox="0 0 36 36" className="brand-icon-svg" fill="none">
                  <circle cx="18" cy="18" r="16" fill="#F4E8E2" stroke="#BFA06C" strokeWidth="1.5" />
                  <circle cx="18" cy="18" r="12" fill="#482C3A" />
                  <path d="M18 10 C18 10 21 14 21 18 C21 21 19 22 18 24 C17 22 15 21 15 18 C15 14 18 10 18 10 Z" fill="#E8D1A7" />
                  <circle cx="18" cy="25" r="1.5" fill="#E8D1A7" />
                </svg>
              </div>
              <div className="brand-text">
                <h1 className="brand-title">{t.siteTitle}</h1>
                <p className="brand-subtitle">{t.siteSubtitle}</p>
              </div>
            </div>
          </div>

          {/* Minimal Navigation: Home, About, Language */}
          <div className="desktop-minimal-links">
            <button 
              onClick={() => handleNav('home')} 
              className={`nav-link ${activeView === 'home' ? 'active' : ''}`}
            >
              <Home size={15} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
              <span>{t.navHome}</span>
            </button>

            <button 
              onClick={() => handleNav('about')} 
              className={`nav-link ${activeView === 'about' ? 'active' : ''}`}
            >
              <Info size={15} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
              <span>{t.navAbout}</span>
            </button>

            {/* If a user is logged in (e.g. admin or dev), provide subtle logout option without pushing login to guests */}
            {user && (
              <div className="auth-profile-pill">
                <span className="auth-user-tag">{user?.profile?.full_name || user?.email?.split('@')[0]}</span>
                <button onClick={onLogout} className="auth-logout-mini" title="Sign out">
                  {t.navLogout}
                </button>
              </div>
            )}
          </div>
        </div>
      </nav>

      <style>{`
        .site-header {
          position: sticky;
          top: 0;
          z-index: 100;
          background: #FFFFFF;
          box-shadow: 0 2px 10px rgba(72, 44, 58, 0.05);
          border-bottom: 1px solid var(--color-border);
        }
        .top-micro-bar {
          background: var(--color-deep-plum);
          color: var(--color-blush-ivory);
          font-size: 0.78rem;
          padding: 6px 0;
        }
        .micro-container {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
        }
        .micro-shloka {
          font-family: var(--font-shloka);
          letter-spacing: 0.04em;
          opacity: 0.95;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .language-selector-wrap {
          display: flex;
          align-items: center;
          gap: 8px;
          white-space: nowrap;
        }
        .globe-icon {
          color: var(--color-antique-gold);
        }
        .lang-label {
          color: var(--color-antique-gold);
          font-weight: 500;
        }
        .lang-btn {
          background: none;
          border: none;
          color: #F4E8E2;
          font-size: 0.78rem;
          cursor: pointer;
          padding: 2px 6px;
          border-radius: 4px;
          transition: all 0.2s ease;
        }
        .lang-btn.active {
          background: var(--color-antique-gold);
          color: var(--color-deep-plum);
          font-weight: 700;
        }
        .lang-divider {
          color: rgba(244, 232, 226, 0.4);
        }
        .main-navbar {
          background: #FFFFFF;
          padding: 10px 0;
        }
        .nav-inner {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 20px;
        }
        .brand-group-wrap {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .sidebar-hamburger-btn {
          display: flex;
          align-items: center;
          gap: 6px;
          background: var(--color-blush-light);
          border: 1px solid var(--color-border);
          color: var(--color-deep-plum);
          padding: 6px 12px;
          border-radius: var(--radius-sm);
          font-size: 0.85rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
        }
        .sidebar-hamburger-btn:hover {
          background: var(--color-sage-light);
          border-color: var(--color-gold-dark);
        }
        .brand-group {
          display: flex;
          align-items: center;
          gap: 10px;
          cursor: pointer;
        }
        .brand-icon-box {
          width: 38px;
          height: 38px;
        }
        .brand-icon-svg {
          width: 100%;
          height: 100%;
        }
        .brand-title {
          font-size: 1.25rem;
          font-family: var(--font-heading);
          color: var(--color-deep-plum);
          margin: 0;
          line-height: 1.1;
          font-weight: 700;
        }
        .brand-subtitle {
          font-size: 0.75rem;
          color: var(--color-text-secondary);
          margin: 0;
          line-height: 1.2;
        }
        .desktop-minimal-links {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .nav-link {
          background: none;
          border: none;
          color: var(--color-deep-plum);
          font-family: var(--font-body);
          font-size: 0.9rem;
          font-weight: 600;
          padding: 6px 14px;
          border-radius: var(--radius-sm);
          cursor: pointer;
          display: flex;
          align-items: center;
          transition: all 0.2s ease;
        }
        .nav-link:hover, .nav-link.active {
          background: var(--color-blush-light);
          color: var(--color-temple-green);
        }
        .auth-profile-pill {
          display: flex;
          align-items: center;
          gap: 8px;
          background: var(--color-blush-light);
          padding: 4px 10px;
          border-radius: 999px;
          border: 1px solid var(--color-border);
          font-size: 0.8rem;
        }
        .auth-user-tag {
          font-weight: 600;
          color: var(--color-deep-plum);
        }
        .auth-logout-mini {
          background: none;
          border: none;
          color: #a02020;
          font-size: 0.75rem;
          cursor: pointer;
          text-decoration: underline;
        }
        @media (max-width: 768px) {
          .brand-subtitle {
            display: none;
          }
          .micro-shloka {
            display: none;
          }
          .micro-container {
            justify-content: flex-end;
          }
        }
      `}</style>
    </header>
  );
}
