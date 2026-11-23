import React, { useState } from 'react';
import { 
  Home, BookOpen, Video, Award, TrendingUp, FileCheck2, 
  CheckCircle2, Circle, ChevronDown, ChevronRight, X, Sparkles, ShieldCheck 
} from 'lucide-react';
import { translations } from '../utils/translations';

export default function PermanentSidebar({
  currentLang = 'en',
  activeTab = 'home',
  onSelectTab,
  completedModules = [],
  isOpenMobile = false,
  onCloseMobile,
  isDesktopCollapsed = false,
  user = null
}) {
  const t = translations[currentLang] || translations.en;
  const [modulesCollapsed, setModulesCollapsed] = useState(false);

  const modulesList = [
    {
      id: 1,
      key: 'module-1',
      num: '1',
      titleEn: '1. Introduction to Kuchipudi',
      titleTe: '1. కూచిపూడి పరిచయం',
      badge: '4 lessons'
    },
    {
      id: 2,
      key: 'module-2',
      num: '2',
      titleEn: '2. Asamyukta Hastas – 28 Single-Hand Gestures',
      titleTe: '2. అసంయుక్త హస్తాలు (28 ఏకహస్త ముద్రలు)',
      badge: '28 lessons'
    },
    {
      id: 3,
      key: 'module-3',
      num: '3',
      titleEn: '3. Samyukta Hastas – 24 Combined-Hand Gestures',
      titleTe: '3. సంయుక్త హస్తాలు (24 ద్విహస్త ముద్రలు)',
      badge: '24 lessons'
    },
    {
      id: 4,
      key: 'module-4',
      num: '4',
      titleEn: '4. Shiro Bhedas – 9 Head Movements',
      titleTe: '4. శిరో భేదాలు (9 శిరస్సు కదలికలు)',
      badge: '9 lessons'
    },
    {
      id: 5,
      key: 'module-5',
      num: '5',
      titleEn: '5. Drishti Bhedas – 8 Eye Movements',
      titleTe: '5. దృష్టి భేదాలు (8 నేత్ర కదలికలు)',
      badge: '8 lessons'
    },
    {
      id: 6,
      key: 'module-6',
      num: '6',
      titleEn: '6. Chaturvidha Abhinaya & Navarasas',
      titleTe: '6. చతుర్విధ అభినయం మరియు నవరసాలు',
      badge: '12 lessons'
    },
    {
      id: 7,
      key: 'module-7',
      num: '7',
      titleEn: '7. Sapta Talamulu',
      titleTe: '7. సప్త తాళములు (35 తాళ వ్యవస్థ)',
      badge: '7 lessons'
    }
  ];

  const handleItemClick = (tabKey) => {
    if (onSelectTab) onSelectTab(tabKey);
    if (onCloseMobile) onCloseMobile();
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpenMobile && (
        <div className="sidebar-backdrop" onClick={onCloseMobile} />
      )}

      <aside 
        className={`permanent-left-sidebar ${isOpenMobile ? 'mobile-visible' : ''} ${isDesktopCollapsed ? 'desktop-collapsed' : ''}`}
        aria-label="Course Curriculum Navigation"
      >
        <div className="sidebar-inner-content">
          {/* Mobile Close Button in Header */}
          <div className="sidebar-mobile-header">
            <span className="sidebar-mobile-title">Nritya AI Navigation</span>
            <button className="sidebar-mobile-close" onClick={onCloseMobile} aria-label="Close menu">
              <X size={18} />
            </button>
          </div>

          {/* Guest Mode Status Chip */}
          <div className="guest-badge-card">
            <div className="guest-status-indicator">
              <span className="live-dot" />
              <span className="guest-title">Public Learning Mode</span>
            </div>
            <p className="guest-sub">Free open access to all 7 modules & AI studio</p>
          </div>

          <nav className="portal-sidebar-nav">
            {/* =========================================================
                SECTION 1: HOME
                ========================================================= */}
            <div className="nav-section-group">
              <span className="nav-section-heading">HOME</span>
              <button
                onClick={() => handleItemClick('home')}
                className={`portal-nav-btn ${activeTab === 'home' ? 'active' : ''}`}
              >
                <Home size={18} className="nav-icon" />
                <span className="nav-label">{t.navHome || 'Home'}</span>
              </button>
            </div>

            {/* =========================================================
                SECTION 2: LEARNING MODULES
                ========================================================= */}
            <div className="nav-section-group">
              <div 
                className="nav-section-heading-row"
                onClick={() => setModulesCollapsed(!modulesCollapsed)}
                style={{ cursor: 'pointer' }}
              >
                <span className="nav-section-heading">LEARNING MODULES</span>
                <span className="toggle-arrow">
                  {modulesCollapsed ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
                </span>
              </div>

              {!modulesCollapsed && (
                <div className="modules-nav-list">
                  {modulesList.map((m) => {
                    const isDone = completedModules.includes(m.id);
                    const isActive = activeTab === m.key;
                    return (
                      <button
                        key={m.key}
                        onClick={() => handleItemClick(m.key)}
                        className={`module-nav-btn ${isActive ? 'active' : ''}`}
                        title={currentLang === 'te' ? m.titleTe : m.titleEn}
                      >
                        <div className="mod-icon-wrapper">
                          {isDone ? (
                            <CheckCircle2 size={14} className="mod-icon-done" />
                          ) : (
                            <Circle size={14} className="mod-icon-pending" />
                          )}
                        </div>
                        <span className="mod-btn-text">
                          {currentLang === 'te' ? m.titleTe : m.titleEn}
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            {/* =========================================================
                SECTION 3: PRACTICE & ASSESSMENT
                ========================================================= */}
            <div className="nav-section-group">
              <span className="nav-section-heading">PRACTICE & ASSESSMENT</span>

              <button
                onClick={() => handleItemClick('practice')}
                className={`portal-nav-btn ${activeTab === 'practice' ? 'active' : ''}`}
              >
                <Video size={18} className="nav-icon icon-video" />
                <span className="nav-label">{t.sbPracticeStudio || 'Practice Studio'}</span>
              </button>

              <button
                onClick={() => handleItemClick('assignments')}
                className={`portal-nav-btn ${activeTab === 'assignments' ? 'active' : ''}`}
              >
                <Award size={18} className="nav-icon icon-award" />
                <span className="nav-label">{t.sbAssignmentsQuizzes || 'Assignments & Quizzes'}</span>
              </button>

              <button
                onClick={() => handleItemClick('progress')}
                className={`portal-nav-btn ${activeTab === 'progress' ? 'active' : ''}`}
              >
                <TrendingUp size={18} className="nav-icon icon-progress" />
                <span className="nav-label">{t.sbMyProgress || 'My Progress'}</span>
              </button>

              <button
                onClick={() => handleItemClick('results')}
                className={`portal-nav-btn ${activeTab === 'results' ? 'active' : ''}`}
              >
                <FileCheck2 size={18} className="nav-icon icon-results" />
                <span className="nav-label">{t.sbResults || 'Results'}</span>
              </button>

              {/* Admin Management (Strictly visible ONLY if logged in user has admin role) */}
              {user?.role === 'admin' && (
                <button
                  onClick={() => handleItemClick('admin')}
                  className={`portal-nav-btn admin-btn ${activeTab === 'admin' ? 'active' : ''}`}
                >
                  <ShieldCheck size={18} className="nav-icon icon-admin" />
                  <span className="nav-label">{t.sbAdminManagement || 'Admin Management'}</span>
                </button>
              )}
            </div>
          </nav>
        </div>

        <style>{`
          .permanent-left-sidebar {
            width: 285px;
            min-width: 285px;
            background: #FFFFFF;
            border-right: 1.5px solid var(--color-border);
            display: flex;
            flex-direction: column;
            position: sticky;
            top: 61px;
            height: calc(100vh - 61px);
            overflow-y: auto;
            z-index: 90;
            box-shadow: 2px 0 12px rgba(72, 44, 58, 0.03);
          }
          .sidebar-inner-content {
            padding: 18px 14px 30px;
            display: flex;
            flex-direction: column;
            gap: 16px;
          }
          .sidebar-mobile-header {
            display: none;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--color-border);
          }
          .sidebar-mobile-title {
            font-family: var(--font-heading);
            font-weight: 700;
            color: var(--color-deep-plum);
            font-size: 0.95rem;
          }
          .sidebar-mobile-close {
            background: none;
            border: none;
            color: var(--color-deep-plum);
            cursor: pointer;
            padding: 4px;
          }
          .guest-badge-card {
            background: linear-gradient(135deg, var(--color-blush-light) 0%, var(--color-sage-light) 100%);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-md);
            padding: 10px 12px;
          }
          .guest-status-indicator {
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 2px;
          }
          .live-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #2e7d32;
            display: inline-block;
            box-shadow: 0 0 0 2px rgba(46, 125, 50, 0.2);
          }
          .guest-title {
            font-weight: 700;
            font-size: 0.8rem;
            color: var(--color-temple-green);
            text-transform: uppercase;
            letter-spacing: 0.05em;
          }
          .guest-sub {
            font-size: 0.72rem;
            color: var(--color-text-secondary);
            margin: 0;
            line-height: 1.3;
          }
          .portal-sidebar-nav {
            display: flex;
            flex-direction: column;
            gap: 20px;
          }
          .nav-section-group {
            display: flex;
            flex-direction: column;
            gap: 4px;
          }
          .nav-section-heading {
            font-size: 0.7rem;
            font-weight: 800;
            color: var(--color-gold-dark);
            letter-spacing: 0.1em;
            text-transform: uppercase;
            padding: 0 10px 4px;
          }
          .nav-section-heading-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 10px 4px;
          }
          .nav-section-heading-row .nav-section-heading {
            padding: 0;
          }
          .toggle-arrow {
            color: var(--color-gold-dark);
          }
          .portal-nav-btn {
            display: flex;
            align-items: center;
            gap: 12px;
            width: 100%;
            padding: 10px 12px;
            background: none;
            border: none;
            border-radius: var(--radius-sm);
            color: var(--color-deep-plum);
            font-family: var(--font-body);
            font-size: 0.88rem;
            font-weight: 600;
            cursor: pointer;
            text-align: left;
            transition: all 0.15s ease;
          }
          .portal-nav-btn:hover {
            background: var(--color-blush-light);
            color: var(--color-temple-green);
          }
          .portal-nav-btn.active {
            background: var(--color-blush-ivory);
            color: var(--color-deep-plum);
            border-left: 3.5px solid var(--color-gold-dark);
            font-weight: 700;
            box-shadow: 0 2px 6px rgba(72, 44, 58, 0.06);
          }
          .portal-nav-btn.admin-btn {
            color: #822727;
            margin-top: 6px;
            border-top: 1px dashed var(--color-border);
            border-radius: 0;
            padding-top: 12px;
          }
          .nav-icon {
            flex-shrink: 0;
            color: var(--color-deep-plum);
          }
          .portal-nav-btn.active .nav-icon {
            color: var(--color-temple-green);
          }
          .modules-nav-list {
            display: flex;
            flex-direction: column;
            gap: 2px;
            padding-left: 4px;
          }
          .module-nav-btn {
            display: flex;
            align-items: flex-start;
            gap: 9px;
            width: 100%;
            padding: 7px 10px;
            background: none;
            border: none;
            border-radius: var(--radius-sm);
            color: var(--color-text-secondary);
            font-family: var(--font-body);
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            text-align: left;
            line-height: 1.35;
            transition: all 0.15s ease;
          }
          .module-nav-btn:hover {
            background: var(--color-sage-light);
            color: var(--color-deep-plum);
          }
          .module-nav-btn.active {
            background: var(--color-sage);
            color: var(--color-deep-plum);
            font-weight: 700;
            border-left: 3px solid var(--color-temple-green);
          }
          .mod-icon-wrapper {
            margin-top: 2px;
            flex-shrink: 0;
          }
          .mod-icon-done {
            color: #2e7d32;
          }
          .mod-icon-pending {
            color: #b0a4a9;
          }
          .mod-btn-text {
            flex-grow: 1;
            word-break: break-word;
          }

          @media (min-width: 993px) {
            .permanent-left-sidebar.desktop-collapsed {
              display: none !important;
            }
          }

          /* Responsive Mobile Drawer */
          @media (max-width: 992px) {
            .permanent-left-sidebar {
              position: fixed;
              top: 0;
              left: 0;
              height: 100vh;
              z-index: 1000;
              transform: translateX(-100%);
              transition: transform 0.25s ease;
              box-shadow: 4px 0 24px rgba(0,0,0,0.15);
            }
            .permanent-left-sidebar.mobile-visible {
              transform: translateX(0);
            }
            .sidebar-mobile-header {
              display: flex;
            }
            .sidebar-backdrop {
              position: fixed;
              inset: 0;
              background: rgba(40, 25, 33, 0.45);
              backdrop-filter: blur(2px);
              z-index: 999;
            }
          }
        `}</style>
      </aside>
    </>
  );
}
