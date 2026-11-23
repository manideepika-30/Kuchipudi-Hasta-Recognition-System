import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import PermanentSidebar from './components/PermanentSidebar';
import HomePage from './pages/HomePage';
import AboutPage from './pages/AboutPage';
import ModuleLessonViewer from './components/ModuleLessonViewer';
import PracticeStudio from './components/PracticeStudio';
import AssignmentViewer from './components/AssignmentViewer';
import ProgressView from './components/ProgressView';
import ResultsView from './components/ResultsView';
import { 
  updateLanguagePreference, 
  fetchCurrentUser, 
  logoutUser, 
  getAuthToken,
  fetchModules,
  fetchProgress,
  toggleLessonProgress,
  fetchQuizHistory,
  fetchPracticeHistory
} from './services/api';
import './styles/theme.css';

export default function App() {
  // 1. Language Preference (Default 'en', persists in localStorage)
  const [currentLang, setCurrentLang] = useState(() => {
    return localStorage.getItem('kuchipudi_lang') || 'en';
  });

  // 2. Active Tab / Route in the Learning Portal
  // Options: 'home', 'module-1'..'module-7', 'practice', 'assignments', 'progress', 'results', 'about', 'admin'
  const [activeTab, setActiveTab] = useState(() => {
    return localStorage.getItem('nritya_active_tab') || 'home';
  });

  // 3. User Session (Preserved if previously logged in; null for public guest)
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem('kuchipudi_user');
    return saved ? JSON.parse(saved) : null;
  });

  // 4. Responsive Mobile & Desktop Sidebar State
  const [sidebarOpenMobile, setSidebarOpenMobile] = useState(false);
  const [sidebarCollapsedDesktop, setSidebarCollapsedDesktop] = useState(false);
  const [isMobile, setIsMobile] = useState(() => typeof window !== 'undefined' && window.innerWidth <= 992);

  // Resize listener for responsive breakpoint and Escape key handler
  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth <= 992);
    };
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        setSidebarOpenMobile(false);
      }
    };
    window.addEventListener('resize', handleResize);
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  const handleToggleSidebar = () => {
    if (isMobile) {
      setSidebarOpenMobile(prev => !prev);
    } else {
      setSidebarCollapsedDesktop(prev => !prev);
    }
  };

  // 5. Curriculum and Progress Data (Guest or Authenticated)
  const [modulesList, setModulesList] = useState([]);
  const [completedModules, setCompletedModules] = useState([]);
  const [completedLessonIds, setCompletedLessonIds] = useState([]);
  const [quizHistory, setQuizHistory] = useState([]);
  const [practiceHistory, setPracticeHistory] = useState([]);
  const [initialPracticeGesture, setInitialPracticeGesture] = useState(null);

  // Verify and sync authentication state on mount if token exists (silent background check)
  useEffect(() => {
    const token = getAuthToken();
    if (token) {
      fetchCurrentUser()
        .then((userData) => {
          setUser(userData);
          localStorage.setItem('kuchipudi_user', JSON.stringify(userData));
          if (userData?.profile?.preferred_language) {
            setCurrentLang(userData.profile.preferred_language);
            localStorage.setItem('kuchipudi_lang', userData.profile.preferred_language);
          }
        })
        .catch(() => {
          logoutUser();
          setUser(null);
        });
    }
  }, []);

  // Load curriculum, progress, and history on mount and tab changes
  useEffect(() => {
    loadPortalData();
  }, [activeTab]);

  const loadPortalData = async () => {
    try {
      const [mods, prog, qHist, pHist] = await Promise.all([
        fetchModules().catch(() => []),
        fetchProgress().catch(() => null),
        fetchQuizHistory().catch(() => []),
        fetchPracticeHistory().catch(() => [])
      ]);

      if (Array.isArray(mods) && mods.length > 0) {
        setModulesList(mods);
      }
      if (prog) {
        setCompletedModules(prog.completed_module_ids || []);
        setCompletedLessonIds(prog.completed_lesson_ids || []);
      }
      if (Array.isArray(qHist)) {
        setQuizHistory(qHist);
      }
      if (Array.isArray(pHist)) {
        setPracticeHistory(pHist);
      }
    } catch (e) {
      console.warn('Portal data load error:', e);
    }
  };

  // Language switch handler
  const handleToggleLang = async (lang) => {
    setCurrentLang(lang);
    localStorage.setItem('kuchipudi_lang', lang);
    if (user && getAuthToken()) {
      try {
        await updateLanguagePreference(lang);
      } catch (e) {
        console.warn('Failed to persist language preference:', e);
      }
    }
  };

  // Navigation tab select handler
  const handleSelectTab = (tab) => {
    setActiveTab(tab);
    localStorage.setItem('nritya_active_tab', tab);
    setSidebarOpenMobile(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Toggle lesson completion handler (supports guest mode persistence)
  const handleToggleLesson = async (moduleId, lessonId) => {
    const isNowDone = !completedLessonIds.includes(lessonId);
    const updatedLessons = isNowDone
      ? [...completedLessonIds, lessonId]
      : completedLessonIds.filter(id => id !== lessonId);
    setCompletedLessonIds(updatedLessons);

    try {
      await toggleLessonProgress(moduleId, lessonId, isNowDone);
      const prog = await fetchProgress();
      if (prog) {
        setCompletedModules(prog.completed_module_ids || []);
        setCompletedLessonIds(prog.completed_lesson_ids || []);
      }
    } catch (e) {
      console.warn('Toggle lesson progress error:', e);
    }
  };

  // Launch practice studio with pre-selected mudra
  const handleLaunchPracticeWithGesture = (gestureName) => {
    setInitialPracticeGesture(gestureName);
    handleSelectTab('practice');
  };

  // Logout handler (resets to guest mode cleanly)
  const handleLogout = () => {
    logoutUser();
    setUser(null);
    handleSelectTab('home');
  };

  // Determine active module object if viewing module-1 .. module-7
  const activeModuleNum = activeTab.startsWith('module-') 
    ? parseInt(activeTab.replace('module-', ''), 10) 
    : null;

  const currentModuleData = activeModuleNum 
    ? modulesList.find(m => m.module_number === activeModuleNum) 
    : null;

  return (
    <div className="nritya-app-root">
      {/* 1. Minimal Top Navigation Header */}
      <Navbar 
        currentLang={currentLang}
        onToggleLang={handleToggleLang}
        user={user}
        onLogout={handleLogout}
        activeView={activeTab}
        onNavigate={handleSelectTab}
        onToggleSidebar={handleToggleSidebar}
        sidebarOpen={isMobile ? sidebarOpenMobile : !sidebarCollapsedDesktop}
      />

      {/* 2. Permanent 2-Column Learning Portal Shell */}
      <div className="portal-shell-container">
        {/* Permanent Left Sidebar (always present across all pages) */}
        <PermanentSidebar 
          currentLang={currentLang}
          activeTab={activeTab}
          onSelectTab={handleSelectTab}
          completedModules={completedModules}
          isOpenMobile={sidebarOpenMobile}
          onCloseMobile={() => setSidebarOpenMobile(false)}
          isDesktopCollapsed={sidebarCollapsedDesktop}
          user={user}
        />

        {/* Dynamic Main Content Area */}
        <main className="portal-main-stage">
          <div className="portal-stage-inner">
            {/* View 1: Public Homepage / Learning Portal Overview */}
            {activeTab === 'home' && (
              <HomePage 
                currentLang={currentLang}
                onStartLearning={(modNum) => handleSelectTab(`module-${modNum || 1}`)}
                onSelectModule={(target) => {
                  const tab = typeof target === 'number' ? `module-${target}` : target;
                  handleSelectTab(tab);
                }}
                completedModules={completedModules}
              />
            )}

            {/* View 2: About Kuchipudi Classical Dance */}
            {activeTab === 'about' && (
              <AboutPage setActivePage={handleSelectTab} />
            )}

            {/* View 3: Individual Module Lessons Viewer (Modules 1 to 7) */}
            {activeModuleNum && (
              <div className="module-view-stage">
                <div className="module-breadcrumb-bar">
                  <button onClick={() => handleSelectTab('home')} className="breadcrumb-btn">
                    Home
                  </button>
                  <span className="breadcrumb-sep">/</span>
                  <span className="breadcrumb-current">
                    Module 0{activeModuleNum}: {currentModuleData?.title_en || `Module ${activeModuleNum}`}
                  </span>
                </div>

                {currentModuleData ? (
                  <ModuleLessonViewer 
                    module={currentModuleData}
                    currentLang={currentLang}
                    completedLessonIds={completedLessonIds}
                    onToggleLesson={(lesId) => handleToggleLesson(currentModuleData.id, lesId)}
                    onLaunchPractice={handleLaunchPracticeWithGesture}
                  />
                ) : (
                  <div className="card-soft loading-card">
                    <p>Loading canonical Kuchipudi lessons for Module {activeModuleNum}...</p>
                  </div>
                )}
              </div>
            )}

            {/* View 4: AI Practice Studio */}
            {activeTab === 'practice' && (
              <PracticeStudio 
                currentLang={currentLang}
                initialTargetGesture={initialPracticeGesture}
              />
            )}

            {/* View 5: Assignments & Quizzes */}
            {activeTab === 'assignments' && (
              <AssignmentViewer 
                currentLang={currentLang}
                onLaunchCameraExam={() => handleSelectTab('practice')}
              />
            )}

            {/* View 6: My Progress */}
            {activeTab === 'progress' && (
              <ProgressView 
                currentLang={currentLang}
                completedModules={completedModules}
                completedLessonIds={completedLessonIds}
                quizHistory={quizHistory}
                practiceHistory={practiceHistory}
                onNavigateToModule={(modId) => handleSelectTab(`module-${modId}`)}
              />
            )}

            {/* View 7: Results */}
            {activeTab === 'results' && (
              <ResultsView 
                currentLang={currentLang}
                quizHistory={quizHistory}
                practiceHistory={practiceHistory}
                onNavigateToAssignments={() => handleSelectTab('assignments')}
                onNavigateToPractice={() => handleSelectTab('practice')}
              />
            )}
          </div>
        </main>
      </div>

      {/* 3. Universal Footer */}
      <Footer currentLang={currentLang} />

      <style>{`
        .nritya-app-root {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          background: var(--color-blush-light);
        }
        .portal-shell-container {
          display: flex;
          flex-grow: 1;
          width: 100%;
          max-width: 1720px;
          margin: 0 auto;
          position: relative;
        }
        .portal-main-stage {
          flex-grow: 1;
          min-width: 0;
          padding: 24px 32px 64px;
        }
        .portal-stage-inner {
          max-width: 1280px;
          margin: 0 auto;
        }
        .module-view-stage {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        .module-breadcrumb-bar {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 0.85rem;
          color: var(--color-text-secondary);
          margin-bottom: 4px;
        }
        .breadcrumb-btn {
          background: none;
          border: none;
          color: var(--color-deep-plum);
          font-weight: 600;
          cursor: pointer;
          padding: 0;
          text-decoration: underline;
        }
        .breadcrumb-sep {
          color: var(--color-gold-dark);
        }
        .breadcrumb-current {
          color: var(--color-temple-green);
          font-weight: 600;
        }
        .loading-card {
          padding: 48px;
          text-align: center;
          color: var(--color-text-secondary);
          font-family: var(--font-heading);
        }
        @media (max-width: 992px) {
          .portal-main-stage {
            padding: 16px 16px 48px;
          }
        }
      `}</style>
    </div>
  );
}
