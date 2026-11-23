import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, BookOpen, Video, Award, TrendingUp, FileCheck2, 
  User, LogOut, ChevronDown, ChevronRight, CheckCircle2, Circle, 
  Sparkles, PlayCircle, ArrowRight, ShieldCheck, Menu, X, 
  Calendar, Clock, BarChart3, ArrowLeft, RefreshCw
} from 'lucide-react';
import { translations } from '../utils/translations';
import TraditionalDivider from '../components/TraditionalDivider';
import ModuleLessonViewer from '../components/ModuleLessonViewer';
import AssignmentViewer from '../components/AssignmentViewer';
import PracticeStudio from '../components/PracticeStudio';
import { 
  fetchModules,
  fetchProgress, 
  toggleLessonProgress, 
  fetchQuizHistory,
  fetchPracticeHistory,
  fetchAdminMetrics, 
  fetchAdminStudents,
  fetchAdminQuizResults,
  updateLanguagePreference
} from '../services/api';

export default function StudentDashboard({ 
  user, 
  currentLang, 
  onLogout, 
  onBackToHome 
}) {
  const t = translations[currentLang] || translations.en;
  
  // Permanent Navigation state:
  // 'dashboard', 'modules' (or 'module-1'..'module-7'), 'practice', 'assignments', 'progress', 'results', 'profile', 'admin'
  const [activeTab, setActiveTab] = useState('dashboard');
  const [modulesSubmenuOpen, setModulesSubmenuOpen] = useState(true);
  const [sidebarOpenMobile, setSidebarOpenMobile] = useState(false);

  // Student progress tracking & results
  const [completedModules, setCompletedModules] = useState([]);
  const [completedLessonIds, setCompletedLessonIds] = useState([]);
  const [modulesList, setModulesList] = useState([]);
  const [quizHistory, setQuizHistory] = useState([]);
  const [practiceHistory, setPracticeHistory] = useState([]);
  const [loadingResults, setLoadingResults] = useState(false);

  // Admin stats (strictly populated if user?.role === 'admin')
  const [adminMetrics, setAdminMetrics] = useState(null);
  const [adminStudents, setAdminStudents] = useState([]);
  const [adminQuizResults, setAdminQuizResults] = useState([]);

  // Canonical full name display
  const displayName = user?.profile?.full_name || user?.name || user?.email?.split('@')[0] || 'Dance Student';

  // Load progress and curriculum data on mount
  useEffect(() => {
    fetchModules()
      .then(mods => {
        if (mods && mods.length > 0) setModulesList(mods);
      })
      .catch(() => {});

    fetchProgress()
      .then(data => {
        if (data) {
          if (Array.isArray(data.completed_module_ids)) {
            setCompletedModules(data.completed_module_ids);
          }
          if (Array.isArray(data.completed_lesson_ids)) {
            setCompletedLessonIds(data.completed_lesson_ids);
          }
        }
      })
      .catch(() => {});

    if (user?.role === 'admin') {
      fetchAdminMetrics().then(setAdminMetrics).catch(() => {});
      fetchAdminStudents().then(setAdminStudents).catch(() => {});
      fetchAdminQuizResults().then(setAdminQuizResults).catch(() => {});
    }
  }, [user?.role]);

  // Load student results whenever results or progress tab is activated
  useEffect(() => {
    if (activeTab === 'results' || activeTab === 'progress') {
      setLoadingResults(true);
      Promise.allSettled([
        fetchQuizHistory().then(res => setQuizHistory(Array.isArray(res) ? res : [])),
        fetchPracticeHistory().then(res => setPracticeHistory(Array.isArray(res) ? res : []))
      ]).finally(() => {
        setLoadingResults(false);
      });
    }
  }, [activeTab]);

  const handleToggleLesson = async (modId, lessonId) => {
    const isNowDone = !completedLessonIds.includes(lessonId);
    const newDone = isNowDone
      ? [...completedLessonIds, lessonId]
      : completedLessonIds.filter(id => id !== lessonId);
    setCompletedLessonIds(newDone);

    try {
      await toggleLessonProgress(modId, lessonId, isNowDone);
      const p = await fetchProgress();
      if (p) {
        setCompletedModules(p.completed_module_ids || []);
        setCompletedLessonIds(p.completed_lesson_ids || []);
      }
    } catch (e) {
      console.warn("Failed to update lesson completion in SQLite:", e);
    }
  };

  const allModules = [
    {
      id: 1,
      title: currentLang === 'te' ? 'విభాగం 1: కూచిపూడి పరిచయం' : 'Module 1: Introduction to Kuchipudi',
      sanskrit: 'नाट्य शास्त्र प्रवेशम्',
      shortTitle: 'Module 1',
      lessonsCount: 4,
      desc: currentLang === 'te' 
        ? 'కూచిపూడి చారిత్రక మూలాలు, సిద్ధేంద్ర యోగి నాట్య సంప్రదాయం, నృత్త, నృత్య మరియు నాట్యాల సంగమం.' 
        : 'Historical origins from Kuchelapuram, Siddhendra Yogi tradition, and the triad of Nritta, Nritya, and Natya.',
      items: [
        'Origins in Kuchelapuram & Siddhendra Yogi',
        'Natya Shastra & Abhinaya Darpana Traditions',
        'Angika, Vachika, Aharya & Sattvika Abhinaya',
        'Foundations of Rasa (Navarasas) & Bhava'
      ]
    },
    {
      id: 2,
      title: currentLang === 'te' ? 'విభాగం 2: అసంయుత హస్తాలు' : 'Module 2: Asamyukta Hasthas',
      sanskrit: 'असंयुत हस्ताः (२८ भेदाः)',
      shortTitle: 'Module 2',
      lessonsCount: 28,
      desc: currentLang === 'te' 
        ? 'అభినయ దర్పణంలోని 28 ఏకహస్త ముద్రలు: పతాక నుండి త్రిశూల వరకు.' 
        : 'The complete canon of 28 single-hand gestures from Pathaka to Trishoola.',
      items: [
        '1. Pathaka (Flag)', '2. Thripathaka', '3. Ardhapathaka', '4. Kartarimukha',
        '5. Mayura (Peacock)', '6. Ardhachandra', '7. Arala', '8. Sukatunda',
        '9. Mushti (Fist)', '10. Shikhara', '11. Kapittha', '12. Katakamukha',
        '13. Soochi (Needle)', '14. Chandrakala', '15. Padmakosha', '16. Sarpasheersha',
        '17. Mrugasheersha', '18. Simhamukha', '19. Kaangula', '20. Alapadma',
        '21. Chatura', '22. Bhramara', '23. Hamsasya', '24. Hamsapaksha',
        '25. Samdamsha', '26. Mukula', '27. Tamrachuda', '28. Trishoola'
      ]
    },
    {
      id: 3,
      title: currentLang === 'te' ? 'విభాగం 3: సంయుత హస్తాలు' : 'Module 3: Samyukta Hasthas',
      sanskrit: 'संयुत हस्ताः (२४ भेदाः)',
      shortTitle: 'Module 3',
      lessonsCount: 24,
      desc: currentLang === 'te' 
        ? '24 సంయుత ద్విహస్త ముద్రలు: అంజలి నుండి అవహితము వరకు.' 
        : 'The 24 combined two-hand gestures from Anjali to Avahithamu.',
      items: [
        '1. Anjali (Salutation)', '2. Kapothamu (Dove)', '3. Karkatamu (Crab)', '4. Swasthikamu',
        '5. Dola Hastamu (Swing)', '6. Pushpaputamu', '7. Uthsungamu', '8. Sivalingamu',
        '9. Katakavardhanamu', '10. Karthariswasthikamu', '11. Sakatamu', '12. Sankhamu (Conch)',
        '13. Chakramu (Discus)', '14. Samputamu', '15. Paasamu', '16. Keelakamu',
        '17. Matsyamu (Fish)', '18. Kurmamu (Tortoise)', '19. Varahamu', '20. Garudamu',
        '21. Nagabandhakamu', '22. Khatva', '23. Bherundamu', '24. Avahithamu'
      ]
    },
    {
      id: 4,
      title: currentLang === 'te' ? 'విభాగం 4: శిరో భేదాలు' : 'Module 4: Shiro Bhedhas',
      sanskrit: 'शिरो भेदाः (९ भेदाः)',
      shortTitle: 'Module 4',
      lessonsCount: 9,
      desc: currentLang === 'te' 
        ? '9 సంప్రదాయ తల కదలికలు: సమ, ఉద్వాహితం, అధోముఖం, ఆలోలితం, ధుతం, కంపితం, పరావృత్తం, ఉత్క్షిప్తం, పరివాహితం.' 
        : 'The 9 classical head and neck movements from Sama to Parivahittam.',
      items: [
        '1. Sama (Level / Centered)', '2. Udvahitam (Raised up)', '3. Adhomukham (Cast down)',
        '4. Alolitam (Circular rolling)', '5. Dhutam (Side to side shake)', '6. Kampitam (Nodding)',
        '7. Paravrittam (Turned aside)', '8. Ukshiptam (Tilted upwards)', '9. Parivahittam (Gently swinging)'
      ]
    },
    {
      id: 5,
      title: currentLang === 'te' ? 'విభాగం 5: దృష్టి భేదాలు' : 'Module 5: Drishti Bhedhas',
      sanskrit: 'दृष्टि भेदाः (८ भेदाः)',
      shortTitle: 'Module 5',
      lessonsCount: 8,
      desc: currentLang === 'te' 
        ? '8 ప్రామాణిక నేత్ర వీక్షణలు: సమ, ఆలోకిత, సాచి, ప్రలోకిత, నిమీలిత, ఉల్లోకిత, అనువృత్త, అవలోకిత.' 
        : 'The 8 foundational eye glances from Sama to Avalokita.',
      items: [
        '1. Sama (Straight gaze)', '2. Alokita (Keen glance all around)', '3. Saachi (Sidelong glance)',
        '4. Pralokita (Looking side to side)', '5. Nimilita (Half-closed meditative gaze)',
        '6. Ullokita (Looking upwards)', '7. Anuvritta (Glancing rapidly up & down)', '8. Avalokita (Looking downwards)'
      ]
    },
    {
      id: 6,
      title: currentLang === 'te' ? 'విభాగం 6: ద్వాదశ హస్త ప్రణామాలు' : 'Module 6: Dvadasha Hasta Pranamas',
      sanskrit: 'द्वादश हस्त प्रणामाः',
      shortTitle: 'Module 6',
      lessonsCount: 12,
      desc: currentLang === 'te' 
        ? '12 శాస్త్రీయ హస్త ప్రణామాలు మరియు ఆచార నమస్కార పద్ధతులు.' 
        : 'The 12 traditional classical hand salutations expressing reverence and offering.',
      items: [
        '1. Prasaranam', '2. Kunchitam', '3. Rechitam', '4. Pumkhitam',
        '5. Apaveshtitam', '6. Preritam', '7. Udveshtitam', '8. Vyavrittam',
        '9. Parivrittam', '10. Sanketam', '11. Chihnam', '12. Padardha Tika'
      ]
    },
    {
      id: 7,
      title: currentLang === 'te' ? 'విభాగం 7: సప్త తాళములు' : 'Module 7: Sapta Talamulu',
      sanskrit: 'सप्त तालमुलु (३५ ताल व्यवस्था)',
      shortTitle: 'Module 7',
      lessonsCount: 35,
      desc: currentLang === 'te' 
        ? '7 ప్రధాన తాళాలు, లఘు-ధ్రుత-అనుధ్రుత అంగాలు, 5 జాతులు మరియు 35 తాళాల సమగ్ర పట్టిక.' 
        : 'The 7 principal Talas, Angamulu (Laghu, Dhrutam), 5 Jaatis, and the complete 35 Tala system.',
      items: [
        '1. Dhruva Tala (I O I I)', '2. Matya Tala (I O I)', '3. Rupaka Tala (O I)',
        '4. Jhampe Tala (I U O)', '5. Triputa Tala (I O O)', '6. Ata Tala (I I O O)', '7. Eka Tala (I)',
        'Pancha Jaatis: Tisra (3), Chaturasra (4), Khanda (5), Misra (7), Sankeerna (9)',
        'Mathematical 35 Tala Matrix'
      ]
    }
  ];

  // Dynamic next incomplete module identification
  const getNextIncompleteModule = () => {
    for (let m of allModules) {
      if (!completedModules.includes(m.id)) {
        return m;
      }
    }
    return null;
  };

  const nextModule = getNextIncompleteModule();
  const progressPercent = Math.round((completedModules.length / allModules.length) * 100);

  // Active individual module view (if activeTab is 'module-1'..'module-7')
  const currentModuleView = allModules.find(m => `module-${m.id}` === activeTab);

  return (
    <div className="lms-layout-wrapper">
      {/* ===================================================================
          PERMANENT LEFT SIDEBAR (8 REQUIRED ITEMS FOR STUDENTS + ADMIN)
          =================================================================== */}
      <aside className={`lms-sidebar ${sidebarOpenMobile ? 'mobile-open' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-brand" onClick={onBackToHome}>
            <div className="sidebar-logo-diya">🪔</div>
            <div>
              <strong className="sidebar-app-name">Nritya AI</strong>
              <span className="sidebar-app-sub">Kuchipudi Learning</span>
            </div>
          </div>
          <button 
            className="sidebar-close-mobile" 
            onClick={() => setSidebarOpenMobile(false)}
            aria-label="Close menu"
          >
            <X size={18} />
          </button>
        </div>

        {/* User Mini Profile Card */}
        <div className="sidebar-user-card">
          <div className="user-avatar-circle">
            <User size={18} />
          </div>
          <div className="user-meta">
            <strong className="user-name">{displayName}</strong>
            <span className="user-role-badge">
              {user?.role === 'admin' ? (t.authAdminRole || 'Instructor / Admin') : (t.authStudentRole || 'Dance Student')}
            </span>
          </div>
        </div>

        {/* Permanent 8-Item Navigation Menu */}
        <nav className="sidebar-nav">
          {/* 1. Dashboard */}
          <button
            onClick={() => { setActiveTab('dashboard'); setSidebarOpenMobile(false); }}
            className={`sb-nav-item ${activeTab === 'dashboard' ? 'active' : ''}`}
          >
            <LayoutDashboard size={18} />
            <span>{t.sbDashboard || 'Dashboard'}</span>
          </button>

          {/* 2. Learning Modules (Modules 1 to 7 with collapsible submenu) */}
          <div className="sb-nav-group">
            <div className={`sb-nav-item has-sub ${activeTab === 'modules' || activeTab.startsWith('module-') ? 'active' : ''}`}>
              <button 
                className="sb-nav-item-btn"
                onClick={() => { setActiveTab('modules'); setSidebarOpenMobile(false); }}
              >
                <BookOpen size={18} />
                <span>{t.sbLearningModules || 'Learning Modules'}</span>
              </button>
              <button 
                className="sb-nav-toggle-btn"
                onClick={(e) => { e.stopPropagation(); setModulesSubmenuOpen(!modulesSubmenuOpen); }}
                aria-label="Toggle modules list"
              >
                {modulesSubmenuOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              </button>
            </div>

            {modulesSubmenuOpen && (
              <div className="sb-sub-list">
                {allModules.map((m) => {
                  const isDone = completedModules.includes(m.id);
                  const isCurrent = `module-${m.id}` === activeTab;
                  return (
                    <button
                      key={m.id}
                      onClick={() => { setActiveTab(`module-${m.id}`); setSidebarOpenMobile(false); }}
                      className={`sb-sub-item ${isCurrent ? 'active' : ''}`}
                    >
                      {isDone ? (
                        <CheckCircle2 size={13} className="icon-done" />
                      ) : (
                        <Circle size={13} className="icon-pending" />
                      )}
                      <span className="sub-item-text">{m.shortTitle}: {m.title.split(':')[1] || m.title}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* 3. Practice Studio */}
          <button
            onClick={() => { setActiveTab('practice'); setSidebarOpenMobile(false); }}
            className={`sb-nav-item ${activeTab === 'practice' ? 'active' : ''}`}
          >
            <Video size={18} />
            <span>{t.sbPracticeStudio || 'Practice Studio'}</span>
          </button>

          {/* 4. Assignments & Quizzes */}
          <button
            onClick={() => { setActiveTab('assignments'); setSidebarOpenMobile(false); }}
            className={`sb-nav-item ${activeTab === 'assignments' ? 'active' : ''}`}
          >
            <Award size={18} />
            <span>{t.sbAssignmentsQuizzes || 'Assignments & Quizzes'}</span>
          </button>

          {/* 5. My Progress */}
          <button
            onClick={() => { setActiveTab('progress'); setSidebarOpenMobile(false); }}
            className={`sb-nav-item ${activeTab === 'progress' ? 'active' : ''}`}
          >
            <TrendingUp size={18} />
            <span>{t.sbMyProgress || 'My Progress'}</span>
          </button>

          {/* 6. Results */}
          <button
            onClick={() => { setActiveTab('results'); setSidebarOpenMobile(false); }}
            className={`sb-nav-item ${activeTab === 'results' ? 'active' : ''}`}
          >
            <FileCheck2 size={18} />
            <span>{t.sbResults || 'Results'}</span>
          </button>

          {/* 7. My Profile */}
          <button
            onClick={() => { setActiveTab('profile'); setSidebarOpenMobile(false); }}
            className={`sb-nav-item ${activeTab === 'profile' ? 'active' : ''}`}
          >
            <User size={18} />
            <span>{t.sbMyProfile || 'My Profile'}</span>
          </button>

          {/* Admin Management (Strictly visible ONLY for role === 'admin') */}
          {user?.role === 'admin' && (
            <div style={{ marginTop: '14px', borderTop: '1px dashed var(--color-border)', paddingTop: '10px' }}>
              <button
                onClick={() => { setActiveTab('admin'); setSidebarOpenMobile(false); }}
                className={`sb-nav-item ${activeTab === 'admin' ? 'active' : ''}`}
                style={{ color: '#822727' }}
              >
                <ShieldCheck size={18} />
                <span>{t.sbAdminManagement || 'Admin Management'}</span>
              </button>
            </div>
          )}
        </nav>

        {/* 8. Logout (Fixed at bottom) */}
        <div className="sidebar-footer">
          <button onClick={onLogout} className="sb-logout-btn">
            <LogOut size={16} />
            <span>{t.sbLogout || 'Logout'}</span>
          </button>
        </div>
      </aside>

      {/* Backdrop for mobile drawer */}
      {sidebarOpenMobile && (
        <div className="mobile-backdrop" onClick={() => setSidebarOpenMobile(false)} />
      )}

      {/* ===================================================================
          MAIN CONTENT AREA (Changes dynamically based on activeTab)
          =================================================================== */}
      <main className="lms-main-content">
        {/* Top Control Bar with breadcrumb and Home link */}
        <div className="lms-top-bar">
          <div className="lms-mobile-trigger">
            <button 
              onClick={() => setSidebarOpenMobile(true)}
              className="btn-open-sidebar"
              aria-label="Open sidebar"
            >
              <Menu size={20} />
              <span>Menu</span>
            </button>
          </div>

          <div className="top-breadcrumb">
            <span>Nritya AI</span>
            <span>/</span>
            <strong className="current-tab-label">
              {activeTab === 'dashboard' && (t.sbDashboard || 'Dashboard')}
              {activeTab === 'modules' && (t.sbLearningModules || 'Learning Modules')}
              {currentModuleView && currentModuleView.title}
              {activeTab === 'practice' && (t.sbPracticeStudio || 'Practice Studio')}
              {activeTab === 'assignments' && (t.sbAssignmentsQuizzes || 'Assignments & Quizzes')}
              {activeTab === 'progress' && (t.sbMyProgress || 'My Progress')}
              {activeTab === 'results' && (t.sbResults || 'Results')}
              {activeTab === 'profile' && (t.sbMyProfile || 'My Profile')}
              {activeTab === 'admin' && (t.sbAdminManagement || 'Admin Management')}
            </strong>
          </div>

          <button onClick={onBackToHome} className="btn-back-home">
            <span>Public Home</span>
          </button>
        </div>

        <div className="lms-content-container">
          {/* ===============================================================
              VIEW 1: DASHBOARD OVERVIEW
              =============================================================== */}
          {activeTab === 'dashboard' && (
            <div className="dash-overview-view">
              {/* Personalized Welcome Banner */}
              <div className="welcome-banner card-soft">
                <div className="welcome-text">
                  <span className="welcome-tag">Kuchipudi Natya Shiksha</span>
                  <h2 className="welcome-title">
                    {t.dashWelcome || 'Welcome back'}, {displayName}!
                  </h2>
                  <p className="welcome-desc">
                    {currentLang === 'te' 
                      ? 'మీ కూచిపూడి అభ్యాస యాత్రను కొనసాగించండి. మీ సిద్ధాంత పురోగతి మరియు క్విజ్ ఫలితాలు ఇక్కడ పొందుపరచబడ్డాయి.'
                      : 'Continue your sacred Kuchipudi journey. Your theory progress and practice assessments are tracked directly in this student portal.'}
                  </p>

                  <div className="welcome-actions">
                    {nextModule ? (
                      <button 
                        onClick={() => setActiveTab(`module-${nextModule.id}`)} 
                        className="btn-primary"
                      >
                        <PlayCircle size={18} />
                        <span>{t.dashContinueLearning || 'Continue Next Lesson'}: {nextModule.shortTitle}</span>
                      </button>
                    ) : (
                      <button 
                        onClick={() => setActiveTab('practice')} 
                        className="btn-primary"
                      >
                        <Video size={18} />
                        <span>Launch AI Practice Studio</span>
                      </button>
                    )}
                  </div>
                </div>

                <div className="progress-ring-card">
                  <div className="progress-circle-wrap">
                    <span className="progress-percent-number">{progressPercent}%</span>
                    <span className="progress-percent-label">Completed</span>
                  </div>
                </div>
              </div>

              {/* Next Incomplete Lesson Alert */}
              {nextModule ? (
                <div className="next-lesson-card card-sage">
                  <div className="next-lesson-info">
                    <span className="next-tag">{t.dashNextLessonTitle || 'Next Incomplete Lesson'}</span>
                    <h3 className="next-title">{nextModule.title}</h3>
                    <p className="next-desc">{nextModule.desc}</p>
                  </div>
                  <button 
                    onClick={() => setActiveTab(`module-${nextModule.id}`)}
                    className="btn-green"
                  >
                    <span>Resume Lesson</span>
                    <ArrowRight size={16} />
                  </button>
                </div>
              ) : (
                <div className="all-completed-banner card-lavender">
                  <Sparkles size={24} className="sparkle-gold" />
                  <div>
                    <strong>{t.dashAllCompleteTitle || 'All Core Theory Modules Completed!'}</strong>
                    <p>{t.dashAllCompleteDesc || 'Congratulations on mastering the foundational theory. Head to the Practice Studio to refine your physical mudras.'}</p>
                  </div>
                </div>
              )}

              {/* 3 Metrics Summary Cards */}
              <div className="dash-metrics-grid">
                <div className="metric-box card-soft" onClick={() => setActiveTab('modules')} style={{ cursor: 'pointer' }}>
                  <div className="metric-icon-wrap m-plum">
                    <BookOpen size={20} />
                  </div>
                  <div>
                    <span className="metric-label">{t.dashCompletedCount || 'Modules Completed'}</span>
                    <strong className="metric-val">{completedModules.length} / {allModules.length}</strong>
                  </div>
                </div>

                <div className="metric-box card-soft" onClick={() => setActiveTab('assignments')} style={{ cursor: 'pointer' }}>
                  <div className="metric-icon-wrap m-green">
                    <Award size={20} />
                  </div>
                  <div>
                    <span className="metric-label">{t.dashPendingAssignments || 'Quizzes & Assignments'}</span>
                    <strong className="metric-val">3 Available</strong>
                  </div>
                </div>

                <div className="metric-box card-soft" onClick={() => setActiveTab('results')} style={{ cursor: 'pointer' }}>
                  <div className="metric-icon-wrap m-gold">
                    <Sparkles size={20} />
                  </div>
                  <div>
                    <span className="metric-label">{t.dashRecentQuizScore || 'Latest Assessment Score'}</span>
                    <strong className="metric-val">
                      {quizHistory.length > 0 ? `${quizHistory[0].percentage}%` : 'Not Taken Yet'}
                    </strong>
                  </div>
                </div>
              </div>

              {/* Modules Progress Table */}
              <div className="modules-status-section card-soft">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px' }}>
                  <h3 className="section-subheading" style={{ margin: 0 }}>
                    {currentLang === 'te' ? 'ఏడు అభ్యాస విభాగాల పురోగతి' : 'Seven Theory Modules Status'}
                  </h3>
                  <button 
                    onClick={() => setActiveTab('modules')} 
                    className="btn-outline" 
                    style={{ fontSize: '0.8rem', padding: '6px 14px' }}
                  >
                    View All Modules
                  </button>
                </div>
                <div className="module-status-table">
                  {allModules.map((m) => {
                    const isDone = completedModules.includes(m.id);
                    return (
                      <div key={m.id} className="module-status-row">
                        <div className="mod-info-col">
                          <span className="mod-id-badge">0{m.id}</span>
                          <div>
                            <strong className="mod-name">{m.title}</strong>
                            <span className="mod-sanskrit">{m.sanskrit}</span>
                          </div>
                        </div>

                        <div className="mod-status-col">
                          {isDone ? (
                            <span className="status-pill completed">
                              <CheckCircle2 size={14} /> Completed
                            </span>
                          ) : (
                            <span className="status-pill pending">
                              Incomplete
                            </span>
                          )}
                        </div>

                        <div className="mod-action-col">
                          <button 
                            onClick={() => setActiveTab(`module-${m.id}`)}
                            className="btn-open-mod"
                          >
                            <span>Open</span>
                            <ChevronRight size={14} />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* ===============================================================
              VIEW 2: LEARNING MODULES CATALOG (All 7 Modules Grid)
              =============================================================== */}
          {activeTab === 'modules' && (
            <div className="modules-catalog-view">
              <div className="section-header">
                <span className="section-tag">Nritya AI Theory Curriculum</span>
                <h2 className="section-title">
                  {currentLang === 'te' ? 'కూచిపూడి సమగ్ర అభ్యాస విభాగాలు (1-7)' : 'Comprehensive Kuchipudi Learning Modules'}
                </h2>
                <TraditionalDivider motif="lotus" />
                <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.92rem', maxWidth: '720px' }}>
                  {currentLang === 'te'
                    ? 'నాట్యశాస్త్రం మరియు అభినయ దర్పణం గ్రంథాల ఆధారంగా రూపొందించబడిన 7 ప్రామాణిక విభాగాలు. ప్రతి విభాగంలో పాఠాలు, ముద్రలు మరియు పరీక్షలు కలవు.'
                    : 'The authentic curriculum compiled from Natya Shastra and Abhinaya Darpana treatises across 7 foundational modules.'}
                </p>
              </div>

              <div className="modules-grid-cards">
                {allModules.map((m) => {
                  const isDone = completedModules.includes(m.id);
                  return (
                    <div key={m.id} className="module-card card-soft">
                      <div className="module-card-top">
                        <span className="mod-number-badge">Module 0{m.id}</span>
                        {isDone ? (
                          <span className="status-pill completed"><CheckCircle2 size={13} /> Completed</span>
                        ) : (
                          <span className="status-pill pending">In Progress</span>
                        )}
                      </div>
                      <h3 className="mod-card-title">{m.title}</h3>
                      <span className="mod-card-sanskrit">{m.sanskrit}</span>
                      <p className="mod-card-desc">{m.desc}</p>
                      
                      <div className="mod-card-meta">
                        <span>{m.lessonsCount} Canonical Lessons</span>
                      </div>

                      <button 
                        onClick={() => setActiveTab(`module-${m.id}`)}
                        className="btn-primary"
                        style={{ width: '100%', marginTop: '16px', display: 'flex', justifyContent: 'center' }}
                      >
                        <span>{isDone ? 'Review Module' : 'Study Module'}</span>
                        <ChevronRight size={16} />
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ===============================================================
              VIEW 2B: INDIVIDUAL MODULE LESSON VIEWER
              =============================================================== */}
          {currentModuleView && (
            <div className="individual-module-wrapper">
              <div style={{ marginBottom: '16px' }}>
                <button 
                  onClick={() => setActiveTab('modules')} 
                  className="btn-back-link"
                  style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--color-deep-plum)', fontWeight: 600, fontSize: '0.85rem' }}
                >
                  <ArrowLeft size={16} />
                  <span>← Back to All Modules</span>
                </button>
              </div>
              <ModuleLessonViewer 
                module={modulesList.find(m => m.module_number === currentModuleView.id) || currentModuleView}
                currentLang={currentLang}
                completedLessonIds={completedLessonIds}
                onToggleLesson={(lesId) => handleToggleLesson(currentModuleView.id, lesId)}
                onLaunchPractice={() => setActiveTab('practice')}
              />
            </div>
          )}

          {/* ===============================================================
              VIEW 3: PRACTICE STUDIO TAB (STAGE 6 REAL AI ENGINE)
              =============================================================== */}
          {activeTab === 'practice' && (
            <PracticeStudio currentLang={currentLang} />
          )}

          {/* ===============================================================
              VIEW 4: ASSIGNMENTS & QUIZZES (STAGE 5 REAL SYSTEM)
              =============================================================== */}
          {activeTab === 'assignments' && (
            <AssignmentViewer 
              currentLang={currentLang}
              onLaunchCameraExam={() => setActiveTab('practice')}
            />
          )}

          {/* ===============================================================
              VIEW 5: MY PROGRESS
              =============================================================== */}
          {activeTab === 'progress' && (
            <div className="lms-progress-view card-soft">
              <div className="section-header">
                <span className="section-tag">Learning Analytics</span>
                <h2 className="section-title">
                  {currentLang === 'te' ? 'నా అభ్యాస పురోగతి మరియు రికార్డులు' : 'My Learning Progress & Records'}
                </h2>
                <TraditionalDivider motif="lotus" />
              </div>

              {/* Summary Stats */}
              <div className="profile-details-grid">
                <div className="profile-field card-blush">
                  <span className="p-label">Modules Completed</span>
                  <strong className="p-val">{completedModules.length} of {allModules.length} Modules ({progressPercent}%)</strong>
                </div>
                <div className="profile-field card-sage">
                  <span className="p-label">Completed Individual Lessons</span>
                  <strong className="p-val">{completedLessonIds.length} of 92 Canonical Lessons</strong>
                </div>
                <div className="profile-field card-lavender">
                  <span className="p-label">Quizzes Completed</span>
                  <strong className="p-val">{quizHistory.length} Submissions</strong>
                </div>
                <div className="profile-field card-sage">
                  <span className="p-label">AI Practice Sessions Logged</span>
                  <strong className="p-val">{practiceHistory.length} Sessions</strong>
                </div>
              </div>

              {/* Detailed Breakdown Per Module */}
              <div style={{ marginTop: '32px' }}>
                <h3 style={{ fontSize: '1.1rem', color: 'var(--color-deep-plum)', marginBottom: '16px' }}>
                  {currentLang === 'te' ? 'విభాగాల వారీగా పురోగతి' : 'Curriculum Breakdown by Module'}
                </h3>
                <div className="module-status-table">
                  {allModules.map((m) => {
                    const isDone = completedModules.includes(m.id);
                    return (
                      <div key={m.id} className="module-status-row" style={{ padding: '16px 20px' }}>
                        <div className="mod-info-col" style={{ flex: 1 }}>
                          <span className="mod-id-badge">0{m.id}</span>
                          <div style={{ width: '100%', maxWidth: '380px' }}>
                            <strong className="mod-name">{m.title}</strong>
                            <span className="mod-sanskrit">{m.sanskrit} ({m.lessonsCount} lessons)</span>
                          </div>
                        </div>

                        <div style={{ width: '160px', margin: '0 20px' }}>
                          <div style={{ height: '8px', background: '#E0D6CE', borderRadius: '4px', overflow: 'hidden' }}>
                            <div style={{ 
                              height: '100%', 
                              width: isDone ? '100%' : '15%', 
                              background: isDone ? 'var(--color-temple-green)' : 'var(--color-gold-dark)',
                              borderRadius: '4px' 
                            }} />
                          </div>
                          <span style={{ fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '4px', display: 'block' }}>
                            {isDone ? '100% Complete' : 'In Progress'}
                          </span>
                        </div>

                        <div className="mod-action-col">
                          <button 
                            onClick={() => setActiveTab(`module-${m.id}`)}
                            className="btn-open-mod"
                          >
                            <span>Resume</span>
                            <ChevronRight size={14} />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* ===============================================================
              VIEW 6: RESULTS (Quiz Submissions & AI Practice History)
              =============================================================== */}
          {activeTab === 'results' && (
            <div className="lms-results-view card-soft">
              <div className="section-header">
                <span className="section-tag">Performance History</span>
                <h2 className="section-title">
                  {currentLang === 'te' ? 'పరీక్షా ఫలితాలు మరియు సాధన లాగ్‌లు' : 'Assessment Results & Practice Logs'}
                </h2>
                <TraditionalDivider motif="lotus" />
                <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.92rem' }}>
                  {currentLang === 'te'
                    ? 'మీరు సమర్పించిన అన్ని క్విజ్‌లు మరియు AI కెమెరా ద్వారా పూర్తి చేసిన సాధన రికార్డులు క్రింద చూపబడ్డాయి.'
                    : 'Review your historical quiz scores, knowledge assessments, and computer vision AI practice records.'}
                </p>
              </div>

              {/* Sub-section 1: Quiz Submissions */}
              <div style={{ marginTop: '24px' }}>
                <h3 style={{ fontSize: '1.1rem', color: 'var(--color-deep-plum)', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Award size={18} />
                  <span>{currentLang === 'te' ? 'క్విజ్ పరీక్షా ఫలితాలు' : 'Quiz & Exam Submissions'}</span>
                </h3>

                {loadingResults ? (
                  <p style={{ color: 'var(--color-text-secondary)', padding: '16px' }}>Loading historical results...</p>
                ) : quizHistory.length > 0 ? (
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
                      <thead>
                        <tr style={{ background: 'var(--color-blush-light)', textAlign: 'left', borderBottom: '1.5px solid var(--color-border)' }}>
                          <th style={{ padding: '12px' }}>Assessment Title</th>
                          <th style={{ padding: '12px' }}>Type</th>
                          <th style={{ padding: '12px' }}>Score</th>
                          <th style={{ padding: '12px' }}>Percentage</th>
                          <th style={{ padding: '12px' }}>Status</th>
                          <th style={{ padding: '12px' }}>Completed At</th>
                        </tr>
                      </thead>
                      <tbody>
                        {quizHistory.map(r => (
                          <tr key={r.id} style={{ borderBottom: '1px solid var(--color-border)' }}>
                            <td style={{ padding: '12px', fontWeight: 600 }}>{r.quiz_title_en}</td>
                            <td style={{ padding: '12px' }}>
                              <span style={{ textTransform: 'uppercase', fontSize: '0.72rem', fontWeight: 700, color: 'var(--plum-muted)' }}>
                                {r.quiz_type}
                              </span>
                            </td>
                            <td style={{ padding: '12px', fontWeight: 700 }}>{r.score} / {r.total_questions}</td>
                            <td style={{ padding: '12px' }}>{r.percentage}%</td>
                            <td style={{ padding: '12px' }}>
                              <span style={{
                                padding: '4px 10px',
                                borderRadius: '12px',
                                fontSize: '0.75rem',
                                fontWeight: 700,
                                background: r.passed ? '#DEF7EC' : '#FDE8E8',
                                color: r.passed ? '#03543F' : '#9B1C1C'
                              }}>
                                {r.passed ? 'PASSED' : 'RETRY'}
                              </span>
                            </td>
                            <td style={{ padding: '12px', color: 'var(--plum-muted)', fontSize: '0.8rem' }}>
                              {r.completed_at ? new Date(r.completed_at).toLocaleString() : 'Recent'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="card-blush" style={{ padding: '24px', textAlign: 'center', borderRadius: 'var(--radius-sm)' }}>
                    <p style={{ color: 'var(--color-deep-plum)', fontWeight: 600, marginBottom: '6px' }}>
                      No quiz submissions found yet.
                    </p>
                    <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.85rem', marginBottom: '14px' }}>
                      Take a module quiz in the Assignments section to test your classical theory knowledge.
                    </p>
                    <button onClick={() => setActiveTab('assignments')} className="btn-primary">
                      Take Quiz Now
                    </button>
                  </div>
                )}
              </div>

              {/* Sub-section 2: AI Practice Sessions */}
              <div style={{ marginTop: '36px' }}>
                <h3 style={{ fontSize: '1.1rem', color: 'var(--color-deep-plum)', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Video size={18} />
                  <span>{currentLang === 'te' ? 'AI సాధన రికార్డులు' : 'AI Camera Practice Sessions'}</span>
                </h3>

                {practiceHistory.length > 0 ? (
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
                      <thead>
                        <tr style={{ background: 'var(--color-blush-light)', textAlign: 'left', borderBottom: '1.5px solid var(--color-border)' }}>
                          <th style={{ padding: '12px' }}>Category</th>
                          <th style={{ padding: '12px' }}>Duration</th>
                          <th style={{ padding: '12px' }}>Accuracy</th>
                          <th style={{ padding: '12px' }}>Gestures Practiced</th>
                          <th style={{ padding: '12px' }}>Timestamp</th>
                        </tr>
                      </thead>
                      <tbody>
                        {practiceHistory.map((s, idx) => (
                          <tr key={s.id || idx} style={{ borderBottom: '1px solid var(--color-border)' }}>
                            <td style={{ padding: '12px', fontWeight: 600, textTransform: 'capitalize' }}>{s.category}</td>
                            <td style={{ padding: '12px' }}>{s.duration_seconds ? `${Math.round(s.duration_seconds / 60)} mins` : 'N/A'}</td>
                            <td style={{ padding: '12px', fontWeight: 700, color: 'var(--color-temple-green)' }}>
                              {s.accuracy_percentage ? `${Math.round(s.accuracy_percentage)}%` : 'Active'}
                            </td>
                            <td style={{ padding: '12px' }}>{s.poses_detected || s.poses_practiced || 1} mudras</td>
                            <td style={{ padding: '12px', color: 'var(--plum-muted)', fontSize: '0.8rem' }}>
                              {s.created_at ? new Date(s.created_at).toLocaleString() : 'Recent'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="card-sage" style={{ padding: '24px', textAlign: 'center', borderRadius: 'var(--radius-sm)' }}>
                    <p style={{ color: 'var(--color-deep-plum)', fontWeight: 600, marginBottom: '6px' }}>
                      No AI practice sessions logged yet.
                    </p>
                    <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.85rem', marginBottom: '14px' }}>
                      Open the Practice Studio to test your hand mudras in real time with our TensorFlow/Keras neural engine.
                    </p>
                    <button onClick={() => setActiveTab('practice')} className="btn-green">
                      Start AI Practice Studio
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ===============================================================
              VIEW 7: MY PROFILE
              =============================================================== */}
          {activeTab === 'profile' && (
            <div className="lms-profile-view card-soft">
              <div className="section-header">
                <span className="section-tag">Student Account</span>
                <h2 className="section-title">{displayName}</h2>
                <TraditionalDivider motif="lotus" />
              </div>

              <div className="profile-details-grid">
                <div className="profile-field card-blush">
                  <span className="p-label">Full Name</span>
                  <strong className="p-val">{displayName}</strong>
                </div>
                <div className="profile-field card-sage">
                  <span className="p-label">Email Address</span>
                  <strong className="p-val">{user?.email || 'student@kuchipudi.edu'}</strong>
                </div>
                <div className="profile-field card-lavender">
                  <span className="p-label">Account Role</span>
                  <strong className="p-val">
                    {user?.role === 'admin' ? 'Administrator / Dance Master' : 'Classical Dance Student'}
                  </strong>
                </div>
                <div className="profile-field card-sage">
                  <span className="p-label">Preferred Interface Language</span>
                  <strong className="p-val">
                    {currentLang === 'te' ? 'తెలుగు (Telugu)' : 'English'}
                  </strong>
                </div>
                <div className="profile-field card-blush">
                  <span className="p-label">Curriculum Completion</span>
                  <strong className="p-val">{progressPercent}% ({completedModules.length} of 7 modules)</strong>
                </div>
                <div className="profile-field card-lavender">
                  <span className="p-label">Registered Since</span>
                  <strong className="p-val">
                    {user?.created_at ? new Date(user.created_at).toLocaleDateString() : 'Active Session'}
                  </strong>
                </div>
              </div>

              {/* Language Switcher Preference */}
              <div style={{ marginTop: '30px', padding: '20px', background: 'var(--color-blush-light)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--color-border)' }}>
                <h4 style={{ color: 'var(--color-deep-plum)', marginBottom: '8px', fontSize: '0.95rem' }}>
                  Interface Language Preference
                </h4>
                <p style={{ color: 'var(--color-text-secondary)', fontSize: '0.82rem', marginBottom: '14px' }}>
                  Choose your preferred language for lesson explanations and Sanskrit translations.
                </p>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button 
                    onClick={() => updateLanguagePreference('en').then(() => window.location.reload())}
                    className={currentLang === 'en' ? 'btn-primary' : 'btn-outline'}
                    style={{ fontSize: '0.85rem', padding: '8px 16px' }}
                  >
                    English
                  </button>
                  <button 
                    onClick={() => updateLanguagePreference('te').then(() => window.location.reload())}
                    className={currentLang === 'te' ? 'btn-primary' : 'btn-outline'}
                    style={{ fontSize: '0.85rem', padding: '8px 16px' }}
                  >
                    తెలుగు (Telugu)
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* ===============================================================
              VIEW 8: ADMIN MANAGEMENT (Strictly Admin Role Only)
              =============================================================== */}
          {activeTab === 'admin' && user?.role === 'admin' && (
            <div className="lms-profile-view card-soft">
              <div className="section-header">
                <span className="section-tag">System Administration</span>
                <h2 className="section-title">Administrator Dashboard & Metrics</h2>
                <TraditionalDivider motif="lotus" />
              </div>

              <div className="profile-details-grid">
                <div className="profile-field card-blush">
                  <span className="p-label">Registered Students</span>
                  <strong className="p-val">{adminMetrics?.total_students ?? 1} Students</strong>
                </div>
                <div className="profile-field card-sage">
                  <span className="p-label">Completed Theory Modules</span>
                  <strong className="p-val">{adminMetrics?.total_completed_lessons ?? 0} Completed</strong>
                </div>
                <div className="profile-field card-lavender">
                  <span className="p-label">Database Status</span>
                  <strong className="p-val">{adminMetrics?.database_status ?? 'Healthy (SQLite kuchipudi.db)'}</strong>
                </div>
                <div className="profile-field card-sage">
                  <span className="p-label">AI Architecture Readiness</span>
                  <strong className="p-val">Stage 6 Prepared (69 Classes)</strong>
                </div>
              </div>

              <div style={{ marginTop: '24px' }}>
                <h3 style={{ fontSize: '1rem', color: 'var(--color-deep-plum)', marginBottom: '12px' }}>Registered Students</h3>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                    <thead>
                      <tr style={{ background: 'var(--color-blush-light)', textAlign: 'left', borderBottom: '1px solid var(--color-border)' }}>
                        <th style={{ padding: '10px 12px' }}>Student Name</th>
                        <th style={{ padding: '10px 12px' }}>Email Address</th>
                        <th style={{ padding: '10px 12px' }}>Language</th>
                        <th style={{ padding: '10px 12px' }}>Experience</th>
                        <th style={{ padding: '10px 12px' }}>Completed Items</th>
                      </tr>
                    </thead>
                    <tbody>
                      {adminStudents && adminStudents.length > 0 ? (
                        adminStudents.map(s => (
                          <tr key={s.id} style={{ borderBottom: '1px solid var(--color-border)' }}>
                            <td style={{ padding: '10px 12px', fontWeight: 600 }}>{s.full_name}</td>
                            <td style={{ padding: '10px 12px' }}>{s.email}</td>
                            <td style={{ padding: '10px 12px' }}>{s.preferred_language === 'te' ? 'Telugu (తెలుగు)' : 'English'}</td>
                            <td style={{ padding: '10px 12px' }}>{s.experience_level}</td>
                            <td style={{ padding: '10px 12px' }}>{s.completed_items} items</td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={5} style={{ padding: '16px', textAlign: 'center', color: 'var(--color-text-secondary)' }}>
                            No student accounts found.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Admin Assignment Results Submissions */}
              <div style={{ marginTop: '32px' }}>
                <h3 style={{ fontSize: '1rem', color: 'var(--color-deep-plum)', marginBottom: '12px' }}>
                  Student Assessment Submissions & Scores
                </h3>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                    <thead>
                      <tr style={{ background: 'var(--color-blush-light)', textAlign: 'left', borderBottom: '1px solid var(--color-border)' }}>
                        <th style={{ padding: '10px 12px' }}>Student</th>
                        <th style={{ padding: '10px 12px' }}>Assignment Title</th>
                        <th style={{ padding: '10px 12px' }}>Type</th>
                        <th style={{ padding: '10px 12px' }}>Score</th>
                        <th style={{ padding: '10px 12px' }}>Status</th>
                        <th style={{ padding: '10px 12px' }}>Submitted At</th>
                      </tr>
                    </thead>
                    <tbody>
                      {adminQuizResults && adminQuizResults.length > 0 ? (
                        adminQuizResults.map(r => (
                          <tr key={r.id} style={{ borderBottom: '1px solid var(--color-border)' }}>
                            <td style={{ padding: '10px 12px', fontWeight: 600 }}>{r.student_name} ({r.student_email})</td>
                            <td style={{ padding: '10px 12px' }}>{r.quiz_title_en}</td>
                            <td style={{ padding: '10px 12px' }}>
                              <span style={{ textTransform: 'uppercase', fontSize: '0.75rem', fontWeight: 700, color: 'var(--plum-muted)' }}>
                                {r.quiz_type}
                              </span>
                            </td>
                            <td style={{ padding: '10px 12px', fontWeight: 700 }}>
                              {r.score} / {r.total_questions} ({r.percentage}%)
                            </td>
                            <td style={{ padding: '10px 12px' }}>
                              <span style={{
                                padding: '3px 8px',
                                borderRadius: '12px',
                                fontSize: '0.75rem',
                                fontWeight: 700,
                                background: r.passed ? '#DEF7EC' : '#FDE8E8',
                                color: r.passed ? '#03543F' : '#9B1C1C'
                              }}>
                                {r.passed ? 'PASSED' : 'FAILED'}
                              </span>
                            </td>
                            <td style={{ padding: '10px 12px', color: 'var(--plum-muted)' }}>
                              {r.completed_at ? new Date(r.completed_at).toLocaleString() : 'N/A'}
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={6} style={{ padding: '16px', textAlign: 'center', color: 'var(--color-text-secondary)' }}>
                            No assignment submissions recorded yet.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      <style>{`
        .lms-layout-wrapper {
          display: flex;
          min-height: calc(100vh - 80px);
          background: var(--color-blush-light);
        }

        /* Permanent Left Sidebar */
        .lms-sidebar {
          width: 290px;
          background: #FFFFFF;
          border-right: 1.5px solid var(--color-border);
          display: flex;
          flex-direction: column;
          flex-shrink: 0;
          position: sticky;
          top: 75px;
          height: calc(100vh - 75px);
          overflow-y: auto;
          z-index: 100;
        }

        .sidebar-header {
          padding: 16px 20px;
          border-bottom: 1px solid var(--color-border);
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        .sidebar-brand {
          display: flex;
          align-items: center;
          gap: 10px;
          cursor: pointer;
        }

        .sidebar-logo-diya {
          font-size: 1.4rem;
        }

        .sidebar-app-name {
          font-family: var(--font-heading);
          font-size: 1rem;
          color: var(--color-deep-plum);
          display: block;
        }

        .sidebar-app-sub {
          font-size: 0.68rem;
          color: var(--color-temple-green);
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .sidebar-close-mobile {
          display: none;
          background: transparent;
          border: none;
          cursor: pointer;
          color: var(--color-deep-plum);
        }

        .sidebar-user-card {
          padding: 14px 20px;
          display: flex;
          align-items: center;
          gap: 12px;
          background: var(--color-blush-light);
          border-bottom: 1px solid var(--color-border);
        }

        .user-avatar-circle {
          width: 36px;
          height: 36px;
          border-radius: 50%;
          background: var(--color-deep-plum);
          color: var(--color-antique-gold);
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .user-name {
          font-size: 0.85rem;
          color: var(--color-deep-plum);
          display: block;
        }

        .user-role-badge {
          font-size: 0.7rem;
          color: var(--color-temple-green);
          font-weight: 600;
        }

        .sidebar-nav {
          padding: 12px 10px;
          display: flex;
          flex-direction: column;
          gap: 4px;
          flex: 1;
        }

        .sb-nav-item {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 10px 14px;
          border: none;
          background: transparent;
          color: var(--color-text-primary);
          font-family: var(--font-body);
          font-size: 0.88rem;
          font-weight: 600;
          border-radius: var(--radius-sm);
          text-align: left;
          cursor: pointer;
          transition: all 0.2s;
          width: 100%;
        }

        .sb-nav-item:hover {
          background: var(--color-blush-ivory);
          color: var(--color-deep-plum);
        }

        .sb-nav-item.active {
          background: var(--color-deep-plum);
          color: #FFFFFF;
          font-weight: 700;
        }

        .sb-nav-group {
          display: flex;
          flex-direction: column;
        }

        .sb-nav-item.has-sub {
          padding: 0;
          display: flex;
          justify-content: space-between;
          align-items: stretch;
        }

        .sb-nav-item-btn {
          flex: 1;
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 10px 14px;
          background: transparent;
          border: none;
          color: inherit;
          font-family: inherit;
          font-size: inherit;
          font-weight: inherit;
          text-align: left;
          cursor: pointer;
        }

        .sb-nav-toggle-btn {
          padding: 10px 12px;
          background: transparent;
          border: none;
          color: inherit;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .sb-sub-list {
          padding: 4px 0 4px 26px;
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .sb-sub-item {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 6px 12px;
          border: none;
          background: transparent;
          color: var(--color-text-secondary);
          font-size: 0.78rem;
          border-radius: 4px;
          text-align: left;
          cursor: pointer;
          transition: all 0.15s;
        }

        .sb-sub-item:hover {
          color: var(--color-deep-plum);
          background: rgba(0,0,0,0.03);
        }

        .sb-sub-item.active {
          color: var(--color-deep-plum);
          font-weight: 700;
          background: #E8DDF0;
        }

        .sub-item-text {
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .icon-done {
          color: var(--color-temple-green);
          flex-shrink: 0;
        }

        .icon-pending {
          color: #BFA06C;
          flex-shrink: 0;
        }

        .sidebar-footer {
          padding: 14px 16px;
          border-top: 1px solid var(--color-border);
        }

        .sb-logout-btn {
          width: 100%;
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 9px 12px;
          border: 1px solid #D9534F;
          background: transparent;
          color: #D9534F;
          font-size: 0.82rem;
          font-weight: 700;
          border-radius: var(--radius-sm);
          cursor: pointer;
          transition: all 0.2s;
        }

        .sb-logout-btn:hover {
          background: #D9534F;
          color: #FFFFFF;
        }

        /* Main Content */
        .lms-main-content {
          flex: 1;
          display: flex;
          flex-direction: column;
          min-width: 0;
        }

        .lms-top-bar {
          background: #FFFFFF;
          border-bottom: 1px solid var(--color-border);
          padding: 12px 30px;
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        .top-breadcrumb {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 0.85rem;
          color: var(--color-text-secondary);
        }

        .current-tab-label {
          color: var(--color-deep-plum);
          font-family: var(--font-heading);
        }

        .btn-back-home {
          padding: 6px 14px;
          background: var(--color-blush-light);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-sm);
          font-size: 0.78rem;
          font-weight: 700;
          color: var(--color-deep-plum);
          cursor: pointer;
        }

        .lms-content-container {
          padding: 30px;
          max-width: 1020px;
          width: 100%;
        }

        /* Overview View */
        .welcome-banner {
          padding: 32px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 30px;
          margin-bottom: 24px;
        }

        .welcome-tag {
          font-size: 0.75rem;
          text-transform: uppercase;
          color: var(--color-temple-green);
          font-weight: 700;
          letter-spacing: 0.1em;
          display: block;
        }

        .welcome-title {
          font-size: 1.8rem;
          color: var(--color-deep-plum);
          margin: 4px 0 8px;
        }

        .welcome-desc {
          font-size: 0.95rem;
          color: var(--color-text-secondary);
          max-width: 540px;
          line-height: 1.5;
        }

        .welcome-actions {
          margin-top: 18px;
        }

        .progress-circle-wrap {
          width: 110px;
          height: 110px;
          border-radius: 50%;
          border: 6px solid var(--color-temple-green);
          background: var(--color-sage-light);
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
        }

        .progress-percent-number {
          font-family: var(--font-heading);
          font-size: 1.8rem;
          font-weight: 800;
          color: var(--color-deep-plum);
          line-height: 1;
        }

        .progress-percent-label {
          font-size: 0.7rem;
          color: var(--color-temple-green);
          font-weight: 700;
          text-transform: uppercase;
        }

        .next-lesson-card {
          padding: 24px 28px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 20px;
          margin-bottom: 24px;
        }

        .next-tag {
          font-size: 0.75rem;
          color: var(--color-temple-green);
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          display: block;
          margin-bottom: 4px;
        }

        .next-title {
          font-size: 1.3rem;
          color: var(--color-deep-plum);
          margin-bottom: 4px;
        }

        .next-desc {
          font-size: 0.88rem;
          color: var(--color-text-secondary);
        }

        .all-completed-banner {
          padding: 20px 24px;
          display: flex;
          align-items: center;
          gap: 16px;
          margin-bottom: 24px;
        }

        .dash-metrics-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 18px;
          margin-bottom: 24px;
        }

        .metric-box {
          padding: 20px;
          display: flex;
          align-items: center;
          gap: 14px;
          transition: transform 0.2s;
        }

        .metric-box:hover {
          transform: translateY(-2px);
        }

        .metric-icon-wrap {
          width: 44px;
          height: 44px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          color: #FFFFFF;
        }
        .metric-icon-wrap.m-plum { background: var(--color-deep-plum); }
        .metric-icon-wrap.m-green { background: var(--color-temple-green); }
        .metric-icon-wrap.m-gold { background: var(--color-gold-dark); }

        .metric-label {
          font-size: 0.75rem;
          color: var(--color-text-secondary);
          display: block;
        }

        .metric-val {
          font-size: 1.25rem;
          color: var(--color-deep-plum);
          font-family: var(--font-heading);
        }

        .modules-status-section {
          padding: 28px;
        }

        .section-subheading {
          font-size: 1.15rem;
          color: var(--color-deep-plum);
        }

        .module-status-table {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .module-status-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 12px 16px;
          background: var(--color-blush-light);
          border-radius: var(--radius-sm);
          border: 1px solid var(--color-border);
        }

        .mod-info-col {
          display: flex;
          align-items: center;
          gap: 14px;
        }

        .mod-id-badge {
          font-family: var(--font-heading);
          font-weight: 800;
          color: var(--color-gold-dark);
          font-size: 1.1rem;
        }

        .mod-name {
          font-size: 0.95rem;
          color: var(--color-deep-plum);
          display: block;
        }

        .mod-sanskrit {
          font-family: var(--font-shloka);
          font-size: 0.85rem;
          color: var(--color-temple-green);
        }

        .status-pill {
          font-size: 0.75rem;
          font-weight: 700;
          padding: 4px 10px;
          border-radius: 12px;
          display: inline-flex;
          align-items: center;
          gap: 4px;
        }

        .status-pill.completed {
          background: var(--color-sage);
          color: var(--color-temple-green-dark);
        }

        .status-pill.pending {
          background: #EEDCBE;
          color: #7A5B28;
        }

        .btn-open-mod {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          padding: 6px 12px;
          background: #FFFFFF;
          border: 1px solid var(--color-border);
          border-radius: var(--radius-sm);
          font-size: 0.78rem;
          font-weight: 700;
          color: var(--color-deep-plum);
          cursor: pointer;
        }

        /* Modules Catalog Grid */
        .modules-grid-cards {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
          gap: 20px;
          margin-top: 24px;
        }

        .module-card {
          padding: 24px;
          display: flex;
          flex-direction: column;
        }

        .module-card-top {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }

        .mod-number-badge {
          font-size: 0.75rem;
          font-weight: 800;
          color: var(--color-gold-dark);
          text-transform: uppercase;
        }

        .mod-card-title {
          font-size: 1.15rem;
          color: var(--color-deep-plum);
          margin-bottom: 4px;
        }

        .mod-card-sanskrit {
          font-family: var(--font-shloka);
          color: var(--color-temple-green);
          font-size: 0.88rem;
          margin-bottom: 8px;
          display: block;
        }

        .mod-card-desc {
          font-size: 0.84rem;
          color: var(--color-text-secondary);
          line-height: 1.5;
          flex: 1;
        }

        .mod-card-meta {
          margin-top: 14px;
          font-size: 0.78rem;
          font-weight: 600;
          color: var(--color-deep-plum);
        }

        /* Profile & Progress Styles */
        .lms-profile-view,
        .lms-progress-view,
        .lms-results-view {
          padding: 32px;
        }

        .profile-details-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 16px;
          margin-top: 20px;
        }

        .profile-field {
          padding: 18px 20px;
          border-radius: var(--radius-sm);
        }

        .p-label {
          font-size: 0.75rem;
          color: var(--color-text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.08em;
          display: block;
        }

        .p-val {
          font-size: 1.05rem;
          color: var(--color-deep-plum);
          margin-top: 4px;
          display: block;
        }

        /* Mobile Responsive Adjustments */
        .lms-mobile-trigger {
          display: none;
        }

        @media (max-width: 900px) {
          .lms-sidebar {
            position: fixed;
            top: 0;
            left: -320px;
            height: 100vh;
            transition: left 0.3s ease;
            box-shadow: var(--shadow-lg);
          }

          .lms-sidebar.mobile-open {
            left: 0;
          }

          .sidebar-close-mobile {
            display: block;
          }

          .lms-mobile-trigger {
            display: block;
          }

          .btn-open-sidebar {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            background: var(--color-blush-light);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-sm);
            font-size: 0.8rem;
            color: var(--color-deep-plum);
            cursor: pointer;
          }

          .mobile-backdrop {
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.4);
            z-index: 90;
          }

          .lms-content-container {
            padding: 16px;
          }

          .welcome-banner {
            flex-direction: column;
            text-align: center;
          }

          .dash-metrics-grid {
            grid-template-columns: 1fr;
          }

          .profile-details-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
}
