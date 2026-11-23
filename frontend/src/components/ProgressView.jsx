import React from 'react';
import { 
  TrendingUp, BookOpen, Award, Sparkles, CheckCircle2, 
  Circle, ChevronRight, Video 
} from 'lucide-react';
import TraditionalDivider from './TraditionalDivider';
import { translations } from '../utils/translations';

export default function ProgressView({
  currentLang = 'en',
  completedModules = [],
  completedLessonIds = [],
  quizHistory = [],
  practiceHistory = [],
  onNavigateToModule
}) {
  const t = translations[currentLang] || translations.en;

  const allModules = [
    { id: 1, title: '1. Introduction to Kuchipudi', titleTe: '1. కూచిపూడి పరిచయం', sanskrit: 'నాట్య శాస్త్ర ప్రవేశమ్', count: 4 },
    { id: 2, title: '2. Asamyukta Hastas (28 Single-Hand)', titleTe: '2. అసంయుక్త హస్తాలు (28 ఏకహస్త ముద్రలు)', sanskrit: 'అసంయుత హస్తాః', count: 28 },
    { id: 3, title: '3. Samyukta Hastas (24 Combined-Hand)', titleTe: '3. సంయుక్త హస్తాలు (24 ద్విహస్త ముద్రలు)', sanskrit: 'సంయుత హస్తాః', count: 24 },
    { id: 4, title: '4. Shiro Bhedas (9 Head Movements)', titleTe: '4. శిరో భేదాలు (9 శిరస్సు కదలికలు)', sanskrit: 'శిరో భేద ప్రకరణమ్', count: 9 },
    { id: 5, title: '5. Drishti Bhedas (8 Eye Movements)', titleTe: '5. దృష్టి భేదాలు (8 నేత్ర కదలికలు)', sanskrit: 'దృష్టి భేద ప్రకరణమ్', count: 8 },
    { id: 6, title: '6. Chaturvidha Abhinaya & Navarasas', titleTe: '6. చతుర్విధ అభినయం మరియు నవరసాలు', sanskrit: 'ద్వాదశ హస్త ప్రణామాః & నవరసాః', count: 12 },
    { id: 7, title: '7. Sapta Talamulu (35 Tala System)', titleTe: '7. సప్త తాళములు (35 తాళ వ్యవస్థ)', sanskrit: 'సప్త తాల ప్రకరణమ్', count: 7 }
  ];

  const totalLessons = 92;
  const progressPercent = Math.round((completedLessonIds.length / totalLessons) * 100);

  return (
    <div className="progress-view-content card-soft">
      <div className="section-header">
        <span className="section-tag">LEARNING ANALYTICS</span>
        <h2 className="section-title">
          {currentLang === 'te' ? 'నా అభ్యాస పురోగతి మరియు రికార్డులు' : 'My Learning Progress & Records'}
        </h2>
        <TraditionalDivider motif="lotus" />
        <p className="section-subtitle">
          {currentLang === 'te'
            ? 'మీరు పూర్తి చేసిన సిద్ధాంత పాఠాలు, అభ్యాస విభాగాలు మరియు క్విజ్ పురోగతి ఇక్కడ భద్రపరచబడ్డాయి.'
            : 'Track your completion across all 92 canonical lessons, 7 theory modules, quizzes, and AI practice sessions.'}
        </p>
      </div>

      {/* Progress Cards Grid */}
      <div className="analytics-metrics-grid">
        <div className="metric-card card-blush">
          <div className="metric-icon-box m-plum">
            <BookOpen size={20} />
          </div>
          <div>
            <span className="m-label">Modules Completed</span>
            <strong className="m-value">{completedModules.length} of {allModules.length} Modules</strong>
            <span className="m-sub">{Math.round((completedModules.length / allModules.length) * 100)}% Complete</span>
          </div>
        </div>

        <div className="metric-card card-sage">
          <div className="metric-icon-box m-green">
            <TrendingUp size={20} />
          </div>
          <div>
            <span className="m-label">Completed Individual Lessons</span>
            <strong className="m-value">{completedLessonIds.length} of {totalLessons} Lessons</strong>
            <span className="m-sub">{progressPercent}% Overall Progress</span>
          </div>
        </div>

        <div className="metric-card card-lavender">
          <div className="metric-icon-box m-gold">
            <Award size={20} />
          </div>
          <div>
            <span className="m-label">Quizzes Completed</span>
            <strong className="m-value">{quizHistory.length} Submissions</strong>
            <span className="m-sub">{quizHistory.filter(q => q.passed).length} Passed</span>
          </div>
        </div>

        <div className="metric-card card-blush">
          <div className="metric-icon-box m-plum">
            <Video size={20} />
          </div>
          <div>
            <span className="m-label">AI Practice Sessions Logged</span>
            <strong className="m-value">{practiceHistory.length} Sessions</strong>
            <span className="m-sub">Stored in Local Session</span>
          </div>
        </div>
      </div>

      {/* Overall Progress Ribbon */}
      <div className="overall-ribbon-card card-sage">
        <div className="ribbon-text">
          <strong>Overall Curriculum Completion</strong>
          <span>{completedLessonIds.length} of {totalLessons} canonical lessons completed</span>
        </div>
        <div className="ribbon-track-wrap">
          <div className="ribbon-track">
            <div className="ribbon-fill" style={{ width: `${progressPercent}%` }} />
          </div>
          <span className="ribbon-percent">{progressPercent}%</span>
        </div>
      </div>

      {/* Module-by-Module Detailed Status Table */}
      <div className="curriculum-breakdown-section">
        <h3 className="breakdown-heading">
          {currentLang === 'te' ? 'విభాగాల వారీగా సమగ్ర పురోగతి' : 'Curriculum Breakdown by Module'}
        </h3>

        <div className="breakdown-list">
          {allModules.map((m) => {
            const isDone = completedModules.includes(m.id);
            return (
              <div key={m.id} className="breakdown-row">
                <div className="row-info-col">
                  <span className="row-num-badge">0{m.id}</span>
                  <div>
                    <strong className="row-title">{currentLang === 'te' ? m.titleTe : m.title}</strong>
                    <span className="row-sub">{m.sanskrit} • {m.count} lessons</span>
                  </div>
                </div>

                <div className="row-status-col">
                  {isDone ? (
                    <span className="status-badge done">
                      <CheckCircle2 size={14} /> Completed
                    </span>
                  ) : (
                    <span className="status-badge pending">
                      <Circle size={14} /> In Progress
                    </span>
                  )}
                </div>

                <div className="row-action-col">
                  <button 
                    onClick={() => onNavigateToModule(m.id)}
                    className="btn-open-row"
                  >
                    <span>Study</span>
                    <ChevronRight size={14} />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <style>{`
        .progress-view-content {
          padding: 32px;
          display: flex;
          flex-direction: column;
          gap: 28px;
        }
        .analytics-metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
          gap: 16px;
        }
        .metric-card {
          padding: 18px;
          border-radius: var(--radius-md);
          display: flex;
          align-items: flex-start;
          gap: 14px;
          border: 1px solid var(--color-border);
        }
        .metric-icon-box {
          width: 42px;
          height: 42px;
          border-radius: 10px;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }
        .m-plum { background: var(--color-blush-ivory); color: var(--color-deep-plum); }
        .m-green { background: var(--color-sage-light); color: var(--color-temple-green); }
        .m-gold { background: #fdf5e6; color: var(--color-gold-dark); }
        .m-label {
          font-size: 0.75rem;
          color: var(--color-text-secondary);
          display: block;
          margin-bottom: 2px;
        }
        .m-value {
          font-size: 1.05rem;
          color: var(--color-deep-plum);
          display: block;
          line-height: 1.2;
        }
        .m-sub {
          font-size: 0.72rem;
          color: var(--color-temple-green);
          font-weight: 600;
          display: block;
          margin-top: 2px;
        }
        .overall-ribbon-card {
          padding: 20px 24px;
          border-radius: var(--radius-md);
          border: 1px solid var(--color-border);
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 24px;
          flex-wrap: wrap;
        }
        .ribbon-text strong {
          display: block;
          font-family: var(--font-heading);
          color: var(--color-deep-plum);
          font-size: 1rem;
        }
        .ribbon-text span {
          font-size: 0.8rem;
          color: var(--color-text-secondary);
        }
        .ribbon-track-wrap {
          display: flex;
          align-items: center;
          gap: 12px;
          flex-grow: 1;
          max-width: 400px;
        }
        .ribbon-track {
          flex-grow: 1;
          height: 10px;
          background: #e4d7cb;
          border-radius: 5px;
          overflow: hidden;
        }
        .ribbon-fill {
          height: 100%;
          background: var(--color-temple-green);
          border-radius: 5px;
          transition: width 0.3s ease;
        }
        .ribbon-percent {
          font-weight: 700;
          font-size: 0.95rem;
          color: var(--color-deep-plum);
        }
        .curriculum-breakdown-section {
          margin-top: 8px;
        }
        .breakdown-heading {
          font-family: var(--font-heading);
          font-size: 1.2rem;
          color: var(--color-deep-plum);
          margin-bottom: 14px;
        }
        .breakdown-list {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }
        .breakdown-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 14px 18px;
          background: #FFFFFF;
          border: 1px solid var(--color-border);
          border-radius: var(--radius-sm);
          gap: 16px;
        }
        .row-info-col {
          display: flex;
          align-items: center;
          gap: 12px;
          flex-grow: 1;
        }
        .row-num-badge {
          background: var(--color-deep-plum);
          color: #FFF;
          font-size: 0.75rem;
          font-weight: 700;
          padding: 4px 8px;
          border-radius: 4px;
        }
        .row-title {
          font-size: 0.92rem;
          color: var(--color-deep-plum);
          display: block;
        }
        .row-sub {
          font-size: 0.75rem;
          color: var(--color-text-secondary);
        }
        .status-badge {
          display: inline-flex;
          align-items: center;
          gap: 5px;
          font-size: 0.75rem;
          font-weight: 600;
          padding: 4px 10px;
          border-radius: 999px;
        }
        .status-badge.done {
          background: #def7ec;
          color: #03543f;
        }
        .status-badge.pending {
          background: var(--color-blush-light);
          color: var(--color-text-secondary);
        }
        .btn-open-row {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          background: var(--color-blush-light);
          border: 1px solid var(--color-border);
          color: var(--color-deep-plum);
          padding: 6px 12px;
          border-radius: var(--radius-sm);
          font-size: 0.8rem;
          font-weight: 600;
          cursor: pointer;
        }
        .btn-open-row:hover {
          background: var(--color-deep-plum);
          color: #FFF;
        }
      `}</style>
    </div>
  );
}
