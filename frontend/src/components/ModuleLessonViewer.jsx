import React, { useState, useEffect } from 'react';
import { 
  ChevronLeft, ChevronRight, CheckCircle2, Circle, 
  Video, BookOpen, Sparkles, Award, PlayCircle 
} from 'lucide-react';
import TraditionalDivider from './TraditionalDivider';

export default function ModuleLessonViewer({
  module,
  currentLang = 'en',
  completedLessonIds = [],
  onToggleLesson,
  onLaunchPractice
}) {
  const lessons = module?.lessons || [];
  
  // Find first incomplete lesson to start on, default to 0
  const findFirstIncomplete = () => {
    const idx = lessons.findIndex(l => !completedLessonIds.includes(l.id));
    return idx >= 0 ? idx : 0;
  };

  const [activeIdx, setActiveIdx] = useState(findFirstIncomplete);

  // Update active lesson when module changes
  useEffect(() => {
    setActiveIdx(findFirstIncomplete());
  }, [module?.id]);

  if (!module || lessons.length === 0) {
    return (
      <div className="card-soft" style={{ padding: '32px', textAlign: 'center' }}>
        <p>Loading canonical module content...</p>
      </div>
    );
  }

  const currentLesson = lessons[activeIdx] || lessons[0];
  const isCurrentDone = completedLessonIds.includes(currentLesson.id);
  const completedCount = lessons.filter(l => completedLessonIds.includes(l.id)).length;
  const percent = Math.round((completedCount / lessons.length) * 100);

  const handlePrev = () => {
    if (activeIdx > 0) setActiveIdx(activeIdx - 1);
  };

  const handleNext = () => {
    if (activeIdx < lessons.length - 1) setActiveIdx(activeIdx + 1);
  };

  return (
    <div className="module-lesson-viewer card-soft">
      {/* Module Header & Progress Ribbon */}
      <div className="viewer-header">
        <div className="header-meta">
          <span className="mod-number-pill">Module 0{module.module_number}</span>
          <h2 className="mod-heading">
            {currentLang === 'te' ? module.title_te : module.title_en}
          </h2>
          <span className="mod-sanskrit-sub">{module.sanskrit_title}</span>
        </div>

        <div className="header-progress-box">
          <div className="progress-info-row">
            <span className="progress-text-label">
              {completedCount} of {lessons.length} {currentLang === 'te' ? 'పూర్తయ్యాయి' : 'Completed'}
            </span>
            <strong className="progress-percent-val">{percent}%</strong>
          </div>
          <div className="progress-bar-track">
            <div className="progress-bar-fill" style={{ width: `${percent}%` }} />
          </div>
        </div>
      </div>

      <TraditionalDivider motif="lotus" />

      {/* Main Interactive Lesson Card */}
      <div className="lesson-stage-grid">
        {/* Visual Media & Shloka Section */}
        <div className="lesson-visual-col">
          <div className="lesson-image-frame card-blush">
            {currentLesson.image_url ? (
              <img 
                src={currentLesson.image_url} 
                alt={currentLesson.name_en} 
                className="lesson-authentic-img"
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.src = '/assets/asamyutha_sample.jpeg';
                }}
              />
            ) : (
              <div className="img-placeholder">
                <BookOpen size={48} className="placeholder-icon" />
                <span>Canonical Indian Dance Mudra</span>
              </div>
            )}
            <div className="image-caption">
              <strong>{currentLesson.name_en}</strong>
              <span>({currentLesson.name_te})</span>
            </div>
          </div>

          {/* Shloka Card */}
          {currentLesson.shloka && (
            <div className="lesson-shloka-card card-lavender">
              <span className="shloka-badge">📜 {currentLang === 'te' ? 'సంస్కృత శ్లోకం' : 'Sanskrit Shlokam'}</span>
              <p className="shloka-text">{currentLesson.shloka}</p>
              {currentLesson.meaning_te && currentLang === 'te' && (
                <p className="shloka-meaning">
                  <strong>భావం:</strong> {currentLesson.meaning_te}
                </p>
              )}
              {currentLesson.meaning_en && currentLang !== 'te' && (
                <p className="shloka-meaning">
                  <strong>Meaning:</strong> {currentLesson.meaning_en}
                </p>
              )}
            </div>
          )}
        </div>

        {/* Detailed Explanation & Actions Column */}
        <div className="lesson-details-col">
          <div className="lesson-title-row">
            <div>
              <span className="lesson-order-tag">
                Lesson {activeIdx + 1} of {lessons.length}
              </span>
              <h3 className="lesson-name-main">
                {currentLesson.name_en} — <span className="te-name">{currentLesson.name_te}</span>
              </h3>
              <span className="sanskrit-canon-tag">{currentLesson.sanskrit_name}</span>
            </div>

            <button
              onClick={() => onToggleLesson(currentLesson.id)}
              className={`btn-toggle-completion ${isCurrentDone ? 'done' : 'undone'}`}
              title={isCurrentDone ? "Mark as Incomplete" : "Mark as Completed"}
            >
              {isCurrentDone ? (
                <>
                  <CheckCircle2 size={18} />
                  <span>{currentLang === 'te' ? 'పూర్తయింది' : 'Completed'}</span>
                </>
              ) : (
                <>
                  <Circle size={18} />
                  <span>{currentLang === 'te' ? 'పూర్తయినట్లు గుర్తించు' : 'Mark Completed'}</span>
                </>
              )}
            </button>
          </div>

          <div className="lesson-desc-body card-soft">
            <h4 className="desc-heading">
              {currentLang === 'te' ? 'వివరణ మరియు నాట్య ప్రదర్శన విధానం' : 'Description & Performance Technique'}
            </h4>
            <p className="desc-content">
              {currentLang === 'te' ? currentLesson.description_te : currentLesson.description_en}
            </p>

            <div className="learning-objective-box card-sage">
              <strong>🎯 {currentLang === 'te' ? 'అభ్యసన లక్ష్యం:' : 'Learning Objective:'}</strong>
              <p>
                {currentLang === 'te'
                  ? `ఈ పాఠం ద్వారా ${currentLesson.name_te} యొక్క ఖచ్చితమైన చేతి అమరిక, శ్లోకం మరియు నాట్య వినియోగాలను అభ్యసించండి.`
                  : `Master the canonical structural alignment, Sanskrit shloka attribution, and dramatic expressions of ${currentLesson.name_en}.`}
              </p>
            </div>
          </div>

          {/* Quick AI Practice CTA */}
          <div className="practice-cta-card card-blush">
            <div>
              <strong>{currentLang === 'te' ? 'ప్రత్యక్ష కెమెరా అభ్యాసం' : 'Real-time Camera Practice'}</strong>
              <p>
                {currentLang === 'te'
                  ? 'AI ప్రాక్టీస్ స్టూడియోలో మీ కెమెరా ముందు ఈ ముద్రను సాధన చేయండి.'
                  : `Practice ${currentLesson.name_en} in front of your webcam with AI landmark detection.`}
              </p>
            </div>
            <button onClick={onLaunchPractice} className="btn-green">
              <Video size={16} />
              <span>{currentLang === 'te' ? 'స్టూడియో ప్రారంభించు' : 'Launch Studio'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Lesson Navigation Tray (Previous, Index Tray, Next) */}
      <div className="viewer-nav-tray">
        <button 
          onClick={handlePrev} 
          disabled={activeIdx === 0}
          className="btn-nav-prev"
        >
          <ChevronLeft size={18} />
          <span>{currentLang === 'te' ? 'మునుపటి పాఠం' : 'Previous'}</span>
        </button>

        {/* Scrollable Lesson Index Tray */}
        <div className="lessons-carousel-strip">
          {lessons.map((les, idx) => {
            const done = completedLessonIds.includes(les.id);
            const active = idx === activeIdx;
            return (
              <button
                key={les.id}
                onClick={() => setActiveIdx(idx)}
                className={`lesson-pill-btn ${active ? 'active' : ''} ${done ? 'done' : ''}`}
                title={`${les.name_en} (${les.name_te})`}
              >
                <span className="pill-num">{idx + 1}</span>
                <span className="pill-name">{les.name_en}</span>
                {done && <CheckCircle2 size={12} className="pill-check" />}
              </button>
            );
          })}
        </div>

        <button 
          onClick={handleNext} 
          disabled={activeIdx === lessons.length - 1}
          className="btn-nav-next"
        >
          <span>{currentLang === 'te' ? 'తదుపరి పాఠం' : 'Next'}</span>
          <ChevronRight size={18} />
        </button>
      </div>

      <style>{`
        .module-lesson-viewer {
          padding: 24px;
          border-radius: var(--radius-lg);
          background: #FFFFFF;
          border: 1px solid var(--color-border);
        }

        .viewer-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          flex-wrap: wrap;
          gap: 16px;
        }

        .mod-number-pill {
          display: inline-block;
          font-size: 0.75rem;
          font-weight: 700;
          color: var(--color-deep-plum);
          background: var(--color-blush-light);
          padding: 3px 8px;
          border-radius: 4px;
          margin-bottom: 4px;
        }

        .mod-heading {
          font-size: 1.4rem;
          color: var(--color-deep-plum);
          margin: 0;
        }

        .mod-sanskrit-sub {
          font-size: 0.9rem;
          font-style: italic;
          color: var(--color-gold-dark);
          font-weight: 600;
        }

        .header-progress-box {
          min-width: 220px;
        }

        .progress-info-row {
          display: flex;
          justify-content: space-between;
          font-size: 0.8rem;
          margin-bottom: 6px;
          color: var(--color-text-secondary);
        }

        .progress-bar-track {
          width: 100%;
          height: 8px;
          background: var(--color-blush-light);
          border-radius: 4px;
          overflow: hidden;
        }

        .progress-bar-fill {
          height: 100%;
          background: var(--color-temple-green);
          border-radius: 4px;
          transition: width 0.3s ease;
        }

        .lesson-stage-grid {
          display: grid;
          grid-template-columns: 1fr 1.3fr;
          gap: 24px;
          margin: 20px 0;
        }

        @media (max-width: 860px) {
          .lesson-stage-grid {
            grid-template-columns: 1fr;
          }
        }

        .lesson-image-frame {
          border-radius: var(--radius-md);
          overflow: hidden;
          border: 1px solid var(--color-border);
          text-align: center;
          padding: 12px;
        }

        .lesson-authentic-img {
          width: 100%;
          max-height: 260px;
          object-fit: contain;
          border-radius: var(--radius-sm);
          background: #FFFFFF;
        }

        .img-placeholder {
          height: 200px;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 8px;
          color: var(--color-text-secondary);
        }

        .image-caption {
          margin-top: 8px;
          display: flex;
          justify-content: center;
          gap: 6px;
          font-size: 0.9rem;
          color: var(--color-deep-plum);
        }

        .lesson-shloka-card {
          margin-top: 16px;
          padding: 14px;
          border-radius: var(--radius-md);
          border: 1px solid var(--color-border);
        }

        .shloka-badge {
          font-size: 0.75rem;
          font-weight: 700;
          color: var(--color-deep-plum);
          text-transform: uppercase;
        }

        .shloka-text {
          font-family: var(--font-heading);
          font-size: 0.92rem;
          color: var(--color-deep-plum);
          line-height: 1.6;
          margin: 8px 0;
          font-weight: 600;
        }

        .shloka-meaning {
          font-size: 0.82rem;
          color: var(--color-text-primary);
          line-height: 1.5;
        }

        .lesson-details-col {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .lesson-title-row {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 12px;
          flex-wrap: wrap;
        }

        .lesson-order-tag {
          font-size: 0.75rem;
          color: var(--color-gold-dark);
          font-weight: 700;
          text-transform: uppercase;
        }

        .lesson-name-main {
          font-size: 1.25rem;
          color: var(--color-deep-plum);
          margin: 2px 0;
        }

        .sanskrit-canon-tag {
          font-size: 0.85rem;
          color: var(--color-temple-green);
          font-style: italic;
        }

        .btn-toggle-completion {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 8px 14px;
          border-radius: var(--radius-sm);
          font-size: 0.82rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }

        .btn-toggle-completion.undone {
          background: #FFFFFF;
          border: 1px solid var(--color-border);
          color: var(--color-text-primary);
        }

        .btn-toggle-completion.undone:hover {
          border-color: var(--color-temple-green);
          color: var(--color-temple-green);
        }

        .btn-toggle-completion.done {
          background: var(--color-sage-light);
          border: 1px solid var(--color-temple-green);
          color: var(--color-temple-green);
        }

        .desc-heading {
          font-size: 0.95rem;
          color: var(--color-deep-plum);
          margin: 0 0 8px 0;
        }

        .desc-content {
          font-size: 0.9rem;
          line-height: 1.6;
          color: var(--color-text-primary);
        }

        .learning-objective-box {
          margin-top: 14px;
          padding: 10px 14px;
          border-radius: var(--radius-sm);
          font-size: 0.85rem;
          color: var(--color-temple-green);
        }

        .practice-cta-card {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 14px 18px;
          border-radius: var(--radius-md);
          gap: 16px;
        }

        .viewer-nav-tray {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          margin-top: 20px;
          padding-top: 16px;
          border-top: 1px solid var(--color-border);
        }

        .btn-nav-prev, .btn-nav-next {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 8px 14px;
          border-radius: var(--radius-sm);
          background: var(--color-blush-light);
          border: 1px solid var(--color-border);
          color: var(--color-deep-plum);
          font-size: 0.82rem;
          font-weight: 600;
          cursor: pointer;
          flex-shrink: 0;
        }

        .btn-nav-prev:disabled, .btn-nav-next:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .lessons-carousel-strip {
          display: flex;
          gap: 8px;
          overflow-x: auto;
          padding: 4px 2px;
          max-width: 60%;
        }

        .lesson-pill-btn {
          display: inline-flex;
          align-items: center;
          gap: 5px;
          padding: 6px 12px;
          border-radius: 20px;
          background: #FFFFFF;
          border: 1px solid var(--color-border);
          font-size: 0.78rem;
          color: var(--color-text-primary);
          cursor: pointer;
          white-space: nowrap;
          transition: all 0.2s;
        }

        .lesson-pill-btn.active {
          background: var(--color-deep-plum);
          color: #FFFFFF;
          border-color: var(--color-deep-plum);
        }

        .lesson-pill-btn.done:not(.active) {
          background: var(--color-sage-light);
          color: var(--color-temple-green);
          border-color: var(--color-sage-green);
        }
      `}</style>
    </div>
  );
}
