import React from 'react';
import { Award, Video, CheckCircle2, AlertCircle, ArrowRight } from 'lucide-react';
import TraditionalDivider from './TraditionalDivider';
import { translations } from '../utils/translations';

export default function ResultsView({
  currentLang = 'en',
  quizHistory = [],
  practiceHistory = [],
  onNavigateToAssignments,
  onNavigateToPractice
}) {
  const t = translations[currentLang] || translations.en;

  return (
    <div className="results-view-content card-soft">
      <div className="section-header">
        <span className="section-tag">PERFORMANCE HISTORY</span>
        <h2 className="section-title">
          {currentLang === 'te' ? 'పరీక్షా ఫలితాలు మరియు సాధన లాగ్‌లు' : 'Assessment Results & Practice Logs'}
        </h2>
        <TraditionalDivider motif="lotus" />
        <p className="section-subtitle">
          {currentLang === 'te'
            ? 'మీరు సమర్పించిన అన్ని క్విజ్‌లు మరియు AI కెమెరా ద్వారా పూర్తి చేసిన సాధన రికార్డులు క్రింద చూపబడ్డాయి.'
            : 'Review your historical quiz scores, theoretical knowledge assessments, and computer vision AI practice records.'}
        </p>
      </div>

      {/* Sub-section 1: Quiz Submissions */}
      <div className="results-sub-block">
        <div className="block-title-row">
          <h3 className="block-heading">
            <Award size={18} />
            <span>{currentLang === 'te' ? 'క్విజ్ పరీక్షా ఫలితాలు' : 'Quiz & Exam Submissions'}</span>
          </h3>
          <button onClick={onNavigateToAssignments} className="btn-outline" style={{ fontSize: '0.8rem', padding: '6px 14px' }}>
            <span>Take New Quiz</span>
            <ArrowRight size={14} />
          </button>
        </div>

        {quizHistory.length > 0 ? (
          <div className="table-responsive">
            <table className="results-table">
              <thead>
                <tr>
                  <th>Assessment Title</th>
                  <th>Score</th>
                  <th>Percentage</th>
                  <th>Status</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {quizHistory.map((r, i) => (
                  <tr key={r.id || i}>
                    <td style={{ fontWeight: 600 }}>{r.quiz_title_en}</td>
                    <td style={{ fontWeight: 700 }}>{r.score} / {r.total_questions}</td>
                    <td>{r.percentage}%</td>
                    <td>
                      <span className={`status-tag ${r.passed ? 'passed' : 'failed'}`}>
                        {r.passed ? 'PASSED ✓' : 'RETRY'}
                      </span>
                    </td>
                    <td style={{ color: 'var(--color-text-secondary)', fontSize: '0.8rem' }}>
                      {r.completed_at ? new Date(r.completed_at).toLocaleString() : 'Recent'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state-box card-blush">
            <p className="empty-title">No quiz submissions found yet.</p>
            <p className="empty-desc">Take a module quiz in the Assignments section to test your classical theory knowledge.</p>
            <button onClick={onNavigateToAssignments} className="btn-primary" style={{ marginTop: '8px' }}>
              <span>Take Quiz Now</span>
            </button>
          </div>
        )}
      </div>

      {/* Sub-section 2: AI Practice Sessions */}
      <div className="results-sub-block" style={{ marginTop: '36px' }}>
        <div className="block-title-row">
          <h3 className="block-heading">
            <Video size={18} />
            <span>{currentLang === 'te' ? 'AI సాధన రికార్డులు' : 'AI Camera Practice Sessions'}</span>
          </h3>
          <button onClick={onNavigateToPractice} className="btn-green" style={{ fontSize: '0.8rem', padding: '6px 14px' }}>
            <span>Launch Studio</span>
            <ArrowRight size={14} />
          </button>
        </div>

        {practiceHistory.length > 0 ? (
          <div className="table-responsive">
            <table className="results-table">
              <thead>
                <tr>
                  <th>Category</th>
                  <th>Gesture / Pose</th>
                  <th>Confidence Score</th>
                  <th>Status / Feedback</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {practiceHistory.map((s, idx) => (
                  <tr key={s.id || idx}>
                    <td style={{ textTransform: 'capitalize', fontWeight: 600 }}>{s.category}</td>
                    <td style={{ fontWeight: 700, color: 'var(--color-deep-plum)' }}>{s.gesture_name}</td>
                    <td>
                      <span style={{ fontWeight: 700, color: 'var(--color-temple-green)' }}>
                        {s.accuracy_percentage ? `${Math.round(s.accuracy_percentage)}%` : `${Math.round((s.confidence_score || 0) * 100)}%`}
                      </span>
                    </td>
                    <td style={{ fontSize: '0.82rem', color: 'var(--color-text-secondary)' }}>
                      {s.feedback_notes || 'Accurate recognition logged'}
                    </td>
                    <td style={{ color: 'var(--color-text-secondary)', fontSize: '0.8rem' }}>
                      {s.created_at ? new Date(s.created_at).toLocaleString() : 'Recent'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state-box card-sage">
            <p className="empty-title">No AI practice sessions logged yet.</p>
            <p className="empty-desc">Open the Practice Studio to test your hand mudras in real time with our neural engine.</p>
            <button onClick={onNavigateToPractice} className="btn-green" style={{ marginTop: '8px' }}>
              <span>Start AI Practice Studio</span>
            </button>
          </div>
        )}
      </div>

      <style>{`
        .results-view-content {
          padding: 32px;
          display: flex;
          flex-direction: column;
          gap: 20px;
        }
        .block-title-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 14px;
        }
        .block-heading {
          font-family: var(--font-heading);
          font-size: 1.15rem;
          color: var(--color-deep-plum);
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .table-responsive {
          overflow-x: auto;
          background: #FFFFFF;
          border: 1px solid var(--color-border);
          border-radius: var(--radius-sm);
        }
        .results-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.88rem;
        }
        .results-table th {
          background: var(--color-blush-light);
          padding: 12px 14px;
          text-align: left;
          font-weight: 700;
          color: var(--color-deep-plum);
          border-bottom: 1.5px solid var(--color-border);
          font-size: 0.8rem;
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }
        .results-table td {
          padding: 12px 14px;
          border-bottom: 1px solid var(--color-border);
        }
        .status-tag {
          display: inline-block;
          font-size: 0.75rem;
          font-weight: 700;
          padding: 4px 10px;
          border-radius: 999px;
        }
        .status-tag.passed {
          background: #def7ec;
          color: #03543f;
        }
        .status-tag.failed {
          background: #fde8e8;
          color: #9b1c1c;
        }
        .empty-state-box {
          padding: 32px;
          text-align: center;
          border-radius: var(--radius-sm);
          border: 1px solid var(--color-border);
        }
        .empty-title {
          font-family: var(--font-heading);
          font-size: 1.05rem;
          color: var(--color-deep-plum);
          margin-bottom: 4px;
        }
        .empty-desc {
          font-size: 0.85rem;
          color: var(--color-text-secondary);
          margin-bottom: 12px;
        }
      `}</style>
    </div>
  );
}
