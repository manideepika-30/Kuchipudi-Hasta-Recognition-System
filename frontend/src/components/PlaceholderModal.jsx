import React from 'react';
import { X, Sparkles, CheckCircle2, ArrowRight } from 'lucide-react';
import TraditionalDivider from './TraditionalDivider';

export default function PlaceholderModal({ isOpen, onClose, title, message, onExploreModules }) {
  if (!isOpen) return null;

  const stages = [
    { num: '1', name: 'Project & AI Model Inspection', status: 'Completed', current: false },
    { num: '2', name: 'Traditional Theme & Homepage Layout', status: 'Completed', current: true },
    { num: '3', name: 'Student Registration & Database Auth', status: 'Next Stage', current: false },
    { num: '4', name: 'All 7 Theory Modules from PDFs', status: 'Upcoming', current: false },
    { num: '5', name: 'Assignments & Theory Quizzes', status: 'Upcoming', current: false },
    { num: '6', name: 'Webcam AI Recognition Studio (69 Classes)', status: 'Upcoming', current: false },
    { num: '7', name: 'Student & Admin Dashboards', status: 'Upcoming', current: false },
  ];

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-temple-card" onClick={(e) => e.stopPropagation()}>
        {/* Close Button */}
        <button className="modal-close-btn" onClick={onClose} aria-label="Close dialog">
          <X size={20} />
        </button>

        {/* Modal Header */}
        <div className="modal-header">
          <div className="modal-diya-badge">🪔</div>
          <span className="modal-tag">Platform Roadmap</span>
          <h3 className="modal-title">{title || 'Kuchipudi Learning Platform'}</h3>
          <TraditionalDivider motif="diya" />
        </div>

        {/* Modal Body */}
        <div className="modal-body">
          <p className="modal-message">
            {message || 'This feature is part of the progressive development stages. You can preview the full homepage layout, explore the 7 learning module descriptions, and test responsive navigation right now.'}
          </p>

          <div className="roadmap-preview">
            <h4 className="roadmap-title">
              <Sparkles size={16} className="sparkle-gold" />
              Development Progression
            </h4>
            <div className="roadmap-list">
              {stages.map((st) => (
                <div key={st.num} className={`roadmap-item ${st.current ? 'current-item' : ''}`}>
                  <span className={`roadmap-step-badge ${st.status === 'Completed' ? 'badge-done' : ''}`}>
                    {st.num}
                  </span>
                  <div className="roadmap-info">
                    <span className="roadmap-name">{st.name}</span>
                    <span className="roadmap-status">{st.status}</span>
                  </div>
                  {st.status === 'Completed' && <CheckCircle2 size={16} className="icon-done" />}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="modal-footer">
          <button 
            className="btn-secondary" 
            onClick={onClose}
            style={{ color: '#2E0D14', borderColor: '#C49A45' }}
          >
            Close
          </button>
          <button 
            className="btn-primary" 
            onClick={() => {
              onClose();
              if (onExploreModules) onExploreModules();
            }}
          >
            <span>Explore 7 Modules</span>
            <ArrowRight size={16} />
          </button>
        </div>
      </div>

      <style>{`
        .modal-backdrop {
          position: fixed;
          inset: 0;
          background: rgba(30, 7, 13, 0.75);
          backdrop-filter: blur(6px);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 2000;
          padding: 20px;
        }

        .modal-temple-card {
          background: #FAF5EC;
          border: 2px solid #C49A45;
          border-radius: var(--radius-md);
          max-width: 540px;
          width: 100%;
          position: relative;
          box-shadow: 0 25px 60px rgba(0, 0, 0, 0.5);
          overflow: hidden;
          animation: modalAppear 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .modal-temple-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 5px;
          background: var(--gold-gradient);
        }

        @keyframes modalAppear {
          from {
            opacity: 0;
            transform: scale(0.92) translateY(10px);
          }
          to {
            opacity: 1;
            transform: scale(1) translateY(0);
          }
        }

        .modal-close-btn {
          position: absolute;
          top: 16px;
          right: 16px;
          background: rgba(69, 21, 31, 0.08);
          border: 1px solid rgba(196, 154, 69, 0.3);
          border-radius: 50%;
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: #45151F;
          cursor: pointer;
          transition: all 0.2s;
        }

        .modal-close-btn:hover {
          background: #45151F;
          color: #F5DE98;
        }

        .modal-header {
          text-align: center;
          padding: 28px 24px 10px;
        }

        .modal-diya-badge {
          font-size: 2rem;
          margin-bottom: 6px;
        }

        .modal-tag {
          font-family: var(--font-serif);
          font-size: 0.75rem;
          color: #C49A45;
          letter-spacing: 0.15em;
          text-transform: uppercase;
          font-weight: 700;
        }

        .modal-title {
          font-family: var(--font-heading);
          font-size: 1.45rem;
          color: #45151F;
          margin-top: 4px;
        }

        .modal-body {
          padding: 0 28px 20px;
        }

        .modal-message {
          font-size: 0.92rem;
          color: #55423A;
          line-height: 1.55;
          text-align: center;
          margin-bottom: 20px;
        }

        .roadmap-preview {
          background: #F4E9D5;
          border: 1px solid rgba(196, 154, 69, 0.3);
          border-radius: 8px;
          padding: 16px;
        }

        .roadmap-title {
          display: flex;
          align-items: center;
          gap: 6px;
          font-family: var(--font-serif);
          font-size: 0.85rem;
          color: #45151F;
          margin-bottom: 12px;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .sparkle-gold {
          color: #C49A45;
        }

        .roadmap-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .roadmap-item {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 6px 10px;
          background: rgba(255, 255, 255, 0.6);
          border-radius: 6px;
          font-size: 0.82rem;
        }

        .roadmap-item.current-item {
          background: rgba(196, 154, 69, 0.2);
          border: 1px solid #C49A45;
        }

        .roadmap-step-badge {
          width: 22px;
          height: 22px;
          border-radius: 50%;
          background: #6B342B;
          color: #FFF;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 0.75rem;
          font-weight: 700;
        }

        .roadmap-step-badge.badge-done {
          background: #45151F;
          color: #F5DE98;
        }

        .roadmap-info {
          flex: 1;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .roadmap-name {
          font-weight: 600;
          color: #2E0D14;
        }

        .roadmap-status {
          font-size: 0.72rem;
          color: #7A6258;
          font-weight: 500;
        }

        .icon-done {
          color: #45151F;
        }

        .modal-footer {
          padding: 16px 28px 24px;
          display: flex;
          justify-content: flex-end;
          gap: 12px;
          background: rgba(69, 21, 31, 0.04);
          border-top: 1px solid rgba(196, 154, 69, 0.2);
        }
      `}</style>
    </div>
  );
}
