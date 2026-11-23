import React, { useState } from 'react';
import TraditionalDivider from '../components/TraditionalDivider';
import { Video, Sparkles, Camera, RefreshCw, ShieldAlert, ArrowLeft, CheckCircle2 } from 'lucide-react';

export default function StudioPlaceholderPage({ setActivePage, onOpenPlaceholder }) {
  const [selectedCategory, setSelectedCategory] = useState('hasthas');

  const categories = [
    { id: 'hasthas', label: 'A. Hasta Practice', count: '52 Classes', desc: '175-D Landmark Feature Space • Asamyutha (28) & Samyutha (24)' },
    { id: 'shiro', label: 'B. Shiro Bheda Practice', count: '9 Classes', desc: '102-D Head Temporal Motion Sequence • 160-frame BiGRU buffer' },
    { id: 'drishti', label: 'C. Drishti Bheda Practice', count: '8 Classes', desc: '132-D Facial Gaze Temporal Motion Sequence • Iris & Eye Aspect Ratio' },
  ];

  return (
    <div className="studio-page-wrapper section-padding">
      <div className="container">
        {/* Back Link */}
        <button onClick={() => setActivePage('home')} className="back-nav-btn">
          <ArrowLeft size={16} /> Back to Homepage
        </button>

        <div className="section-header" style={{ marginTop: '16px' }}>
          <span className="section-tag">Stage 6 Integration Preview</span>
          <h1 className="section-title">AI Practice Studio</h1>
          <TraditionalDivider motif="diya" />
          <p className="section-subtitle">
            Practise classical gestures and expressions in real-time through your computer camera. Evaluated by our unified 69-class deep neural model.
          </p>
        </div>

        {/* Practice Studio Layout Container */}
        <div className="studio-main-grid">
          {/* Left Column: Category & Gesture Selection */}
          <div className="studio-controls-panel temple-card">
            <h3 className="panel-title">1. Select Practice Category</h3>
            <div className="category-tabs-column">
              {categories.map((c) => (
                <button
                  key={c.id}
                  onClick={() => setSelectedCategory(c.id)}
                  className={`category-select-card ${selectedCategory === c.id ? 'active' : ''}`}
                >
                  <div className="cat-title-row">
                    <strong className="cat-label">{c.label}</strong>
                    <span className="cat-count-badge">{c.count}</span>
                  </div>
                  <p className="cat-desc">{c.desc}</p>
                </button>
              ))}
            </div>

            <div className="pipeline-notice-box">
              <div className="notice-icon">⚙️</div>
              <div>
                <strong>Zero Fake Predictions Rule</strong>
                <p>
                  Per project specifications, this studio connects directly to your verified Python AI backend in <strong>Stage 6</strong>. No simulated results will be displayed.
                </p>
              </div>
            </div>

            <div className="studio-actions-list">
              <button 
                onClick={() => onOpenPlaceholder('AI Practice Studio Camera', 'Stage 6 will integrate your browser webcam feed with deployment_clean/nritya_ai_weights.keras for real-time 69-class recognition.')} 
                className="btn-primary" 
                style={{ width: '100%' }}
              >
                <Camera size={18} />
                <span>Request Camera & Start (Stage 6)</span>
              </button>
            </div>
          </div>

          {/* Right Column: Camera Viewport & Recognition HUD Preview */}
          <div className="studio-camera-panel temple-card">
            <div className="camera-header-bar">
              <div className="camera-status-indicator">
                <span className="status-bullet ready" />
                <span className="status-text">Studio Standby • Stage 2 Layout</span>
              </div>
              <span className="camera-cat-active">Focus: {selectedCategory.toUpperCase()}</span>
            </div>

            <div className="camera-viewport-box">
              {/* Reference Preview Image */}
              <img 
                src={selectedCategory === 'hasthas' ? '/assets/hastha_anjali.jpeg' : (selectedCategory === 'shiro' ? '/assets/shiro_sama.jpeg' : '/assets/drishti_sama.jpeg')} 
                alt="Dance Gesture Reference" 
                className="viewport-sample-image" 
              />

              {/* Viewport Overlay Guide */}
              <div className="viewport-overlay-guide">
                <div className="guide-target-box">
                  <div className="corner-bracket top-left" />
                  <div className="corner-bracket top-right" />
                  <div className="corner-bracket bottom-left" />
                  <div className="corner-bracket bottom-right" />
                  <span className="guide-instruction">Target Gesture Alignment Area</span>
                </div>

                <div className="studio-hud-feedback">
                  <div className="hud-badge-row">
                    <span className="hud-chip">MODEL: nritya_ai_weights.keras</span>
                    <span className="hud-chip">FPS: 30</span>
                  </div>
                  <div className="hud-result-box">
                    <strong>Target: Pathaka Hasta (पताक)</strong>
                    <span>Hold pose steadily for 6 consecutive frames to register stable recognition</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Studio Footer Controls */}
            <div className="camera-footer-controls">
              <button className="ctrl-btn" disabled>
                <RefreshCw size={16} /> Reset Buffer
              </button>
              <div className="info-text">
                Webcam stream processing will be fully operational in Stage 6.
              </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .studio-page-wrapper {
          background: #FAF5EC;
          min-height: 85vh;
        }

        .back-nav-btn {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          background: transparent;
          border: none;
          color: #6B342B;
          font-family: var(--font-serif);
          font-weight: 700;
          font-size: 0.88rem;
          cursor: pointer;
          transition: transform 0.2s;
        }

        .back-nav-btn:hover {
          color: #45151F;
          transform: translateX(-3px);
        }

        .studio-main-grid {
          display: grid;
          grid-template-columns: 0.9fr 1.1fr;
          gap: 30px;
        }

        @media (max-width: 1024px) {
          .studio-main-grid {
            grid-template-columns: 1fr;
          }
        }

        .studio-controls-panel,
        .studio-camera-panel {
          padding: 28px;
        }

        .panel-title {
          font-family: var(--font-heading);
          font-size: 1.25rem;
          color: #45151F;
          margin-bottom: 18px;
        }

        .category-tabs-column {
          display: flex;
          flex-direction: column;
          gap: 12px;
          margin-bottom: 24px;
        }

        .category-select-card {
          padding: 16px;
          border-radius: 8px;
          border: 1.5px solid rgba(196, 154, 69, 0.3);
          background: #FAF5EC;
          text-align: left;
          cursor: pointer;
          transition: all 0.25s ease;
        }

        .category-select-card:hover {
          border-color: #C49A45;
          background: #F4E9D5;
        }

        .category-select-card.active {
          border-color: #45151F;
          background: #F4E9D5;
          box-shadow: 0 4px 15px rgba(69, 21, 31, 0.1);
        }

        .cat-title-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 4px;
        }

        .cat-label {
          font-family: var(--font-serif);
          font-size: 0.95rem;
          color: #45151F;
        }

        .cat-count-badge {
          font-size: 0.72rem;
          font-weight: 700;
          color: #8A463A;
          background: rgba(196, 154, 69, 0.2);
          padding: 2px 8px;
          border-radius: 10px;
        }

        .cat-desc {
          font-size: 0.78rem;
          color: #6B5B52;
          line-height: 1.4;
        }

        .pipeline-notice-box {
          display: flex;
          align-items: flex-start;
          gap: 12px;
          background: rgba(196, 154, 69, 0.15);
          border: 1px solid rgba(196, 154, 69, 0.4);
          border-radius: 8px;
          padding: 14px;
          margin-bottom: 24px;
        }

        .notice-icon {
          font-size: 1.3rem;
        }

        .pipeline-notice-box strong {
          display: block;
          font-family: var(--font-serif);
          font-size: 0.85rem;
          color: #45151F;
        }

        .pipeline-notice-box p {
          font-size: 0.78rem;
          color: #55443B;
          margin-top: 2px;
          line-height: 1.45;
        }

        .camera-header-bar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 16px;
        }

        .camera-status-indicator {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .status-bullet {
          width: 10px;
          height: 10px;
          border-radius: 50%;
        }

        .status-bullet.ready {
          background: #C49A45;
        }

        .status-text {
          font-family: var(--font-serif);
          font-size: 0.82rem;
          font-weight: 600;
          color: #45151F;
        }

        .camera-cat-active {
          font-size: 0.75rem;
          font-weight: 700;
          color: #8A463A;
          background: #F4E9D5;
          padding: 3px 10px;
          border-radius: 12px;
          border: 1px solid rgba(196, 154, 69, 0.3);
        }

        .camera-viewport-box {
          position: relative;
          height: 400px;
          background: #250A10;
          border-radius: 12px;
          overflow: hidden;
          border: 2px solid #C49A45;
        }

        .viewport-sample-image {
          width: 100%;
          height: 100%;
          object-fit: cover;
          opacity: 0.85;
        }

        .viewport-overlay-guide {
          position: absolute;
          inset: 0;
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          padding: 20px;
          background: linear-gradient(180deg, rgba(0,0,0,0.5) 0%, transparent 40%, rgba(0,0,0,0.85) 100%);
        }

        .guide-target-box {
          position: relative;
          width: 240px;
          height: 200px;
          margin: 0 auto;
          display: flex;
          align-items: center;
          justify-content: center;
          border: 1px dashed rgba(245, 222, 152, 0.4);
        }

        .corner-bracket {
          position: absolute;
          width: 16px;
          height: 16px;
          border-color: #F5DE98;
          border-style: solid;
        }

        .corner-bracket.top-left { top: -2px; left: -2px; border-width: 3px 0 0 3px; }
        .corner-bracket.top-right { top: -2px; right: -2px; border-width: 3px 3px 0 0; }
        .corner-bracket.bottom-left { bottom: -2px; left: -2px; border-width: 0 0 3px 3px; }
        .corner-bracket.bottom-right { bottom: -2px; right: -2px; border-width: 0 3px 3px 0; }

        .guide-instruction {
          font-size: 0.72rem;
          color: #FAF5EC;
          background: rgba(0, 0, 0, 0.6);
          padding: 4px 8px;
          border-radius: 4px;
        }

        .studio-hud-feedback {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .hud-badge-row {
          display: flex;
          gap: 8px;
        }

        .hud-chip {
          font-size: 0.68rem;
          font-family: var(--font-serif);
          background: rgba(43, 12, 19, 0.85);
          border: 1px solid #C49A45;
          color: #F5DE98;
          padding: 3px 8px;
          border-radius: 4px;
        }

        .hud-result-box {
          background: rgba(43, 12, 19, 0.9);
          border: 1.5px solid #C49A45;
          border-radius: 8px;
          padding: 10px 14px;
          color: #F4E9D5;
        }

        .hud-result-box strong {
          display: block;
          font-family: var(--font-heading);
          color: #F5DE98;
          font-size: 0.95rem;
        }

        .hud-result-box span {
          font-size: 0.75rem;
          color: #D3C4B0;
        }

        .camera-footer-controls {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-top: 16px;
        }

        .ctrl-btn {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 8px 16px;
          background: #FAF5EC;
          border: 1px solid #C49A45;
          border-radius: 6px;
          font-family: var(--font-serif);
          font-size: 0.82rem;
          color: #45151F;
          cursor: not-allowed;
          opacity: 0.6;
        }

        .info-text {
          font-size: 0.75rem;
          color: #7A6258;
        }
      `}</style>
    </div>
  );
}
