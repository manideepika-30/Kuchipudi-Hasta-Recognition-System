import React, { useState } from 'react';
import TraditionalDivider from '../components/TraditionalDivider';
import { Award, HelpCircle, Eye, Video, ArrowLeft, CheckCircle2, ChevronRight } from 'lucide-react';

export default function AssignmentsPlaceholderPage({ setActivePage, onOpenPlaceholder }) {
  const [activeTab, setActiveTab] = useState('theory');

  const sampleQuestions = [
    {
      q: 'Which of the following is the first gesture in the Asamyutha Hastha (single-hand) canon?',
      options: ['1. Thripathaka', '2. Pathaka', '3. Ardhapathaka', '4. Mushti'],
      correct: 1,
      explanation: 'Pathaka (the flag gesture) is the foundational first gesture of the 28 Asamyutha Hasthas according to the Abhinaya Darpana.'
    },
    {
      q: 'Which Shiro Bheda head movement is characterized by rolling the head in a complete circle?',
      options: ['1. Sama', '2. Dhutam', '3. Alolitam', '4. Kampitam'],
      correct: 2,
      explanation: 'Alolitam refers to the circular rolling movement of the head, used to signify passion, dizziness, intoxication, or deep emotion.'
    },
    {
      q: 'How many total variations exist in the Carnatic / Kuchipudi 35 Sapta Tala system?',
      options: ['1. 7', '2. 28', '3. 35 (7 Talas × 5 Jaatis)', '4. 69'],
      correct: 2,
      explanation: 'The 7 principal Talas multiplied across the 5 Pancha Jaatis (Tisra, Chaturasra, Khanda, Misra, Sankeerna) produce exactly 35 Tala variations.'
    }
  ];

  return (
    <div className="assignments-page-wrapper section-padding">
      <div className="container">
        <button onClick={() => setActivePage('home')} className="back-nav-btn">
          <ArrowLeft size={16} /> Back to Homepage
        </button>

        <div className="section-header" style={{ marginTop: '16px' }}>
          <span className="section-tag">Stage 5 Architecture Preview</span>
          <h1 className="section-title">Assignments & Examination Studio</h1>
          <TraditionalDivider motif="diya" />
          <p className="section-subtitle">
            Structured evaluations across theory, visual diagnostic recognition, and live practical performance.
          </p>
        </div>

        {/* Tab Selection */}
        <div className="assignment-tabs-bar">
          <button 
            onClick={() => setActiveTab('theory')}
            className={`tab-btn ${activeTab === 'theory' ? 'active' : ''}`}
          >
            <HelpCircle size={18} />
            <span>Type A: Theory Quiz</span>
          </button>
          <button 
            onClick={() => setActiveTab('image')}
            className={`tab-btn ${activeTab === 'image' ? 'active' : ''}`}
          >
            <Eye size={18} />
            <span>Type B: Image Identification</span>
          </button>
          <button 
            onClick={() => setActiveTab('practical')}
            className={`tab-btn ${activeTab === 'practical' ? 'active' : ''}`}
          >
            <Video size={18} />
            <span>Type C: Practical Camera Test</span>
          </button>
        </div>

        {/* Tab Content Display */}
        <div className="tab-display-card temple-card">
          {activeTab === 'theory' && (
            <div>
              <div className="tab-card-header">
                <div>
                  <h3 className="tab-title">Theory Multiple-Choice Quiz Preview</h3>
                  <span className="tab-sub">Comprehensive testing on all 7 theoretical modules</span>
                </div>
                <span className="badge-stage">Stage 5 Interactive</span>
              </div>

              <div className="questions-sample-list">
                {sampleQuestions.map((sq, idx) => (
                  <div key={idx} className="sample-question-box">
                    <span className="q-number">Question 0{idx + 1}</span>
                    <h4 className="q-text">{sq.q}</h4>
                    <div className="options-grid">
                      {sq.options.map((opt, optIdx) => (
                        <div key={optIdx} className={`sample-option ${optIdx === sq.correct ? 'is-correct' : ''}`}>
                          <span>{opt}</span>
                          {optIdx === sq.correct && <CheckCircle2 size={16} className="opt-check" />}
                        </div>
                      ))}
                    </div>
                    <div className="explanation-callout">
                      <strong>Explanation:</strong> {sq.explanation}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'image' && (
            <div>
              <div className="tab-card-header">
                <div>
                  <h3 className="tab-title">Image Identification Assignment Preview</h3>
                  <span className="tab-sub">Diagnose and identify classical gestures from authentic photographs</span>
                </div>
                <span className="badge-stage">Stage 5 Interactive</span>
              </div>

              <div className="image-test-preview-box">
                <div className="image-test-sample-photo">
                  <img src="/assets/asamyutha_sample.jpeg" alt="Test Gesture" />
                  <span className="photo-tag">Sample Test Image #01</span>
                </div>

                <div className="image-test-quiz-side">
                  <span className="q-number">Identify This Mudra</span>
                  <h4 className="q-text">Which classical one-hand gesture (Asamyutha Hastha) is shown?</h4>
                  
                  <div className="options-vertical">
                    <div className="sample-option is-correct">
                      <span>A. Pathaka (Flag Gesture)</span>
                      <CheckCircle2 size={16} className="opt-check" />
                    </div>
                    <div className="sample-option">
                      <span>B. Thripathaka (Triple Flag)</span>
                    </div>
                    <div className="sample-option">
                      <span>C. Ardhapathaka (Half Flag)</span>
                    </div>
                    <div className="sample-option">
                      <span>D. Mayura (Peacock)</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'practical' && (
            <div>
              <div className="tab-card-header">
                <div>
                  <h3 className="tab-title">Practical Performance Examination Preview</h3>
                  <span className="tab-sub">Webcam-based AI evaluated test</span>
                </div>
                <span className="badge-stage">Stage 5 Interactive</span>
              </div>

              <div className="practical-workflow-grid">
                <div className="workflow-step">
                  <div className="step-badge">1</div>
                  <strong>Target Assignment</strong>
                  <p>System assigns you a random gesture (e.g. <em>"Perform Katakamukha Hasta"</em>).</p>
                </div>
                <div className="workflow-step">
                  <div className="step-badge">2</div>
                  <strong>Perform Live</strong>
                  <p>You present the hand formation to your camera. AI analyzes joint angles in real time.</p>
                </div>
                <div className="workflow-step">
                  <div className="step-badge">3</div>
                  <strong>Automated Grading</strong>
                  <p>Model scores stability, angle precision, and saves grade directly to your student record.</p>
                </div>
              </div>
            </div>
          )}

          <div className="assignments-footer-bar">
            <span className="footer-note">
              Full scoring engines, answer validation, and database tracking will be connected in <strong>Stage 5</strong>.
            </span>
            <button 
              onClick={() => onOpenPlaceholder('Assignment Submission System', 'Stage 5 will implement full MCQ evaluation, image matching quizzes, and student grade recording into the SQLite database.')} 
              className="btn-primary"
            >
              <span>Explore Assignment Mechanics</span>
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </div>

      <style>{`
        .assignments-page-wrapper {
          background: #FAF5EC;
          min-height: 85vh;
        }

        .assignment-tabs-bar {
          display: flex;
          justify-content: center;
          gap: 12px;
          margin-bottom: 30px;
          flex-wrap: wrap;
        }

        .tab-btn {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 12px 24px;
          background: #F4E9D5;
          border: 1.5px solid rgba(196, 154, 69, 0.4);
          border-radius: var(--radius-sm);
          font-family: var(--font-serif);
          font-size: 0.9rem;
          font-weight: 700;
          color: #45151F;
          cursor: pointer;
          transition: all 0.25s ease;
        }

        .tab-btn:hover {
          border-color: #C49A45;
          background: #EEDCBE;
        }

        .tab-btn.active {
          background: #45151F;
          color: #F5DE98;
          border-color: #C49A45;
          box-shadow: 0 4px 15px rgba(69, 21, 31, 0.25);
        }

        .tab-display-card {
          padding: 36px;
        }

        @media (max-width: 768px) {
          .tab-display-card {
            padding: 20px;
          }
        }

        .tab-card-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 28px;
          border-bottom: 1.5px solid rgba(196, 154, 69, 0.3);
          padding-bottom: 16px;
        }

        .tab-title {
          font-family: var(--font-heading);
          font-size: 1.4rem;
          color: #45151F;
        }

        .tab-sub {
          font-size: 0.85rem;
          color: #6B5B52;
        }

        .badge-stage {
          background: rgba(196, 154, 69, 0.2);
          border: 1px solid #C49A45;
          color: #8A463A;
          font-family: var(--font-serif);
          font-size: 0.72rem;
          font-weight: 700;
          padding: 4px 10px;
          border-radius: 12px;
          text-transform: uppercase;
        }

        .questions-sample-list {
          display: flex;
          flex-direction: column;
          gap: 24px;
        }

        .sample-question-box {
          background: #F4E9D5;
          border: 1px solid rgba(196, 154, 69, 0.35);
          border-radius: 8px;
          padding: 20px;
        }

        .q-number {
          font-family: var(--font-serif);
          font-size: 0.75rem;
          color: #8A463A;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          display: block;
          margin-bottom: 4px;
        }

        .q-text {
          font-family: var(--font-heading);
          font-size: 1.05rem;
          color: #2E0D14;
          margin-bottom: 14px;
        }

        .options-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 10px;
          margin-bottom: 12px;
        }

        @media (max-width: 640px) {
          .options-grid {
            grid-template-columns: 1fr;
          }
        }

        .sample-option {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 10px 14px;
          background: #FAF5EC;
          border: 1px solid rgba(196, 154, 69, 0.3);
          border-radius: 6px;
          font-size: 0.85rem;
          color: #45151F;
        }

        .sample-option.is-correct {
          background: rgba(75, 181, 67, 0.12);
          border-color: #4BB543;
          font-weight: 600;
        }

        .opt-check {
          color: #4BB543;
        }

        .explanation-callout {
          background: #FAF5EC;
          border-left: 3px solid #C49A45;
          padding: 8px 12px;
          font-size: 0.8rem;
          color: #55443B;
          border-radius: 4px;
        }

        .image-test-preview-box {
          display: grid;
          grid-template-columns: 0.8fr 1.2fr;
          gap: 30px;
          align-items: center;
        }

        @media (max-width: 768px) {
          .image-test-preview-box {
            grid-template-columns: 1fr;
          }
        }

        .image-test-sample-photo {
          position: relative;
          height: 320px;
          border-radius: 8px;
          overflow: hidden;
          border: 2px solid #C49A45;
        }

        .image-test-sample-photo img {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }

        .photo-tag {
          position: absolute;
          bottom: 10px;
          left: 10px;
          background: rgba(43, 12, 19, 0.85);
          color: #F5DE98;
          font-size: 0.72rem;
          padding: 4px 8px;
          border-radius: 4px;
        }

        .options-vertical {
          display: flex;
          flex-direction: column;
          gap: 10px;
          margin-top: 14px;
        }

        .practical-workflow-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 20px;
          margin: 30px 0;
        }

        @media (max-width: 768px) {
          .practical-workflow-grid {
            grid-template-columns: 1fr;
          }
        }

        .workflow-step {
          background: #F4E9D5;
          border: 1px solid rgba(196, 154, 69, 0.35);
          border-radius: 8px;
          padding: 24px;
          text-align: center;
        }

        .step-badge {
          width: 36px;
          height: 36px;
          border-radius: 50%;
          background: #45151F;
          color: #F5DE98;
          font-family: var(--font-heading);
          font-size: 1.1rem;
          font-weight: 700;
          display: flex;
          align-items: center;
          justify-content: center;
          margin: 0 auto 12px;
          border: 1.5px solid #C49A45;
        }

        .workflow-step strong {
          display: block;
          font-family: var(--font-serif);
          font-size: 1rem;
          color: #45151F;
          margin-bottom: 6px;
        }

        .workflow-step p {
          font-size: 0.82rem;
          color: #5A4A42;
          line-height: 1.5;
        }

        .assignments-footer-bar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-top: 32px;
          padding-top: 20px;
          border-top: 1px solid rgba(196, 154, 69, 0.25);
        }

        @media (max-width: 640px) {
          .assignments-footer-bar {
            flex-direction: column;
            gap: 16px;
            text-align: center;
          }
        }

        .footer-note {
          font-size: 0.82rem;
          color: #6B5B52;
        }
      `}</style>
    </div>
  );
}
