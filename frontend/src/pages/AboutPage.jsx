import React from 'react';
import TraditionalDivider from '../components/TraditionalDivider';
import { Sparkles, BookOpen, Heart, Award } from 'lucide-react';

export default function AboutPage({ setActivePage }) {
  return (
    <div className="about-page-wrapper section-padding">
      <div className="container">
        <div className="section-header">
          <span className="section-tag">Classical Indian Heritage</span>
          <h1 className="section-title">About Kuchipudi Classical Dance</h1>
          <TraditionalDivider motif="lotus" />
          <p className="section-subtitle">
            A spiritual confluence of sacred drama, expressive mime, rhythmic footwork, and devotion originating from Andhra Pradesh.
          </p>
        </div>

        <div className="about-content-card temple-card">
          <div className="about-grid">
            <div className="about-text-column">
              <h2 className="about-h2">1. Origin & Spiritual Evolution</h2>
              <p>
                Kuchipudi is one of the eight officially recognized classical dance styles of India. Originating in <strong>Kuchelapuram</strong>, a tranquil village in the Krishna district of Andhra Pradesh, it derives its lineage from the venerable <em>Natya Shastra</em> authored by sage Bharata Muni.
              </p>
              <p>
                In the fourteenth to sixteenth century, <strong>Siddhendra Yogi</strong>, an ascetic and devotional scholar, codified the tradition into dance-dramas (Yakshaganas) dedicated to Lord Krishna. Brahmin male performers (known as <em>Bhagavathulu</em>) travelled from village to village staging sacred performances, infusing rigorous classical technique with profound spiritual fervor.
              </p>

              <h2 className="about-h2" style={{ marginTop: '28px' }}>2. The Triad of Classical Expression</h2>
              <div className="triad-grid">
                <div className="triad-item">
                  <span className="triad-num">I</span>
                  <strong>Nritta (नृत्त)</strong>
                  <p>Pure rhythmic dance showcasing crisp body geometry, sculptural poses, and complex footwork set to Tala.</p>
                </div>
                <div className="triad-item">
                  <span className="triad-num">II</span>
                  <strong>Nritya (नृत्य)</strong>
                  <p>Expressive solo storytelling employing hand gestures (Hastas), facial expressions (Mukhaja Abhinaya), and emotive glances.</p>
                </div>
                <div className="triad-item">
                  <span className="triad-num">III</span>
                  <strong>Natya (नाट्य)</strong>
                  <p>Dramatic representation combining dance, spoken dialogue, and theatrical character portrayal.</p>
                </div>
              </div>

              <h2 className="about-h2" style={{ marginTop: '28px' }}>3. The Abhinaya Tradition in Our AI Platform</h2>
              <p>
                According to the <em>Abhinaya Darpana</em> (Mirror of Gesture) by Nandikesvara, bodily expression is articulated through four modes: <em>Angika</em> (gestures and body posture), <em>Vachika</em> (song and speech), <em>Aharya</em> (costume and ornamentation), and <em>Sattvika</em> (genuine emotional states).
              </p>
              <p>
                Our AI learning platform concentrates on the sacred precision of <strong>Angika Abhinaya</strong>:
              </p>
              <ul className="about-bullet-list">
                <li><strong>Hastas:</strong> 28 Asamyutha (one-hand) and 24 Samyutha (two-hand) mudras.</li>
                <li><strong>Shiro Bhedas:</strong> 9 distinct head and neck kinetic movements.</li>
                <li><strong>Drishti Bhedas:</strong> 8 expressive eye glances.</li>
                <li><strong>Sapta Talas:</strong> The 35 rhythmic tala systems governing foot percussion.</li>
              </ul>
            </div>

            <div className="about-image-column">
              <div className="about-dancer-frame">
                <img src="/assets/samyutha_sample.jpeg" alt="Kuchipudi Dance Performance" className="about-dancer-img" />
                <div className="frame-caption">
                  <span>Traditional Anjali & Samyutha Hasta Mudras</span>
                </div>
              </div>

              <div className="shloka-box" style={{ marginTop: '24px' }}>
                <p className="shloka-sanskrit">
                  आङ्गिकं भुवनं यस्य वाचिकं सर्ववाङ्गमयम् ।<br />
                  आहार्यं चन्द्रतारादि तं नुमः सात्त्विकं शिवम् ॥
                </p>
                <p className="shloka-meaning">
                  "Whose bodily movement is the universe, whose speech is all language, whose ornaments are the moon and the stars; to that pure and luminous Shiva, we bow."
                </p>
              </div>

              <div className="cta-about-box">
                <button onClick={() => setActivePage('modules')} className="btn-primary" style={{ width: '100%' }}>
                  <BookOpen size={18} />
                  <span>Begin Theory Modules</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .about-page-wrapper {
          background: #FAF5EC;
          min-height: 80vh;
        }

        .about-content-card {
          padding: 48px;
        }

        @media (max-width: 768px) {
          .about-content-card {
            padding: 24px;
          }
        }

        .about-grid {
          display: grid;
          grid-template-columns: 1.25fr 0.75fr;
          gap: 40px;
        }

        @media (max-width: 1024px) {
          .about-grid {
            grid-template-columns: 1fr;
          }
        }

        .about-h2 {
          font-family: var(--font-heading);
          font-size: 1.45rem;
          color: #45151F;
          margin-bottom: 12px;
          border-bottom: 1.5px solid rgba(196, 154, 69, 0.3);
          padding-bottom: 6px;
        }

        .about-text-column p {
          font-size: 0.96rem;
          color: #4A3A32;
          line-height: 1.7;
          margin-bottom: 14px;
        }

        .triad-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 16px;
          margin: 18px 0;
        }

        @media (max-width: 640px) {
          .triad-grid {
            grid-template-columns: 1fr;
          }
        }

        .triad-item {
          background: #F4E9D5;
          border: 1px solid rgba(196, 154, 69, 0.4);
          border-radius: 8px;
          padding: 16px;
          text-align: center;
        }

        .triad-num {
          display: inline-block;
          font-family: var(--font-heading);
          font-size: 1.1rem;
          color: #C49A45;
          font-weight: 700;
          margin-bottom: 4px;
        }

        .triad-item strong {
          display: block;
          font-family: var(--font-serif);
          font-size: 0.95rem;
          color: #45151F;
          margin-bottom: 6px;
        }

        .triad-item p {
          font-size: 0.78rem !important;
          color: #5A4A42 !important;
          line-height: 1.45 !important;
          margin-bottom: 0 !important;
        }

        .about-bullet-list {
          list-style: none;
          display: flex;
          flex-direction: column;
          gap: 8px;
          margin-top: 10px;
        }

        .about-bullet-list li {
          font-size: 0.92rem;
          color: #45151F;
          padding-left: 20px;
          position: relative;
        }

        .about-bullet-list li::before {
          content: '◆';
          position: absolute;
          left: 0;
          color: #C49A45;
          font-size: 0.8rem;
        }

        .about-dancer-frame {
          border-radius: 12px;
          overflow: hidden;
          border: 2px solid #C49A45;
          box-shadow: 0 10px 25px rgba(69, 21, 31, 0.2);
          position: relative;
        }

        .about-dancer-img {
          width: 100%;
          height: 380px;
          object-fit: cover;
          display: block;
        }

        .frame-caption {
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          background: rgba(43, 12, 19, 0.85);
          color: #F5DE98;
          font-family: var(--font-serif);
          font-size: 0.82rem;
          padding: 10px 16px;
          text-align: center;
          border-top: 1px solid #C49A45;
        }

        .cta-about-box {
          margin-top: 24px;
        }
      `}</style>
    </div>
  );
}
