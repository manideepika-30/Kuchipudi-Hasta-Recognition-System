import React from 'react';
import { 
  BookOpen, Video, Award, Sparkles, ArrowRight, CheckCircle2, 
  Circle, PlayCircle, Eye, Compass, Flame, Clock, Layers 
} from 'lucide-react';
import { translations } from '../utils/translations';
import TraditionalDivider from '../components/TraditionalDivider';

export default function HomePage({ 
  currentLang = 'en', 
  onStartLearning, 
  onSelectModule,
  completedModules = []
}) {
  const t = translations[currentLang] || translations.en;

  const modules = [
    {
      id: 1,
      num: '01',
      titleEn: 'Introduction to Kuchipudi',
      titleTe: 'కూచిపూడి పరిచయం',
      sanskrit: 'నాట్య శాస్త్ర ప్రవేశమ్',
      lessonsCount: 4,
      tag: 'Foundations',
      descEn: 'Historical origins from Kuchelapuram, Siddhendra Yogi tradition, Nritta, Nritya, Natya triad, and Chaturvidha Abhinaya.',
      descTe: 'కూచిపూడి చారిత్రక మూలాలు, సిద్ధేంద్ర యోగి నాట్య సంప్రదాయం, నృత్త-నృత్య-నాట్యాల త్రివేణి సంగమం.',
      highlights: ['Kuchelapuram & Siddhendra Yogi', 'Nritta, Nritya & Natya', 'Chaturvidha Abhinaya', 'Navarasa Foundations']
    },
    {
      id: 2,
      num: '02',
      titleEn: 'Asamyukta Hastas – 28 Single-Hand Gestures',
      titleTe: 'అసంయుక్త హస్తాలు (28 ఏకహస్త ముద్రలు)',
      sanskrit: 'అసంయుత హస్తాః (28 భేదాః)',
      lessonsCount: 28,
      tag: 'Hand Gestures',
      descEn: 'The authoritative canon of 28 single-hand mudras from Abhinaya Darpana: Pathaka to Trishoola with Sanskrit shlokas and usages.',
      descTe: 'అభినయ దర్పణంలోని 28 ఏకహస్త ముద్రలు: పతాక నుండి త్రిశూల వరకు శ్లోకాలు మరియు వినియోగాలు.',
      highlights: ['Pathaka, Tripathaka, Ardhapathaka', 'Mushti, Shikhara, Kapittha', 'Alapadma, Hamsasya, Mukula', 'All 28 Authentic Mudras']
    },
    {
      id: 3,
      num: '03',
      titleEn: 'Samyukta Hastas – 24 Combined-Hand Gestures',
      titleTe: 'సంయుక్త హస్తాలు (24 ద్విహస్త ముద్రలు)',
      sanskrit: 'సంయుత హస్తాః (24 భేదాః)',
      lessonsCount: 24,
      tag: 'Two-Hand Mudras',
      descEn: 'The 24 coordinate two-hand mudras from Abhinaya Darpana: Anjali to Avahittha requiring spatial alignment of both hands.',
      descTe: 'రెండు చేతుల కలయికతో రూపొందే 24 సంయుక్త హస్తాలు: అంజలి నుండి అవహిత్థ వరకు.',
      highlights: ['Anjali, Kapotha, Karkata', 'Shivalinga, Katakavardhana', 'Matsya, Kurma, Garuda', 'All 24 Dual-Hand Mudras']
    },
    {
      id: 4,
      num: '04',
      titleEn: 'Shiro Bhedas – 9 Head Movements',
      titleTe: 'శిరో భేదాలు (9 శిరస్సు కదలికలు)',
      sanskrit: 'శిరో భేద ప్రకరణమ్',
      lessonsCount: 9,
      tag: 'Head Movements',
      descEn: 'The 9 canonical head and neck kinetic gestures from Natyashastra expressing emotional inflection, inquiry, affirmation, and awe.',
      descTe: 'నాట్యశాస్త్రంలోని 9 శిరో భేదాలు: సమ, ఉద్వాహిత, అధోముఖ, ఆలోలిత, ధూత, కంపిత, పరావృత్త, ఉత్క్షిప్త, పరివాహిత.',
      highlights: ['Sama & Udvahita', 'Adhomukha & Alolita', 'Dhuta, Kampita & Paravritta', 'Ukshipta & Parivahita']
    },
    {
      id: 5,
      num: '05',
      titleEn: 'Drishti Bhedas – 8 Eye Movements',
      titleTe: 'దృష్టి భేదాలు (8 నేత్ర కదలికలు)',
      sanskrit: 'దృష్టి భేద ప్రకరణమ్',
      lessonsCount: 8,
      tag: 'Eye Glances',
      descEn: 'The 8 expressive eye glances directing Sattvika Abhinaya and guiding emotional resonance (Bhava and Rasa) in classical performance.',
      descTe: 'రసావిష్కరణకు మూలమైన 8 సునిశిత నేత్ర వీక్షణలు: సమ, ఆలోకిత, సాచి, ప్రలోకిత, నిమీలిత, ఉల్లోకిత, అనువృత్త, అవలోకిత.',
      highlights: ['Sama (Level Gaze)', 'Alokita (Circular Scan)', 'Sachi (Corner Glance)', 'Nimilita, Ullokita & Avalokita']
    },
    {
      id: 6,
      num: '06',
      titleEn: 'Chaturvidha Abhinaya & Navarasas',
      titleTe: 'చతుర్విధ అభినయం మరియు నవరసాలు',
      sanskrit: 'ద్వాదశ హస్త ప్రణామాః & నవరసాః',
      lessonsCount: 12,
      tag: 'Expression & Salutations',
      descEn: 'The 12 classical Dvadasha Hasta Pranamas (ritual salutations) combined with the theory of 9 primary Rasas.',
      descTe: '12 శాస్త్రీయ హస్త ప్రణామాలు, ఆచార నమస్కారాలు మరియు నవరస భావ సిద్ధాంతం.',
      highlights: ['12 Pranamas (Salutations)', 'Prasaranam & Kunchitam', 'Navarasas (9 Emotions)', 'Sattvika Expression']
    },
    {
      id: 7,
      num: '07',
      titleEn: 'Sapta Talamulu – 35 Tala System',
      titleTe: 'సప్త తాళములు (35 తాళ వ్యవస్థ)',
      sanskrit: 'సప్త తాల ప్రకరణమ్',
      lessonsCount: 7,
      tag: 'Rhythm & Talas',
      descEn: 'The 7 principal Talas (Dhruva, Matya, Rupaka, Jhampe, Triputa, Ata, Eka), Angas (Laghu, Dhrutam), and the 35 Tala Carnatic matrix.',
      descTe: '7 ప్రధాన తాళాలు, లఘు-ధ్రుత అంగాలు, 5 జాతులు (తిస్ర, చతురస్ర, ఖండ, మిశ్ర, సంకీర్ణ) మరియు 35 తాళాల సమగ్ర వ్యవస్థ.',
      highlights: ['Dhruva, Matya & Rupaka', 'Jhampe, Triputa, Ata, Eka', '5 Jaatis (3, 4, 5, 7, 9)', '35 Tala Matrix']
    }
  ];

  return (
    <div className="portal-homepage-content">
      {/* ===================================================================
          1. HERO SECTION (Classical Dancer, Shloka, & Start Learning)
          =================================================================== */}
      <section className="portal-hero card-soft">
        <div className="hero-grid">
          {/* Left Text Column */}
          <div className="hero-text-col">
            <div className="hero-branding-pill">
              <Sparkles size={14} className="gold-sparkle" />
              <span>Public Learning Portal • ఉచిత అభ్యాస వేదిక</span>
            </div>

            <h1 className="hero-main-title">
              {currentLang === 'te' 
                ? 'కూచిపూడి నాట్య సౌందర్యాన్ని ఆస్వాదించండి' 
                : 'Welcome to Nritya AI'}
            </h1>
            
            <p className="hero-subtitle">
              {currentLang === 'te'
                ? 'కృత్రిమ మేధ ఆధారిత కూచిపూడి నాట్య అభ్యాస, గుర్తింపు మరియు విశ్లేషణ వేదిక'
                : 'AI-Based Kuchipudi Dance Learning, Recognition & Analysis System'}
            </p>

            <p className="hero-desc">
              {currentLang === 'te'
                ? 'సిద్ధేంద్ర యోగి సంప్రదాయం నుండి ఉద్భవించిన ప్రామాణిక సిద్ధాంత పాఠాలు, 52 హస్త ముద్రలు, శిరో-దృష్టి భేదాలు మరియు కంప్యూటర్ విజన్ AI కెమెరా ద్వారా కూచిపూడి నృత్యాన్ని నేర్చుకోండి.'
                : 'Master authentic Kuchipudi theory, 52 canonical hand mudras, head kinetics, eye glances, and rhythmic tala cycles with computer vision AI guidance. No login required to begin.'}
            </p>

            {/* Sacred Sanskrit Shloka Card */}
            <div className="shloka-quote-card">
              <p className="shloka-verse">
                यतो हस्तस्ततो दृष्टिर्यतो दृष्टिस्ततो मनः ।<br />
                यतो मनस्ततो भावो यतो भावस्ततो रसः ॥
              </p>
              <p className="shloka-meaning">
                {t.footerShlokaMeaning || '"Where the hands go, the eyes follow; where the eyes go, the mind follows; where the mind goes, expressions emerge; and where expressions arise, Rasa is born."'}
              </p>
            </div>

            {/* Direct CTA Action Buttons */}
            <div className="hero-actions-row">
              <button onClick={() => onStartLearning(1)} className="btn-primary hero-btn-main">
                <PlayCircle size={20} />
                <span>{currentLang === 'te' ? 'పాఠ్యాంశాలు ప్రారంభించండి' : 'Start Learning (Module 1)'}</span>
              </button>

              <button onClick={() => onSelectModule('practice')} className="btn-secondary hero-btn-sub">
                <Video size={18} />
                <span>{t.btnPracticeStudio || 'AI Practice Studio'}</span>
              </button>
            </div>
          </div>

          {/* Right Hero Visual Column */}
          <div className="hero-visual-col">
            <div className="dancer-hero-frame">
              <img 
                src="/assets/asamyutha_sample.jpeg" 
                alt="Authentic Kuchipudi Dance" 
                className="dancer-hero-img"
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.src = '/assets/hastas/asamyukta_01_pathaka.jpg';
                }}
              />
              <div className="dancer-frame-badge">
                <strong className="badge-title">Pathaka Mudra • పతాకః</strong>
                <span className="badge-sub">Canonical Kuchipudi Asamyukta Hasta</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ===================================================================
          2. QUICK STATS BAR
          =================================================================== */}
      <div className="portal-stats-bar">
        <div className="stat-item">
          <strong className="stat-number">7</strong>
          <span className="stat-text">Learning Modules</span>
        </div>
        <div className="stat-divider" />
        <div className="stat-item">
          <strong className="stat-number">92</strong>
          <span className="stat-text">Theory Lessons</span>
        </div>
        <div className="stat-divider" />
        <div className="stat-item">
          <strong className="stat-number">52</strong>
          <span className="stat-text">Authentic Hastas</span>
        </div>
        <div className="stat-divider" />
        <div className="stat-item">
          <strong className="stat-number">69</strong>
          <span className="stat-text">AI Recognition Classes</span>
        </div>
        <div className="stat-divider" />
        <div className="stat-item">
          <strong className="stat-number">100%</strong>
          <span className="stat-text">Open Guest Access</span>
        </div>
      </div>

      {/* ===================================================================
          3. SEVEN LEARNING MODULES GRID
          =================================================================== */}
      <section className="modules-catalog-section">
        <div className="section-title-wrap">
          <div>
            <span className="section-eyebrow">CURRICULUM OVERVIEW</span>
            <h2 className="section-main-heading">
              {currentLang === 'te' ? 'ఏడు ప్రామాణిక అభ్యాస విభాగాలు' : 'Seven Canonical Learning Modules'}
            </h2>
            <p className="section-desc">
              {currentLang === 'te'
                ? 'క్రింది ఏ విభాగంపైనైనా క్లిక్ చేసి వెంటనే పాఠాలను ఉచితంగా చదవవచ్చు మరియు అభ్యసించవచ్చు.'
                : 'Click any module card below to open its verified lessons, authentic photographs, Sanskrit shlokas, and Telugu explanations.'}
            </p>
          </div>
        </div>

        <div className="modules-card-grid">
          {modules.map((m) => {
            const isCompleted = completedModules.includes(m.id);
            return (
              <div key={m.id} className="module-catalog-card card-soft">
                <div className="mod-card-top">
                  <div className="mod-num-chip">Module {m.num}</div>
                  <span className="mod-category-pill">{m.tag}</span>
                  {isCompleted && (
                    <span className="completed-badge" title="Completed">
                      <CheckCircle2 size={16} />
                    </span>
                  )}
                </div>

                <h3 className="mod-card-title">
                  {currentLang === 'te' ? m.titleTe : m.titleEn}
                </h3>
                <span className="mod-card-sanskrit">{m.sanskrit}</span>

                <p className="mod-card-desc">
                  {currentLang === 'te' ? m.descTe : m.descEn}
                </p>

                <div className="mod-highlights-wrap">
                  {m.highlights.map((h, i) => (
                    <span key={i} className="highlight-pill">• {h}</span>
                  ))}
                </div>

                <div className="mod-card-footer">
                  <span className="lessons-count-text">
                    <strong>{m.lessonsCount}</strong> lessons
                  </span>
                  <button 
                    onClick={() => onSelectModule(m.id)} 
                    className="btn-open-module"
                    aria-label={`Open Module ${m.id}`}
                  >
                    <span>Open Module</span>
                    <ArrowRight size={15} />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* ===================================================================
          4. PRACTICE STUDIO INTRODUCTION CARD
          =================================================================== */}
      <section className="practice-intro-section">
        <div className="practice-feature-card card-sage">
          <div className="practice-feature-content">
            <span className="practice-tag">COMPUTER VISION & AI</span>
            <h2 className="practice-title">AI Practice Studio (Camera & Image Upload)</h2>
            <p className="practice-desc">
              Practice physical mudras in front of your webcam or test with images. Our consolidated TensorFlow Keras model classifies 52 Hastas, 9 Shiro Bhedas, and 8 Drishti Bhedas in real time without simulating fake predictions.
            </p>
            <div className="practice-perks">
              <div className="perk-item">
                <CheckCircle2 size={16} className="perk-icon" />
                <span>Real-time hand and face landmark analysis</span>
              </div>
              <div className="perk-item">
                <CheckCircle2 size={16} className="perk-icon" />
                <span>Instant confidence percentage and posture feedback</span>
              </div>
              <div className="perk-item">
                <CheckCircle2 size={16} className="perk-icon" />
                <span>Guest session history stored locally in browser</span>
              </div>
            </div>
            <button onClick={() => onSelectModule('practice')} className="btn-green practice-launch-btn">
              <Video size={18} />
              <span>Launch Practice Studio</span>
            </button>
          </div>

          <div className="practice-feature-visual">
            <div className="studio-mini-preview">
              <div className="preview-header">
                <span className="preview-dot red" />
                <span className="preview-dot yellow" />
                <span className="preview-dot green" />
                <span className="preview-title">Real-Time AI Vision Feed</span>
              </div>
              <div className="preview-body">
                <img src="/assets/hastas/asamyukta_01_pathaka.jpg" alt="Pathaka Preview" className="preview-img" />
                <div className="preview-overlay">
                  <span className="overlay-tag">Detected: Pathaka (94.2%)</span>
                  <span className="overlay-sub">Pose Matched ✓</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ===================================================================
          5. ABOUT KUCHIPUDI HERITAGE SUMMARY
          =================================================================== */}
      <section className="heritage-summary-section card-soft">
        <div className="heritage-inner">
          <span className="section-eyebrow">ABOUT KUCHIPUDI</span>
          <h2 className="section-main-heading">{t.aboutTitle || 'About Kuchipudi Classical Dance'}</h2>
          <p className="heritage-p">
            {t.aboutP1 || 'Kuchipudi is one of the eight major Indian classical dances, originating from the historic village of Kuchelapuram in Andhra Pradesh. Codified by Siddhendra Yogi, it uniquely synthesizes pure rhythmic footwork (Nritta), expressive mime (Nritya), and sacred dramatic storytelling (Natya).'}
          </p>
          <div className="heritage-actions">
            <button onClick={() => onSelectModule('about')} className="btn-outline">
              <span>Read Full Heritage Guide</span>
              <ArrowRight size={15} />
            </button>
          </div>
        </div>
      </section>

      <style>{`
        .portal-homepage-content {
          display: flex;
          flex-direction: column;
          gap: 32px;
          padding: 24px 0 48px;
        }

        /* Hero */
        .portal-hero {
          background: linear-gradient(135deg, #FFFFFF 0%, var(--color-blush-light) 60%, var(--color-sage-light) 100%);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg);
          padding: 40px;
          box-shadow: 0 4px 20px rgba(72, 44, 58, 0.05);
        }
        .hero-grid {
          display: grid;
          grid-template-columns: 1.25fr 0.75fr;
          gap: 36px;
          align-items: center;
        }
        .hero-branding-pill {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          background: var(--color-blush-ivory);
          color: var(--color-temple-green);
          font-size: 0.78rem;
          font-weight: 700;
          padding: 5px 12px;
          border-radius: 999px;
          border: 1px solid var(--color-border);
          margin-bottom: 14px;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
        .gold-sparkle {
          color: var(--color-gold-dark);
        }
        .hero-main-title {
          font-size: 2.2rem;
          font-family: var(--font-heading);
          color: var(--color-deep-plum);
          line-height: 1.2;
          margin-bottom: 8px;
        }
        .hero-subtitle {
          font-size: 1rem;
          font-weight: 600;
          color: var(--color-temple-green);
          margin-bottom: 12px;
        }
        .hero-desc {
          font-size: 0.95rem;
          color: var(--color-text-secondary);
          line-height: 1.6;
          margin-bottom: 20px;
        }
        .shloka-quote-card {
          background: #FFFDF9;
          border-left: 4px solid var(--color-gold-dark);
          padding: 12px 18px;
          border-radius: 4px;
          margin-bottom: 24px;
          border: 1px solid rgba(191, 160, 108, 0.25);
          border-left-width: 4px;
        }
        .shloka-verse {
          font-family: var(--font-shloka);
          font-size: 1.15rem;
          color: var(--color-deep-plum);
          margin-bottom: 4px;
          line-height: 1.4;
          font-weight: 600;
        }
        .shloka-meaning {
          font-size: 0.8rem;
          color: var(--color-text-secondary);
          font-style: italic;
          margin: 0;
          line-height: 1.4;
        }
        .hero-actions-row {
          display: flex;
          align-items: center;
          gap: 14px;
          flex-wrap: wrap;
        }
        .hero-btn-main {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 0.95rem;
          padding: 12px 24px;
        }
        .hero-btn-sub {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 0.95rem;
          padding: 12px 20px;
        }
        .hero-visual-col {
          display: flex;
          justify-content: center;
        }
        .dancer-hero-frame {
          width: 100%;
          max-width: 320px;
          border-radius: var(--radius-md);
          overflow: hidden;
          background: #FFFFFF;
          border: 2px solid var(--color-border);
          box-shadow: 0 8px 24px rgba(72, 44, 58, 0.12);
        }
        .dancer-hero-img {
          width: 100%;
          height: 380px;
          object-fit: cover;
          display: block;
        }
        .dancer-frame-badge {
          background: var(--color-deep-plum);
          color: #FFFFFF;
          padding: 10px 14px;
          display: flex;
          flex-direction: column;
        }
        .badge-title {
          font-size: 0.9rem;
          color: var(--color-antique-gold);
          font-family: var(--font-heading);
        }
        .badge-sub {
          font-size: 0.75rem;
          opacity: 0.85;
        }

        /* Stats Bar */
        .portal-stats-bar {
          display: flex;
          align-items: center;
          justify-content: space-around;
          background: #FFFFFF;
          border: 1px solid var(--color-border);
          border-radius: var(--radius-md);
          padding: 16px 20px;
          box-shadow: var(--shadow-sm);
        }
        .stat-item {
          text-align: center;
        }
        .stat-number {
          display: block;
          font-family: var(--font-heading);
          font-size: 1.5rem;
          color: var(--color-deep-plum);
          line-height: 1.1;
        }
        .stat-text {
          font-size: 0.75rem;
          color: var(--color-text-secondary);
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }
        .stat-divider {
          width: 1px;
          height: 32px;
          background: var(--color-border);
        }

        /* Modules Grid */
        .modules-catalog-section {
          display: flex;
          flex-direction: column;
          gap: 20px;
        }
        .section-eyebrow {
          font-size: 0.72rem;
          font-weight: 800;
          color: var(--color-gold-dark);
          text-transform: uppercase;
          letter-spacing: 0.1em;
          display: block;
          margin-bottom: 4px;
        }
        .section-main-heading {
          font-size: 1.6rem;
          font-family: var(--font-heading);
          color: var(--color-deep-plum);
          margin-bottom: 6px;
        }
        .section-desc {
          font-size: 0.9rem;
          color: var(--color-text-secondary);
          max-width: 800px;
          line-height: 1.5;
        }
        .modules-card-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
          gap: 20px;
        }
        .module-catalog-card {
          background: #FFFFFF;
          border: 1px solid var(--color-border);
          border-radius: var(--radius-md);
          padding: 22px;
          display: flex;
          flex-direction: column;
          transition: transform 0.2s ease, box-shadow 0.2s ease;
          position: relative;
        }
        .module-catalog-card:hover {
          transform: translateY(-3px);
          box-shadow: 0 8px 24px rgba(72, 44, 58, 0.09);
        }
        .mod-card-top {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 12px;
        }
        .mod-num-chip {
          background: var(--color-deep-plum);
          color: #FFFFFF;
          font-size: 0.75rem;
          font-weight: 700;
          padding: 3px 8px;
          border-radius: 4px;
        }
        .mod-category-pill {
          background: var(--color-sage-light);
          color: var(--color-temple-green);
          font-size: 0.72rem;
          font-weight: 600;
          padding: 3px 8px;
          border-radius: 4px;
        }
        .completed-badge {
          margin-left: auto;
          color: #2e7d32;
        }
        .mod-card-title {
          font-family: var(--font-heading);
          font-size: 1.15rem;
          color: var(--color-deep-plum);
          margin-bottom: 4px;
          line-height: 1.3;
        }
        .mod-card-sanskrit {
          font-size: 0.82rem;
          color: var(--color-gold-dark);
          font-style: italic;
          margin-bottom: 10px;
          display: block;
        }
        .mod-card-desc {
          font-size: 0.85rem;
          color: var(--color-text-secondary);
          line-height: 1.5;
          margin-bottom: 14px;
          flex-grow: 1;
        }
        .mod-highlights-wrap {
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
          margin-bottom: 18px;
        }
        .highlight-pill {
          font-size: 0.72rem;
          background: var(--color-blush-light);
          color: var(--color-deep-plum);
          padding: 2px 7px;
          border-radius: 4px;
        }
        .mod-card-footer {
          display: flex;
          align-items: center;
          justify-content: space-between;
          border-top: 1px solid var(--color-border);
          padding-top: 14px;
          margin-top: auto;
        }
        .lessons-count-text {
          font-size: 0.82rem;
          color: var(--color-text-secondary);
        }
        .btn-open-module {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          background: var(--color-blush-light);
          border: 1px solid var(--color-border);
          color: var(--color-deep-plum);
          padding: 6px 14px;
          border-radius: var(--radius-sm);
          font-size: 0.85rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
        }
        .btn-open-module:hover {
          background: var(--color-deep-plum);
          color: #FFFFFF;
        }

        /* Practice Feature Card */
        .practice-feature-card {
          border-radius: var(--radius-lg);
          padding: 36px;
          display: grid;
          grid-template-columns: 1.2fr 0.8fr;
          gap: 32px;
          align-items: center;
          border: 1px solid var(--color-border);
        }
        .practice-tag {
          font-size: 0.72rem;
          font-weight: 800;
          color: var(--color-temple-green);
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }
        .practice-title {
          font-family: var(--font-heading);
          font-size: 1.6rem;
          color: var(--color-deep-plum);
          margin: 6px 0 12px;
        }
        .practice-desc {
          font-size: 0.9rem;
          color: var(--color-text-secondary);
          line-height: 1.6;
          margin-bottom: 16px;
        }
        .practice-perks {
          display: flex;
          flex-direction: column;
          gap: 8px;
          margin-bottom: 24px;
        }
        .perk-item {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 0.85rem;
          color: var(--color-text-primary);
        }
        .perk-icon {
          color: var(--color-temple-green);
          flex-shrink: 0;
        }
        .practice-launch-btn {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 12px 24px;
          font-size: 0.95rem;
        }
        .studio-mini-preview {
          background: #FFFFFF;
          border: 1px solid var(--color-border);
          border-radius: var(--radius-md);
          overflow: hidden;
          box-shadow: 0 6px 20px rgba(0,0,0,0.06);
        }
        .preview-header {
          background: var(--color-deep-plum);
          padding: 8px 12px;
          display: flex;
          align-items: center;
          gap: 6px;
        }
        .preview-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }
        .preview-dot.red { background: #ff5f56; }
        .preview-dot.yellow { background: #ffbd2e; }
        .preview-dot.green { background: #27c93f; }
        .preview-title {
          font-size: 0.72rem;
          color: #F4E8E2;
          margin-left: 8px;
          font-weight: 500;
        }
        .preview-body {
          position: relative;
          height: 240px;
          background: #000;
        }
        .preview-img {
          width: 100%;
          height: 100%;
          object-fit: cover;
          opacity: 0.9;
        }
        .preview-overlay {
          position: absolute;
          bottom: 12px;
          left: 12px;
          right: 12px;
          background: rgba(40, 25, 33, 0.88);
          color: #FFF;
          padding: 8px 12px;
          border-radius: 6px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 0.8rem;
          backdrop-filter: blur(4px);
        }
        .overlay-tag { font-weight: 700; color: #a5d6a7; }
        .overlay-sub { font-size: 0.75rem; color: #E8D1A7; }

        /* Heritage Summary */
        .heritage-summary-section {
          background: #FFFFFF;
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg);
          padding: 32px 36px;
        }
        .heritage-p {
          font-size: 0.95rem;
          color: var(--color-text-secondary);
          line-height: 1.7;
          margin-bottom: 18px;
          max-width: 900px;
        }

        @media (max-width: 992px) {
          .hero-grid {
            grid-template-columns: 1fr;
          }
          .hero-visual-col {
            display: none;
          }
          .practice-feature-card {
            grid-template-columns: 1fr;
          }
          .practice-feature-visual {
            display: none;
          }
          .portal-stats-bar {
            flex-wrap: wrap;
            gap: 16px;
          }
          .stat-divider {
            display: none;
          }
        }
      `}</style>
    </div>
  );
}
