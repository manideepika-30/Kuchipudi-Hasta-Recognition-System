import React, { useState } from 'react';
import TraditionalDivider from '../components/TraditionalDivider';
import { BookOpen, Video, Award, ChevronDown, ChevronUp, CheckCircle, Sparkles } from 'lucide-react';

export default function ModulesPage({ setActivePage }) {
  const [expandedModule, setExpandedModule] = useState(2); // default open Asamyutha Hasthas

  const modulesList = [
    {
      id: 1,
      tag: 'Module 01',
      title: 'Introduction to Kuchipudi',
      sanskrit: 'नाट्य शास्त्र प्रवेशम्',
      pdfSource: 'Curriculum Foundations',
      totalItems: 'Theory & History',
      description: 'Foundational introduction to classical Indian dance, history of Kuchipudi from Kuchelapuram, Natya Shastra lineages, the spiritual purpose of dance, and the synthesis of Nritta, Nritya, and Natya.',
      topics: [
        'What is Kuchipudi? — Spiritual roots and revival by Siddhendra Yogi',
        'Introduction to Classical Indian Dance and Natya Shastra tradition',
        'Importance of Hastas and Mudras in communication',
        'Importance of Mukhaja Abhinaya (facial expressions and nine Rasas)',
        'Introduction to Shiro Bhedhas (head) and Drishti Bhedhas (eye glances)',
        'Introduction to Tala, Laya, and rhythmic metre'
      ]
    },
    {
      id: 2,
      tag: 'Module 02',
      title: 'Asamyutha Hasthas (Single-Hand Gestures)',
      sanskrit: 'असंयुत हस्ताः (२८ भेदाः)',
      pdfSource: 'Asamyutha_Hasthas.pdf',
      totalItems: '28 Gestures',
      description: 'The 28 canonical single-hand gestures with formations, symbolic meanings, and classical usages according to the Abhinaya Darpana.',
      gestures: [
        '1. Pathaka (Flag)', '2. Thripathaka (Triple Flag)', '3. Ardhapathaka (Half Flag)', '4. Kartarimukha (Scissor Face)',
        '5. Mayura (Peacock)', '6. Ardhachandra (Half Moon)', '7. Arala (Bent)', '8. Sukatunda (Parrot Beak)',
        '9. Mushti (Fist)', '10. Shikhara (Peak/Spire)', '11. Kapittha (Wood Apple)', '12. Katakamukha (Opening in a Link)',
        '13. Soochi (Needle)', '14. Chandrakala (Crescent Moon)', '15. Padmakosha (Lotus Bud)', '16. Sarpasheersha (Snake Hood)',
        '17. Mrugasheersha (Deer Head)', '18. Simhamukha (Lion Face)', '19. Kaangula (Bell/Tail)', '20. Alapadma (Full Bloomed Lotus)',
        '21. Chatura (Four Sides)', '22. Bhramara (Bee)', '23. Hamsasya (Swan Beak)', '24. Hamsapaksha (Swan Wing)',
        '25. Samdamsha (Pincers/Grip)', '26. Mukula (Blossom)', '27. Tamrachuda (Rooster Crest)', '28. Trishoola (Trident)'
      ]
    },
    {
      id: 3,
      tag: 'Module 03',
      title: 'Samyutha Hasthas (Two-Hand Gestures)',
      sanskrit: 'संयुत हस्ताः (२४ भेदाः)',
      pdfSource: 'Samyutha Hasthas.pdf',
      totalItems: '24 Gestures',
      description: 'The 24 combined two-hand gestures where both hands collaborate to represent deities, animals, cosmic relationships, and emotional states.',
      gestures: [
        '1. Anjali (Salutation)', '2. Kapothamu (Dove)', '3. Karkatamu (Crab)', '4. Swasthikamu (Crossed)',
        '5. Dola Hastamu (Swing)', '6. Pushpaputamu (Flower Casket)', '7. Uthsungamu (Embrace)', '8. Sivalingamu (Phallic Icon)',
        '9. Katakavardhanamu (Link of Increase)', '10. Karthariswasthikamu (Crossed Shears)', '11. Sakatamu (Car/Wheel)', '12. Sankhamu (Conch)',
        '13. Chakramu (Discus)', '14. Samputamu (Casket/Box)', '15. Paasamu (Noose)', '16. Keelakamu (Axle/Bond)',
        '17. Matsyamu (Fish)', '18. Kurmamu (Tortoise)', '19. Varahamu (Boar)', '20. Garudamu (Sacred Eagle)',
        '21. Nagabandhakamu (Serpent Coil)', '22. Khatva (Cot/Bed)', '23. Bherundamu (Two-headed Bird)', '24. Avahithamu (Dissimulation)'
      ]
    },
    {
      id: 4,
      tag: 'Module 04',
      title: 'Shiro Bhedhas (Head Movements)',
      sanskrit: 'शिरो भेदाः (९ भेदाः)',
      pdfSource: 'Shiro Bhedhas.pdf',
      totalItems: '9 Movements',
      description: 'The nine distinct head and neck kinetic movements governing spatial focus, emotional inflection, and bodily poise.',
      gestures: [
        '1. Sama (Level / Natural)', '2. Udvahitam (Raised upwards / Pride & Sky)', '3. Adhomukham (Cast downwards / Modesty & Sorrow)',
        '4. Alolitam (Circular rolling / Passion & Dizziness)', '5. Dhutam (Side-to-side shake / Denial & Dismay)', '6. Kampitam (Up & down nodding / Inquiry & Welcome)',
        '7. Paravrittam (Turned aside / Aversion & Respect)', '8. Ukshiptam (Tilted upwards / Longing & Height)', '9. Parivahitam (Gently swinging side to side / Ecstasy)'
      ]
    },
    {
      id: 5,
      tag: 'Module 05',
      title: 'Drishti Bhedhas (Eye Movements)',
      sanskrit: 'दृष्टि भेदाः (८ भेदाः)',
      pdfSource: 'Drishti Bhedhas.pdf',
      totalItems: '8 Eye Movements',
      description: 'The eight canonical eye movements (glances) essential for expressing the subtle psychological states (Bhavas) and Rasas.',
      gestures: [
        '1. Sama (Straight gaze / Beginning of dance)', '2. Alokita (Swiftly looking all around / Surprise)', '3. Saachi (Sideways glance / Amor & Secret intent)',
        '4. Pralokita (Looking from side to side / Wonder & Restlessness)', '5. Nimilita (Half-closed meditative eyes / Prayer & Sagehood)',
        '6. Ullokita (Gazing upwards / Moon, stars & divine vision)', '7. Anuvritta (Glancing rapidly up & down / Scrutiny & Fury)', '8. Avalokita (Looking downwards / Contemplation & Earth)'
      ]
    },
    {
      id: 6,
      tag: 'Module 06',
      title: 'Dvadasha Hasta Pranamas (Hand Salutations)',
      sanskrit: 'द्वादश हस्त प्रणामाः (१२ भेदाः)',
      pdfSource: 'Dvadasha Hasta Pranamas.pdf',
      totalItems: '12 Salutations',
      description: 'The twelve classical hand salutations expressing reverence, offering, and ritual devotion before Gurus, deities, and the audience.',
      gestures: [
        '1. Prasaranam (Extending forward)', '2. Kunchitam (Contracting / Folding)', '3. Rechitam (Whirling / Flourish)',
        '4. Pumkhitam (Steadily vibrating)', '5. Apaveshtitam (Inward circling)', '6. Preritam (Directed impulse)',
        '7. Udveshtitam (Outward turning)', '8. Vyavrittam (Side unfolding)', '9. Parivrittam (Full rotation)',
        '10. Sanketam (Symbolic indication)', '11. Chihnam (Identification emblem)', '12. Padardha Tika (Expository verse salutation)'
      ]
    },
    {
      id: 7,
      tag: 'Module 07',
      title: 'Sapta Talamulu & 35 Tala Matrix',
      sanskrit: 'सप्त तालमुलु (३५ ताल व्यवस्था)',
      pdfSource: 'Sapta Talamulu.pdf',
      totalItems: '7 Talas • 35 Systems',
      description: 'The rhythmic science of Carnatic music and Kuchipudi dance. The 7 principal Talas, the 3 Angamulu, the 5 Jaatis, and the complete 35 Tala table.',
      topics: [
        'The 7 Core Talas: Dhruva, Matya, Rupaka, Jhampe, Triputa, Ata, and Eka Tala',
        'Angamulu (Tala limbs): Laghu (|), Dhrutam (O), and Anudhrutam (U)',
        'Pancha Jaatis: Tisra (3), Chaturasra (4), Khanda (5), Misra (7), and Sankeerna (9)',
        'The 35 Tala Mathematical Chart (7 Talas × 5 Jaatis = 35 Variations)',
        'Tala Kartalu and synchronization with Nattuvangam and cymbals'
      ]
    }
  ];

  return (
    <div className="modules-page-wrapper section-padding">
      <div className="container">
        <div className="section-header">
          <span className="section-tag">Academic Dance Curriculum</span>
          <h1 className="section-title">Kuchipudi Learning Modules</h1>
          <TraditionalDivider motif="lotus" />
          <p className="section-subtitle">
            Study all 7 modules covering 52 Hastas, 9 Shiro Bhedas, 8 Drishti Bhedas, 12 Pranamas, and 35 Talas with authentic traditional terminology.
          </p>
        </div>

        <div className="modules-accordion-list">
          {modulesList.map((m) => {
            const isExpanded = expandedModule === m.id;
            return (
              <div key={m.id} className="module-accordion-card temple-card">
                <div 
                  className="accordion-header"
                  onClick={() => setExpandedModule(isExpanded ? null : m.id)}
                >
                  <div className="header-left">
                    <span className="acc-number">0{m.id}</span>
                    <div>
                      <span className="acc-tag">{m.tag} • Source: {m.pdfSource}</span>
                      <h3 className="acc-title">{m.title}</h3>
                      <span className="acc-sanskrit">{m.sanskrit}</span>
                    </div>
                  </div>

                  <div className="header-right">
                    <span className="acc-count-chip">{m.totalItems}</span>
                    <button className="expand-toggle-icon" aria-label="Toggle details">
                      {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                    </button>
                  </div>
                </div>

                {isExpanded && (
                  <div className="accordion-body">
                    <p className="acc-desc">{m.description}</p>

                    {m.gestures && (
                      <div className="gestures-section">
                        <h4 className="gestures-grid-title">
                          <Sparkles size={16} /> Classical Gestures List ({m.gestures.length})
                        </h4>
                        <div className="gestures-grid">
                          {m.gestures.map((g, idx) => (
                            <div key={idx} className="gesture-item-pill">
                              <span className="gesture-bullet">◆</span>
                              <span>{g}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {m.topics && (
                      <div className="topics-section">
                        <h4 className="gestures-grid-title">
                          <BookOpen size={16} /> Key Study Topics
                        </h4>
                        <div className="topics-list">
                          {m.topics.map((t, idx) => (
                            <div key={idx} className="topic-item">
                              <CheckCircle size={16} className="topic-check" />
                              <span>{t}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="accordion-actions">
                      <button onClick={() => setActivePage('practice')} className="btn-primary">
                        <Video size={16} />
                        <span>Practise This Module in AI Studio</span>
                      </button>
                      <button onClick={() => setActivePage('assignments')} className="btn-secondary" style={{ color: '#45151F', borderColor: '#45151F' }}>
                        <Award size={16} />
                        <span>Take Module Quiz</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <style>{`
        .modules-page-wrapper {
          background: #FAF5EC;
          min-height: 85vh;
        }

        .modules-accordion-list {
          display: flex;
          flex-direction: column;
          gap: 20px;
        }

        .module-accordion-card {
          border-radius: var(--radius-md);
          overflow: hidden;
          transition: all 0.3s ease;
        }

        .accordion-header {
          padding: 24px 32px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          cursor: pointer;
          background: #FAF5EC;
          user-select: none;
        }

        @media (max-width: 768px) {
          .accordion-header {
            padding: 18px 20px;
            flex-direction: column;
            align-items: flex-start;
            gap: 16px;
          }
          .header-right {
            width: 100%;
            justify-content: space-between;
          }
        }

        .header-left {
          display: flex;
          align-items: center;
          gap: 20px;
        }

        .acc-number {
          font-family: var(--font-heading);
          font-size: 2rem;
          font-weight: 900;
          color: #C49A45;
          min-width: 48px;
        }

        .acc-tag {
          font-family: var(--font-serif);
          font-size: 0.75rem;
          color: #8A463A;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          display: block;
        }

        .acc-title {
          font-family: var(--font-heading);
          font-size: 1.35rem;
          color: #45151F;
          margin: 2px 0;
        }

        .acc-sanskrit {
          font-family: var(--font-shloka);
          font-size: 1.05rem;
          color: #C49A45;
          display: block;
        }

        .header-right {
          display: flex;
          align-items: center;
          gap: 16px;
        }

        .acc-count-chip {
          background: #F4E9D5;
          border: 1px solid #C49A45;
          color: #45151F;
          font-family: var(--font-serif);
          font-size: 0.78rem;
          font-weight: 700;
          padding: 6px 14px;
          border-radius: 20px;
        }

        .expand-toggle-icon {
          background: rgba(69, 21, 31, 0.08);
          border: 1px solid rgba(196, 154, 69, 0.3);
          border-radius: 50%;
          width: 36px;
          height: 36px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: #45151F;
          cursor: pointer;
        }

        .accordion-body {
          padding: 0 32px 32px 96px;
          border-top: 1px solid rgba(196, 154, 69, 0.2);
          background: #FAF5EC;
        }

        @media (max-width: 768px) {
          .accordion-body {
            padding: 20px;
          }
        }

        .acc-desc {
          font-size: 0.95rem;
          color: #55443B;
          line-height: 1.6;
          margin: 18px 0;
        }

        .gestures-grid-title {
          display: flex;
          align-items: center;
          gap: 8px;
          font-family: var(--font-serif);
          font-size: 0.9rem;
          color: #45151F;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-bottom: 14px;
        }

        .gestures-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 10px;
          margin-bottom: 24px;
        }

        @media (max-width: 900px) {
          .gestures-grid {
            grid-template-columns: repeat(2, 1fr);
          }
        }

        @media (max-width: 550px) {
          .gestures-grid {
            grid-template-columns: 1fr;
          }
        }

        .gesture-item-pill {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 12px;
          background: #F4E9D5;
          border: 1px solid rgba(196, 154, 69, 0.3);
          border-radius: 6px;
          font-size: 0.82rem;
          color: #2E0D14;
        }

        .gesture-bullet {
          color: #C49A45;
          font-size: 0.7rem;
        }

        .topics-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
          margin-bottom: 24px;
        }

        .topic-item {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 8px 12px;
          background: #F4E9D5;
          border-radius: 6px;
          font-size: 0.88rem;
          color: #38241C;
        }

        .topic-check {
          color: #6B342B;
          flex-shrink: 0;
        }

        .accordion-actions {
          display: flex;
          gap: 14px;
          margin-top: 20px;
        }

        @media (max-width: 640px) {
          .accordion-actions {
            flex-direction: column;
          }
        }
      `}</style>
    </div>
  );
}
