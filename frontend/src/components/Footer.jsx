import React from 'react';
import { translations } from '../utils/translations';

export default function Footer({ currentLang }) {
  const t = translations[currentLang] || translations.en;

  return (
    <footer className="refined-footer">
      <div className="container footer-inner">
        <div className="footer-shloka-box">
          <p className="footer-sanskrit">{t.footerShloka}</p>
          <p className="footer-meaning">{t.footerShlokaMeaning}</p>
          <span className="footer-citation">— Abhinaya Darpana (अभिनय दर्पणम्)</span>
        </div>

        <div className="footer-bottom-line">
          <p className="footer-copy">{t.footerCopyright}</p>
          <div className="footer-tech-tag">
            <span>SEDS Project</span>
            <span>•</span>
            <span>69 Kuchipudi Gesture Classes</span>
          </div>
        </div>
      </div>

      <style>{`
        .refined-footer {
          background: #FFFFFF;
          border-top: 1.5px solid var(--color-border);
          padding: 40px 0 25px;
          color: var(--color-text-secondary);
        }

        .footer-shloka-box {
          text-align: center;
          padding: 20px 24px;
          background: var(--color-blush-light);
          border: 1px solid rgba(191, 160, 108, 0.35);
          border-radius: var(--radius-md);
          max-width: 800px;
          margin: 0 auto 30px;
        }

        .footer-sanskrit {
          font-family: var(--font-shloka);
          font-size: 1.2rem;
          color: var(--color-deep-plum);
          line-height: 1.6;
        }

        .footer-meaning {
          font-size: 0.85rem;
          color: var(--color-text-secondary);
          font-style: italic;
          margin-top: 6px;
        }

        .footer-citation {
          display: inline-block;
          font-family: var(--font-serif);
          font-size: 0.72rem;
          color: var(--color-temple-green);
          letter-spacing: 0.12em;
          text-transform: uppercase;
          margin-top: 8px;
        }

        .footer-bottom-line {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding-top: 20px;
          border-top: 1px solid var(--color-border);
          font-size: 0.8rem;
          color: var(--color-text-secondary);
        }

        @media (max-width: 640px) {
          .footer-bottom-line {
            flex-direction: column;
            gap: 8px;
            text-align: center;
          }
        }

        .footer-tech-tag {
          display: flex;
          gap: 8px;
          color: var(--color-gold-dark);
          font-weight: 600;
        }
      `}</style>
    </footer>
  );
}
