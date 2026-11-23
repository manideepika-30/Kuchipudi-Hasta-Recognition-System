import React from 'react';
import { Sparkles } from 'lucide-react';

export default function HeroDancer() {
  return (
    <div className="hero-dancer-container">
      {/* Decorative Rotating Mandala Background */}
      <div className="mandala-aura animate-spin-slow">
        <svg viewBox="0 0 200 200" className="mandala-svg">
          <circle cx="100" cy="100" r="90" stroke="rgba(196, 154, 69, 0.25)" strokeWidth="1" fill="none" strokeDasharray="4 4" />
          <circle cx="100" cy="100" r="75" stroke="rgba(245, 222, 152, 0.3)" strokeWidth="1.5" fill="none" />
          <circle cx="100" cy="100" r="60" stroke="rgba(196, 154, 69, 0.4)" strokeWidth="1" fill="none" strokeDasharray="8 6" />
          {[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330].map((deg) => (
            <g key={deg} transform={`rotate(${deg} 100 100)`}>
              <path d="M100 18 C95 35 105 35 100 50 C95 35 105 35 100 18 Z" fill="rgba(196, 154, 69, 0.3)" />
              <circle cx="100" cy="14" r="2.5" fill="rgba(245, 222, 152, 0.6)" />
            </g>
          ))}
        </svg>
      </div>

      {/* Temple Arch Frame */}
      <div className="temple-arch-frame">
        {/* Glow halo */}
        <div className="glow-halo animate-pulse-gold" />

        {/* Dancer Image Card */}
        <div className="dancer-visual-card">
          <div className="dancer-image-wrapper">
            <img 
              src="/assets/asamyutha_sample.jpeg" 
              alt="Kuchipudi Classical Dancer demonstrating sacred Hasta Mudra" 
              className="dancer-main-image"
              onError={(e) => {
                // Fallback to hastha_anjali if sample fails
                e.target.src = '/assets/hastha_anjali.jpeg';
              }}
            />
            {/* Elegant vignette and maroon gradient overlay */}
            <div className="image-vignette" />
          </div>

          {/* Floating Live Badge */}
          <div className="dancer-badge-top">
            <Sparkles size={14} className="sparkle-icon" />
            <span>69 Classical Dance Classes</span>
          </div>

          {/* Floating Mudra Info Card */}
          <div className="dancer-floating-card">
            <div className="mudra-symbol-chip">हस्त</div>
            <div>
              <div className="mudra-name">Pathaka Hasta (पताक)</div>
              <div className="mudra-meaning">Flag Gesture • Forest, Night & Blessings</div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .hero-dancer-container {
          position: relative;
          width: 100%;
          max-width: 480px;
          margin: 0 auto;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .mandala-aura {
          position: absolute;
          width: 520px;
          height: 520px;
          pointer-events: none;
          z-index: 1;
        }

        .mandala-svg {
          width: 100%;
          height: 100%;
        }

        .temple-arch-frame {
          position: relative;
          z-index: 2;
          width: 100%;
          padding: 12px;
          background: linear-gradient(135deg, rgba(196, 154, 69, 0.5) 0%, rgba(69, 21, 31, 0.8) 50%, rgba(196, 154, 69, 0.5) 100%);
          border-radius: 200px 200px 24px 24px;
          box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6);
        }

        .glow-halo {
          position: absolute;
          inset: 0;
          border-radius: 200px 200px 24px 24px;
          pointer-events: none;
        }

        .dancer-visual-card {
          position: relative;
          border-radius: 190px 190px 18px 18px;
          overflow: hidden;
          background: #2E0D14;
          border: 2px solid #F5DE98;
        }

        .dancer-image-wrapper {
          position: relative;
          width: 100%;
          height: 480px;
          overflow: hidden;
        }

        @media (max-width: 640px) {
          .dancer-image-wrapper {
            height: 380px;
          }
          .mandala-aura {
            width: 380px;
            height: 380px;
          }
        }

        .dancer-main-image {
          width: 100%;
          height: 100%;
          object-fit: cover;
          object-position: center 20%;
          transition: transform 0.8s ease;
        }

        .hero-dancer-container:hover .dancer-main-image {
          transform: scale(1.04);
        }

        .image-vignette {
          position: absolute;
          inset: 0;
          background: linear-gradient(
            180deg, 
            rgba(43, 12, 19, 0.1) 0%, 
            rgba(43, 12, 19, 0.3) 60%, 
            rgba(30, 7, 13, 0.95) 100%
          );
        }

        .dancer-badge-top {
          position: absolute;
          top: 20px;
          right: 20px;
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 6px 14px;
          background: rgba(43, 12, 19, 0.85);
          backdrop-filter: blur(8px);
          border: 1px solid #C49A45;
          border-radius: 20px;
          color: #F5DE98;
          font-family: var(--font-serif);
          font-size: 0.75rem;
          font-weight: 700;
          letter-spacing: 0.05em;
          box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        }

        .sparkle-icon {
          color: #F5DE98;
        }

        .dancer-floating-card {
          position: absolute;
          bottom: 20px;
          left: 20px;
          right: 20px;
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 12px 18px;
          background: rgba(43, 12, 19, 0.92);
          backdrop-filter: blur(10px);
          border: 1px solid #C49A45;
          border-radius: 12px;
          color: #F4E9D5;
          box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
        }

        .mudra-symbol-chip {
          width: 40px;
          height: 40px;
          border-radius: 8px;
          background: var(--gold-gradient);
          color: #2E0D14;
          font-family: var(--font-shloka);
          font-size: 1.25rem;
          font-weight: 700;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }

        .mudra-name {
          font-family: var(--font-serif);
          font-size: 0.95rem;
          font-weight: 700;
          color: #F5DE98;
        }

        .mudra-meaning {
          font-size: 0.75rem;
          color: #E6D7BE;
          opacity: 0.9;
        }
      `}</style>
    </div>
  );
}
