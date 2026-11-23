import React from 'react';

export default function TraditionalDivider({ motif = "lotus", className = "" }) {
  return (
    <div className={`traditional-divider-wrapper ${className}`}>
      <div className="divider-line left" />
      <div className="divider-motif">
        {motif === "lotus" && (
          <svg viewBox="0 0 40 24" className="motif-svg">
            {/* Traditional Indian Lotus motif */}
            <path d="M20 4 C18 10 14 14 8 18 C14 18 18 16 20 20 C22 16 26 18 32 18 C26 14 22 10 20 4 Z" fill="#C49A45" />
            <circle cx="20" cy="12" r="2.5" fill="#F5DE98" />
            <circle cx="10" cy="17" r="1.5" fill="#C49A45" />
            <circle cx="30" cy="17" r="1.5" fill="#C49A45" />
          </svg>
        )}
        {motif === "diya" && (
          <svg viewBox="0 0 36 24" className="motif-svg">
            {/* Traditional Diya flame */}
            <path d="M18 2 C19 6 22 10 22 13 C22 15.5 20.2 17 18 17 C15.8 17 14 15.5 14 13 C14 10 17 6 18 2 Z" fill="#F5DE98" />
            <path d="M18 6 C18.5 8 20 10.5 20 12.5 C20 14 19 15 18 15 C17 15 16 14 16 12.5 C16 10.5 17.5 8 18 6 Z" fill="#C49A45" />
            <path d="M6 16 C10 22 26 22 30 16 C31 18 28 23 18 23 C8 23 5 18 6 16 Z" fill="#C49A45" />
          </svg>
        )}
      </div>
      <div className="divider-line right" />

      <style>{`
        .traditional-divider-wrapper {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 16px;
          margin: 20px auto 35px;
          max-width: 320px;
        }

        .divider-line {
          flex: 1;
          height: 1.5px;
          background: linear-gradient(90deg, transparent, #C49A45);
        }

        .divider-line.right {
          background: linear-gradient(90deg, #C49A45, transparent);
        }

        .divider-motif {
          width: 36px;
          height: 24px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .motif-svg {
          width: 100%;
          height: 100%;
          filter: drop-shadow(0 2px 4px rgba(196, 154, 69, 0.4));
        }
      `}</style>
    </div>
  );
}
