import React, { useState } from 'react';
import { X, LogIn, UserPlus, Lock, Mail, User, Shield, AlertCircle } from 'lucide-react';
import { translations } from '../utils/translations';
import { loginUser, registerUser } from '../services/api';

export default function AuthModal({ 
  isOpen, 
  onClose, 
  initialMode = 'login', 
  onLoginSuccess, 
  currentLang 
}) {
  if (!isOpen) return null;

  const t = translations[currentLang] || translations.en;
  const [mode, setMode] = useState(initialMode);
  const [errorMsg, setErrorMsg] = useState('');
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: ''
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMsg('');
    setLoading(true);

    try {
      if (mode === 'register') {
        const result = await registerUser({
          email: formData.email,
          password: formData.password,
          full_name: formData.name,
          preferred_language: currentLang || 'en'
        });
        onLoginSuccess(result.user);
        onClose();
      } else {
        const result = await loginUser(formData.email, formData.password);
        onLoginSuccess(result.user);
        onClose();
      }
    } catch (err) {
      setErrorMsg(err.message || 'Authentication failed. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  const handleSwitchMode = (newMode) => {
    setMode(newMode);
    setErrorMsg('');
  };

  return (
    <div className="auth-modal-overlay" onClick={onClose}>
      <div className="auth-modal-card card-soft" onClick={(e) => e.stopPropagation()}>
        {/* Close Button */}
        <button className="auth-close-btn" onClick={onClose} aria-label="Close dialog">
          <X size={18} />
        </button>

        {/* Modal Header */}
        <div className="auth-modal-header">
          <div className="auth-diya-icon">🪔</div>
          <h2 className="auth-title">
            {mode === 'login' ? t.authLoginTitle : t.authRegisterTitle}
          </h2>
          <p className="auth-subtitle">
            {mode === 'login' 
              ? (currentLang === 'te' ? 'కూచిపూడి అభ్యాస వేదికలోకి ప్రవేశించండి' : 'Enter your credentials to continue learning')
              : (currentLang === 'te' ? 'కొత్త విద్యార్థిగా నమోదు చేసుకోండి' : 'Register as a new student to start your journey')}
          </p>
        </div>

        {/* Error notification banner */}
        {errorMsg && (
          <div className="auth-error-banner">
            <AlertCircle size={16} />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="auth-form">
          {mode === 'register' && (
            <div className="form-group">
              <label className="form-label">{t.authFullName}</label>
              <div className="input-wrap">
                <User size={16} className="input-icon" />
                <input 
                  type="text" 
                  required 
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="form-input" 
                  placeholder={currentLang === 'te' ? 'మీ పూర్తి పేరు' : 'e.g. Ananya Sharma'}
                />
              </div>
            </div>
          )}

          <div className="form-group">
            <label className="form-label">{t.authEmail}</label>
            <div className="input-wrap">
              <Mail size={16} className="input-icon" />
              <input 
                type="email" 
                required 
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                className="form-input" 
                placeholder="student@kuchipudi.edu"
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">{t.authPassword}</label>
            <div className="input-wrap">
              <Lock size={16} className="input-icon" />
              <input 
                type="password" 
                required 
                minLength={6}
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                className="form-input" 
                placeholder="••••••••"
              />
            </div>
            {mode === 'register' && (
              <span className="password-hint">
                {currentLang === 'te' ? 'కనీసం 6 అక్షరాలు ఉండాలి' : 'Must be at least 6 characters'}
              </span>
            )}
          </div>

          <button 
            type="submit" 
            className="btn-primary" 
            style={{ width: '100%', marginTop: '8px' }}
            disabled={loading}
          >
            {loading ? (
              <span>{currentLang === 'te' ? 'వేచి ఉండండి...' : 'Please wait...'}</span>
            ) : mode === 'login' ? (
              <>
                <LogIn size={16} />
                <span>{t.authSubmitLogin}</span>
              </>
            ) : (
              <>
                <UserPlus size={16} />
                <span>{t.authSubmitRegister}</span>
              </>
            )}
          </button>
        </form>

        {/* Toggle Mode */}
        <div className="auth-switch-box">
          {mode === 'login' ? (
            <button onClick={() => handleSwitchMode('register')} className="auth-switch-link">
              {t.authSwitchToRegister}
            </button>
          ) : (
            <button onClick={() => handleSwitchMode('login')} className="auth-switch-link">
              {t.authSwitchToLogin}
            </button>
          )}
        </div>
      </div>

      <style>{`
        .auth-modal-overlay {
          position: fixed;
          inset: 0;
          background: rgba(41, 60, 53, 0.65);
          backdrop-filter: blur(5px);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 2000;
          padding: 20px;
        }

        .auth-modal-card {
          background: #FFFFFF;
          border: 1.5px solid var(--color-border);
          border-radius: var(--radius-lg);
          max-width: 440px;
          width: 100%;
          padding: 32px;
          position: relative;
          box-shadow: var(--shadow-lg);
        }

        .auth-close-btn {
          position: absolute;
          top: 18px;
          right: 18px;
          background: var(--color-blush-ivory);
          border: none;
          width: 30px;
          height: 30px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          color: var(--color-deep-plum);
          transition: background 0.2s;
        }

        .auth-close-btn:hover {
          background: var(--color-antique-gold);
        }

        .auth-modal-header {
          text-align: center;
          margin-bottom: 16px;
        }

        .auth-diya-icon {
          font-size: 1.8rem;
          margin-bottom: 4px;
        }

        .auth-title {
          font-size: 1.35rem;
          color: var(--color-deep-plum);
        }

        .auth-subtitle {
          font-size: 0.8rem;
          color: var(--color-temple-green);
          margin-top: 4px;
          background: var(--color-sage-light);
          padding: 4px 8px;
          border-radius: 4px;
          display: inline-block;
        }

        .auth-error-banner {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 10px 12px;
          background: #FEE2E2;
          border: 1px solid #FCA5A5;
          border-radius: var(--radius-sm);
          color: #991B1B;
          font-size: 0.82rem;
          margin-bottom: 14px;
        }

        .auth-form {
          display: flex;
          flex-direction: column;
          gap: 14px;
        }

        .form-group {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .form-label {
          font-size: 0.8rem;
          font-weight: 600;
          color: var(--color-deep-plum);
        }

        .password-hint {
          font-size: 0.72rem;
          color: var(--color-text-secondary);
        }

        .input-wrap {
          position: relative;
          display: flex;
          align-items: center;
        }

        .input-icon {
          position: absolute;
          left: 12px;
          color: var(--color-gold-dark);
        }

        .form-input {
          width: 100%;
          padding: 10px 12px 10px 36px;
          border: 1px solid var(--color-border);
          border-radius: var(--radius-sm);
          font-family: var(--font-body);
          font-size: 0.9rem;
          color: var(--color-text-primary);
          background: #FFFFFF;
          outline: none;
          transition: border-color 0.2s;
        }

        .form-input:focus {
          border-color: var(--color-deep-plum);
          box-shadow: 0 0 0 2px rgba(72, 44, 58, 0.1);
        }

        .auth-switch-box {
          text-align: center;
          margin-top: 18px;
          padding-top: 14px;
          border-top: 1px solid var(--color-border);
        }

        .auth-switch-link {
          background: transparent;
          border: none;
          color: var(--color-temple-green);
          font-size: 0.82rem;
          font-weight: 600;
          cursor: pointer;
          text-decoration: underline;
        }
      `}</style>
    </div>
  );
}
