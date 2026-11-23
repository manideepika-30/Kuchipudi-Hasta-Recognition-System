import React, { useState, useEffect } from 'react';
import { 
  fetchQuizzes, 
  fetchQuizDetail, 
  submitQuiz, 
  fetchQuizHistory 
} from '../services/api';
import TraditionalDivider from './TraditionalDivider';

export default function AssignmentViewer({ currentLang = 'en', onLaunchCameraExam }) {
  const [quizzes, setQuizzes] = useState([]);
  const [history, setHistory] = useState([]);
  const [activeQuiz, setActiveQuiz] = useState(null); // active quiz object being taken
  const [activeQuestions, setActiveQuestions] = useState([]);
  const [currentQuestionIdx, setCurrentQuestionIdx] = useState(0);
  const [answers, setAnswers] = useState({}); // { [question_id]: "A" }
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [quizResult, setQuizResult] = useState(null); // result after submission
  const [viewMode, setViewMode] = useState('list'); // 'list', 'taking', 'result', 'history'
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState('');

  // Load quizzes and student history on mount
  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    setErrorMsg('');
    try {
      const [qList, hList] = await Promise.all([
        fetchQuizzes().catch(() => []),
        fetchQuizHistory().catch(() => [])
      ]);
      setQuizzes(qList);
      setHistory(hList);
    } catch (err) {
      setErrorMsg(err.message || 'Failed to load assignments.');
    } finally {
      setLoading(false);
    }
  };

  const handleStartQuiz = async (quiz) => {
    if (quiz.quiz_type === 'practical_camera' && onLaunchCameraExam) {
      onLaunchCameraExam();
      return;
    }
    setLoading(true);
    setErrorMsg('');
    try {
      const detail = await fetchQuizDetail(quiz.id);
      setActiveQuiz(detail);
      setActiveQuestions(detail.questions || []);
      setCurrentQuestionIdx(0);
      setAnswers({});
      setQuizResult(null);
      setViewMode('taking');
    } catch (err) {
      setErrorMsg(err.message || 'Failed to load quiz questions.');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectOption = (questionId, optionLetter) => {
    setAnswers(prev => ({
      ...prev,
      [String(questionId)]: optionLetter
    }));
  };

  const handleSubmitQuiz = async () => {
    if (!activeQuiz) return;
    const unansweredCount = activeQuestions.filter(q => !answers[String(q.id)]).length;
    if (unansweredCount > 0) {
      const confirmSubmit = window.confirm(
        currentLang === 'te'
          ? `మీరు ${unansweredCount} ప్రశ్నలకు సమాధానం ఇవ్వలేదు. పరీక్షను సమర్పించాలనుకుంటున్నారా?`
          : `You have ${unansweredCount} unanswered question(s). Are you sure you want to submit?`
      );
      if (!confirmSubmit) return;
    }

    setIsSubmitting(true);
    setErrorMsg('');
    try {
      const result = await submitQuiz(activeQuiz.id, answers);
      setQuizResult(result);
      setViewMode('result');
      // Refresh list to update best score badge
      loadData();
    } catch (err) {
      setErrorMsg(err.message || 'Failed to evaluate quiz.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const t = {
    tag: currentLang === 'te' ? 'విద్యార్థి మూల్యాంకనాలు' : 'Student Assessments',
    title: currentLang === 'te' ? 'అసైన్‌మెంట్లు & పరీక్షలు' : 'Assignments & Examinations',
    subtitle: currentLang === 'te'
      ? 'శాస్త్రీయ సిద్ధాంత బహుళైచ్ఛిక పరీక్షలు, ముద్రల చిత్ర గుర్తింపు మరియు ప్రత్యక్ష కెమెరా మూల్యాంకనాలు.'
      : 'Classical theory MCQ quizzes, gesture image identification, and practical camera benchmark assessments.',
    typeA: currentLang === 'te' ? 'విభాగం A • సిద్ధాంత పరీక్ష' : 'Type A • Theory MCQ Exam',
    typeB: currentLang === 'te' ? 'విభాగం B • చిత్ర గుర్తింపు' : 'Type B • Gesture Image Identification',
    typeC: currentLang === 'te' ? 'విభాగం C • ప్రత్యక్ష కెమెరా పరీక్ష' : 'Type C • Practical Camera Exam',
    passScore: currentLang === 'te' ? 'ఉత్తీర్ణత మార్కులు' : 'Passing Threshold',
    questions: currentLang === 'te' ? 'ప్రశ్నలు' : 'Questions',
    timeLimit: currentLang === 'te' ? 'సమయం' : 'Time Limit',
    min: currentLang === 'te' ? 'నిమిషాలు' : 'mins',
    startTest: currentLang === 'te' ? 'పరీక్ష ప్రారంభించండి' : 'Start Assessment',
    retakeTest: currentLang === 'te' ? 'మళ్లీ వ్రాయండి' : 'Retake Assessment',
    launchExam: currentLang === 'te' ? 'కెమెరా పరీక్ష ప్రారంభించండి' : 'Launch Camera Exam',
    bestScore: currentLang === 'te' ? 'ఉత్తమ స్కోరు' : 'Best Score',
    passed: currentLang === 'te' ? 'ఉత్తీర్ణత (Passed)' : 'Passed',
    failed: currentLang === 'te' ? 'అనుత్తీర్ణత (Try Again)' : 'Try Again',
    attempts: currentLang === 'te' ? 'ప్రయత్నాలు' : 'Attempts',
    historyTab: currentLang === 'te' ? 'పరీక్షల చరిత్ర' : 'Assessment History',
    assignmentsTab: currentLang === 'te' ? 'అన్ని అసైన్‌మెంట్లు' : 'All Assignments',
    questionNum: currentLang === 'te' ? 'ప్రశ్న' : 'Question',
    of: currentLang === 'te' ? 'మొత్తం' : 'of',
    prev: currentLang === 'te' ? 'మునుపటి' : 'Previous',
    next: currentLang === 'te' ? 'తరువాతి' : 'Next',
    submit: currentLang === 'te' ? 'సమర్పించండి' : 'Submit Assessment',
    submitting: currentLang === 'te' ? 'మూల్యాంకనం జరుగుతోంది...' : 'Evaluating answers...',
    backToList: currentLang === 'te' ? 'అసైన్‌మెంట్ల జాబితాకు వెళ్లండి' : 'Back to Assignments',
    reviewHeader: currentLang === 'te' ? 'సమగ్ర సమాధానాల విశ్లేషణ' : 'Question-by-Question Detailed Review',
    yourChoice: currentLang === 'te' ? 'మీ సమాధానం' : 'Your Answer',
    correctChoice: currentLang === 'te' ? 'సరైన సమాధానం' : 'Correct Answer',
    explanation: currentLang === 'te' ? 'వివరణ' : 'Explanation',
    correct: currentLang === 'te' ? 'సరైనది ✓' : 'Correct ✓',
    incorrect: currentLang === 'te' ? 'తప్పు ✗' : 'Incorrect ✗',
    noHistory: currentLang === 'te' ? 'మీరు ఇంకా ఎటువంటి పరీక్షలను పూర్తి చేయలేదు.' : 'You have not taken any assignments yet.',
  };

  const getQuizBadge = (type) => {
    switch (type) {
      case 'theory_mcq': return { label: t.typeA, color: 'card-sage' };
      case 'image_identification': return { label: t.typeB, color: 'card-blush' };
      case 'practical_camera': return { label: t.typeC, color: 'card-lavender' };
      default: return { label: type, color: 'card-soft' };
    }
  };

  if (loading && viewMode === 'list') {
    return (
      <div className="card-soft" style={{ padding: '40px', textAlign: 'center' }}>
        <p style={{ color: 'var(--plum-dark)', fontFamily: 'Cinzel, serif' }}>Loading Kuchipudi Assignments...</p>
      </div>
    );
  }

  return (
    <div className="lms-assignments-view">
      {/* HEADER SECTION */}
      <div className="section-header">
        <span className="section-tag">{t.tag}</span>
        <h2 className="section-title">{t.title}</h2>
        <TraditionalDivider motif="diya" />
        <p className="section-subtitle">{t.subtitle}</p>

        {errorMsg && (
          <div style={{
            background: '#FEE2E2',
            color: '#B91C1C',
            padding: '12px 16px',
            borderRadius: '8px',
            margin: '16px auto',
            maxWidth: '600px',
            fontSize: '0.9rem'
          }}>
            {errorMsg}
          </div>
        )}

        {/* View Switcher Tabs (Only visible when not taking a test) */}
        {viewMode !== 'taking' && (
          <div style={{ display: 'flex', gap: '12px', justifyContent: 'center', marginTop: '16px' }}>
            <button 
              className={`btn-subtle ${viewMode === 'list' || viewMode === 'result' ? 'active' : ''}`}
              onClick={() => setViewMode('list')}
              style={{
                background: viewMode === 'list' || viewMode === 'result' ? 'var(--temple-green)' : 'transparent',
                color: viewMode === 'list' || viewMode === 'result' ? '#fff' : 'var(--plum-dark)',
                border: '1px solid var(--antique-gold)',
                padding: '8px 18px',
                borderRadius: '20px',
                cursor: 'pointer',
                fontWeight: '600'
              }}
            >
              {t.assignmentsTab} ({quizzes.length})
            </button>
            <button 
              className={`btn-subtle ${viewMode === 'history' ? 'active' : ''}`}
              onClick={() => setViewMode('history')}
              style={{
                background: viewMode === 'history' ? 'var(--temple-green)' : 'transparent',
                color: viewMode === 'history' ? '#fff' : 'var(--plum-dark)',
                border: '1px solid var(--antique-gold)',
                padding: '8px 18px',
                borderRadius: '20px',
                cursor: 'pointer',
                fontWeight: '600'
              }}
            >
              {t.historyTab} ({history.length})
            </button>
          </div>
        )}
      </div>

      {/* ===============================================================
          VIEW 1: ASSIGNMENTS CATALOG LIST
          =============================================================== */}
      {viewMode === 'list' && (
        <div className="assignments-list-vertical" style={{ display: 'flex', flexDirection: 'column', gap: '20px', marginTop: '24px' }}>
          {quizzes.map((quiz) => {
            const badge = getQuizBadge(quiz.quiz_type);
            const title = currentLang === 'te' ? quiz.title_te : quiz.title_en;
            const desc = currentLang === 'te' ? quiz.description_te : quiz.description_en;

            return (
              <div 
                key={quiz.id} 
                className={`assignment-item-card ${badge.color}`}
                style={{
                  border: '1px solid var(--antique-gold)',
                  borderRadius: '12px',
                  padding: '24px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  flexWrap: 'wrap',
                  gap: '16px',
                  boxShadow: '0 4px 15px rgba(72,44,58,0.06)'
                }}
              >
                <div className="assign-meta" style={{ flex: '1 1 500px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                    <span style={{
                      background: 'rgba(72,44,58,0.1)',
                      color: 'var(--plum-dark)',
                      padding: '4px 10px',
                      borderRadius: '12px',
                      fontSize: '0.75rem',
                      fontWeight: '700',
                      letterSpacing: '0.5px'
                    }}>
                      {badge.label}
                    </span>
                    <span style={{ fontSize: '0.8rem', color: 'var(--plum-muted)' }}>
                      {quiz.total_questions} {t.questions} • {quiz.time_limit_minutes} {t.min}
                    </span>
                  </div>

                  <h3 style={{ margin: '0 0 6px', color: 'var(--plum-dark)', fontSize: '1.25rem' }}>
                    {title}
                  </h3>
                  <p style={{ margin: 0, color: 'var(--plum-muted)', fontSize: '0.92rem', lineHeight: '1.5' }}>
                    {desc}
                  </p>

                  <div style={{ display: 'flex', gap: '16px', marginTop: '12px', fontSize: '0.85rem' }}>
                    <span>{t.passScore}: <strong>{quiz.passing_score_pct}%</strong></span>
                    {quiz.attempts_count > 0 && (
                      <span>{t.attempts}: <strong>{quiz.attempts_count}</strong></span>
                    )}
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '10px' }}>
                  {quiz.user_best_score !== null && (
                    <div style={{
                      padding: '6px 14px',
                      borderRadius: '20px',
                      fontSize: '0.85rem',
                      fontWeight: '700',
                      background: quiz.user_passed ? '#DEF7EC' : '#FDE8E8',
                      color: quiz.user_passed ? '#03543F' : '#9B1C1C',
                      border: `1px solid ${quiz.user_passed ? '#31C48D' : '#F98080'}`
                    }}>
                      {t.bestScore}: {quiz.user_best_score}% • {quiz.user_passed ? t.passed : t.failed}
                    </div>
                  )}

                  <button 
                    className="btn-primary"
                    onClick={() => handleStartQuiz(quiz)}
                    style={{
                      background: 'var(--plum-dark)',
                      color: '#fff',
                      padding: '10px 22px',
                      borderRadius: '8px',
                      border: 'none',
                      fontWeight: '600',
                      cursor: 'pointer',
                      fontSize: '0.9rem',
                      boxShadow: '0 2px 8px rgba(72,44,58,0.2)'
                    }}
                  >
                    {quiz.quiz_type === 'practical_camera' 
                      ? t.launchExam
                      : (quiz.attempts_count > 0 ? t.retakeTest : t.startTest)
                    }
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ===============================================================
          VIEW 2: ACTIVE QUIZ RUNNER (Taking Test)
          =============================================================== */}
      {viewMode === 'taking' && activeQuiz && activeQuestions.length > 0 && (
        <div className="quiz-runner-container card-soft" style={{
          border: '1px solid var(--antique-gold)',
          borderRadius: '16px',
          padding: '30px',
          marginTop: '20px',
          background: '#fff'
        }}>
          {/* Quiz Runner Header */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            borderBottom: '1px solid rgba(232,209,167,0.4)',
            paddingBottom: '16px',
            marginBottom: '24px'
          }}>
            <div>
              <span style={{ fontSize: '0.8rem', color: 'var(--plum-muted)', textTransform: 'uppercase' }}>
                {activeQuiz.quiz_type === 'image_identification' ? t.typeB : t.typeA}
              </span>
              <h3 style={{ margin: '4px 0 0', color: 'var(--plum-dark)' }}>
                {currentLang === 'te' ? activeQuiz.title_te : activeQuiz.title_en}
              </h3>
            </div>
            <div style={{ textAlign: 'right' }}>
              <span style={{ fontSize: '1rem', fontWeight: '700', color: 'var(--temple-green)' }}>
                {t.questionNum} {currentQuestionIdx + 1} {t.of} {activeQuestions.length}
              </span>
            </div>
          </div>

          {/* Current Question Body */}
          {(() => {
            const curQ = activeQuestions[currentQuestionIdx];
            if (!curQ) return null;
            const qText = currentLang === 'te' ? curQ.question_text_te : curQ.question_text_en;
            const chosen = answers[String(curQ.id)] || '';

            return (
              <div className="current-question-body">
                {/* Image plate for Type B */}
                {curQ.image_url && (
                  <div style={{
                    textAlign: 'center',
                    marginBottom: '24px',
                    padding: '16px',
                    background: 'var(--blush-ivory)',
                    borderRadius: '12px',
                    border: '1px solid var(--antique-gold)'
                  }}>
                    <img 
                      src={curQ.image_url} 
                      alt="Classical Kuchipudi Plate"
                      style={{
                        maxHeight: '260px',
                        maxWidth: '100%',
                        borderRadius: '8px',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                      }} 
                    />
                  </div>
                )}

                <h4 style={{
                  fontSize: '1.2rem',
                  color: 'var(--plum-dark)',
                  lineHeight: '1.6',
                  marginBottom: '20px'
                }}>
                  {curQ.order_num}. {qText}
                </h4>

                {/* Options List */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '30px' }}>
                  {['A', 'B', 'C', 'D'].map(opt => {
                    const optTextEn = curQ[`option_${opt.toLowerCase()}_en`];
                    const optTextTe = curQ[`option_${opt.toLowerCase()}_te`];
                    const optText = currentLang === 'te' ? optTextTe : optTextEn;
                    const isSelected = chosen === opt;

                    return (
                      <div
                        key={opt}
                        onClick={() => handleSelectOption(curQ.id, opt)}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '14px',
                          padding: '14px 20px',
                          borderRadius: '10px',
                          border: `2px solid ${isSelected ? 'var(--temple-green)' : 'rgba(232,209,167,0.6)'}`,
                          background: isSelected ? 'rgba(41,60,53,0.06)' : '#fff',
                          cursor: 'pointer',
                          transition: 'all 0.2s ease'
                        }}
                      >
                        <div style={{
                          width: '28px',
                          height: '28px',
                          borderRadius: '50%',
                          border: `2px solid ${isSelected ? 'var(--temple-green)' : 'var(--plum-muted)'}`,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontWeight: '700',
                          fontSize: '0.85rem',
                          background: isSelected ? 'var(--temple-green)' : 'transparent',
                          color: isSelected ? '#fff' : 'var(--plum-dark)'
                        }}>
                          {opt}
                        </div>
                        <span style={{ fontSize: '1rem', color: 'var(--plum-dark)', fontWeight: isSelected ? '600' : '400' }}>
                          {optText}
                        </span>
                      </div>
                    );
                  })}
                </div>

                {/* Navigation Stepper Controls */}
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  borderTop: '1px solid rgba(232,209,167,0.4)',
                  paddingTop: '20px'
                }}>
                  <button
                    disabled={currentQuestionIdx === 0}
                    onClick={() => setCurrentQuestionIdx(prev => Math.max(0, prev - 1))}
                    style={{
                      padding: '8px 20px',
                      borderRadius: '6px',
                      border: '1px solid var(--antique-gold)',
                      background: 'transparent',
                      cursor: currentQuestionIdx === 0 ? 'not-allowed' : 'pointer',
                      opacity: currentQuestionIdx === 0 ? 0.4 : 1,
                      fontWeight: '600',
                      color: 'var(--plum-dark)'
                    }}
                  >
                    ← {t.prev}
                  </button>

                  <div style={{ display: 'flex', gap: '6px' }}>
                    {activeQuestions.map((q, idx) => (
                      <button
                        key={q.id}
                        onClick={() => setCurrentQuestionIdx(idx)}
                        style={{
                          width: '32px',
                          height: '32px',
                          borderRadius: '6px',
                          border: idx === currentQuestionIdx ? '2px solid var(--temple-green)' : '1px solid #ddd',
                          background: answers[String(q.id)] ? 'var(--sage-green)' : '#f9f9f9',
                          fontWeight: '700',
                          fontSize: '0.8rem',
                          cursor: 'pointer'
                        }}
                      >
                        {idx + 1}
                      </button>
                    ))}
                  </div>

                  {currentQuestionIdx < activeQuestions.length - 1 ? (
                    <button
                      onClick={() => setCurrentQuestionIdx(prev => Math.min(activeQuestions.length - 1, prev + 1))}
                      style={{
                        padding: '8px 20px',
                        borderRadius: '6px',
                        background: 'var(--temple-green)',
                        color: '#fff',
                        border: 'none',
                        cursor: 'pointer',
                        fontWeight: '600'
                      }}
                    >
                      {t.next} →
                    </button>
                  ) : (
                    <button
                      disabled={isSubmitting}
                      onClick={handleSubmitQuiz}
                      style={{
                        padding: '10px 24px',
                        borderRadius: '6px',
                        background: 'var(--plum-dark)',
                        color: '#fff',
                        border: 'none',
                        cursor: isSubmitting ? 'not-allowed' : 'pointer',
                        fontWeight: '700',
                        boxShadow: '0 4px 12px rgba(72,44,58,0.25)'
                      }}
                    >
                      {isSubmitting ? t.submitting : t.submit}
                    </button>
                  )}
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* ===============================================================
          VIEW 3: RESULTS & EXPLANATIONS REVIEW
          =============================================================== */}
      {viewMode === 'result' && quizResult && (
        <div className="quiz-result-container card-soft" style={{
          border: '1px solid var(--antique-gold)',
          borderRadius: '16px',
          padding: '30px',
          marginTop: '20px',
          background: '#fff'
        }}>
          {/* Summary Score Card */}
          <div style={{
            textAlign: 'center',
            padding: '30px 20px',
            background: quizResult.passed ? 'rgba(217,229,213,0.4)' : 'rgba(244,232,226,0.6)',
            borderRadius: '12px',
            marginBottom: '30px',
            border: `2px solid ${quizResult.passed ? 'var(--temple-green)' : 'var(--plum-dark)'}`
          }}>
            <span style={{
              display: 'inline-block',
              padding: '6px 16px',
              borderRadius: '20px',
              fontSize: '0.85rem',
              fontWeight: '700',
              background: quizResult.passed ? 'var(--temple-green)' : '#9B1C1C',
              color: '#fff',
              marginBottom: '12px'
            }}>
              {quizResult.passed ? t.passed : t.failed}
            </span>

            <h2 style={{ fontSize: '2.4rem', margin: '6px 0', color: 'var(--plum-dark)' }}>
              {quizResult.score} / {quizResult.total_questions} ({quizResult.percentage}%)
            </h2>
            <p style={{ color: 'var(--plum-muted)', margin: 0, fontSize: '0.95rem' }}>
              {currentLang === 'te' ? quizResult.quiz_title_te : quizResult.quiz_title_en}
            </p>

            <button
              onClick={() => setViewMode('list')}
              style={{
                marginTop: '18px',
                padding: '10px 24px',
                borderRadius: '8px',
                border: '1px solid var(--plum-dark)',
                background: '#fff',
                color: 'var(--plum-dark)',
                cursor: 'pointer',
                fontWeight: '600'
              }}
            >
              ← {t.backToList}
            </button>
          </div>

          {/* Detailed Question Review */}
          <h3 style={{ color: 'var(--plum-dark)', borderBottom: '1px solid #eee', paddingBottom: '10px', marginBottom: '20px' }}>
            {t.reviewHeader}
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {quizResult.review_details.map((rev, idx) => {
              const qText = currentLang === 'te' ? rev.question_text_te : rev.question_text_en;
              const expl = currentLang === 'te' ? rev.explanation_te : rev.explanation_en;

              return (
                <div 
                  key={rev.question_id}
                  style={{
                    border: `1px solid ${rev.is_correct ? '#31C48D' : '#F98080'}`,
                    background: rev.is_correct ? '#F3FAF7' : '#FDF2F2',
                    borderRadius: '10px',
                    padding: '18px'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <strong>{idx + 1}. {qText}</strong>
                    <span style={{
                      fontWeight: '700',
                      color: rev.is_correct ? '#03543F' : '#9B1C1C'
                    }}>
                      {rev.is_correct ? t.correct : t.incorrect}
                    </span>
                  </div>

                  {rev.image_url && (
                    <div style={{ marginBottom: '12px' }}>
                      <img 
                        src={rev.image_url} 
                        alt="Question Gesture"
                        style={{ maxHeight: '140px', borderRadius: '6px', border: '1px solid #ddd' }} 
                      />
                    </div>
                  )}

                  <div style={{ display: 'flex', gap: '20px', fontSize: '0.9rem', marginBottom: '8px' }}>
                    <span>{t.yourChoice}: <strong>Option {rev.student_choice || 'None'}</strong></span>
                    <span>{t.correctChoice}: <strong>Option {rev.correct_option}</strong></span>
                  </div>

                  {expl && (
                    <div style={{
                      background: 'rgba(255,255,255,0.8)',
                      padding: '10px 14px',
                      borderRadius: '6px',
                      fontSize: '0.88rem',
                      color: 'var(--plum-dark)',
                      lineHeight: '1.5'
                    }}>
                      <strong>{t.explanation}:</strong> {expl}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ===============================================================
          VIEW 4: ASSESSMENT HISTORY LIST
          =============================================================== */}
      {viewMode === 'history' && (
        <div className="quiz-history-container" style={{ marginTop: '20px' }}>
          {history.length === 0 ? (
            <div className="card-soft" style={{ padding: '40px', textAlign: 'center', color: 'var(--plum-muted)' }}>
              {t.noHistory}
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              {history.map((item) => {
                const title = currentLang === 'te' ? item.quiz_title_te : item.quiz_title_en;
                const formattedDate = item.completed_at 
                  ? new Date(item.completed_at).toLocaleDateString(undefined, { 
                      year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' 
                    })
                  : '';

                return (
                  <div 
                    key={item.id}
                    className="card-soft"
                    style={{
                      border: '1px solid var(--antique-gold)',
                      borderRadius: '10px',
                      padding: '18px 24px',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      background: '#fff'
                    }}
                  >
                    <div>
                      <span style={{ fontSize: '0.75rem', color: 'var(--plum-muted)', textTransform: 'uppercase' }}>
                        {item.quiz_type}
                      </span>
                      <h4 style={{ margin: '4px 0', color: 'var(--plum-dark)' }}>{title}</h4>
                      <span style={{ fontSize: '0.8rem', color: 'var(--plum-muted)' }}>{formattedDate}</span>
                    </div>

                    <div style={{ textAlign: 'right' }}>
                      <div style={{
                        display: 'inline-block',
                        padding: '4px 12px',
                        borderRadius: '16px',
                        fontSize: '0.85rem',
                        fontWeight: '700',
                        background: item.passed ? '#DEF7EC' : '#FDE8E8',
                        color: item.passed ? '#03543F' : '#9B1C1C'
                      }}>
                        {item.score} / {item.total_questions} ({item.percentage}%) • {item.passed ? t.passed : t.failed}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
