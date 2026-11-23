import React, { useState, useEffect, useRef } from 'react';
import { 
  fetchAIStatus, 
  fetchSupportedClasses, 
  predictCameraFrame, 
  recordPracticeSession, 
  fetchPracticeHistory 
} from '../services/api';
import TraditionalDivider from './TraditionalDivider';
import { 
  Camera, RefreshCw, CheckCircle2, AlertCircle, Sparkles, 
  Save, Video, VideoOff, Upload, Play, Square, Loader2 
} from 'lucide-react';

export default function PracticeStudio({ currentLang = 'en', initialTargetGesture = null }) {
  const [selectedCategory, setSelectedCategory] = useState('hasthas'); // 'hasthas', 'shiro', 'drishti'
  const [targetGesture, setTargetGesture] = useState(initialTargetGesture || '');
  const [supportedClasses, setSupportedClasses] = useState(null);
  const [aiStatus, setAiStatus] = useState(null);
  
  // Camera lifecycle states: 'standby' | 'requesting' | 'active' | 'error'
  const [cameraState, setCameraState] = useState('standby');
  const [cameraError, setCameraError] = useState('');
  const [isInferencing, setIsInferencing] = useState(false);
  const [lastPrediction, setLastPrediction] = useState(null);
  const [practiceLog, setPracticeLog] = useState([]);
  const [savedSuccess, setSavedSuccess] = useState('');

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const inferIntervalRef = useRef(null);

  // Load AI system status, supported classes, and practice history
  useEffect(() => {
    fetchAIStatus().then(setAiStatus).catch(() => {});
    fetchSupportedClasses().then(setSupportedClasses).catch(() => {});
    fetchPracticeHistory().then(setPracticeLog).catch(() => {});

    return () => {
      stopCamera();
    };
  }, []);

  // Update target when initialTargetGesture changes
  useEffect(() => {
    if (initialTargetGesture) {
      setTargetGesture(initialTargetGesture);
    }
  }, [initialTargetGesture]);

  const startCamera = async () => {
    setCameraError('');
    setCameraState('requesting');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraState('error');
      setCameraError(
        currentLang === 'te'
          ? 'మీ బ్రౌజర్‌లో వెబ్‌క్యామ్ మద్దతు అందుబాటులో లేదు. దయచేసి క్రింది చిత్రం అప్‌లోడ్ ద్వారా పరీక్షించండి.'
          : 'Webcam API is not supported in this browser environment. You can test with image upload below.'
      );
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraState('active');
      // Start real-time prediction loop every 350ms
      startInferenceLoop();
    } catch (err) {
      console.warn('Camera access error:', err);
      setCameraState('error');
      let msg = currentLang === 'te'
        ? 'కెమెరా ప్రాప్యత అనుమతి నిరాకరించబడింది లేదా వెబ్‌క్యామ్ కనుగొనబడలేదు. మీరు క్రింది చిత్రం అప్‌లోడ్ ద్వారా పరీక్షించవచ్చు.'
        : 'Webcam permission denied or no camera device found. You can test any gesture using image upload below.';
      
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        msg = currentLang === 'te'
          ? 'కెమెరా అనుమతి నిరాకరించబడింది. బ్రౌజర్ సెట్టింగ్స్‌లో అనుమతిని ప్రారంభించండి.'
          : 'Camera permission was denied. Please allow camera access in your browser address bar.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        msg = currentLang === 'te'
          ? 'కెమెరా పరికరం కనుగొనబడలేదు. దయచేసి వెబ్‌క్యామ్‌ను తనిఖీ చేయండి.'
          : 'No camera hardware detected on this device. Please connect a webcam or use image upload.';
      }
      setCameraError(msg);
    }
  };

  const stopCamera = () => {
    if (inferIntervalRef.current) {
      clearInterval(inferIntervalRef.current);
      inferIntervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraState('standby');
    clearCanvas();
  };

  const clearCanvas = () => {
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  const captureFrameBase64 = () => {
    if (!videoRef.current || videoRef.current.readyState < 2) return null;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.85);
  };

  const startInferenceLoop = () => {
    if (inferIntervalRef.current) clearInterval(inferIntervalRef.current);

    inferIntervalRef.current = setInterval(async () => {
      if (isInferencing) return;
      const frameBase64 = captureFrameBase64();
      if (!frameBase64) return;

      setIsInferencing(true);
      try {
        const result = await predictCameraFrame(frameBase64, selectedCategory, targetGesture || null);
        setLastPrediction(result);
        drawFeedbackOverlay(result);
      } catch (e) {
        console.warn('Real-time inference skipped frame:', e);
      } finally {
        setIsInferencing(false);
      }
    }, 400);
  };

  const drawFeedbackOverlay = (result) => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    if (result && result.detected) {
      const isMatch = result.target_matched !== false;
      ctx.lineWidth = 3;
      ctx.strokeStyle = isMatch ? '#10B981' : '#F59E0B';
      ctx.strokeRect(30, 30, canvasRef.current.width - 60, canvasRef.current.height - 60);

      ctx.fillStyle = isMatch ? 'rgba(16, 185, 129, 0.9)' : 'rgba(245, 158, 11, 0.9)';
      ctx.fillRect(30, 30, 260, 42);

      ctx.fillStyle = '#FFFFFF';
      ctx.font = 'bold 16px "Plus Jakarta Sans", sans-serif';
      const label = result.predicted_gesture || result.predicted_movement || result.predicted_glance || 'Pose Detected';
      ctx.fillText(`${label} (${result.confidence_percentage}%)`, 42, 58);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setCameraError('');
    const reader = new FileReader();
    reader.onload = async (e) => {
      const base64 = e.target.result;
      try {
        const result = await predictCameraFrame(base64, selectedCategory, targetGesture || null);
        setLastPrediction(result);
        if (canvasRef.current && videoRef.current) {
          const img = new Image();
          img.onload = () => {
            const ctx = canvasRef.current.getContext('2d');
            ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height);
            drawFeedbackOverlay(result);
          };
          img.src = base64;
        }
      } catch (err) {
        setCameraError('AI inference failed on uploaded image: ' + err.message);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleSaveSession = async () => {
    if (!lastPrediction || !lastPrediction.detected) return;

    const gestureName = lastPrediction.predicted_gesture || lastPrediction.predicted_movement || lastPrediction.predicted_glance || 'Practice Mudra';
    const confScore = (lastPrediction.confidence_percentage || 75.0) / 100.0;
    const accuracy = lastPrediction.confidence_percentage || 75.0;
    const feedback = lastPrediction.feedback_message || `Detected ${gestureName} with high confidence`;

    try {
      await recordPracticeSession({
        category: selectedCategory,
        gesture_name: gestureName,
        confidence_score: confScore,
        accuracy_percentage: accuracy,
        feedback_notes: feedback
      });
      setSavedSuccess(
        currentLang === 'te' 
          ? `అభ్యాసం విజయవంతంగా భద్రపరచబడింది!` 
          : `Practice session for ${gestureName} saved successfully!`
      );
      setTimeout(() => setSavedSuccess(''), 4000);
      fetchPracticeHistory().then(setPracticeLog).catch(() => {});
    } catch (e) {
      setCameraError('Failed to save practice session: ' + e.message);
    }
  };

  const t = {
    tag: currentLang === 'te' ? 'ప్రత్యక్ష ఏఐ అభ్యాస కేంద్రం' : 'Live AI Practice Studio',
    title: currentLang === 'te' ? 'కూచిపూడి ఏఐ సాధన కేంద్రం' : 'Kuchipudi AI Practice Studio',
    subtitle: currentLang === 'te'
      ? 'మీ వెబ్‌క్యామ్ ద్వారా ముద్రలు మరియు శిరో-దృష్టి కదలికలను ప్రదర్శించండి. మా 69-క్లాస్ డీప్ న్యూరల్ మోడల్ ద్వారా ప్రత్యక్ష విశ్లేషణ పొందండి.'
      : 'Perform hand mudras, head movements, and eye glances before your webcam with real-time classification by your 69-class deep neural model.',
    hastasTab: currentLang === 'te' ? 'హస్త ముద్రలు (52 తరగతులు)' : 'Hastas (52 Classes)',
    shiroTab: currentLang === 'te' ? 'శిరో భేదాలు (9 తరగతులు)' : 'Shiro Bhedas (9 Classes)',
    drishtiTab: currentLang === 'te' ? 'దృష్టి భేదాలు (8 తరగతులు)' : 'Drishti Bhedas (8 Classes)',
    targetSelect: currentLang === 'te' ? 'లక్ష్య ముద్రను ఎంచుకోండి (ఐచ్ఛికం):' : 'Select Target Gesture (Optional):',
    anyGesture: currentLang === 'te' ? 'స్వేచ్ఛా అభ్యాసం (ఏదైనా ముద్ర)' : 'Free Practice (Any Gesture)',
    startCam: currentLang === 'te' ? 'START CAMERA' : 'START CAMERA',
    stopCam: currentLang === 'te' ? 'STOP CAMERA' : 'STOP CAMERA',
    uploadTest: currentLang === 'te' ? 'చిత్రం అప్‌లోడ్ చేసి పరీక్షించండి' : 'Test With Image Upload',
    statusStandby: currentLang === 'te' ? 'Camera Standby' : 'Camera Standby',
    statusRequesting: currentLang === 'te' ? 'Requesting Permission...' : 'Requesting Permission...',
    statusActive: currentLang === 'te' ? 'Camera Active' : 'Camera Active',
    statusError: currentLang === 'te' ? 'Camera Error' : 'Camera Error',
    predicted: currentLang === 'te' ? 'గుర్తించబడిన ముద్ర' : 'Recognized Gesture',
    confidence: currentLang === 'te' ? 'ఖచ్చితత్వ శాతం' : 'Confidence',
    latency: currentLang === 'te' ? 'విశ్లేషణ సమయం' : 'Inference Latency',
    saveBtn: currentLang === 'te' ? 'అభ్యాసాన్ని రికార్డ్ చేయండి' : 'Save Session to History',
    historyTitle: currentLang === 'te' ? 'ఇటీవలి అభ్యాస చరిత్ర' : 'Recent Practice History',
    noHistory: currentLang === 'te' ? 'ఇంకా ఎటువంటి అభ్యాస రికార్డులు లేవు.' : 'No practice sessions recorded yet.',
  };

  const getTargetOptions = () => {
    if (!supportedClasses) return [];
    if (selectedCategory === 'hasthas') return supportedClasses.hasthas || [];
    if (selectedCategory === 'shiro') return supportedClasses.shiro_bhedhas || [];
    if (selectedCategory === 'drishti') return supportedClasses.drishti_bhedhas || [];
    return [];
  };

  const isCameraActive = cameraState === 'active';

  return (
    <div className="practice-studio-wrapper card-soft">
      {/* Studio Header */}
      <div className="section-header">
        <span className="section-tag">{t.tag}</span>
        <h2 className="section-title">{t.title}</h2>
        <TraditionalDivider motif="diya" />
        <p className="section-subtitle">{t.subtitle}</p>

        {aiStatus && (
          <div className="model-status-chip">
            <Sparkles size={14} className="sparkle-gold" />
            <span>
              Consolidated Keras Model: <strong>{aiStatus.weights_file}</strong> ({aiStatus.supported_classes.total} Classes Active)
            </span>
          </div>
        )}
      </div>

      {savedSuccess && (
        <div className="alert-banner success">
          <CheckCircle2 size={18} />
          <span>{savedSuccess}</span>
        </div>
      )}

      {cameraError && (
        <div className="alert-banner error">
          <AlertCircle size={18} />
          <span>{cameraError}</span>
        </div>
      )}

      {/* Category Selection Tabs (Hastas 52, Shiro 9, Drishti 8) */}
      <div className="category-tabs-container">
        {[
          { id: 'hasthas', label: t.hastasTab },
          { id: 'shiro', label: t.shiroTab },
          { id: 'drishti', label: t.drishtiTab },
        ].map(cat => {
          const isSelected = selectedCategory === cat.id;
          return (
            <button
              key={cat.id}
              onClick={() => {
                setSelectedCategory(cat.id);
                setTargetGesture('');
                setLastPrediction(null);
              }}
              className={`category-tab-btn ${isSelected ? 'selected' : 'unselected'}`}
              aria-selected={isSelected}
            >
              {cat.label}
            </button>
          );
        })}
      </div>

      {/* Target Gesture Dropdown */}
      <div className="target-select-bar">
        <label className="target-label">
          {t.targetSelect}
        </label>
        <select
          value={targetGesture}
          onChange={(e) => setTargetGesture(e.target.value)}
          className="target-dropdown"
        >
          <option value="">{t.anyGesture}</option>
          {getTargetOptions().map(opt => (
            <option key={opt.id} value={opt.name}>
              {opt.name}
            </option>
          ))}
        </select>
      </div>

      {/* Practice Studio Main Interactive Canvas Layout */}
      <div className="studio-canvas-layout">
        {/* Left Column: Camera Feed & Overlay */}
        <div className="camera-feed-card card-soft">
          <div className="camera-header-row">
            <div className="camera-status-pill-wrap">
              <span className={`status-indicator-dot ${cameraState}`} />
              <strong className="status-indicator-text">
                {cameraState === 'active' && t.statusActive}
                {cameraState === 'requesting' && t.statusRequesting}
                {cameraState === 'standby' && t.statusStandby}
                {cameraState === 'error' && t.statusError}
              </strong>
            </div>

            {lastPrediction?.latency_ms && (
              <span className="latency-indicator">
                {t.latency}: <strong>{lastPrediction.latency_ms} ms</strong>
              </span>
            )}
          </div>

          {/* Video Feed Box with Canvas Landmark Overlay */}
          <div className="video-feed-box">
            <video 
              ref={videoRef}
              playsInline
              muted
              className={`live-video-elem ${isCameraActive ? 'visible' : 'hidden'}`}
            />

            <canvas 
              ref={canvasRef}
              width={640}
              height={480}
              className={`live-canvas-elem ${isCameraActive ? 'visible' : 'hidden'}`}
            />

            {!isCameraActive && (
              <div className="camera-standby-placeholder">
                <div className="standby-icon-box">
                  <Camera size={52} />
                </div>
                <h4 className="standby-title">Camera Feed Inactive</h4>
                <p className="standby-desc">
                  Click the button below to activate your webcam and start real-time Kuchipudi pose detection.
                </p>
                
                {cameraState === 'requesting' ? (
                  <button className="btn-start-camera requesting" disabled>
                    <Loader2 size={18} className="spinner-icon" />
                    <span>REQUESTING PERMISSION...</span>
                  </button>
                ) : (
                  <button onClick={startCamera} className="btn-start-camera">
                    <Camera size={18} />
                    <span>START CAMERA</span>
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Camera Controls Bar */}
          <div className="camera-controls-bar">
            {!isCameraActive ? (
              cameraState === 'requesting' ? (
                <button className="btn-control-cam requesting" disabled>
                  <Loader2 size={18} className="spinner-icon" />
                  <span>REQUESTING...</span>
                </button>
              ) : (
                <button onClick={startCamera} className="btn-control-cam start">
                  <Camera size={18} />
                  <span>{t.startCam}</span>
                </button>
              )
            ) : (
              <button onClick={stopCamera} className="btn-control-cam stop">
                <VideoOff size={18} />
                <span>{t.stopCam}</span>
              </button>
            )}

            <label className="btn-upload-label">
              <Upload size={16} />
              <span>{t.uploadTest}</span>
              <input 
                type="file" 
                accept="image/*" 
                onChange={handleFileUpload} 
                style={{ display: 'none' }} 
              />
            </label>
          </div>
        </div>

        {/* Right Column: Live Recognition Results & Feedback */}
        <div className="recognition-results-col">
          <div className="recognition-output-card card-soft">
            <span className="output-eyebrow">Real-Time AI Output</span>
            <h3 className="output-heading">{t.predicted}</h3>

            {lastPrediction && lastPrediction.detected ? (
              <div className="output-details-wrap">
                <div className="predicted-gesture-row">
                  <h2 className="predicted-gesture-name">
                    {lastPrediction.predicted_gesture || lastPrediction.predicted_movement || lastPrediction.predicted_glance}
                  </h2>
                  <span className="predicted-conf-pill">
                    {lastPrediction.confidence_percentage}%
                  </span>
                </div>

                {/* Progress confidence bar */}
                <div className="conf-progress-track">
                  <div 
                    className="conf-progress-bar"
                    style={{
                      width: `${Math.min(100, lastPrediction.confidence_percentage)}%`,
                      background: lastPrediction.confidence_percentage >= 70 ? 'var(--color-temple-green)' : 'var(--color-gold-dark)'
                    }} 
                  />
                </div>

                {/* Target Match & Feedback */}
                {lastPrediction.feedback_message && (
                  <div className={`feedback-message-box ${lastPrediction.target_matched ? 'matched' : 'unmatched'}`}>
                    <strong>Feedback:</strong> {lastPrediction.feedback_message}
                  </div>
                )}

                {/* Top 3 Predictions */}
                {lastPrediction.top_predictions && lastPrediction.top_predictions.length > 0 && (
                  <div className="top-predictions-box">
                    <span className="top-pred-heading">Top Neural Probabilities:</span>
                    <div className="top-pred-list">
                      {lastPrediction.top_predictions.map(p => (
                        <div key={p.gesture_id} className="top-pred-row">
                          <span className="top-pred-name">{p.gesture_name}</span>
                          <strong className="top-pred-val">{p.confidence}%</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <button onClick={handleSaveSession} className="btn-save-practice-session">
                  <Save size={16} />
                  <span>{t.saveBtn}</span>
                </button>
              </div>
            ) : (
              <div className="empty-prediction-state">
                <p className="empty-pred-text">
                  {isCameraActive 
                    ? 'Position your hand clearly within the camera frame to view real-time recognition.'
                    : 'Start the camera or upload an image to see neural network inference.'}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        .practice-studio-wrapper {
          padding: 32px;
          display: flex;
          flex-direction: column;
          gap: 20px;
        }

        .model-status-chip {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          background: rgba(41, 60, 53, 0.08);
          border: 1px solid var(--color-antique-gold);
          padding: 6px 14px;
          border-radius: 20px;
          font-size: 0.8rem;
          color: var(--color-temple-green);
          font-weight: 600;
          margin-top: 10px;
        }

        .alert-banner {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 12px 18px;
          border-radius: var(--radius-sm);
          font-size: 0.9rem;
          font-weight: 600;
          max-width: 750px;
          margin: 0 auto;
          width: 100%;
        }
        .alert-banner.success {
          background: #DEF7EC;
          color: #03543F;
          border: 1px solid #31C48D;
        }
        .alert-banner.error {
          background: #FDE8E8;
          color: #9B1C1C;
          border: 1px solid #F87171;
        }

        /* Category Selection Tabs - High Contrast & Clearly Visible */
        .category-tabs-container {
          display: flex;
          gap: 14px;
          justify-content: center;
          margin: 16px 0;
          flex-wrap: wrap;
        }
        .category-tab-btn {
          padding: 11px 22px;
          border-radius: 24px;
          font-family: var(--font-body);
          font-size: 0.9rem;
          cursor: pointer;
          transition: all 0.2s ease;
          display: inline-flex;
          align-items: center;
          justify-content: center;
        }
        .category-tab-btn.selected {
          background: #482C3A !important;
          color: #FFFFFF !important;
          border: 1.5px solid #482C3A !important;
          font-weight: 700;
          box-shadow: 0 4px 14px rgba(72, 44, 58, 0.25);
        }
        .category-tab-btn.unselected {
          background: #F4E8E2;
          color: #482C3A;
          border: 1.5px solid #E8D1A7;
          font-weight: 600;
        }
        .category-tab-btn.unselected:hover {
          background: #EDF3EB;
          color: #293C35;
          border-color: #BFA06C;
        }

        /* Target select bar */
        .target-select-bar {
          text-align: center;
          margin-bottom: 20px;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 12px;
          flex-wrap: wrap;
        }
        .target-label {
          font-size: 0.9rem;
          font-weight: 600;
          color: var(--color-deep-plum);
        }
        .target-dropdown {
          padding: 9px 16px;
          border-radius: var(--radius-sm);
          border: 1.5px solid var(--color-antique-gold);
          background: #FFFFFF;
          color: var(--color-deep-plum);
          font-size: 0.9rem;
          font-weight: 600;
          cursor: pointer;
          max-width: 320px;
        }

        /* Studio Canvas Layout */
        .studio-canvas-layout {
          display: grid;
          grid-template-columns: 1.2fr 0.8fr;
          gap: 24px;
          align-items: start;
        }

        /* Camera Feed Card */
        .camera-feed-card {
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg);
          padding: 20px;
          background: #FFFFFF;
          box-shadow: var(--shadow-sm);
        }
        .camera-header-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 14px;
        }
        .camera-status-pill-wrap {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .status-indicator-dot {
          width: 10px;
          height: 10px;
          border-radius: 50%;
        }
        .status-indicator-dot.active {
          background: #10B981;
          box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.25);
        }
        .status-indicator-dot.standby {
          background: #6B7280;
        }
        .status-indicator-dot.requesting {
          background: #D97706;
          box-shadow: 0 0 0 3px rgba(217, 119, 6, 0.25);
        }
        .status-indicator-dot.error {
          background: #EF4444;
        }
        .status-indicator-text {
          font-size: 0.85rem;
          color: var(--color-deep-plum);
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }
        .latency-indicator {
          font-size: 0.8rem;
          color: var(--color-text-secondary);
        }

        /* Video feed box */
        .video-feed-box {
          position: relative;
          width: 100%;
          height: 400px;
          background: #111827;
          border-radius: var(--radius-md);
          overflow: hidden;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .live-video-elem {
          width: 100%;
          height: 100%;
          object-fit: cover;
          transform: scaleX(-1);
        }
        .live-video-elem.hidden, .live-canvas-elem.hidden {
          display: none;
        }
        .live-canvas-elem {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          transform: scaleX(-1);
          pointer-events: none;
        }

        /* Standby placeholder overlay */
        .camera-standby-placeholder {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          padding: 24px;
          color: #E5E7EB;
          gap: 12px;
          max-width: 420px;
        }
        .standby-icon-box {
          color: var(--color-antique-gold);
          opacity: 0.8;
        }
        .standby-title {
          font-family: var(--font-heading);
          font-size: 1.2rem;
          color: #FFFFFF;
          margin: 0;
        }
        .standby-desc {
          font-size: 0.85rem;
          color: #9CA3AF;
          line-height: 1.5;
          margin: 0;
        }

        /* Prominent Start Camera Buttons */
        .btn-start-camera {
          display: inline-flex;
          align-items: center;
          gap: 10px;
          background: #293C35;
          color: #FFFFFF;
          border: 1.5px solid #3F5950;
          padding: 12px 28px;
          border-radius: 8px;
          font-family: var(--font-body);
          font-size: 0.95rem;
          font-weight: 700;
          cursor: pointer;
          letter-spacing: 0.05em;
          box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
          transition: all 0.2s ease;
          margin-top: 6px;
        }
        .btn-start-camera:hover {
          background: #1C2B25;
          border-color: var(--color-antique-gold);
          transform: translateY(-2px);
          box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45);
        }
        .btn-start-camera.requesting {
          background: #BFA06C;
          border-color: #E8D1A7;
          cursor: not-allowed;
        }

        /* Camera Controls Bar */
        .camera-controls-bar {
          display: flex;
          gap: 12px;
          margin-top: 18px;
          flex-wrap: wrap;
        }
        .btn-control-cam {
          flex: 1 1 200px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
          padding: 12px 20px;
          border-radius: var(--radius-sm);
          font-family: var(--font-body);
          font-size: 0.92rem;
          font-weight: 700;
          letter-spacing: 0.04em;
          cursor: pointer;
          border: none;
          transition: all 0.2s ease;
        }
        .btn-control-cam.start {
          background: #293C35;
          color: #FFFFFF;
          box-shadow: 0 4px 12px rgba(41, 60, 53, 0.25);
        }
        .btn-control-cam.start:hover {
          background: #1C2B25;
          transform: translateY(-2px);
        }
        .btn-control-cam.stop {
          background: #9B1C1C;
          color: #FFFFFF;
          box-shadow: 0 4px 12px rgba(155, 28, 28, 0.25);
        }
        .btn-control-cam.stop:hover {
          background: #771D1D;
        }
        .btn-control-cam.requesting {
          background: #BFA06C;
          color: #FFFFFF;
          cursor: not-allowed;
        }
        .btn-upload-label {
          flex: 1 1 200px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          padding: 12px 20px;
          border-radius: var(--radius-sm);
          border: 1.5px solid var(--color-antique-gold);
          background: #FFFFFF;
          color: var(--color-deep-plum);
          font-family: var(--font-body);
          font-size: 0.88rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
        }
        .btn-upload-label:hover {
          background: var(--color-blush-light);
          border-color: var(--color-gold-dark);
        }

        /* Right Column Output Card */
        .recognition-output-card {
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg);
          padding: 24px;
          background: #FFFFFF;
          box-shadow: var(--shadow-sm);
        }
        .output-eyebrow {
          font-size: 0.75rem;
          font-weight: 800;
          color: var(--color-gold-dark);
          text-transform: uppercase;
          letter-spacing: 0.08em;
          display: block;
        }
        .output-heading {
          margin: 4px 0 16px;
          color: var(--color-deep-plum);
          font-family: var(--font-heading);
          font-size: 1.25rem;
        }
        .predicted-gesture-row {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          margin-bottom: 12px;
        }
        .predicted-gesture-name {
          font-size: 1.8rem;
          font-family: var(--font-heading);
          color: var(--color-temple-green);
          margin: 0;
        }
        .predicted-conf-pill {
          font-size: 1.1rem;
          font-weight: 700;
          color: var(--color-deep-plum);
        }
        .conf-progress-track {
          width: 100%;
          height: 10px;
          background: #E5E7EB;
          border-radius: 5px;
          overflow: hidden;
          margin-bottom: 16px;
        }
        .conf-progress-bar {
          height: 100%;
          border-radius: 5px;
          transition: width 0.3s ease;
        }
        .feedback-message-box {
          padding: 12px 16px;
          border-radius: var(--radius-sm);
          font-size: 0.88rem;
          line-height: 1.5;
          margin-bottom: 16px;
        }
        .feedback-message-box.matched {
          background: #DEF7EC;
          color: #03543F;
          border: 1px solid #31C48D;
        }
        .feedback-message-box.unmatched {
          background: var(--color-blush-light);
          color: var(--color-deep-plum);
          border: 1px solid var(--color-antique-gold);
        }
        .top-predictions-box {
          margin-top: 16px;
          background: var(--color-blush-light);
          padding: 12px 14px;
          border-radius: var(--radius-sm);
        }
        .top-pred-heading {
          font-size: 0.78rem;
          color: var(--color-text-secondary);
          font-weight: 700;
          text-transform: uppercase;
        }
        .top-pred-list {
          display: flex;
          flex-direction: column;
          gap: 6px;
          margin-top: 8px;
        }
        .top-pred-row {
          display: flex;
          justify-content: space-between;
          font-size: 0.85rem;
        }
        .top-pred-name {
          color: var(--color-deep-plum);
        }
        .top-pred-val {
          color: var(--color-temple-green);
        }
        .btn-save-practice-session {
          width: 100%;
          margin-top: 20px;
          padding: 12px;
          border-radius: var(--radius-sm);
          background: #293C35;
          color: #FFFFFF;
          border: none;
          font-weight: 700;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          font-size: 0.9rem;
          transition: all 0.2s ease;
        }
        .btn-save-practice-session:hover {
          background: #1C2B25;
        }
        .empty-prediction-state {
          padding: 32px 16px;
          text-align: center;
          color: var(--color-text-secondary);
          background: var(--color-blush-light);
          border-radius: var(--radius-sm);
          border: 1px dashed var(--color-border);
        }
        .empty-pred-text {
          margin: 0;
          font-size: 0.9rem;
          line-height: 1.5;
        }
        .spinner-icon {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        @media (max-width: 992px) {
          .studio-canvas-layout {
            grid-template-columns: 1fr;
          }
          .video-feed-box {
            height: 320px;
          }
        }
      `}</style>
    </div>
  );
}
