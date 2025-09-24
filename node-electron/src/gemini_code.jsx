import React, { useState, useEffect } from 'react';

// Main App Component
function Idk() {
  const [isTracking, setIsTracking] = useState(false);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(false);
  const [systemStatus, setSystemStatus] = useState('green');
  const [authorizationColor, setAuthorizationColor] = useState('bg-orange-500');
  const [workerCount, setWorkerCount] = useState(2);
  const [troubleText, setTroubleText] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [showSettingsPanel, setShowSettingsPanel] = useState(false);
  const [showPasswordPrompt, setShowPasswordPrompt] = useState(false);
  const [passwordInput, setPasswordInput] = useState('');
  const [passwordError, setPasswordError] = useState('');

  const correctPassword = 'admin';

  const toggleTracking = () => {
    const newTrackingState = !isTracking;
    setIsTracking(newTrackingState);
    setShowBoundingBoxes(newTrackingState);
  };

  const handleEmergencyStop = () => {
    setTroubleText("Emergency Stop Activated. All robotic operations have been halted. A full system check is recommended.");
    setShowModal(true);
    setIsTracking(false);
    setShowBoundingBoxes(false);
  };

  const troubleshootIssue = async () => {
    setIsLoading(true);
    setTroubleText(null);
    setShowModal(true);

    let queryPrompt = `The SCORBOT Gesture Control Interface is reporting a '${systemStatus}' status. There are ${workerCount} workers in the workspace. Provide a concise, single-paragraph explanation and a potential solution.`;

    if (systemStatus === 'red') {
      queryPrompt = `The SCORBOT Gesture Control Interface has a critical '${systemStatus}' status. There are ${workerCount} workers in the workspace, and the system detected a person outside of a safety zone with a confidence score of 'Person 0.81'. Provide a concise, single-paragraph explanation and a potential solution.`;
    } else if (systemStatus === 'yellow') {
      queryPrompt = `The SCORBOT Gesture Control Interface has a warning '${systemStatus}' status. There are ${workerCount} workers in the workspace, and the system detected a person approaching a safety zone with a confidence score of 'Person 0.78'. Provide a concise, single-paragraph explanation and a potential solution.`;
    }

    try {
      const payload = {
        contents: [{ parts: [{ text: queryPrompt }] }],
        tools: [{ "google_search": {} }],
        systemInstruction: {
          parts: [{ text: "Act as a world-class robotic systems analyst. Provide a concise, single-paragraph summary of the key findings and a potential solution." }]
        },
      };

      const apiKey = "";
      const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${apiKey}`;
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const result = await response.json();
      const text = result?.candidates?.[0]?.content?.parts?.[0]?.text || "No solution found.";
      setTroubleText(text);
    } catch (error) {
      console.error("Error with LLM API:", error);
      setTroubleText("An error occurred while fetching the solution. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSettingsClick = () => {
    setShowPasswordPrompt(true);
  };
  
  const handlePasswordSubmit = (e) => {
    e.preventDefault();
    if (passwordInput === correctPassword) {
      setShowPasswordPrompt(false);
      setShowSettingsPanel(true);
      setPasswordError('');
      setPasswordInput('');
    } else {
      setPasswordError('Invalid password. Please try again.');
      setPasswordInput('');
    }
  };
  
  const closeSettingsPanel = () => {
    setShowSettingsPanel(false);
  };

  useEffect(() => {
    const statusInterval = setInterval(() => {
      const statuses = ['green', 'yellow', 'red'];
      const newStatus = statuses[Math.floor(Math.random() * statuses.length)];
      setSystemStatus(newStatus);
    }, 10000);

    setWorkerCount(2);

    return () => {
      clearInterval(statusInterval);
    };
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case 'green':
        return 'green';
      case 'yellow':
        return 'yellow';
      case 'red':
        return 'red';
      default:
        return 'gray';
    }
  };


  return (
    <>
      <div className="main-screen-container">
        <div className="main-layout">
          <div className="header-content-section">
            <header className="header">
              <h1 className="title">SCORBOT Gesture Control Interface</h1>
            </header>
            <div className="main-content-layout">
              <div className="main-screen-left-column-container">
                <aside className="control-panel">
                  <div className="button-group">
                    <span className="button-label">START/PAUSE LIVE TRACKING</span>
                    <div className="button-ring">
                      <button className={`button ${isTracking ? 'button-gold' : 'button-green'}`} onClick={toggleTracking}>
                        {isTracking ? (
                          <>
                            <svg xmlns="http://www.w3.org/2000/svg" className="button-icon" viewBox="0 0 24 24" fill="currentColor">
                              <path d="M6.75 5.25a.75.75 0 01.75-.75H9a.75.75 0 01.75.75v13.5a.75.75 0 01-.75.75H7.5a.75.75 0 01-.75-.75V5.25zm7.5 0A.75.75 0 0115 4.5h1.5a.75.75 0 01.75.75v13.5a.75.75 0 01-.75.75H15a.75.75 0 01-.75-.75V5.25z" />
                            </svg>
                            PAUSE
                          </>
                        ) : (
                          <>
                            <svg xmlns="http://www.w3.org/2000/svg" className="button-icon" viewBox="0 0 24 24" fill="currentColor">
                              <path d="M5.25 5.653c0-.856.917-1.353 1.637-.976l10.353 5.347a1.5 1.5 0 010 2.606L6.887 19.323A1.5 1.5 0 015.25 18.006V5.653z" />
                            </svg>
                            START
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                  <div className="button-group">
                    <span className="button-label">EMERGENCY STOP</span>
                    <div className="button-ring">
                      <button className="button-octagon" onClick={handleEmergencyStop}>
                        <svg xmlns="http://www.w3.org/2000/svg" className="button-icon" viewBox="0 0 24 24" fill="currentColor">
                          <path d="M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25zM9.75 9.75a.75.75 0 000 1.5h4.5a.75.75 0 000-1.5h-4.5z" />
                        </svg>
                        STOP
                      </button>
                    </div>
                  </div>
                  <div className="button-group">
                    <span className="button-label">SETTINGS</span>
                    <div className="button-ring">
                      <button className="button button-blue" onClick={handleSettingsClick}>
                        <svg xmlns="http://www.w3.org/2000/svg" className="button-icon" viewBox="0 0 24 24" fill="currentColor">
                          <path d="M11.666 21.054a.75.75 0 01-.896-.282l-2.454-3.793A9 9 1.5 0 013.5 14.5a9 9 0 0111.411-8.567l.19-.053-.19.053a9 9 0 01-11.411 8.567z" />
                          <path d="M12.992 10.978a.75.75 0 01-.896-.282l-2.454-3.793A9 9 0 0110.5 4.5a9 9 0 0111.411 8.567l.19.053-.19-.053a9 9 0 01-11.411-8.567z" />
                        </svg>
                        SETTINGS
                      </button>
                    </div>
                  </div>
                </aside>
              </div>
              <div className="middle-video-content-container">
                <div className="video-container">
                  <div className="video-text">
                    <img
                      src="https://placehold.co/1080x720/4a5568/ffffff?text=Video+Feed"
                      alt="Live Gesture Tracking"
                      className="video-image"
                    />
                  </div>
                  {showBoundingBoxes && (
                    <div className="overlay">
                      <BoundingBox/>
                    </div>
                  )}
                </div>
              </div>
              <div className="main-screen-right-column-container">
                <aside className="status-panel">
                  <div className="status-group">
                    <span className="status-label">SYSTEM STATUS</span>
                    <div className="status-lights-container">
                      <div className={`status-light ${getStatusColor(systemStatus) === 'green' ? 'status-light-green' : 'status-light-off'}`}></div>
                      <div className={`status-light ${getStatusColor(systemStatus) === 'yellow' ? 'status-light-yellow' : 'status-light-off'}`}></div>
                      <div className={`status-light ${getStatusColor(systemStatus) === 'red' ? 'status-light-red' : 'status-light-off'}`}></div>
                    </div>
                  </div>
                  <div className="status-group">
                    <span className="status-label">AUTHORIZATION COLOR</span>
                    <div className="status-circle status-circle-orange"></div>
                  </div>
                  <div className="status-group">
                    <span className="status-label">WORKERS IN WORKSPACE</span>
                    <div className="status-circle status-circle-blue">
                      <span className="status-count">{workerCount}</span>
                    </div>
                  </div>
                </aside>
              </div>
            </div>
          </div>
          {showModal && (
            <div className="modal-overlay">
              <div className="modal-content">
                <h2 className="modal-title">System Analysis</h2>
                <div className="modal-body">
                  {isLoading ? (
                    <p>Analyzing system... please wait.</p>
                  ) : (
                    <p>{troubleText}</p>
                  )}
                </div>
                <button className="modal-close-button" onClick={() => setShowModal(false)}>Close</button>
              </div>
            </div>
          )}
        </div>
      </div>
      {showPasswordPrompt && (
        <div className="password-prompt-overlay">
          <div className="password-prompt-panel">
            <h2 className="password-prompt-title">Authentication Required</h2>
            <p className="password-prompt-label">
                Please enter the password to access settings.
            </p>
            <form onSubmit={handlePasswordSubmit}>
              <input
                type="password"
                className="password-input"
                placeholder="Enter password"
                value={passwordInput}
                onChange={(e) => setPasswordInput(e.target.value)}
              />
              {passwordError && <p className="password-error">{passwordError}</p>}
              <button type="submit" className="password-submit-button">Submit</button>
            </form>
            <button className="close-password-prompt" onClick={() => setShowPasswordPrompt(false)}>
              <svg xmlns="http://www.w3.org/2000/svg" className="close-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}
      <SettingsPanel isOpen={showSettingsPanel} onClose={closeSettingsPanel} />
    </>
  );
}
export default Idk;
