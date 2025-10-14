import React, { useState, useRef, useEffect } from 'react';
import './styles/SettingsPanel.css';
import AuthenticationPanel from './tabs/AuthenticationPanel';
import GesturesTab from './tabs/GesturesTab';
import MatlabTab from './tabs/MatlabTab';
import KinectTab from './tabs/KinectTab';

const TABS = {
  Authentication: AuthenticationPanel,
  Gestures: GesturesTab,
  MATLAB: MatlabTab,
  Kinect: KinectTab,
};

export default function SettingsPanel({
  isOpen,
  onClose,
  initialWidth = 360,
  minWidth = 240,
  maxWidth = 900,
  onWidthChange,
}) {
  const [width, setWidth] = useState(initialWidth);
  const isResizing = useRef(false);
  const startX = useRef(0);
  const startWidth = useRef(initialWidth);

  useEffect(() => {
    onWidthChange?.(width);
  }, [width, onWidthChange]);

  useEffect(() => {
    function onMouseMove(e) {
      if (!isResizing.current) return;
      // panel is anchored to the right, dragging left-edge: increase when dragging left
      const delta = startX.current - e.clientX;
      const newW = Math.max(minWidth, Math.min(maxWidth, startWidth.current + delta));
      setWidth(newW);
    }
    function onMouseUp() {
      if (!isResizing.current) return;
      isResizing.current = false;
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    }
    if (isResizing.current) {
      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [minWidth, maxWidth]);

  function startResize(e) {
    isResizing.current = true;
    startX.current = e.clientX;
    startWidth.current = width;
    e.preventDefault();
  }

  if (!isOpen) return null;

  const Active = TABS['Authentication'];

  return (
    <div className="settings-overlay" aria-hidden={!isOpen}>
      <div
        className="settings-panel"
        style={{ width: `${width}px`, right: 0, top: 0, bottom: 0, height: '100vh', position: 'fixed' }}
      >
        {/* draggable left-edge handle */}
        <div
          className="settings-resize-handle"
          title="Drag to resize"
          onMouseDown={startResize}
          role="separator"
          aria-orientation="vertical"
        />
        <header className="settings-header">
          <h2 className="settings-title">System Settings</h2>
          <button className="close-button" onClick={onClose}>✕</button>
        </header>

        <div className="settings-main-area">
          <nav className="settings-tabs-nav">
            {Object.keys(TABS).map((tab) => (
              <button key={tab} className="tab-button" onClick={() => {}}>
                {tab}
              </button>
            ))}
          </nav>

          <div className="settings-content-container">
            <Active />
          </div>
        </div>
      </div>
    </div>
  );
}