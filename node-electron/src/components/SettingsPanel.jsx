import React, { useState } from 'react';
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

export default function SettingsPanel({ isOpen, onClose }) {
  const [activeTab, setActiveTab] = useState('Authentication');

  if (!isOpen) return null;

  const Active = TABS[activeTab];

  return (
    <div className="settings-overlay">
      <div className="settings-panel">
        <header className="settings-header">
          <h2 className="settings-title">System Settings</h2>
          <button className="close-button" onClick={onClose}>✕</button>
        </header>

        <div className="settings-main-area">
          <nav className="settings-tabs-nav">
            {Object.keys(TABS).map((tab) => (
              <button 
                key={tab} 
                className={`tab-button ${activeTab === tab ? 'active-tab' : ''}`} 
                onClick={() => setActiveTab(tab)}
              >
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