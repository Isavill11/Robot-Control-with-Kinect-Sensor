import React, { useState } from 'react';
import './styles.css'; // Use the same stylesheet

 // Settings Panel Component
  const SettingsPanel = ({ isOpen, onClose }) => {
    const [activeTab, setActiveTab] = useState('Authentication');
  
    if (!isOpen) {
      return null;
    }
  
  const renderContent = () => {
    switch (activeTab) {
      case 'Authentication':
        return (
          <div className="settings-content-body">
            <h3>User Access & Roles</h3>
            <p>Manage users, assign authentication colors, and set permissions.</p>
            <button className="settings-action-button button-green">Add New User</button>
            <div className="user-log">
                <p>Last accessed: *User 1* on 2025/09/17</p>
            </div>
          </div>
        );
      case 'Gestures':
        return (
          <div className="settings-content-body">
            <h3>Gesture Mapping</h3>
            <p>Configure and recalibrate hand gestures for robotic commands (e.g., 'Open Gripper' gesture sensitivity).</p>
            <button className="settings-action-button button-blue">Recalibrate Kinect</button>
          </div>
        );
      case 'MATLAB':
        return (
            <div className="settings-content-body">
                <h3>Script Execution</h3>
                <p>Upload, select, or manage MATLAB scripts for advanced custom operations (e.g., 'Align 3 Blocks').</p>
                <button className="settings-action-button button-gold">Upload Script</button>
            </div>
        );
      case 'Kinect':
        return (
            <div className="settings-content-body">
                <h3>Camera Configuration</h3>
                <p>Adjust camera field-of-view, resolution, and bounding box detection parameters.</p>
                <button className="settings-action-button button-red">Reset Camera</button>
            </div>
        );
      default:
        return <div>Select a setting category.</div>;
    }
  };
  
  return (
    <div className="settings-overlay">
      <div className="settings-panel">
        <header className="settings-header">
          <h2 className="settings-title">System Settings</h2>
          <button className="close-button" onClick={onClose}>
            <svg xmlns="http://www.w3.org/2000/svg" className="close-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </header>
        
        <div className="settings-main-area">
          <nav className="settings-tabs-nav">
            {['Authentication', 'Gestures', 'MATLAB', 'Kinect'].map((tab) => (
              <button 
                key={tab}
                className={`tab-button ${activeTab === tab ? 'tab-active' : ''}`}
                onClick={() => setActiveTab(tab)}
              >
                {tab === 'Authentication' ? 'User Authentication' : 
                  tab === 'Gestures' ? 'Hand Gestures' : 
                  tab === 'MATLAB' ? 'MATLAB Scripts' : 
                  'XBOX KINECT Camera'}
              </button>
            ))}
          </nav>
          <div className="settings-content-container">
            {renderContent()}
          </div>
        </div>
      </div>
    </div>
  );
};


export default SettingsPanel; 