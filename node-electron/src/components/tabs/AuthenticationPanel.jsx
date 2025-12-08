// src/tabs/AuthenticationPanel.js

import React from 'react';

// Use a placeholder icon for the add button for now (a simple plus sign)
const AddIcon = () => <span className="add-icon">+</span>;

// Placeholder for user details (like what commands they have access to)
const UserAccessList = () => (
    <div className="user-access-list">
        <h4 className="access-title">ACCESS</h4>
        <div className="command-button home">Home Robot</div>
        <div className="command-button initialize">Initialize</div>
        <div className="command-button shutdown">Safe Shutdown</div>
        <div className="command-button open-gripper">Open Gripper</div>
        <div className="command-button open-gripper-2">Open Gripper</div>
        <div className="command-button handover">Handover Block</div>
        <div className="command-button align">Align 3 Blocks</div>
        <div className="command-button stack">Stack 3 Blocks</div>
        <button className="add-button small-add-button" title="Add Access">+</button>
    </div>
);


export default function AuthenticationPanel() {
  return (
    <div className="auth-panel-content">
      <div className="user-section">
        <div className="current-user-settings">
            <h3 className="user-id">USER 1:</h3>
            <p className="color-assigned">COLOR ASSIGNED: <span className="color-label orange">ORANGE</span></p>
            <UserAccessList />
        </div>

        <div className="add-user-section">
            <h3 className="add-user-title">ADD USER</h3>
            <button className="add-button" title="Add New User">
                <AddIcon />
            </button>
        </div>
      </div>
      
      <div className="user-log-section">
        <h3 className="log-title">USER TIME LOG</h3>
        <div className="user-time-log">
            <p><strong>Last accessed by:</strong> [User Name]</p>
            <p><strong>On:</strong> [Date/Time]</p>
            <br/>
            <p><strong>Earlier accessed by:</strong> [User Name]</p>
            <p><strong>On:</strong> [Date/Time]</p>
            {/* More log entries would go here */}
        </div>
      </div>
      
    </div>
  );
}