import React from 'react';
import './style.css'; // Correct way to import CSS

function SettingsScreen() {

    return (
        <div className="settings-container">
            <div className="settings-sidebar">
                <h2 className="sidebar-title">Settings</h2>
                <div className="sidebar-tabs">
                    <button className="tab-button active" data-tab="user-authentication">User Authentication</button>
                    <button className="tab-button" data-tab="hand-gestures">Hand Gestures</button>
                    <button className="tab-button" data-tab="matlab-scripts">MATLAB Scripts</button>
                    <button className="tab-button" data-tab="xbox-kinect-camera">XBOX KINECT Camera</button>
                </div>
                <div className="sidebar-bottom">
                    <button className="back-button">
                        <span className="button-icon">&#x2190;</span> Back to Main
                    </button>
                </div>
            </div>
            <div className="settings-content">
                <div id="user-authentication" className="content-panel active">
                    <h3>User Authentication Settings</h3>
                    <div className="content-section">
                        <h4>User Profiles</h4>
                        <div className="user-profile-list">
                            <div className="user-profile-card">
                                <span className="user-name">User 1</span>
                                <span className="user-status">Status: Active</span>
                                <span className="user-color">Color Assigned: Orange</span>
                                <button className="edit-button">Edit</button>
                            </div>
                            <div className="user-profile-card">
                                <span className="user-name">User 2</span>
                                <span className="user-status">Status: Inactive</span>
                                <span className="user-color">Color Assigned: Blue</span>
                                <button className="edit-button">Edit</button>
                            </div>
                        </div>
                        <button className="add-user-button">+ Add New User</button>
                    </div>
                    <div className="content-section">
                        <h4>User Time Log</h4>
                        <div className="time-log-box">
                            <p>Last accessed by: User 1</p>
                            <p>On: 2025-09-15</p>
                            <p>Earlier accessed by: User 2</p>
                            <p>On: 2025-09-14</p>
                        </div>
                    </div>
                </div>
                <div id="hand-gestures" className="content-panel">
                    <h3>Hand Gestures Control</h3>
                    <div className="content-section">
                        <h4>Gesture Mapping</h4>
                        <div className="gesture-mapping-list">
                            <div className="gesture-item">
                                <span className="gesture-name">Home Robot</span>
                                <button className="reassign-button">Reassign Gesture</button>
                            </div>
                            <div className="gesture-item">
                                <span className="gesture-name">Initialize</span>
                                <button className="reassign-button">Reassign Gesture</button>
                            </div>
                            <div className="gesture-item">
                                <span className="gesture-name">Safe Shutdown</span>
                                <button className="reassign-button">Reassign Gesture</button>
                            </div>
                        </div>
                        <button className="add-gesture-button">+ Add New Gesture</button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default SettingsScreen;