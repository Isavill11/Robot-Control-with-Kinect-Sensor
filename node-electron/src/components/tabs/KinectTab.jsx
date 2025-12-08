// src/tabs/KinectTab.js

import React from 'react';

export default function KinectTab() {
  return (
    <div className="kinect-tab-content">
      <div className="kinect-stream-options">
        {/* Options to change video stream */}
        <div className="stream-type-buttons">
            <div className="stream-button active">INFRARED</div>
            <div className="stream-button">COLOR</div>
            <div className="stream-button">DEPTH</div>
        </div>
      </div>

      <div className="kinect-preview-area">
        <h3 className="preview-title">Preview</h3>
        <div className="video-placeholder">
            Video
        </div>
      </div>
    </div>
  );
}