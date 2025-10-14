import React from 'react';

export default function KinectTab() {
  return (
    <div className="settings-content-body">
      <h3>Camera Configuration</h3>
      <p>Adjust camera, FOV, resolution and detection bounds.</p>
      <button className="settings-action-button button-red">Reset Camera</button>
    </div>
  );
}