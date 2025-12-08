// src/tabs/GesturesTab.js

import React from 'react';

const AddIcon = () => <span className="add-icon">+</span>;

// A component to display the current gestures
const CurrentGestures = () => (
    <div className="gestures-list-container">
        <div className="gesture-button">ASL C-Sign</div>
        <div className="gesture-button">Thumbs Up Sign</div>
        <div className="gesture-button">Thumbs Down Sign</div>
        <div className="gesture-button">ASL A-Sign</div>
        <div className="gesture-button">ASL B-Sign</div>
        <div className="gesture-button">Palm Up Sign</div>
        <div className="gesture-button">Okay Sign</div>
        <div className="gesture-button">Peace Sign</div>
        <button className="add-button small-add-button" title="Add More Gestures">+</button>
    </div>
);

export default function GesturesTab() {
  return (
    <div className="gestures-tab-content">

      <div className="tab-column-container current-gestures-column">
        <h3 className="column-title">CURRENT GESTURES</h3>
        <CurrentGestures />
      </div>

      {/* Container 2: Create New Gestures */}
      <div className="tab-column-container create-gestures-column">
        <h3 className="column-title">CREATE NEW GESTURES</h3>
        <button className="add-button large-add-button" title="Create New Gesture">
            <AddIcon />
        </button>
      </div>
    </div>
  );
}