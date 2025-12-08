// src/tabs/MatlabTab.js

import React from 'react';

const AddIcon = () => <span className="add-icon">+</span>;

// A component to display the current commands/scripts
const CurrentScripts = () => (
    <div className="scripts-list-container">
        <div className="command-button home">Home Robot</div>
        <div className="command-button initialize">Initialize</div>
        <div className="command-button shutdown">Safe Shutdown</div>
        <div className="command-button open-gripper">Open Gripper</div>
        <div className="command-button open-gripper-2">Open Gripper</div>
        <div className="command-button handover">Handover Block</div>
        <div className="command-button align">Align 3 Blocks</div>
        <div className="command-button stack">Stack 3 Blocks</div>
        <button className="add-button small-add-button" title="Add More Scripts">+</button>
    </div>
);

export default function MatlabTab() {
  return (
    <div className="matlab-tab-content three-column-layout">
      {/* Container 1: Current Scripts */}
      <div className="tab-column-container current-scripts-column">
        <h3 className="column-title">CURRENT SCRIPTS</h3>
        <CurrentScripts />
      </div>

      {/* Container 2: Create New Scripts */}
      <div className="tab-column-container create-scripts-column">
        <h3 className="column-title">CREATE NEW SCRIPTS</h3>
        <button className="add-button large-add-button" title="Create New Script">
            <AddIcon />
        </button>
      </div>

      {/* Container 3: Speed of Commands */}
      <div className="tab-column-container speed-column">
        <h3 className="column-title">SPEED OF COMMANDS</h3>
        <div className="speed-slider-container">
            <input type="range" min="0" max="100" defaultValue="50" className="speed-slider"/>
            <div className="speed-labels">
                <span>0</span>
                <span>100</span>
            </div>
        </div>
      </div>
    </div>
  );
}