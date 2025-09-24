import React from 'react';
import MainScreen from './MainScreen'; // Assuming MainScreen.js is in the same directory
import Settings from './Settings'; // A new component for your settings screen

function App() {
  const [showSettings, setShowSettings] = useState(false); // A state to manage which screen to show

  return (
    <div className="app-container">
      {showSettings ? (
        <Settings onClose={() => setShowSettings(false)} />
      ) : (
        <MainScreen onSettingsClick={() => setShowSettings(true)} />
      )}
    </div>
  );
}

export default App;