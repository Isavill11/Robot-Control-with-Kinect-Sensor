import React, { useState } from 'react';
import MainScreen from './components/MainScreen';



export default function App() {
  const [showSettings, setShowSettings] = useState(false);
  return (
    <div className="app-container">
      <MainScreen />
    </div>
  );
}
