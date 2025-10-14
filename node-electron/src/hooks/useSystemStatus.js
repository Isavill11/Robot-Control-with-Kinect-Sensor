import { useState, useEffect } from 'react';

export default function useSystemStatus(initial = 'green', intervalMs = 10000) {
  const [systemStatus, setSystemStatus] = useState(initial);

  useEffect(() => {
    const statuses = ['green', 'yellow', 'red'];
    const id = setInterval(() => {
      const newStatus = statuses[Math.floor(Math.random() * statuses.length)];
      setSystemStatus(newStatus);
    }, intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);

  return [systemStatus, setSystemStatus];
}