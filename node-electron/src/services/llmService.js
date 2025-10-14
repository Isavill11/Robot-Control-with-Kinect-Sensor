export async function fetchSystemAnalysis(systemStatus, workerCount, apiKey = '') {
  const makePrompt = (status) => {
    if (status === 'red') {
      return `The SCORBOT Gesture Control Interface has a critical '${status}' status. There are ${workerCount} workers in the workspace, and the system detected a person outside of a safety zone with a confidence score of 'Person 0.81'. Provide a concise, single-paragraph explanation and a potential solution.`;
    }
    if (status === 'yellow') {
      return `The SCORBOT Gesture Control Interface has a warning '${status}' status. There are ${workerCount} workers in the workspace, and the system detected a person approaching a safety zone with a confidence score of 'Person 0.78'. Provide a concise, single-paragraph explanation and a potential solution.`;
    }
    return `The SCORBOT Gesture Control Interface is reporting a '${status}' status. There are ${workerCount} workers in the workspace. Provide a concise, single-paragraph explanation and a potential solution.`;
  };

  const payload = {
    contents: [{ parts: [{ text: makePrompt(systemStatus) }] }],
    tools: [{ google_search: {} }],
    systemInstruction: {
      parts: [{ text: "Act as a world-class robotic systems analyst. Provide a concise, single-paragraph summary of the key findings and a potential solution." }]
    }
  };

  const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${apiKey}`;
  try {
    const res = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const json = await res.json();
    return json?.candidates?.[0]?.content?.parts?.[0]?.text || 'No solution found.';
  } catch (err) {
    console.error('LLM service error:', err);
    throw err;
  }
}