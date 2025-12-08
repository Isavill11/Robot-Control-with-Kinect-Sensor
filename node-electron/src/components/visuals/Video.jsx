import "../styles/Video.css";

function Video({ image = 'https://placehold.co/1080x720/4a5568/ffffff?text=Video+Feed', type = "default", onClick }) {
    // Construct the class name based on the 'type' prop (e.g., video-container--default)
    const containerClass = `video-container video-container--${type}`;

    return (
        // The outer div can handle the click event
        <div className="middle-video-content-container" onClick={onClick}>
            <div className={containerClass}>
                {/* The <img> tag is used here. 
                  Its 'src' is set by the 'image' prop (which defaults to your placeholder URL).
                  Its 'alt' attribute is set directly in the JSX.
                */}
                <img 
                    src={image} 
                    alt="Live Gesture Tracking Placeholder" 
                    className="video-image" 
                />
                
                {/* Optional: Add a text overlay or live indicator if needed */}
                <div className="video-overlay">
                    <span className="live-indicator">LIVE</span>
                </div>
            </div>
        </div>
    );
}

export default Video;