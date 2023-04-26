import React from "react";
import VideoInput from "./VideoInput";
import "./styles.css";

import Footer from "./Footer";

export default function App() {
  return (
    <div className="App">
      <h2 className="main-header">Video Upsampling module</h2>
      <VideoInput width={400} height={600} />

      <Footer />
    </div>
  );
}
