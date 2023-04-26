import React, { useState, useRef } from "react";


const VideoUploader = () => {
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const handleFormSubmit = async (event) => {
    event.preventDefault();
    
    const formData = new FormData();
    formData.append('video', fileInputRef.current.files[0]);
  
    try {
      const response = await fetch('http://127.0.0.1:8000/api/process_video/', {
        method: 'POST',
        body: formData,
      });
      
      const blob = await response.blob();
      
      // create a URL for the blob object
      const videoUrl = URL.createObjectURL(blob);
      
      // set the URL as the source for a video element
      videoRef.current.src = videoUrl;
    } catch (error) {
      console.error('Error:', error);
    }
  };
  
  return (
    <form onSubmit={handleFormSubmit}>
      {/* <input type="file" ref={fileInputRef} accept="video/*" />
      <button type="submit">Submit</button> */}
      <video ref={videoRef} controls></video>
    </form>
  );
};

export default VideoUploader;
