import React, { useState, useRef } from "react";
import LoadingSpinner from "./Spinner";

export default function VideoInput(props) {
  const { width, height } = props;

  // const inputRef = React.useRef();

  const [source, setSource] = React.useState();
  const [isLoading, setIsLoading] = useState(false);  

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    const url = URL.createObjectURL(file);
    setSource(url);
  };

 

  // const handleSubmit = (event) => {
  //   //implement
  // }


  const fileInputRef = useRef(null);
  const videoRef = useRef(null);

   const handleChoose = (event) => {
    fileInputRef.current.click();
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    
    const formData = new FormData();
    formData.append('video', fileInputRef.current.files[0]);
    setIsLoading(true);
  
    try {
      const response = await fetch('http://127.0.0.1:8000/api/process_video/', {
        method: 'POST',
        body: formData,
      });
      setIsLoading(false);

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
    <div className="container">
      <h5 className="top-header">RD-BasicVSR: BasicVSR with RRDB blocks to upsample videos</h5>
      <div className="container-main">
        <div className="VideoInput">
          <input
            ref={fileInputRef}
            className="VideoInput_input"
            type="file"
            onChange={handleFileChange}
            accept=".mov,.mp4"
          />
          {!source && <button className="Choose-button" onClick={handleChoose}>Upload Video</button>}
          {source && (
            <video
              className="VideoInput_video"
              width="100%"
              height={height}
              controls
              src={source}
            />
            
            
          )}

          {/* {source && (
            <button className="Submit-button" onClick={handleSubmit}>Submit</button>
          )} */}
          {/* <div className="VideoInput_footer">{source || "Nothing selected"}</div> */}
        </div>


        {/* Output frames */}
        <div className="Video-output">
 
          {!source && 
          
          <div className="Video-output-box">
            <p className="output-text">Output here!</p>
              
              
          </div>
          
          }

          {/* {source && (
            <video ref={videoRef}  controls></video>
          )} */}

          { source && isLoading && 
          <div>
            <p className="processing-text">Processing...hang on!</p>
            <LoadingSpinner /> 
          </div>
          }

          {
            source && !isLoading &&
            <div>
              <video ref={videoRef}  controls></video>
            </div>
          }
          {/* <div className="VideoInput_footer">{source || "Nothing selected"}</div> */}
        </div>

      </div>
      <button className="Submit-button" onClick={handleSubmit}>Submit</button>
    </div>
  );
}
