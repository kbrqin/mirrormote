import React, { useState, useRef, useEffect } from 'react';
import Webcam from 'react-webcam'; 
import {
  HandLandmarker,
  FilesetResolver
} from "@mediapipe/tasks-vision";

import { drawHand } from './utilities';

const HandLandmarkerComponent: React.FC = () => {
  const [isWebcamOn, setIsWebcamOn] = useState<boolean>(false);
  const webcamRef = useRef<Webcam | null>(null); 
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [handLandmarker, setHandLandmarker] = useState<HandLandmarker | null>(null); // State for handLandmarker

  // Function to toggle the webcam on and off
  const toggleWebcam = async () => {
    setIsWebcamOn(!isWebcamOn);

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  // Initialize HandLandmarker
  useEffect(() => {
    const createHandLandmarker = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
      );
      const handLandmarkerInstance = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `../public/models/hand_landmarker.task`,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
      });
      setHandLandmarker(handLandmarkerInstance); // Store the instance in state
      console.log("hand landmarker created");
    };

    createHandLandmarker();

  }, []); // Only run once when the component is mounted

  const detectHands = async (handLandmarker: HandLandmarker) => {
    if (!(typeof webcamRef.current !== "undefined" && webcamRef.current !== null && 
      webcamRef.current.video !== null && webcamRef.current.video.readyState === 4)) return;

    const video = webcamRef.current.video;
    const videoWidth = webcamRef.current.video.videoWidth;
    const videoHeight = webcamRef.current.video.videoHeight;

    webcamRef.current.video.width = videoWidth;
    webcamRef.current.video.height = videoHeight;

    if (!(canvasRef.current !== null)) return;

    canvasRef.current.width = videoWidth;
    canvasRef.current.height = videoHeight;
    
    console.log("setup done");

    const result = await handLandmarker.detectForVideo(video, performance.now()); // Pass the timestamp
    console.log(result);

    const ctx = canvasRef.current.getContext("2d");
    
    drawHand(result, ctx, videoWidth, videoHeight); // Draw the hand landmarks on the canvas
  };

  useEffect(() => {
    if (handLandmarker && isWebcamOn) {
      const interval = setInterval(() => {
        detectHands(handLandmarker); // Run detection for every frame
      }, 100); // Adjust time interval as needed for performance

      return () => clearInterval(interval); // Cleanup interval on unmount or when webcam is off
    }
  }, [handLandmarker, isWebcamOn]); // Run only when handLandmarker or webcam state changes

  return (
    <div className="flex flex-col items-center relative">
      <button className="text-gray-50 bg-indigo-600 px-6 py-4 cursor-pointer" onClick={toggleWebcam}>
        {isWebcamOn ? 'Turn off Webcam' : 'Turn on Webcam'}
      </button>
      <div className="mt-4 relative">
        {/* react-webcam component */}
        {isWebcamOn && (
          <Webcam
          mirrored
            ref={webcamRef}
            className="w-[400px] h-auto"
          />
        )}

        {/* Canvas overlay */}
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-[400px] h-auto"
        />
      </div>
    </div>
  );
};

export default HandLandmarkerComponent;