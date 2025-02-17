import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import * as tf from "@tensorflow/tfjs";

import { drawHand } from "./utilities";

const HandLandmarkerComponent: React.FC = () => {
  const [isWebcamOn, setIsWebcamOn] = useState<boolean>(false);
  const webcamRef = useRef<Webcam | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [handLandmarker, setHandLandmarker] = useState<HandLandmarker | null>(null); // State for handLandmarker
  const [model, setModel] = useState<any>(null); // State for the loaded model
  const [predictedClass, setPredictedClass] = useState<string>(""); // State for predicted class (emoji)

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
      const handLandmarkerInstance = await HandLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath: `../public/models/hand_landmarker.task`,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
        }
      );
      setHandLandmarker(handLandmarkerInstance); // Store the instance in state
      console.log("hand landmarker created");
    };

    createHandLandmarker();
  }, []); // Only run once when the component is mounted

  const detectHands = async (handLandmarker: HandLandmarker) => {
    if (
      !(
        typeof webcamRef.current !== "undefined" &&
        webcamRef.current !== null &&
        webcamRef.current.video !== null &&
        webcamRef.current.video.readyState === 4
      )
    )
      return;

    const video = webcamRef.current.video;
    const videoWidth = webcamRef.current.video.videoWidth;
    const videoHeight = webcamRef.current.video.videoHeight;

    webcamRef.current.video.width = videoWidth;
    webcamRef.current.video.height = videoHeight;

    if (!(canvasRef.current !== null)) return;

    canvasRef.current.width = videoWidth;
    canvasRef.current.height = videoHeight;

    console.log("setup done");

    const result = await handLandmarker.detectForVideo(
      video,
      performance.now()
    ); // Pass the timestamp
    console.log(result);

    if (result.landmarks) {
      // Iterate over each hand and extract 2D landmark coordinates
      const landmarks = result.landmarks.flatMap(
        (hand) => hand.map((landmark) => [landmark.x, landmark.y]) // Extract x, y coordinates for each landmark
      );
      const flattenedLandmarks = landmarks.flat(); // Flatten to a 1D array
      console.log("Landmarks:", flattenedLandmarks);

      // Call the model prediction function
      if (model && result.landmarks.length > 0) {
        await detect2(model, processLandmarks(flattenedLandmarks));
      }
    }

    const ctx = canvasRef.current.getContext("2d");
    drawHand(result, ctx, videoWidth, videoHeight); // Draw the hand landmarks on the canvas
  };

  function processLandmarks(landmarkArray: number[]): number[] {
    // Step 1: Convert to relative coordinates
    let baseX = 0, baseY = 0;

    for (let i = 0; i < landmarkArray.length; i += 2) {
      if (i === 0) {
        baseX = landmarkArray[i];
        baseY = landmarkArray[i + 1];
      }

      // Convert to relative coordinates (shift everything based on the first landmark)
      landmarkArray[i] -= baseX; // Relative x
      landmarkArray[i + 1] -= baseY; // Relative y
    }

    // Step 2: Normalize the coordinates to be in the range [-1, 1]
    const maxValue = Math.max(...landmarkArray.map(Math.abs)); // Get the maximum absolute value
    const normalizedLandmarks = landmarkArray.map((value) => value / maxValue);

    return normalizedLandmarks;
  }

  useEffect(() => {
    if (handLandmarker && isWebcamOn) {
      const interval = setInterval(() => {
        detectHands(handLandmarker); // Run detection for every frame
      }, 100);

      return () => clearInterval(interval); // Cleanup interval on unmount or when webcam is off
    }
  }, [handLandmarker, isWebcamOn]);

  const loadModel2 = async () => {
    const model = await tf.loadLayersModel("../public/models/model.json");
    console.log("model loaded");
    model.summary();

    setModel(model); // Save the model in state for use
  };

  // Model prediction function
  const detect2 = async (model: tf.LayersModel, landmarks_2d: number[]): Promise<void> => {
    const input_data = tf.tensor2d([landmarks_2d], [1, 42]); // Reshape to (1, 42)
    const predictionTensor = model.predict(input_data);

    if (!Array.isArray(predictionTensor)) {
      const prediction = await predictionTensor.data();
      const probabilities = Array.from(prediction); // Convert to plain array

      const options = ["✋", "👊", "🤓👆"]; // Emojis for classification options
      const predictedClassIndex = probabilities.indexOf(Math.max(...probabilities));
      const predictedClass = options[predictedClassIndex];

      setPredictedClass(predictedClass); // Update the state with the predicted class
      console.log("Predicted class: ", predictedClass);
    }
  };

  useEffect(() => {
    loadModel2();
  }, []);

  return (
    <div className="flex flex-col gap-4 items-center relative">
      <button
        className="text-gray-50 bg-indigo-600 px-6 py-4 cursor-pointer"
        onClick={toggleWebcam}
      >
        {isWebcamOn ? "Turn off Webcam" : "Turn on Webcam"}
      </button>
      <div className="mt-4 relative">
        {isWebcamOn && (
          <Webcam mirrored ref={webcamRef} className="w-[720px] h-auto" />
        )}

        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-[720px] h-auto"
        />
      </div>
      <div>
        <h1 className="text-5xl">{predictedClass}</h1> {/* Display the predicted class */}
      </div>
    </div>
  );
};

export default HandLandmarkerComponent;