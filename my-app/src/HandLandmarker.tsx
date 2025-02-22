import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import * as tf from "@tensorflow/tfjs";

import { drawHand } from "./utilities";

const HandLandmarkerComponent: React.FC = () => {
  const [isWebcamOn, setIsWebcamOn] = useState<boolean>(false);
  const webcamRef = useRef<Webcam | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [handLandmarker, setHandLandmarker] = useState<HandLandmarker | null>(
    null
  );
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [predictedClass1, setPredictedClass1] = useState<string>("");
  const [predictedClass2, setPredictedClass2] = useState<string>("");

  // Toggle webcam on/off
  const toggleWebcam = async () => {
    setIsWebcamOn(!isWebcamOn);

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx)
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
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
            modelAssetPath: `../models/hand_landmarker.task`,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
        }
      );
      setHandLandmarker(handLandmarkerInstance);
      console.log("Hand landmarker created");
    };

    createHandLandmarker();
  }, []);

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
  
    const result = await handLandmarker.detectForVideo(video, performance.now());
    console.log(result);
  
    if (result.landmarks.length === 0) return; // No hands detected
  
    // Extract landmarks and handedness
    const hands = result.landmarks.map((landmark, index) => ({
      landmarks: landmark.flatMap((point) => [point.x, point.y]), // Flatten landmarks to 1D array
      handedness: result.handedness[index][0].displayName, // "Left" or "Right"
    }));
  
    // Sort hands so that left hand is first, right hand is second (but handednes is mirrored)
    hands.sort((a, b) => (a.handedness === "Right" ? -1 : 1));
  
    // Extract only landmarks in sorted order
    const sortedLandmarks = hands.map((hand) => hand.landmarks);
  
    console.log("Sorted Hands:", hands.map((h) => h.handedness)); // Should log ["Left", "Right"]
  
    // Run predictions with sorted order
    if (model) {
      const predictions = await Promise.all(
        sortedLandmarks.map((landmarks) => detect2(model, processLandmarks(landmarks)))
      );
  
      setPredictedClass1(predictions[0] || ""); // If left hand exists
      setPredictedClass2(predictions[1] || ""); // If right hand exists
    }
  
    const ctx = canvasRef.current.getContext("2d");
    drawHand(result, ctx, videoWidth, videoHeight);
  };

  function processLandmarks(landmarkArray: number[]): number[] {
    // Step 1: Convert to relative coordinates
    let baseX = landmarkArray[0],
      baseY = landmarkArray[1];

    for (let i = 0; i < landmarkArray.length; i += 2) {
      landmarkArray[i] -= baseX; // Relative x
      landmarkArray[i + 1] -= baseY; // Relative y
    }

    // Step 2: Normalize to range [-1, 1]
    const maxValue = Math.max(...landmarkArray.map(Math.abs));
    return landmarkArray.map((value) => value / maxValue);
  }

  useEffect(() => {
    if (handLandmarker && isWebcamOn) {
      const interval = setInterval(() => {
        detectHands(handLandmarker);
      }, 100);

      return () => clearInterval(interval);
    }
  }, [handLandmarker, isWebcamOn]);

  const loadModel = async () => {
    const model = await tf.loadLayersModel("../public/models/model.json");
    console.log("Model loaded");
    model.summary();
    setModel(model);
  };

  // Model prediction function
  const detect2 = async (
    model: tf.LayersModel,
    landmarks_2d: number[]
  ): Promise<string> => {
    const input_data = tf.tensor2d([landmarks_2d], [1, 42]);
    const predictionTensor = model.predict(input_data);

    if (!Array.isArray(predictionTensor)) {
      const prediction = await predictionTensor.data();
      const probabilities = Array.from(prediction);

      const options = ["✋", "👊", "🤓👆"]; // Classification labels
      const predictedClassIndex = probabilities.indexOf(
        Math.max(...probabilities)
      );
      return options[predictedClassIndex] || "";
    }
    return "";
  };

  useEffect(() => {
    loadModel();
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
        <h1 className="text-5xl">
          {predictedClass1 !== "" && predictedClass1}{" "}
          {predictedClass2 !== "" && predictedClass2}
        </h1>
      </div>
    </div>
  );
};

export default HandLandmarkerComponent;
