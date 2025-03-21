import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import * as tf from "@tensorflow/tfjs";
import neutral from "./assets/neutral.png";
import wave_l from "./assets/wave_l.png";
import wave_r from "./assets/wave_r.png";
import wave_b from "./assets/wave_b.png";
import fist_l from "./assets/fist_l.png";
import fist_r from "./assets/fist_r.png";
import fist_b from "./assets/fist_b.png";
import nerd_l from "./assets/nerd_l.png";
import nerd_r from "./assets/nerd_r.png";
import nerd_b from "./assets/nerd_b.png";
import fist_l_nerd_r from "./assets/fist_l_nerd_r.png";
import fist_l_wave_r from "./assets/fist_l_wave_r.png";
import nerd_l_fist_r from "./assets/nerd_l_fist_r.png";
import nerd_l_wave_r from "./assets/nerd_l_wave_r.png";
import wave_l_nerd_r from "./assets/wave_l_nerd_r.png";
import wave_l_fist_r from "./assets/wave_l_fist_r.png";

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
  const [imgName, setImgName] = useState<string>("neutral");

  const imageMap = {
    neutral,
    wave_l,
    wave_r,
    wave_b,
    fist_l,
    fist_r,
    fist_b,
    nerd_l,
    nerd_r,
    nerd_b,
    fist_l_nerd_r,
    fist_l_wave_r,
    nerd_l_fist_r,
    nerd_l_wave_r,
    wave_l_nerd_r,
    wave_l_fist_r,
  };

  const toggleWebcam = async () => {
    setIsWebcamOn(!isWebcamOn);

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx)
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

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

    if (!canvasRef.current) return;

    canvasRef.current.width = videoWidth;
    canvasRef.current.height = videoHeight;

    const result = await handLandmarker.detectForVideo(
      video,
      performance.now()
    );

    if (result.landmarks.length === 0) {
      setPredictedClass1("");
      setPredictedClass2("");
      return;
    }

    if (result.landmarks.length === 0) return;

    let leftHandLandmarks: number[] | null = null;
    let rightHandLandmarks: number[] | null = null;

    result.landmarks.forEach((landmarks, index) => {
      const handedness = result.handedness[index][0].displayName;
      const flatLandmarks = landmarks.flatMap((point) => [point.x, point.y]);

      if (handedness === "Left") {
        leftHandLandmarks = flatLandmarks;
      } else if (handedness === "Right") {
        rightHandLandmarks = flatLandmarks;
      }
    });

    if (model) {
      const leftPrediction = leftHandLandmarks
        ? await detect(model, processLandmarks(leftHandLandmarks))
        : "";
      const rightPrediction = rightHandLandmarks
        ? await detect(model, processLandmarks(rightHandLandmarks))
        : "";

      // camera is mirrored so left goes to 2, right goes to 1
      setPredictedClass2(leftPrediction);
      setPredictedClass1(rightPrediction);
    }

    const ctx = canvasRef.current.getContext("2d");
    drawHand(result, ctx, videoWidth, videoHeight);
  };

  function processLandmarks(landmarkArray: number[]): number[] {
    let baseX = landmarkArray[0],
      baseY = landmarkArray[1];

    for (let i = 0; i < landmarkArray.length; i += 2) {
      landmarkArray[i] -= baseX;
      landmarkArray[i + 1] -= baseY;
    }

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

  useEffect(() => {
    if (predictedClass1 === "✋" && predictedClass2 === "") {
      setImgName("wave_l");
    } else if (predictedClass1 === "" && predictedClass2 === "✋") {
      setImgName("wave_r");
    } else if (predictedClass1 === "✋" && predictedClass2 === "✋") {
      setImgName("wave_b");
    } else if (predictedClass1 === "👊" && predictedClass2 === "") {
      setImgName("fist_l");
    } else if (predictedClass1 === "" && predictedClass2 === "👊") {
      setImgName("fist_r");
    } else if (predictedClass1 === "👊" && predictedClass2 === "👊") {
      setImgName("fist_b");
    } else if (predictedClass1 === "👆" && predictedClass2 === "") {
      setImgName("nerd_l");
    } else if (predictedClass1 === "" && predictedClass2 === "👆") {
      setImgName("nerd_r");
    } else if (predictedClass1 === "👆" && predictedClass2 === "👆") {
      setImgName("nerd_b");
    } else if (predictedClass1 === "👆" && predictedClass2 === "👊") {
      setImgName("nerd_l_fist_r");
    } else if (predictedClass1 === "👆" && predictedClass2 === "✋") {
      setImgName("nerd_l_wave_r");
    } else if (predictedClass1 === "✋" && predictedClass2 === "👊") {
      setImgName("wave_l_fist_r");
    } else if (predictedClass1 === "✋" && predictedClass2 === "👆") {
      setImgName("wave_l_nerd_r");
    } else if (predictedClass1 === "👊" && predictedClass2 === "👆") {
      setImgName("fist_l_nerd_r");
    } else if (predictedClass1 === "👊" && predictedClass2 === "✋") {
      setImgName("fist_l_wave_r");
    } else {
      setImgName("neutral"); // default
    }
  }, [predictedClass1, predictedClass2]);

  const loadModel = async () => {
    const model = await tf.loadLayersModel("../public/models/model.json");
    model.summary();
    setModel(model);
  };

  const detect = async (
    model: tf.LayersModel,
    landmarks_2d: number[]
  ): Promise<string> => {
    const input_data = tf.tensor2d([landmarks_2d], [1, 42]);
    const predictionTensor = model.predict(input_data);

    if (!Array.isArray(predictionTensor)) {
      const prediction = await predictionTensor.data();
      const probabilities = Array.from(prediction);

      const options = ["✋", "👊", "👆"];
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
    <div className="flex flex-col gap-12 items-center justify-center relative p-20">
      <h1 className="text-4xl font-bold text-center text-indigo-800">
        mirrormote :]
      </h1>
      <button
        className="text-gray-50 bg-indigo-600 px-4 py-2 cursor-pointer rounded-lg"
        onClick={toggleWebcam}
      >
        {isWebcamOn ? "stop webcam !" : "start webcam !"}
      </button>
      <div className="flex flex-row gap-8 items-center justify-center">
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
          <img
            src={imageMap[imgName as keyof typeof imageMap] || neutral}
            alt={imgName}
            className="h-48 w-48"
          />
        </div>
      </div>
    </div>
  );
};

export default HandLandmarkerComponent;
