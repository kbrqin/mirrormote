const handMesh = {
  thumb: [0, 1, 2, 3, 4],
  indexFinger: [5, 6, 7, 8],
  middleFinger: [9, 10, 11, 12],
  ringFinger: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
  palm: [0, 5, 9, 13, 17, 0],
};

export const drawHand = (
  predictions: any,
  ctx: any,
  videoWidth: number,
  videoHeight: number
) => {
  if (predictions.landmarks && predictions.landmarks.length > 0) {
    const landmarks = predictions.landmarks;

    landmarks.forEach((landmark: any) => {
      ctx.save(); // Save current state
      ctx.scale(-1, 1); // Flip horizontally
      ctx.translate(-videoWidth, 0); // Shift back into place

      // Iterate through each finger in handMesh
      for (const key in handMesh) {
        const fingerJoints = handMesh[key as keyof typeof handMesh];

        for (let i = 0; i < fingerJoints.length - 1; i++) {
          const firstJointIndex = fingerJoints[i];
          const secondJointIndex = fingerJoints[i + 1];

          // Check if the landmarks exist and have valid x, y values
        //   if (landmark[firstJointIndex] && landmark[secondJointIndex]) {
            const x1 = landmark[firstJointIndex].x * videoWidth;
            const y1 = landmark[firstJointIndex].y * videoHeight;
            const x2 = landmark[secondJointIndex].x * videoWidth;
            const y2 = landmark[secondJointIndex].y * videoHeight;

            // Draw line between the two joints
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.lineWidth = 2;
            ctx.strokeStyle = "white";
            ctx.stroke();
        //   }
        }
      }

      // Draw landmarks (no need to manually adjust X now)
      for (let i = 0; i < landmark.length; i++) {
        // Ensure landmark[i] exists and has x, y values
        if (landmark[i]) {
          const x = landmark[i].x * videoWidth;
          const y = landmark[i].y * videoHeight;

          // Draw each landmark point
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, 2 * Math.PI);
          ctx.fillStyle = "white";
          ctx.fill();
        }
      }

      ctx.restore(); // Restore the original state
    });
  }
};
