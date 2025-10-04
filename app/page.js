'use client'; // Required for App Router (client-side component)

import { useState, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import styles from './page.module.css'; // Adjust to App Router naming

export default function Home() {
  const [prediction, setPrediction] = useState(null);
  const [image, setImage] = useState(null);
  const fileInputRef = useRef(null);
  const modelRef = useRef(null);

  // Load the TensorFlow.js GraphModel
  const loadModel = async () => {
    try {
      modelRef.current = await tf.loadGraphModel('/models/tfjs_model/model.json');
      console.log('GraphModel loaded successfully');
    } catch (error) {
      console.error('Error loading model:', error);
    }
  };

  // Preprocess the image to match training conditions (150x150, RGB, [0, 1])
  const preprocessImage = (imgElement) => {
    const canvas = document.createElement('canvas');
    canvas.width = 150;
    canvas.height = 150;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgElement, 0, 0, 150, 150);
    const imageData = ctx.getImageData(0, 0, 150, 150);
    const tensor = tf.browser.fromPixels(imageData).toFloat().div(255.0);
    return tensor.expandDims(0); // Add batch dimension
  };

  // Handle image upload and prediction
  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const img = new Image();
    img.src = URL.createObjectURL(file);
    setImage(img.src);

    img.onload = async () => {
      try {
        if (!modelRef.current) await loadModel();
        const tensor = preprocessImage(img);
        const output = await modelRef.current.predict(tensor); // GraphModel.predict returns a tensor
        const prediction = await output.data(); // Extract data
        const result = prediction[0] > 0.5 ? 'Tuberculosis' : 'Normal';
        const confidence = (prediction[0] * 100).toFixed(2);
        setPrediction(`Prediction: ${result} (${confidence}% confidence)`);
        tensor.dispose(); // Clean up tensor
        output.dispose(); // Clean up output tensor
      } catch (error) {
        console.error('Error during prediction:', error);
        setPrediction('Error during prediction');
      }
    };
  };

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Tuberculosis Detection</h1>
      <input
        type="file"
        accept="image/*"
        ref={fileInputRef}
        onChange={handleImageUpload}
        className={styles.input}
      />
      {image && (
        <div>
          <img src={image} alt="Uploaded" className={styles.image} />
        </div>
      )}
      {prediction && <p className={styles.prediction}>{prediction}</p>}
    </div>
  );
}