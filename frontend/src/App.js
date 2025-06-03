// frontend/src/App.js
import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post(
        'http://localhost:8000/predict',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      );
      
      setPrediction(response.data.prediction);
      setConfidence(response.data.confidence);
    } catch (error) {
      console.error(error);
    }
  };

  const handleMisclassification = async () => {
    if (!prediction) return;

    try {
      await axios.post('http://localhost:8000/report-error', {
        prediction: prediction,
        confidence: confidence,
        actualClass: prompt('Please enter the correct class:')
      });
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="container">
      <h1>Image Classification</h1>
      
      <div className="upload-section">
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleImageUpload}
        />
      </div>

      {prediction && (
        <div className="result-section">
          <h2>Prediction: {prediction}</h2>
          <p>Confidence: {(confidence * 100).toFixed(2)}%</p>
          
          <button onClick={handleMisclassification}>
            Report Misclassification
          </button>
        </div>
      )}
    </div>
  );
}

export default App;