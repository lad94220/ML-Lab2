import { useRef, useState } from "react";
import { ReactSketchCanvas, type ReactSketchCanvasRef } from "react-sketch-canvas";
import { PredictionDisplay } from "./PredictionDisplay";

export const DrawField = () => {
  const canvasRef = useRef<ReactSketchCanvasRef>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<any>(null);

  const handleClear = () => {
    canvasRef.current?.clearCanvas();
    setPrediction(null);
  };

  const handleRecognize = async () => {
    if (!canvasRef.current) return;

    try {
      setIsLoading(true);
      setPrediction(null);
      
      // Export canvas as base64 image
      const imageData = await canvasRef.current.exportImage("png");

      // Send base64 directly to server
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URI}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      });

      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      console.error('Error:', error);
      setPrediction({ error: 'Failed to connect to server' });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-center text-2xl font-semibold pb-4">
        DRAW 
      </div>
      <div className="flex justify-center">
        <ReactSketchCanvas
          ref={canvasRef}
          strokeWidth={20}
          width="280px"
          height="280px"
          strokeColor="black"
          canvasColor="white"
          style={{ border: '2px solid #e5e7eb' }}
        />
      </div>
      <div className="flex justify-between">
        <button 
          onClick={handleClear}
          className="px-4 py-1 bg-red-500 cursor-pointer text-white rounded hover:bg-red-600 transition-colors"
        >
          Clear
        </button>
        <button
          onClick={handleRecognize}
          disabled={isLoading}
          className="px-6 py-2 bg-blue-500 text-white rounded cursor-pointer hover:bg-blue-600 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {isLoading ? 'Processing...' : 'Recognize'}
        </button>
      </div>

      <PredictionDisplay prediction={prediction} />
    </div>
  )
}