import { useRef } from "react";
import { ReactSketchCanvas, type ReactSketchCanvasRef } from "react-sketch-canvas";

const styles = {
  border: '0.0625rem solid #9c9c9c',
  borderRadius: '0.25rem',
  color: 'black',
};

export const DrawField = () => {
  const canvasRef = useRef<ReactSketchCanvasRef>(null);

  const handleClear = () => {
    canvasRef.current?.clearCanvas();
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-center text-2xl font-semibold pb-2">
        DRAW 
      </div>
      <ReactSketchCanvas
        ref={canvasRef}
        style={styles}
        strokeWidth={4}
        strokeColor="black"
      />
      <div className="flex justify-between">
        <button 
          onClick={handleClear}
          className="px-4 py-1 bg-red-500 cursor-pointer text-white rounded hover:bg-red-600 transition-colors"
        >
          Clear
        </button>
        <button
          className="px-4 py-1 bg-blue-500 text-white rounded cursor-pointer hover:bg-blue-600 transition-colors"
        >
          Recognize
        </button>
      </div>
    </div>
  )
}