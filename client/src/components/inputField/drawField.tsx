import { useRef } from "react";
import { ReactSketchCanvas, type ReactSketchCanvasRef } from "react-sketch-canvas";

export const DrawField = () => {
  const canvasRef = useRef<ReactSketchCanvasRef>(null);

  const handleClear = () => {
    canvasRef.current?.clearCanvas();
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-center text-2xl font-semibold pb-4">
        DRAW 
      </div>
      <ReactSketchCanvas
        ref={canvasRef}
        strokeWidth={4}
        height="96px"
        strokeColor="black"
        className="min-h-96"
      />
      <div className="flex justify-between">
        <button 
          onClick={handleClear}
          className="px-4 py-1 bg-red-500 cursor-pointer text-white rounded hover:bg-red-600 transition-colors"
        >
          Clear
        </button>
        <button
          className="px-6 py-2 bg-blue-500 text-white rounded cursor-pointer hover:bg-blue-600 transition-colors"
        >
          Recognize
        </button>
      </div>
    </div>
  )
}