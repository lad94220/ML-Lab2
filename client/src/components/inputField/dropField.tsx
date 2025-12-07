import { useRef, useState } from "react";
import Uppy from "@uppy/core";
import { PredictionDisplay } from "./PredictionDisplay";

export const DropField = () => {
  const uppyRef = useRef<Uppy | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<{ name: string; size: number; preview?: string } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<any>(null);

  if (!uppyRef.current) {
    uppyRef.current = new Uppy({
      restrictions: {
        maxNumberOfFiles: 1,
        allowedFileTypes: ['image/*']
      }
    });

    uppyRef.current.on('file-added', (uppyFile) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        setFile({
          name: uppyFile.name,
          size: uppyFile.size as number,
          preview: e.target?.result as string
        });
      };
      reader.readAsDataURL(uppyFile.data as Blob);
    });
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('image/')) {
      uppyRef.current?.addFile({
        name: droppedFile.name,
        type: droppedFile.type,
        data: droppedFile,
      });
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      uppyRef.current?.addFile({
        name: selectedFile.name,
        type: selectedFile.type,
        data: selectedFile,
      });
    }
  };

  const handleRemove = () => {
    uppyRef.current?.cancelAll();
    setFile(null);
    setPrediction(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleRecognize = async () => {
    if (!uppyRef.current) return;

    const files = uppyRef.current.getFiles();
    if (files.length === 0) return;

    const formData = new FormData();
    formData.append('file', files[0].data as Blob, files[0].name);

    setIsLoading(true);
    setPrediction(null);

    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URI}/predict`, {
        method: 'POST',
        body: formData,
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
    <div className="space-y-6">
      <div className="flex items-center justify-center text-2xl font-semibold">
        INPUT FILES
      </div>
      
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`min-h-80 flex items-center justify-center border-dashed border-2 rounded-lg cursor-pointer transition-all duration-200 
          ${isDragging 
            ? 'border-blue-500 bg-blue-50' 
            : 'border-gray-400 bg-white/50 hover:bg-white/70'
          }`}
      >
        {!file ? (
          <div className="h-full flex flex-col items-center justify-center p-8 text-center">
            <svg className="w-16 h-16 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <p className="text-lg text-gray-600 mb-2">Drop an image here or click to select</p>
            <p className="text-sm text-gray-400">Supports: JPG, PNG, GIF</p>
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center p-8">
            {file.preview && (
              <img src={file.preview} alt={file.name} className="max-h-48 mb-4 rounded shadow-md" />
            )}
            <p className="text-gray-700 font-medium">{file.name}</p>
            <p className="text-gray-500 text-sm">{(file.size / 1024).toFixed(2)} KB</p>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleRemove();
              }}
              className="mt-4 text-red-500 hover:text-red-700 underline"
            >
              Remove file
            </button>
          </div>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="hidden"
      />

      <div className="flex justify-end">
        <button
          className="px-6 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors font-medium disabled:bg-gray-400 disabled:cursor-not-allowed"
          disabled={!file || isLoading}
          onClick={handleRecognize}
        >
          {isLoading ? 'Processing...' : 'Recognize'}
        </button>
      </div>

      <PredictionDisplay prediction={prediction} />
    </div>
  )
}