import Dropzone from "dropzone";
import "dropzone/dist/dropzone.css";
import { useEffect, useRef } from "react";

Dropzone.autoDiscover = false;

export const DropField = () => {
  const dropzoneRef = useRef<HTMLFormElement>(null);

  useEffect(() => {
    if (!dropzoneRef.current) return;

    const dropzone = new Dropzone(dropzoneRef.current, {
      url: "/file/post",
      maxFiles: 1,
      acceptedFiles: "image/*",
      addRemoveLinks: true,
      dictDefaultMessage: "Drop files here or click to upload",
    }); 

    return () => {
      dropzone.destroy();
    }
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-center text-2xl font-semibold pb-8">
        INPUT FILES
      </div>
      <form 
        ref={dropzoneRef} 
        className="flex items-center justify-center border-dashed border-2 border-gray-400 rounded p-8 bg-white/50 hover:bg-white/70 transition-colors cursor-pointer min-h-96"
        >
        <div className="dz-message">
          Drop files here or click to upload
        </div>
      </form>
      <div className="flex justify-between relative mb-8">
        <button
          className="px-4 py-1 absolute right-0 bg-blue-500 text-white rounded cursor-pointer hover:bg-blue-600 transition-colors"
        >
          Recognize
        </button>
      </div>
    </div>
  )
}