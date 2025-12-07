import { useState } from "react";
import { DropField } from "./dropField";
import { DrawField } from "./drawField";

type Field = "drop" | "draw";

export const InputField = () => {
  const [field, setField] = useState<Field>("drop");

  const handleChangeSwitch = (newField: Field) => {
    setField(newField);
  }

  return (
    <div className="space-y-8 w-3xl p-8 rounded-lg shadow-lg relative bg-white/30 mt-20">
      <div className="absolute left-0 -top-13 flex flex-row gap-2">
        <button 
          className={`w-26 h-9 cursor-pointer rounded-2xl border transition-all
                      ${field === "drop" 
                        ? "bg-blue-400 text-white border-blue-400" 
                        : "bg-white/30"}`}
          onClick={() => handleChangeSwitch("drop")}
        >
          Input File
        </button>
        <button 
          className={`w-26 h-9 cursor-pointer rounded-2xl border transition-all
                      ${field === "draw" 
                        ? "bg-blue-400 text-white border-blue-400" 
                        : "bg-white/30"}`}
          onClick={() => handleChangeSwitch("draw")}
        >
          Draw
        </button>
      </div>
      
      {field === "drop" ? <DropField /> : <DrawField />}      
    </div>
  )
}