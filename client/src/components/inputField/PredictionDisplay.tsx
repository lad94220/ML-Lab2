interface PredictionResult {
  raw?: { prediction: number; probabilities: number[] };
  edges?: { prediction: number; probabilities: number[] };
  pca?: { prediction: number; probabilities: number[] };
  error?: string;
}

interface PredictionDisplayProps {
  prediction: PredictionResult | null;
}

const ModelPrediction = ({ 
  title, 
  prediction, 
  probabilities, 
  color 
}: { 
  title: string; 
  prediction: number; 
  probabilities: number[]; 
  color: string;
}) => {
  return (
    <div className="p-4 border rounded-lg bg-white shadow-sm">
      <div className="font-medium text-gray-700 mb-2">{title}</div>
      <div className={`text-4xl font-bold ${color} mb-4`}>{prediction}</div>
      <div className="text-xs text-gray-500 mb-2">Probabilities:</div>
      <div className="grid grid-cols-5 gap-1 text-xs">
        {probabilities.map((prob, idx) => (
          <div key={idx} className="text-center">
            <div className="font-semibold text-gray-700">{idx}</div>
            <div className="text-gray-500">{(prob * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export const PredictionDisplay = ({ prediction }: PredictionDisplayProps) => {
  if (!prediction) return null;

  return (
    <div className="mt-4 p-6 from-gray-50 to-gray-100 rounded-lg shadow-lg">
      {prediction.error ? (
        <div className="text-red-500 font-medium">Error: {prediction.error}</div>
      ) : (
        <div className="space-y-6">
          <h3 className="font-bold text-xl text-gray-800">Predictions:</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {prediction.raw && (
              <ModelPrediction
                title="Raw Model"
                prediction={prediction.raw.prediction}
                probabilities={prediction.raw.probabilities}
                color="text-blue-600"
              />
            )}
            {prediction.edges && (
              <ModelPrediction
                title="Edges Model"
                prediction={prediction.edges.prediction}
                probabilities={prediction.edges.probabilities}
                color="text-green-600"
              />
            )}
            {prediction.pca && (
              <ModelPrediction
                title="PCA Model"
                prediction={prediction.pca.prediction}
                probabilities={prediction.pca.probabilities}
                color="text-purple-600"
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
};
