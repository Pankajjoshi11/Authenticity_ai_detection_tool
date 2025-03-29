import { useState } from "react";
import Loading from "./Loading";

const AIDetection = () => {
  const [text, setText] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleDetect = async () => {
    if (!text.trim()) return alert("Please enter some text!");

    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:5000/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
      alert("AI Detection failed.");
    }
    setLoading(false);
  };

  return (
    <div className="p-4 bg-gray-100 rounded-lg shadow">
      <h2 className="text-lg font-bold">AI Text Detection</h2>
      <textarea
        className="w-full p-2 mt-2 border rounded"
        rows={4}
        placeholder="Enter text to check..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <button
        className="px-4 py-2 mt-2 text-white bg-red-500 rounded hover:bg-red-600"
        onClick={handleDetect}
        disabled={loading}
      >
        {loading ? "Detecting..." : "Check AI Content"}
      </button>

      {loading && <Loading />}

      {result && (
        <div className="mt-4 p-2 bg-white border rounded">
          <p><strong>Prediction:</strong> {result.prediction}</p>
          <p><strong>AI Probability:</strong> {result.ai_probability}</p>

          {/* AI-Generated Sentences Section */}
          {result.ai_detected_sentences && result.ai_detected_sentences.length > 0 && (
            <div className="mt-4 p-2 bg-red-100 border border-red-400 rounded">
              <h3 className="font-semibold text-red-600">AI-Generated Sentences</h3>
              <ul className="list-disc ml-4">
                {result.ai_detected_sentences.map((item: any, index: number) => (
                  <li key={index}>
                    <span className="font-medium">{item.sentence}</span> - 
                    <span className="text-red-500"> {item.ai_probability}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Human-Written Sentences Section */}
          {result.ai_detected_sentences && result.ai_detected_sentences.length < text.split(". ").length && (
            <div className="mt-4 p-2 bg-green-100 border border-green-400 rounded">
              <h3 className="font-semibold text-green-600">Human-Written Sentences</h3>
              <ul className="list-disc ml-4">
                {text.split(". ").map((sentence, index) => {
                  const isAISentence = result.ai_detected_sentences.some(
                    (aiItem: any) => aiItem.sentence === sentence
                  );
                  return !isAISentence && (
                    <li key={index} className="text-green-700">{sentence}</li>
                  );
                })}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AIDetection;
