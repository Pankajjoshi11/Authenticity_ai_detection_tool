import { useState } from "react";
import Loading from "./Loading";

const AIDetection = () => {
  const [text, setText] = useState("");
  const [result, setResult] = useState<{
    prediction?: string;
    ai_probability?: string;
    ai_detected_sentences?: { sentence: string; ai_probability: string }[];
  } | null>(null);
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
        className="px-4 py-2 mt-2 text-white bg-blue-500 rounded hover:bg-blue-600"
        onClick={handleDetect}
        disabled={loading}
      >
        {loading ? "Analyzing..." : "Check AI Content"}
      </button>

      {loading && (
        <div className="flex justify-center mt-2">
          <Loading />
        </div>
      )}

      {result && (
        <div className="mt-2 p-2 bg-white border rounded">
          <p>
            <strong>Prediction:</strong> {result?.prediction || "N/A"}
          </p>
          <p>
            <strong>AI Probability:</strong>{" "}
            <span className="text-blue-500 font-semibold">
              {result?.ai_probability || "N/A"}
            </span>
          </p>

          {result?.ai_detected_sentences?.length ? (
            <div className="mt-2 p-2 bg-red-50 border-l-4 border-red-500 rounded">
              <h3 className="text-red-600 font-semibold">
                AI-Generated Sentences
              </h3>
              <ul className="list-disc ml-5 mt-1 text-gray-700">
                {result?.ai_detected_sentences?.map((item, index) => (
                  <li key={index}>
                    <span className="font-medium">{item.sentence}</span> -{" "}
                    <span className="text-red-500">{item.ai_probability}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="text-green-600 mt-2">
              No AI-generated content detected.
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default AIDetection;
