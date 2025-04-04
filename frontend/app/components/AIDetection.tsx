import { useState } from "react";
import Loading from "./Loading";
import * as pdfjsLib from "pdfjs-dist";

pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

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

  const handlePDFUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async () => {
      const typedarray = new Uint8Array(reader.result as ArrayBuffer);
      const pdf = await pdfjsLib.getDocument(typedarray).promise;

      let extractedText = "";
      for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
        const page = await pdf.getPage(pageNum);
        const content = await page.getTextContent();
        const pageText = content.items.map((item: any) => item.str).join(" ");
        extractedText += pageText + "\n";
      }

      setText(extractedText);
    };

    reader.readAsArrayBuffer(file);
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-md max-w-3xl mx-auto">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">
        AI Text Detection
      </h2>

      {/* Upload Button */}
      <div className="flex justify-between items-center mb-3">
        <label
          htmlFor="pdf-upload"
          className="text-sm text-white bg-black hover:bg-gray-800 px-4 py-2 rounded cursor-pointer transition"
        >
          Upload PDF
        </label>
        <input
          type="file"
          id="pdf-upload"
          accept="application/pdf"
          onChange={handlePDFUpload}
          className="hidden"
        />
      </div>

      {/* Textarea */}
      <textarea
        className="w-full p-3 border border-gray-300 rounded mb-4 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
        rows={6}
        placeholder="Enter or paste text to check..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      {/* Submit Button */}
      <button
        className="w-full py-2 text-white bg-blue-500 hover:bg-blue-600 rounded transition"
        onClick={handleDetect}
        disabled={loading}
      >
        {loading ? "Analyzing..." : "Check AI Content"}
      </button>

      {/* Loading Spinner */}
      {loading && (
        <div className="flex justify-center mt-4">
          <Loading />
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="mt-6 p-4 bg-gray-50 border rounded">
          <p className="mb-2">
            <strong>Prediction:</strong> {result?.prediction || "N/A"}
          </p>
          <p className="mb-2">
            <strong>AI Probability:</strong>{" "}
            <span className="text-blue-600 font-medium">
              {result?.ai_probability || "N/A"}
            </span>
          </p>

          {result?.ai_detected_sentences?.length ? (
            <div className="mt-3 bg-red-50 border-l-4 border-red-400 p-3 rounded">
              <h3 className="text-red-600 font-semibold mb-2">
                AI-Generated Sentences
              </h3>
              <ul className="list-disc ml-5 space-y-1 text-gray-700 text-sm">
                {result?.ai_detected_sentences.map((item, idx) => (
                  <li key={idx}>
                    <span className="font-medium">{item.sentence}</span> –{" "}
                    <span className="text-red-500">{item.ai_probability}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="text-green-600 mt-2 text-sm">
              No AI-generated content detected.
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default AIDetection;
