// Correct way for TypeScript
import { useState } from "react";
import * as pdfjsLib from "pdfjs-dist";
// Use legacy for Vite compatibility

// @ts-ignore is used below to silence TypeScript's complaint about unknown worker types
// This is safe for client-side usage in a Vite project with a custom worker
// You can also use vite-plugin-pdfjs if needed for advanced usage
// @ts-ignore
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

const Summarization = () => {
  const [text, setText] = useState("");
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSummarize = async () => {
    if (!text.trim()) return alert("Please enter some text!");

    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:5000/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      setSummary(data.summary);
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to summarize text.");
    }
    setLoading(false);
  };

  const handlePDFUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async function () {
      const typedArray = new Uint8Array(reader.result as ArrayBuffer);
      const pdf = await pdfjsLib.getDocument({ data: typedArray }).promise;
      let textContent = "";

      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        const pageText = content.items.map((item: any) => item.str).join(" ");
        textContent += pageText + " ";
      }

      setText(textContent.trim());
    };
    reader.readAsArrayBuffer(file);
  };

  return (
    <div className="p-4 bg-gray-100 rounded-lg shadow">
      <h2 className="text-lg font-bold text-gray-800">Summarization</h2>

      <div className="flex items-center gap-2 mt-2">
        <button
          className="px-4 py-2 text-white bg-black rounded hover:bg-gray-800"
          onClick={handleSummarize}
          disabled={loading}
        >
          {loading ? "Summarizing..." : "Summarize"}
        </button>

        <label className="px-4 py-2 text-sm text-black border border-blue-500 rounded cursor-pointer hover:bg-blue-100">
          Upload PDF
          <input
            type="file"
            accept="application/pdf"
            className="hidden"
            onChange={handlePDFUpload}
          />
        </label>
      </div>

      <textarea
        className="w-full p-2 mt-3 border rounded"
        rows={4}
        placeholder="Enter text or upload PDF..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      {summary && (
        <p className="mt-3 p-3 bg-white border rounded text-gray-800">
          {summary}
        </p>
      )}
    </div>
  );
};

export default Summarization;
