import { useState } from "react";

const Summarization = () => {
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSummarize = async () => {
    if (!text.trim() && !file) {
      return alert("Please enter some text or upload a PDF!");
    }

    setLoading(true);
    try {
      let response;

      if (file) {
        const formData = new FormData();
        formData.append("file", file);

        response = await fetch("http://127.0.0.1:5000/summarize", {
          method: "POST",
          body: formData,
        });
      } else {
        response = await fetch("http://127.0.0.1:5000/summarize", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
      }

      const data = await response.json();
      setSummary(data.summary);
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to summarize text.");
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile && selectedFile.type !== "application/pdf") {
      alert("Only PDF files are allowed.");
      return;
    }
    setFile(selectedFile || null);
    setText(""); // Clear text if file is selected
  };

  return (
    <div className="p-4 bg-gray-100 rounded-lg shadow">
      <h2 className="text-lg font-bold">Summarization</h2>

      <textarea
        className="w-full p-2 mt-2 border rounded"
        rows={4}
        placeholder="Enter text to summarize..."
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          setFile(null); // Clear file if text is typed
        }}
        disabled={!!file}
      />

      <div className="mt-2">
        <input
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-700
            file:mr-4 file:py-2 file:px-4
            file:rounded file:border-0
            file:text-sm file:font-semibold
            file:bg-blue-50 file:text-blue-700
            hover:file:bg-blue-100"
        />
      </div>

      <button
        className="px-4 py-2 mt-3 text-white bg-blue-500 rounded hover:bg-blue-600"
        onClick={handleSummarize}
        disabled={loading}
      >
        {loading ? "Summarizing..." : "Summarize"}
      </button>

      {summary && (
        <p className="mt-4 p-3 bg-white border rounded">{summary}</p>
      )}
    </div>
  );
};

export default Summarization;
