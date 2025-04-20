import { useState } from "react";

const CheckGrammar = () => {
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<{
    status: string;
    incorrect_sentences: string[];
    ml_warnings?: string[];
  } | null>(null);
  const [loading, setLoading] = useState(false);

  const handleCheckGrammar = async () => {
    if (!text.trim() && !file) {
      return alert("Please enter some text or upload a PDF!");
    }

    setLoading(true);
    setResult(null); // Clear previous results

    try {
      let response;

      if (file) {
        const formData = new FormData();
        formData.append("file", file);

        response = await fetch("http://127.0.0.1:5000/check-grammar", {
          method: "POST",
          body: formData,
        });
      } else {
        response = await fetch("http://127.0.0.1:5000/check-grammar", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
      }

      const data = await response.json();

      if (response.ok) {
        setResult(data);
      } else {
        alert(data.error || "Failed to check grammar.");
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to check grammar.");
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
      <h2 className="text-lg font-bold">Grammar Check</h2>

      <textarea
        className="w-full p-2 mt-2 border rounded"
        rows={4}
        placeholder="Enter text to check grammar..."
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
        onClick={handleCheckGrammar}
        disabled={loading}
      >
        {loading ? "Checking..." : "Check Grammar"}
      </button>

      {result && (
        <div className="mt-4 p-3 bg-white border rounded">
          <h3 className="text-md font-semibold">{result.status}</h3>
          {result.incorrect_sentences.length > 0 ? (
            <div>
              <p className="mt-2 font-medium">Incorrect Sentences:</p>
              <ul className="list-disc pl-5 mt-1">
                {result.incorrect_sentences.map((sentence, index) => (
                  <li key={index} className="text-gray-700">{sentence}</li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="mt-2 text-gray-700">No grammatical errors found.</p>
          )}
          {result.ml_warnings && result.ml_warnings.length > 0 && (
            <div className="mt-3">
              <p className="font-medium text-yellow-700">Warnings:</p>
              <ul className="list-disc pl-5 mt-1 text-yellow-700">
                {result.ml_warnings.map((warning, index) => (
                  <li key={index}>{warning}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default CheckGrammar;