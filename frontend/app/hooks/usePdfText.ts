import { useState, useEffect } from "react";

// 👇 Import the custom Vite-friendly PDF worker setup
import "./pdfWorker";

export const usePdfText = (file: File | null) => {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) return;

    const extractText = async () => {
      setLoading(true);
      setError(null);

      try {
        const pdfjsLib = await import("pdfjs-dist");

        const fileReader = new FileReader();
        fileReader.onload = async function () {
          const typedarray = new Uint8Array(this.result as ArrayBuffer);
          const pdf = await pdfjsLib.getDocument(typedarray).promise;

          let fullText = "";

          for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const content = await page.getTextContent();
            const strings = content.items.map((item: any) => item.str);
            fullText += strings.join(" ") + "\n\n";
          }

          setText(fullText);
          setLoading(false);
        };

        fileReader.readAsArrayBuffer(file);
      } catch (err) {
        console.error("PDF parsing error:", err);
        setError("Failed to parse PDF.");
        setLoading(false);
      }
    };

    extractText();
  }, [file]);

  return { text, loading, error };
};
