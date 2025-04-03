import { useState } from "react";
import type { Route } from "./+types/home";
import Summarization from "../components/Summarization";
import AIDetection from "../components/AIDetection";
import Report from "../components/Report";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "AI Processing App" },
    { name: "description", content: "Summarization and AI Detection in React" },
  ];
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<"summarization" | "ai-detection">(
    "summarization"
  );
  const [report, setReport] = useState(null);

  return (
    <div className="w-full min-h-screen bg-white text-gray-900 flex flex-col items-center">
      {/* 🔹 Navbar */}
      <header className="w-full max-w-6xl flex justify-between items-center py-6 px-8">
        <h1 className="text-2xl font-semibold tracking-wide">AI Processing</h1>
      </header>

      {/* 🔹 Hero Section */}
      <section className="w-full max-w-6xl text-center mt-12 mb-10">
        <h2 className="text-5xl font-bold tracking-tight leading-snug">
          AI-Powered Text Processing
        </h2>
        <p className="text-lg text-gray-600 mt-4">
          Summarize and analyze text with advanced AI models.
        </p>
      </section>

      {/* 🔹 Toggle Buttons */}
      <div className="flex gap-6 border-b border-gray-300 pb-4">
        <button
          className={`px-6 py-2 text-lg font-medium rounded-full transition ${
            activeTab === "summarization"
              ? "bg-black text-white shadow-lg"
              : "bg-gray-200 text-gray-700 hover:bg-gray-300"
          }`}
          onClick={() => setActiveTab("summarization")}
        >
          Summarization
        </button>
        <button
          className={`px-6 py-2 text-lg font-medium rounded-full transition ${
            activeTab === "ai-detection"
              ? "bg-black text-white shadow-lg"
              : "bg-gray-200 text-gray-700 hover:bg-gray-300"
          }`}
          onClick={() => setActiveTab("ai-detection")}
        >
          AI Detection
        </button>
      </div>

      {/* 🔹 Content Section */}
      <div className="w-full max-w-3xl mt-12 px-6">
        {activeTab === "summarization" ? <Summarization /> : <AIDetection />}
      </div>

      {/* 🔹 Report Section (Only If Available) */}
      {report && (
        <div className="w-full max-w-3xl mt-16 px-6">
          <Report report={report} />
        </div>
      )}

      {/* 🔹 Footer */}
      <footer className="w-full max-w-6xl text-center mt-20 py-6 text-gray-500 text-sm">
        &copy; {new Date().getFullYear()} AI Processing App. All rights reserved.
      </footer>
    </div>
  );
}
