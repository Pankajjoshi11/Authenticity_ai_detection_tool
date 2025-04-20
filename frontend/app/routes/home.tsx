import { useState } from "react";
import type { Route } from "./+types/home";
import Summarization from "../components/Summarization";
import AIDetection from "../components/AIDetection";
import CheckGrammar from "~/components/CheckGrammar";
import CheckPlagiarism from "../components/CheckPlagiarism";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "AI Processing App" },
    {
      name: "description",
      content: "Summarize, analyze, check grammar, and detect plagiarism with advanced AI models.",
    },
  ];
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<"summarization" | "ai-detection" | "grammar-check" | "plagiarism-check">(
    "summarization"
  );

  return (
    <div className="w-full min-h-screen bg-white text-gray-900 flex flex-col items-center">
      {/* 🔹 Navbar */}
      <header className="w-full max-w-6xl flex justify-between items-center py-6 px-8" role="banner">
        <h1 className="text-2xl font-semibold tracking-wide">AI Processing</h1>
      </header>

      {/* 🔹 Hero Section */}
      <section className="w-full max-w-6xl text-center mt-12 mb-10" aria-labelledby="hero-title">
        <h2 id="hero-title" className="text-5xl font-bold tracking-tight leading-snug">
          AI-Powered Text Processing
        </h2>
        <p className="text-lg text-gray-600 mt-4">
          Summarize, analyze, check grammar, and detect plagiarism with advanced AI models.
        </p>
      </section>

      {/* 🔹 Tab Navigation */}
      <nav className="flex flex-wrap gap-4 border-b border-gray-300 pb-4" role="tablist">
        {[
          { id: "summarization", label: "Summarization" },
          { id: "ai-detection", label: "AI Detection" },
          { id: "grammar-check", label: "Grammar Check" },
          { id: "plagiarism-check", label: "Plagiarism Check" },
        ].map((tab) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls={`${tab.id}-panel`}
            className={`px-6 py-2 text-lg font-medium rounded-full transition ${
              activeTab === tab.id
                ? "bg-black text-white shadow-lg"
                : "bg-gray-200 text-gray-700 hover:bg-gray-300"
            }`}
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {/* 🔹 Content Section */}
      <div className="w-full max-w-3xl mt-12 px-6">
        <div
          role="tabpanel"
          id={`${activeTab}-panel`}
          aria-labelledby={`${activeTab}-tab`}
        >
          {activeTab === "summarization" && <Summarization />}
          {activeTab === "ai-detection" && <AIDetection />}
          {activeTab === "grammar-check" && <CheckGrammar />}
          {activeTab === "plagiarism-check" && <CheckPlagiarism />}
        </div>
      </div>

      {/* 🔹 Footer */}
      <footer className="w-full max-w-6xl text-center mt-20 py-6 text-gray-500 text-sm">
        © 2025 AI Processing App
      </footer>
    </div>
  );
}