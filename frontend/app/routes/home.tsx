import type { Route } from "./+types/home";
import { Welcome } from "../welcome/welcome";
import Summarization from "../components/Summarization";
import AIDetection from "../components/AIDetection";
import { useState } from "react";
import Report from "../components/Report";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "AI Processing App" },
    { name: "description", content: "Summarization and AI Detection in React" },
  ];
}

export default function Home() {
  const [report, setReport] = useState(null);

  return (
    <div className="p-6">
      <Welcome />
      <h1 className="text-2xl font-bold text-center mt-4">AI Processing App</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        <Summarization />
        <AIDetection />
      </div>
      {report && <Report report={report} />}
    </div>
  );
}
