const Report = ({ report }: { report: any }) => {
    if (!report || report.ai_detected_sentences.length === 0) {
      return <p className="p-4 text-center">No AI-generated sentences detected.</p>;
    }
  
    return (
      <div className="p-4 bg-white rounded-lg shadow">
        <h3 className="text-lg font-bold">AI Detection Report</h3>
        <ul className="mt-2">
          {report.ai_detected_sentences.map((item: any, index: number) => (
            <li key={index} className="p-2 border-b">
              <p><strong>Sentence:</strong> {item.sentence}</p>
              <p><strong>AI Probability:</strong> {item.ai_probability}</p>
            </li>
          ))}
        </ul>
      </div>
    );
  };
  
  export default Report;
  