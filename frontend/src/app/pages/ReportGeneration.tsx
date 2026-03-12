import { FileText, Download, Send, Save } from "lucide-react";

export function ReportGeneration() {
  const sampleReport = {
    patient: {
      name: "Michael Chen",
      age: 45,
      gender: "Male",
      id: "P-2024-1523",
    },
    scan: {
      date: "2026-03-11",
      prediction: "Melanoma",
      confidence: 87.5,
      riskLevel: "High" as const,
    },
    doctorNotes: "",
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="mb-2">Report Generation</h1>
        <p className="text-gray-600">Generate and send patient reports</p>
      </div>

      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
          {/* Report Header */}
          <div className="p-8 border-b border-gray-200" style={{ backgroundColor: "#caf0f8" }}>
            <div className="flex items-center gap-3 mb-4">
              <FileText className="w-8 h-8" style={{ color: "#0077b6" }} />
              <h2 style={{ color: "#0077b6" }}>Skin Cancer Screening Report</h2>
            </div>
            <p className="text-gray-700">DermAI Screening Tool - Medical Report</p>
          </div>

          {/* Report Content */}
          <div className="p-8 space-y-8">
            {/* Patient Details */}
            <div>
              <h3 className="mb-4">Patient Details</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 rounded-lg bg-gray-50">
                  <p className="text-sm text-gray-600 mb-1">Patient Name</p>
                  <p className="text-gray-900">{sampleReport.patient.name}</p>
                </div>
                <div className="p-4 rounded-lg bg-gray-50">
                  <p className="text-sm text-gray-600 mb-1">Patient ID</p>
                  <p className="text-gray-900">{sampleReport.patient.id}</p>
                </div>
                <div className="p-4 rounded-lg bg-gray-50">
                  <p className="text-sm text-gray-600 mb-1">Age</p>
                  <p className="text-gray-900">{sampleReport.patient.age} years</p>
                </div>
                <div className="p-4 rounded-lg bg-gray-50">
                  <p className="text-sm text-gray-600 mb-1">Gender</p>
                  <p className="text-gray-900">{sampleReport.patient.gender}</p>
                </div>
              </div>
            </div>

            {/* Scan Image */}
            <div>
              <h3 className="mb-4">Uploaded Image</h3>
              <div className="rounded-lg border-2 p-4" style={{ borderColor: "#00b4d8" }}>
                <div className="bg-gray-100 rounded-lg h-64 flex items-center justify-center">
                  <p className="text-gray-500">Lesion image placeholder</p>
                </div>
              </div>
            </div>

            {/* AI Prediction */}
            <div>
              <h3 className="mb-4">AI Analysis Results</h3>
              <div className="space-y-4">
                <div className="p-4 rounded-lg" style={{ backgroundColor: "#caf0f8" }}>
                  <p className="text-sm text-gray-600 mb-1">Scan Date</p>
                  <p className="text-gray-900">{sampleReport.scan.date}</p>
                </div>
                <div className="p-4 rounded-lg" style={{ backgroundColor: "#caf0f8" }}>
                  <p className="text-sm text-gray-600 mb-1">AI Prediction</p>
                  <p className="text-xl" style={{ color: "#0077b6" }}>
                    {sampleReport.scan.prediction}
                  </p>
                </div>
                <div className="p-4 rounded-lg" style={{ backgroundColor: "#caf0f8" }}>
                  <p className="text-sm text-gray-600 mb-1">Confidence Score</p>
                  <p className="text-xl" style={{ color: "#0077b6" }}>
                    {sampleReport.scan.confidence}%
                  </p>
                </div>
                <div className="p-4 rounded-lg" style={{ backgroundColor: "#caf0f8" }}>
                  <p className="text-sm text-gray-600 mb-2">Risk Assessment</p>
                  <span
                    className="inline-block px-4 py-2 rounded-full"
                    style={{
                      backgroundColor: "#0077b6",
                      color: "white",
                    }}
                  >
                    {sampleReport.scan.riskLevel} Risk
                  </span>
                </div>
              </div>
            </div>

            {/* Doctor Notes */}
            <div>
              <h3 className="mb-4">Doctor Notes</h3>
              <textarea
                placeholder="Add your professional assessment and recommendations here..."
                rows={6}
                className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-[#0077b6]"
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="p-8 border-t border-gray-200 bg-gray-50">
            <div className="flex flex-wrap gap-4">
              <button
                className="flex items-center gap-2 px-6 py-3 rounded-lg text-white transition-all hover:opacity-90"
                style={{ backgroundColor: "#0077b6" }}
              >
                <Download className="w-5 h-5" />
                Generate PDF
              </button>
              <button
                className="flex items-center gap-2 px-6 py-3 rounded-lg text-white transition-all hover:opacity-90"
                style={{ backgroundColor: "#00b4d8" }}
              >
                <Send className="w-5 h-5" />
                Send to Patient
              </button>
              <button
                className="flex items-center gap-2 px-6 py-3 rounded-lg border border-gray-300 text-gray-700 hover:bg-white transition-colors"
              >
                <Save className="w-5 h-5" />
                Save Report
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
