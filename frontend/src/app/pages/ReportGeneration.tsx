import { FileText, Download, Send, Save, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router";

interface Scan {
  id: number;
  image_path: string;
  scan_date: string;
}

interface Patient {
  id: number;
  name: string;
  age: number;
  gender: string;
  scans: Scan[];
}

export function ReportGeneration() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>("");
  const [doctorNotes, setDoctorNotes] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    fetch("http://localhost:8000/api/patients/")
      .then((res) => res.json())
      .then((data) => setPatients(data))
      .catch((err) => console.error("Error fetching patients:", err));
  }, []);

  const selectedPatient = patients.find((p) => p.id.toString() === selectedPatientId);
  
  // Get latest scan
  const latestScan = selectedPatient?.scans?.length 
    ? [...selectedPatient.scans].sort((a, b) => new Date(b.scan_date).getTime() - new Date(a.scan_date).getTime())[0]
    : null;

  const handleGenerate = async () => {
    if (!selectedPatient) {
      alert("Please select a patient first.");
      return;
    }

    setIsGenerating(true);
    try {
      const payload = {
        patient_id: selectedPatient.id,
        scan_id: latestScan ? latestScan.id : null,
        doctor_notes: doctorNotes,
      };

      const res = await fetch("http://localhost:8000/api/reports/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        const data = await res.json();
        alert("Report generated successfully!");
        navigate(`/dashboard/patients/${selectedPatient.id}`);
      } else {
        alert("Failed to generate report.");
      }
    } catch (error) {
      console.error("Error generating report:", error);
      alert("An error occurred.");
    } finally {
      setIsGenerating(false);
    }
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
            {/* Patient Selection */}
            <div>
              <h3 className="mb-4">Select Patient</h3>
              <select
                className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-[#0077b6] bg-white text-gray-900"
                value={selectedPatientId}
                onChange={(e) => setSelectedPatientId(e.target.value)}
              >
                <option value="">-- Choose a patient --</option>
                {patients.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} (P-{p.id.toString().padStart(3, "0")})
                  </option>
                ))}
              </select>
            </div>

            {/* Patient Details */}
            {selectedPatient && (
              <div>
                <h3 className="mb-4">Patient Details</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-lg bg-gray-50">
                    <p className="text-sm text-gray-600 mb-1">Patient Name</p>
                    <p className="text-gray-900">{selectedPatient.name}</p>
                  </div>
                  <div className="p-4 rounded-lg bg-gray-50">
                    <p className="text-sm text-gray-600 mb-1">Patient ID</p>
                    <p className="text-gray-900">P-{selectedPatient.id.toString().padStart(3, "0")}</p>
                  </div>
                  <div className="p-4 rounded-lg bg-gray-50">
                    <p className="text-sm text-gray-600 mb-1">Age</p>
                    <p className="text-gray-900">{selectedPatient.age || "N/A"} years</p>
                  </div>
                  <div className="p-4 rounded-lg bg-gray-50">
                    <p className="text-sm text-gray-600 mb-1">Gender</p>
                    <p className="text-gray-900">{selectedPatient.gender || "N/A"}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Scan Image */}
            {selectedPatient && latestScan ? (
              <div>
                <h3 className="mb-4">Latest Uploaded Image</h3>
                <div className="rounded-lg border-2 p-4" style={{ borderColor: "#00b4d8" }}>
                  <div className="bg-gray-100 rounded-lg h-64 flex items-center justify-center overflow-hidden">
                    <img
                      src={`http://localhost:8000${latestScan.image_path}`}
                      alt="Latest scan"
                      className="w-full h-full object-contain"
                    />
                  </div>
                </div>
              </div>
            ) : selectedPatient ? (
              <div>
                <h3 className="mb-4">Uploaded Image</h3>
                <div className="rounded-lg border-2 p-4 border-gray-200">
                  <div className="bg-gray-50 rounded-lg h-32 flex items-center justify-center">
                    <p className="text-gray-500">No scan image found for this patient.</p>
                  </div>
                </div>
              </div>
            ) : null}

            {/* Doctor Notes */}
            {selectedPatient && (
              <div>
                <h3 className="mb-4">Doctor Notes</h3>
                <textarea
                  placeholder="Add your professional assessment and recommendations here..."
                  rows={6}
                  value={doctorNotes}
                  onChange={(e) => setDoctorNotes(e.target.value)}
                  className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-[#0077b6]"
                />
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div className="p-8 border-t border-gray-200 bg-gray-50">
            <div className="flex flex-wrap gap-4">
              <button
                onClick={handleGenerate}
                disabled={!selectedPatient || isGenerating}
                className="flex items-center gap-2 px-6 py-3 rounded-lg text-white transition-all hover:opacity-90 disabled:opacity-50"
                style={{ backgroundColor: "#0077b6" }}
              >
                {isGenerating ? <Loader2 className="w-5 h-5 animate-spin" /> : <Download className="w-5 h-5" />}
                {isGenerating ? "Generating..." : "Generate PDF"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
