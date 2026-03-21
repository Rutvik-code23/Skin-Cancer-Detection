import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router";
import { ArrowLeft, User, Phone, Mail, MapPin, Trash2 } from "lucide-react";

interface Scan {
  id: number;
  date: string;
  image_path: string;
  diagnosis: string;
  confidence: number;
  risk_level: string;
  scan_date: string;
}

interface Report {
  id: number;
  pdf_path: string;
  created_at: string;
  doctor_notes: string;
}

interface Patient {
  id: number;
  name: string;
  age: number;
  gender: string;
  phone: string;
  email: string;
  address: string;
  notes: string;
  created_at: string;
  scans: Scan[];
  reports: Report[];
}

export function PatientRecord() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [patient, setPatient] = useState<Patient | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [activeTab, setActiveTab] = useState<"scans" | "reports">("scans");

  useEffect(() => {
    if (id) {
      fetch(`http://localhost:8000/api/patients/${id}`)
        .then((res) => res.json())
        .then((data) => setPatient(data))
        .catch((err) => console.error("Error fetching patient:", err));
    }
  }, [id]);

  if (!patient) {
    return <div className="p-8">Loading patient details...</div>;
  }

  // Compute overall status from scans
  // If there are no scans, we just default to "No Data"
  // Otherwise, we take the most recent scan's diagnosis and risk level.
  let overallDiagnosis = "No Scans Yet";
  let overallRisk = "Low";

  if (patient.scans && patient.scans.length > 0) {
    // Assuming scans are appended, the last one is the latest. Or sort by date.
    // Let's sort by date descending to be safe.
    const sortedScans = [...patient.scans].sort((a, b) =>
      new Date(b.scan_date).getTime() - new Date(a.scan_date).getTime()
    );
    const latestScan = sortedScans[0];
    overallDiagnosis = latestScan.diagnosis;
    overallRisk = latestScan.risk_level;
  }

  const handleDelete = async () => {
    if (window.confirm("Are you sure you want to permanently delete this user and all their scans?")) {
      setIsDeleting(true);
      try {
        const response = await fetch(`http://localhost:8000/api/patients/${id}`, {
          method: "DELETE",
        });

        if (response.ok) {
          navigate("/dashboard/patients");
        } else {
          console.error("Failed to delete patient");
          setIsDeleting(false);
        }
      } catch (error) {
        console.error("Error deleting patient:", error);
        setIsDeleting(false);
      }
    }
  };

  return (
    <div className="p-8">
      <button
        onClick={() => navigate("/dashboard/patients")}
        className="flex items-center gap-2 text-gray-600 hover:text-gray-900 mb-6"
      >
        <ArrowLeft className="w-5 h-5" />
        Back to Patients
      </button>

      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="mb-2">Patient Record</h1>
          <p className="text-gray-600">Complete patient history and medical records</p>
        </div>

        <button
          onClick={handleDelete}
          disabled={isDeleting}
          className="flex items-center gap-2 px-4 py-2 rounded-lg border border-red-500 text-red-500 hover:bg-red-50 transition-colors disabled:opacity-50"
        >
          <Trash2 className="w-5 h-5" />
          {isDeleting ? "Deleting..." : "Delete User"}
        </button>
      </div>

      {/* Patient Details Card */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-8 mb-6">
        <div className="flex items-start gap-6">
          <div
            className="w-20 h-20 rounded-full flex items-center justify-center text-2xl text-white"
            style={{ backgroundColor: "#90e0ef" }}
          >
            <User className="w-10 h-10" />
          </div>
          <div className="flex-1">
            <div className="flex items-center justify-between mb-1">
              <h2>{patient.name}</h2>
              <span
                className="px-4 py-1.5 rounded-full text-sm font-medium"
                style={{
                  backgroundColor:
                    overallRisk === "Low"
                      ? "#dcfce7" // green-100
                      : overallRisk === "Medium"
                        ? "#fef08a" // yellow-200
                        : "#fee2e2", // red-100
                  color: 
                    overallRisk === "Low"
                      ? "#166534" // green-800
                      : overallRisk === "Medium"
                        ? "#854d0e" // yellow-800
                        : "#991b1b", // red-800
                }}
              >
                {overallDiagnosis}
              </span>
            </div>
            <p className="text-gray-600 mb-4">Patient ID: P-{patient.id.toString().padStart(3, '0')}</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center gap-2 text-gray-700">
                <User className="w-5 h-5 text-gray-400" />
                {patient.age || "N/A"} years, {patient.gender || "Not specified"}
              </div>
              <div className="flex items-center gap-2 text-gray-700">
                <Phone className="w-5 h-5 text-gray-400" />
                {patient.phone || "No phone"}
              </div>
              <div className="flex items-center gap-2 text-gray-700">
                <Mail className="w-5 h-5 text-gray-400" />
                {patient.email || "No email"}
              </div>
              <div className="flex items-center gap-2 text-gray-700">
                <MapPin className="w-5 h-5 text-gray-400" />
                {patient.address || "No address"}
              </div>
            </div>
          </div>
        </div>
        {patient.notes && (
          <div className="mt-6 p-4 rounded-lg" style={{ backgroundColor: "#caf0f8" }}>
            <p className="text-sm text-gray-600 mb-1">Medical Notes</p>
            <p className="text-gray-900">{patient.notes}</p>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
        <div className="border-b border-gray-200">
          <div className="flex">
            <button 
              onClick={() => setActiveTab("scans")}
              className={`px-6 py-4 border-b-2 transition-colors ${activeTab === "scans" ? "border-[#0077b6] text-[#0077b6]" : "border-transparent text-gray-600 hover:text-gray-900"}`}
            >
              Past Scans
            </button>
            <button 
              onClick={() => setActiveTab("reports")}
              className={`px-6 py-4 border-b-2 transition-colors ${activeTab === "reports" ? "border-[#0077b6] text-[#0077b6]" : "border-transparent text-gray-600 hover:text-gray-900"}`}
            >
              Reports History
            </button>
          </div>
        </div>

        {activeTab === "scans" && (
          <div className="p-6">
            <h3 className="mb-4">Past Scans</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-[#caf0f8]">
                <tr>
                  <th className="px-6 py-3 text-left text-gray-700">Date</th>
                  <th className="px-6 py-3 text-left text-gray-700">Diagnosis</th>
                  <th className="px-6 py-3 text-left text-gray-700">Confidence</th>
                  <th className="px-6 py-3 text-left text-gray-700">Risk Level</th>
                  <th className="px-6 py-3 text-left text-gray-700">Image</th>
                  <th className="px-6 py-3 text-left text-gray-700">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {patient.scans && patient.scans.map((scan) => (
                  <tr key={scan.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 text-gray-900">
                      {new Date(scan.scan_date).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 text-gray-900">{scan.diagnosis}</td>
                    <td className="px-6 py-4 text-gray-600">{scan.confidence.toFixed(1)}%</td>
                    <td className="px-6 py-4">
                      <span
                        className="px-3 py-1 rounded-full text-sm font-medium"
                        style={{
                          backgroundColor:
                            scan.risk_level === "Low"
                              ? "#dcfce7"
                              : scan.risk_level === "Medium"
                                ? "#fef08a"
                                : "#fee2e2",
                          color: 
                            scan.risk_level === "Low"
                              ? "#166534"
                              : scan.risk_level === "Medium"
                                ? "#854d0e"
                                : "#991b1b",
                        }}
                      >
                        {scan.risk_level}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="w-16 h-16 rounded overflow-hidden">
                        <img
                          src={`http://localhost:8000${scan.image_path}`}
                          alt="Scan"
                          className="w-full h-full object-cover"
                        />
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <button
                        className="px-4 py-2 rounded-lg text-white transition-all hover:opacity-90"
                        style={{ backgroundColor: "#0077b6" }}
                      >
                        View Report
                      </button>
                    </td>
                  </tr>
                ))}
                {(!patient.scans || patient.scans.length === 0) && (
                  <tr>
                    <td colSpan={6} className="px-6 py-4 text-center text-gray-500">
                      No scans available.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
        )}

        {/* Reports History Table */}
        {activeTab === "reports" && (
          <div className="p-6">
            <h3 className="mb-4">Reports History</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-[#caf0f8]">
                  <tr>
                    <th className="px-6 py-3 text-left text-gray-700">Date Generated</th>
                    <th className="px-6 py-3 text-left text-gray-700">Doctor Notes</th>
                    <th className="px-6 py-3 text-left text-gray-700">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {patient.reports && patient.reports.map((report) => (
                    <tr key={report.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 text-gray-900">
                        {new Date(report.created_at).toLocaleDateString()}
                      </td>
                      <td className="px-6 py-4 text-gray-600 max-w-md truncate">
                        {report.doctor_notes || "No notes"}
                      </td>
                      <td className="px-6 py-4">
                        <button
                          onClick={() => window.open(`http://localhost:8000${report.pdf_path}`, "_blank")}
                          className="px-4 py-2 rounded-lg text-white transition-all hover:opacity-90"
                          style={{ backgroundColor: "#0077b6" }}
                        >
                          View PDF
                        </button>
                      </td>
                    </tr>
                  ))}
                  {(!patient.reports || patient.reports.length === 0) && (
                    <tr>
                      <td colSpan={3} className="px-6 py-4 text-center text-gray-500">
                        No reports generated yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
