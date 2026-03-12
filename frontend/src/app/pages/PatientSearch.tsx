import { useState } from "react";
import { Search, Eye, Upload, FileText } from "lucide-react";
import { Link } from "react-router";

export function PatientSearch() {
  const [searchTerm, setSearchTerm] = useState("");
  const [filterType, setFilterType] = useState("all");

  const patients = [
    { id: "P-001", name: "Sarah Johnson", age: 34, lastScan: "2026-03-11", diagnosis: "Benign Nevus", risk: "Low" },
    { id: "P-002", name: "Michael Chen", age: 45, lastScan: "2026-03-11", diagnosis: "Melanoma", risk: "High" },
    { id: "P-003", name: "Emily Davis", age: 28, lastScan: "2026-03-10", diagnosis: "Seborrheic Keratosis", risk: "Low" },
    { id: "P-004", name: "James Wilson", age: 52, lastScan: "2026-03-10", diagnosis: "Basal Cell Carcinoma", risk: "Medium" },
    { id: "P-005", name: "Anna Martinez", age: 41, lastScan: "2026-03-09", diagnosis: "Actinic Keratosis", risk: "Medium" },
    { id: "P-006", name: "Robert Taylor", age: 38, lastScan: "2026-03-08", diagnosis: "Benign Nevus", risk: "Low" },
    { id: "P-007", name: "Lisa Anderson", age: 47, lastScan: "2026-03-08", diagnosis: "Melanoma", risk: "High" },
    { id: "P-008", name: "David Brown", age: 55, lastScan: "2026-03-07", diagnosis: "Squamous Cell Carcinoma", risk: "Medium" },
  ];

  const filteredPatients = patients.filter((patient) => {
    const matchesSearch =
      patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      patient.id.toLowerCase().includes(searchTerm.toLowerCase());

    if (filterType === "all") return matchesSearch;
    if (filterType === "high") return matchesSearch && patient.risk === "High";
    if (filterType === "medium") return matchesSearch && patient.risk === "Medium";
    if (filterType === "low") return matchesSearch && patient.risk === "Low";
    return matchesSearch;
  });

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="mb-2">Patient Search</h1>
        <p className="text-gray-600">Search and manage patient records</p>
      </div>

      {/* Search and Filters */}
      <div className="mb-6 flex flex-col md:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search by name, ID, or phone..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-[#fb6f92]"
          />
        </div>
        <select
          value={filterType}
          onChange={(e) => setFilterType(e.target.value)}
          className="px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-[#fb6f92]"
        >
          <option value="all">All Patients</option>
          <option value="high">High Risk</option>
          <option value="medium">Medium Risk</option>
          <option value="low">Low Risk</option>
        </select>
        <Link
          to="/dashboard/patients/create"
          className="px-6 py-3 rounded-lg text-white transition-all hover:opacity-90 whitespace-nowrap text-center"
          style={{ backgroundColor: "#fb6f92" }}
        >
          + Create Patient
        </Link>
      </div>

      {/* Results Table */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-[#ffe5ec]">
              <tr>
                <th className="px-6 py-4 text-left text-gray-700">Patient ID</th>
                <th className="px-6 py-4 text-left text-gray-700">Name</th>
                <th className="px-6 py-4 text-left text-gray-700">Age</th>
                <th className="px-6 py-4 text-left text-gray-700">Last Scan</th>
                <th className="px-6 py-4 text-left text-gray-700">Diagnosis</th>
                <th className="px-6 py-4 text-left text-gray-700">Risk Level</th>
                <th className="px-6 py-4 text-left text-gray-700">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {filteredPatients.map((patient) => (
                <tr key={patient.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 text-gray-900">{patient.id}</td>
                  <td className="px-6 py-4 text-gray-900">{patient.name}</td>
                  <td className="px-6 py-4 text-gray-600">{patient.age}</td>
                  <td className="px-6 py-4 text-gray-600">{patient.lastScan}</td>
                  <td className="px-6 py-4 text-gray-900">{patient.diagnosis}</td>
                  <td className="px-6 py-4">
                    <span
                      className="px-3 py-1 rounded-full text-sm"
                      style={{
                        backgroundColor:
                          patient.risk === "Low"
                            ? "#ffe5ec"
                            : patient.risk === "Medium"
                            ? "#ffc2d1"
                            : "#fb6f92",
                        color: patient.risk === "High" ? "white" : "#fb6f92",
                      }}
                    >
                      {patient.risk}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex gap-2">
                      <Link
                        to={`/dashboard/patients/${patient.id}`}
                        className="p-2 rounded-lg hover:bg-[#ffe5ec] transition-colors"
                        title="View Profile"
                      >
                        <Eye className="w-5 h-5 text-gray-600" />
                      </Link>
                      <Link
                        to="/dashboard/upload"
                        className="p-2 rounded-lg hover:bg-[#ffe5ec] transition-colors"
                        title="Upload Scan"
                      >
                        <Upload className="w-5 h-5 text-gray-600" />
                      </Link>
                      <Link
                        to="/dashboard/reports"
                        className="p-2 rounded-lg hover:bg-[#ffe5ec] transition-colors"
                        title="View Reports"
                      >
                        <FileText className="w-5 h-5 text-gray-600" />
                      </Link>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {filteredPatients.length === 0 && (
          <div className="text-center py-12 text-gray-500">
            <p>No patients found matching your search criteria</p>
          </div>
        )}
      </div>
    </div>
  );
}
