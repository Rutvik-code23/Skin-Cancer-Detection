import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router";
import { Plus, Search, User } from "lucide-react";

interface Patient {
  id: number;
  name: string;
  age: number;
  phone: string;
  scans: {
    scan_date: string;
    diagnosis: string;
  }[];
}

export function PatientList() {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState("");
  const [patients, setPatients] = useState<Patient[]>([]);

  useEffect(() => {
    fetchPatients();
  }, []);

  const fetchPatients = () => {
    fetch("http://localhost:8000/api/patients/")
      .then((res) => res.json())
      .then((data) => setPatients(data))
      .catch((err) => console.error("Error fetching patients:", err));
  };

  const handleDelete = async (id: number) => {
    if (window.confirm("Are you sure you want to delete this patient and all their scans?")) {
      try {
        const response = await fetch(`http://localhost:8000/api/patients/${id}`, {
          method: "DELETE",
        });

        if (response.ok) {
          fetchPatients();
        } else {
          console.error("Failed to delete patient");
        }
      } catch (error) {
        console.error("Error deleting patient:", error);
      }
    }
  };

  const filteredPatients = patients.filter(
    (patient) =>
      patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      patient.id.toString().includes(searchTerm.toLowerCase()) ||
      (patient.phone && patient.phone.includes(searchTerm))
  );

  return (
    <div className="p-8">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="mb-2">Patients</h1>
          <p className="text-gray-600">Manage patient records and information</p>
        </div>
        <Link
          to="/dashboard/patients/create"
          className="flex items-center gap-2 px-6 py-3 rounded-lg text-white transition-all hover:opacity-90"
          style={{ backgroundColor: "#0077b6" }}
        >
          <Plus className="w-5 h-5" />
          Create Patient
        </Link>
      </div>

      {/* Search Bar */}
      <div className="mb-6">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            placeholder="Search by name, ID, or phone number..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-12 pr-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-[#0077b6]"
          />
        </div>
      </div>

      {/* Patient Table */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-[#caf0f8]">
              <tr>
                <th className="px-6 py-4 text-left text-gray-700">Patient ID</th>
                <th className="px-6 py-4 text-left text-gray-700">Name</th>
                <th className="px-6 py-4 text-left text-gray-700">Age</th>
                <th className="px-6 py-4 text-left text-gray-700">Phone</th>
                <th className="px-6 py-4 text-left text-gray-700">Last Scan</th>
                <th className="px-6 py-4 text-left text-gray-700">Diagnosis</th>
                <th className="px-6 py-4 text-left text-gray-700">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {filteredPatients.length > 0 ? (
                filteredPatients.map((patient) => (
                  <tr key={patient.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 text-gray-900">{patient.id}</td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div
                          className="w-10 h-10 rounded-full flex items-center justify-center text-white"
                          style={{ backgroundColor: "#90e0ef" }}
                        >
                          <User className="w-5 h-5" />
                        </div>
                        <span className="text-gray-900">{patient.name}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-gray-600">{patient.age}</td>
                    <td className="px-6 py-4 text-gray-600">{patient.phone}</td>
                    <td className="px-6 py-4 text-gray-600">
                      {patient.scans && patient.scans.length > 0 
                        ? new Date(
                            [...patient.scans].sort(
                              (a, b) => new Date(b.scan_date).getTime() - new Date(a.scan_date).getTime()
                            )[0].scan_date
                          ).toLocaleDateString() 
                        : "No scans"}
                    </td>
                    <td className="px-6 py-4 text-gray-900">
                      {patient.scans && patient.scans.length > 0
                        ? [...patient.scans].sort(
                            (a, b) => new Date(b.scan_date).getTime() - new Date(a.scan_date).getTime()
                          )[0].diagnosis
                        : "N/A"}
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex gap-2">
                        <button
                          onClick={() => navigate(`/dashboard/patients/${patient.id}`)}
                          className="px-4 py-2 rounded-lg text-white transition-all hover:opacity-90"
                          style={{ backgroundColor: "#fb6f92" }}
                        >
                          View Profile
                        </button>
                        <button
                          onClick={() => handleDelete(patient.id)}
                          className="px-4 py-2 rounded-lg text-white transition-all hover:opacity-90"
                          style={{ backgroundColor: "#ef4444" }}
                        >
                          Delete User
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} className="px-6 py-12 text-center text-gray-500">
                    No patients found matching your search.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
