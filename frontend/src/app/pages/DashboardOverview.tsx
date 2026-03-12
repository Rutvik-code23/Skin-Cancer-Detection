import { Users, FileText, Calendar, Activity } from "lucide-react";
import { Link } from "react-router";

export function DashboardOverview() {
  const stats = [
    { label: "Total Patients", value: "1,248", icon: Users, color: "#caf0f8" },
    { label: "Scans Performed", value: "3,567", icon: Activity, color: "#caf0f8" },
    { label: "Reports Generated", value: "2,891", icon: FileText, color: "#caf0f8" },
    { label: "Today's Appointments", value: "12", icon: Calendar, color: "#caf0f8" },
  ];

  const quickActions = [
    { label: "Upload New Skin Image", path: "/dashboard/upload", color: "#03045e" },
    { label: "View Patients", path: "/dashboard/patients", color: "#0077b6" },
    { label: "View Reports", path: "/dashboard/reports", color: "#00b4d8" },
    { label: "Schedule Appointment", path: "/dashboard/appointments", color: "#90e0ef" },
  ];

  const recentActivity = [
    { patient: "Sarah Johnson", date: "2026-03-11", diagnosis: "Benign Nevus", status: "Completed", risk: "Low" },
    { patient: "Michael Chen", date: "2026-03-11", diagnosis: "Melanoma", status: "Pending", risk: "High" },
    { patient: "Emily Davis", date: "2026-03-10", diagnosis: "Seborrheic Keratosis", status: "Completed", risk: "Low" },
    { patient: "James Wilson", date: "2026-03-10", diagnosis: "Basal Cell Carcinoma", status: "Completed", risk: "Medium" },
    { patient: "Anna Martinez", date: "2026-03-09", diagnosis: "Actinic Keratosis", status: "Completed", risk: "Medium" },
  ];

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl mb-2">Dashboard Overview</h1>
        <p className="text-gray-600">Welcome back, Dr. Smith</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <div
              key={stat.label}
              className="rounded-xl p-6 shadow-sm border border-gray-100"
              style={{ backgroundColor: stat.color }}
            >
              <div className="flex items-center justify-between mb-4">
                <Icon className="w-8 h-8" style={{ color: "#03045e" }} />
              </div>
              <p className="text-3xl mb-1" style={{ color: "#03045e" }}>{stat.value}</p>
              <p className="text-gray-700">{stat.label}</p>
            </div>
          );
        })}
      </div>

      {/* Quick Actions */}
      <div className="mb-8">
        <h2 className="mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {quickActions.map((action) => (
            <Link
              key={action.label}
              to={action.path}
              className="p-6 rounded-xl text-white text-center transition-all hover:opacity-90 shadow-sm"
              style={{ backgroundColor: action.color }}
            >
              {action.label}
            </Link>
          ))}
        </div>
      </div>

      {/* Recent Activity Table */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
        <div className="p-6 border-b border-gray-200">
          <h2>Recent Activity</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-[#caf0f8]">
              <tr>
                <th className="px-6 py-4 text-left text-gray-700">Patient Name</th>
                <th className="px-6 py-4 text-left text-gray-700">Scan Date</th>
                <th className="px-6 py-4 text-left text-gray-700">Diagnosis Result</th>
                <th className="px-6 py-4 text-left text-gray-700">Risk Level</th>
                <th className="px-6 py-4 text-left text-gray-700">Report Status</th>
                <th className="px-6 py-4 text-left text-gray-700">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {recentActivity.map((activity, index) => (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="px-6 py-4 text-gray-900">{activity.patient}</td>
                  <td className="px-6 py-4 text-gray-600">{activity.date}</td>
                  <td className="px-6 py-4 text-gray-900">{activity.diagnosis}</td>
                  <td className="px-6 py-4">
                    <span
                      className="px-3 py-1 rounded-full text-sm"
                      style={{
                        backgroundColor:
                          activity.risk === "Low"
                            ? "#caf0f8"
                            : activity.risk === "Medium"
                            ? "#90e0ef"
                            : "#0077b6",
                        color: activity.risk === "High" ? "white" : "#03045e",
                      }}
                    >
                      {activity.risk} Risk
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <span
                      className={`px-3 py-1 rounded-full text-sm ${
                        activity.status === "Completed"
                          ? "bg-green-100 text-green-700"
                          : "bg-yellow-100 text-yellow-700"
                      }`}
                    >
                      {activity.status}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <button
                      className="px-4 py-2 rounded-lg text-white transition-all hover:opacity-90"
                      style={{ backgroundColor: "#0077b6" }}
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
