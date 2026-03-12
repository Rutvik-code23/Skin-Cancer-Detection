import { useState } from "react";
import { Calendar as CalendarIcon, Plus, Clock } from "lucide-react";

export function Appointments() {
  const [showAddForm, setShowAddForm] = useState(false);
  const [selectedDate, setSelectedDate] = useState("2026-03-11");

  const appointments = [
    {
      id: 1,
      patient: "Sarah Johnson",
      date: "2026-03-11",
      time: "09:00 AM",
      type: "Follow-up Scan",
      status: "Confirmed",
    },
    {
      id: 2,
      patient: "Michael Chen",
      date: "2026-03-11",
      time: "10:30 AM",
      type: "Initial Consultation",
      status: "Confirmed",
    },
    {
      id: 3,
      patient: "Emily Davis",
      date: "2026-03-11",
      time: "02:00 PM",
      type: "Report Discussion",
      status: "Pending",
    },
    {
      id: 4,
      patient: "James Wilson",
      date: "2026-03-12",
      time: "11:00 AM",
      type: "Screening",
      status: "Confirmed",
    },
    {
      id: 5,
      patient: "Anna Martinez",
      date: "2026-03-12",
      time: "03:30 PM",
      type: "Follow-up",
      status: "Pending",
    },
  ];

  const upcomingAppointments = appointments.filter(
    (apt) => new Date(apt.date) >= new Date(selectedDate)
  );

  return (
    <div className="p-8">
      <div className="mb-8 flex justify-between items-center">
        <div>
          <h1 className="mb-2">Appointments</h1>
          <p className="text-gray-600">Manage patient appointments and scheduling</p>
        </div>
        <button
          onClick={() => setShowAddForm(!showAddForm)}
          className="flex items-center gap-2 px-6 py-3 rounded-lg text-white transition-all hover:opacity-90"
          style={{ backgroundColor: "#0077b6" }}
        >
          <Plus className="w-5 h-5" />
          Add Appointment
        </button>
      </div>

      {/* Add Appointment Form */}
      {showAddForm && (
        <div className="mb-8 bg-white rounded-xl shadow-sm border border-gray-100 p-6">
          <h2 className="mb-4">Schedule New Appointment</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-700 mb-2">Patient Name</label>
              <select className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-[#0077b6]">
                <option>Select patient</option>
                <option>Sarah Johnson</option>
                <option>Michael Chen</option>
                <option>Emily Davis</option>
              </select>
            </div>
            <div>
              <label className="block text-gray-700 mb-2">Appointment Type</label>
              <select className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-[#0077b6]">
                <option>Select type</option>
                <option>Initial Consultation</option>
                <option>Follow-up Scan</option>
                <option>Report Discussion</option>
                <option>Screening</option>
              </select>
            </div>
            <div>
              <label className="block text-gray-700 mb-2">Date</label>
              <input
                type="date"
                className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-[#0077b6]"
              />
            </div>
            <div>
              <label className="block text-gray-700 mb-2">Time</label>
              <input
                type="time"
                className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-[#0077b6]"
              />
            </div>
            <div className="md:col-span-2">
              <label className="block text-gray-700 mb-2">Notes</label>
              <textarea
                rows={3}
                placeholder="Additional notes or special instructions"
                className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-[#0077b6]"
              />
            </div>
          </div>
          <div className="flex gap-4 mt-4">
            <button
              className="px-6 py-3 rounded-lg text-white transition-all hover:opacity-90"
              style={{ backgroundColor: "#0077b6" }}
            >
              Save Appointment
            </button>
            <button
              onClick={() => setShowAddForm(false)}
              className="px-6 py-3 rounded-lg border border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Calendar View */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <div className="flex items-center gap-2 mb-4">
              <CalendarIcon className="w-5 h-5" style={{ color: "#0077b6" }} />
              <h2>Calendar</h2>
            </div>
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-[#0077b6] mb-4"
            />
            <div className="space-y-2">
              <div className="p-3 rounded-lg" style={{ backgroundColor: "#caf0f8" }}>
                <p className="text-sm text-gray-600">Today's Appointments</p>
                <p className="text-2xl" style={{ color: "#0077b6" }}>
                  {appointments.filter((a) => a.date === "2026-03-11").length}
                </p>
              </div>
              <div className="p-3 rounded-lg" style={{ backgroundColor: "#caf0f8" }}>
                <p className="text-sm text-gray-600">This Week</p>
                <p className="text-2xl" style={{ color: "#0077b6" }}>
                  {appointments.length}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Appointments List */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
            <div className="p-6 border-b border-gray-200">
              <h2>Upcoming Appointments</h2>
            </div>
            <div className="divide-y divide-gray-100">
              {upcomingAppointments.map((appointment) => (
                <div key={appointment.id} className="p-6 hover:bg-gray-50 transition-colors">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h3 className="mb-1">{appointment.patient}</h3>
                      <div className="flex flex-wrap gap-3 text-sm text-gray-600 mb-2">
                        <div className="flex items-center gap-1">
                          <CalendarIcon className="w-4 h-4" />
                          {appointment.date}
                        </div>
                        <div className="flex items-center gap-1">
                          <Clock className="w-4 h-4" />
                          {appointment.time}
                        </div>
                      </div>
                      <p className="text-sm text-gray-700">{appointment.type}</p>
                    </div>
                    <div className="flex flex-col items-end gap-2">
                      <span
                        className={`px-3 py-1 rounded-full text-sm ${
                          appointment.status === "Confirmed"
                            ? "bg-green-100 text-green-700"
                            : "bg-yellow-100 text-yellow-700"
                        }`}
                      >
                        {appointment.status}
                      </span>
                      <button
                        className="text-sm hover:underline"
                        style={{ color: "#0077b6" }}
                      >
                        View Details
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
