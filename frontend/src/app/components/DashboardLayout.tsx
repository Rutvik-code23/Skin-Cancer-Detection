import { Outlet, Link, useLocation, useNavigate } from "react-router";
import {
  LayoutDashboard,
  Users,
  Upload,
  FileText,
  Calendar,
  Settings,
  LogOut,
  Activity
} from "lucide-react";

export function DashboardLayout() {
  const location = useLocation();
  const navigate = useNavigate();

  const menuItems = [
    { icon: LayoutDashboard, label: "Dashboard", path: "/dashboard" },
    { icon: Users, label: "Patients", path: "/dashboard/patients" },
    { icon: Upload, label: "Upload Scan", path: "/dashboard/upload" },
    { icon: FileText, label: "Reports", path: "/dashboard/reports" },
    { icon: Calendar, label: "Appointments", path: "/dashboard/appointments" },
  ];

  const handleLogout = () => {
    navigate("/");
  };

  return (
    <div className="flex h-screen bg-white">
      {/* Sidebar */}
      <div className="w-64 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-8 h-8" style={{ color: "#0077b6" }} />
            <h1 className="text-xl" style={{ color: "#0077b6" }}>DermAI</h1>
          </div>
          <p className="text-sm text-gray-600">Skin Cancer Screening</p>
        </div>

        <nav className="flex-1 p-4 space-y-1">
          {menuItems.map((item) => {
            const isActive = location.pathname === item.path;
            const Icon = item.icon;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                  isActive
                    ? "text-white"
                    : "text-gray-700 hover:bg-[#caf0f8]"
                }`}
                style={isActive ? { backgroundColor: "#0077b6" } : {}}
              >
                <Icon className="w-5 h-5" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="p-4 border-t border-gray-200 space-y-1">
          <button className="flex items-center gap-3 px-4 py-3 rounded-lg text-gray-700 hover:bg-[#caf0f8] w-full transition-colors">
            <Settings className="w-5 h-5" />
            <span>Settings</span>
          </button>
          <button
            onClick={handleLogout}
            className="flex items-center gap-3 px-4 py-3 rounded-lg text-gray-700 hover:bg-[#caf0f8] w-full transition-colors"
          >
            <LogOut className="w-5 h-5" />
            <span>Logout</span>
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto">
        <Outlet />
      </div>
    </div>
  );
}
