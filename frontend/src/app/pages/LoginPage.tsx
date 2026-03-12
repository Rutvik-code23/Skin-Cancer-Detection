import { useState } from "react";
import { useNavigate } from "react-router";
import { Activity, Lock, Mail } from "lucide-react";

export function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    navigate("/dashboard");
  };

  return (
    <div className="min-h-screen bg-white flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div
          className="bg-white rounded-2xl p-8 shadow-lg border-2"
          style={{ borderColor: "#00b4d8" }}
        >
          {/* Logo and Title */}
          <div className="text-center mb-8">
            <div className="flex justify-center mb-4">
              <div
                className="w-16 h-16 rounded-full flex items-center justify-center"
                style={{ backgroundColor: "#caf0f8" }}
              >
                <Activity className="w-8 h-8" style={{ color: "#0077b6" }} />
              </div>
            </div>
            <h1 className="text-2xl mb-2" style={{ color: "#0077b6" }}>
              Skin Cancer AI Screening Tool
            </h1>
            <p className="text-gray-600">Doctor Portal</p>
          </div>

          {/* Login Form */}
          <form onSubmit={handleLogin} className="space-y-6">
            <div>
              <label className="block text-gray-700 mb-2">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="doctor@example.com"
                  className="w-full pl-10 pr-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2"
                  style={{ focusRing: "#fb6f92" }}
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-gray-700 mb-2">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  className="w-full pl-10 pr-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2"
                  style={{ focusRing: "#fb6f92" }}
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              className="w-full py-3 rounded-lg text-white transition-all hover:opacity-90"
              style={{ backgroundColor: "#0077b6" }}
            >
              Login
            </button>

            <div className="text-center">
              <a href="#" className="text-sm hover:underline" style={{ color: "#0077b6" }}>
                Forgot Password?
              </a>
            </div>
          </form>

          {/* Security Note */}
          <div className="mt-6 p-3 rounded-lg" style={{ backgroundColor: "#caf0f8" }}>
            <p className="text-sm text-gray-700 text-center">
              🔒 Secure access - Your data is encrypted and protected
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
