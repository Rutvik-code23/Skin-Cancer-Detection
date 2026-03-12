import { useState, useEffect } from "react";
import { Upload, AlertCircle, CheckCircle } from "lucide-react";

interface Patient {
  id: number;
  name: string;
}

export function UploadScan() {
  const [selectedPatientId, setSelectedPatientId] = useState("");
  const [patients, setPatients] = useState<Patient[]>([]);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<{
    prediction: string;
    confidence: number;
    riskLevel: "Low" | "Medium" | "High";
  } | null>(null);

  useEffect(() => {
    fetch("http://localhost:8000/api/patients/")
      .then((res) => res.json())
      .then((data) => setPatients(data))
      .catch((err) => console.error("Error fetching patients:", err));
  }, []);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setUploadedImage(reader.result as string);
        setResult(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRunDetection = async () => {
    if (!selectedFile || !selectedPatientId) return;

    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append("patient_id", selectedPatientId);
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://localhost:8000/api/scans/", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setResult({
          prediction: data.diagnosis,
          confidence: data.confidence,
          riskLevel: data.risk_level as "Low" | "Medium" | "High",
        });
      } else {
        console.error("Analysis failed");
      }
    } catch (error) {
      console.error("Error running detection:", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="mb-2">Upload Skin Lesion Image</h1>
        <p className="text-gray-600">Select a patient and upload an image for AI-powered skin cancer detection</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left: Upload Section */}
        <div className="space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <label className="block text-gray-700 mb-2">
              Select Patient *
            </label>
            <select
              value={selectedPatientId}
              onChange={(e) => setSelectedPatientId(e.target.value)}
              className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-[#0077b6] mb-6"
            >
              <option value="">Choose a patient</option>
              {patients.map((patient) => (
                <option key={patient.id} value={patient.id.toString()}>
                  {patient.name}
                </option>
              ))}
            </select>

            <label className="block text-gray-700 mb-4">
              Upload Lesion Image *
            </label>

            {!uploadedImage ? (
              <label
                className="border-2 border-dashed rounded-xl p-12 flex flex-col items-center justify-center cursor-pointer hover:bg-[#caf0f8] transition-colors"
                style={{ borderColor: "#00b4d8" }}
              >
                <Upload className="w-12 h-12 mb-4" style={{ color: "#0077b6" }} />
                <p className="mb-2" style={{ color: "#0077b6" }}>
                  Click to upload or drag and drop
                </p>
                <p className="text-sm text-gray-500">PNG, JPG up to 10MB</p>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                />
              </label>
            ) : (
              <div className="space-y-4">
                <div className="relative rounded-xl overflow-hidden border-2" style={{ borderColor: "#00b4d8" }}>
                  <img src={uploadedImage} alt="Uploaded lesion" className="w-full h-64 object-cover" />
                </div>
                <button
                  onClick={() => setUploadedImage(null)}
                  className="text-sm text-gray-600 hover:text-gray-900"
                >
                  Remove image
                </button>
              </div>
            )}

            <button
              onClick={handleRunDetection}
              disabled={!uploadedImage || !selectedPatientId || isAnalyzing}
              className="w-full mt-6 px-6 py-3 rounded-lg text-white transition-all hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
              style={{ backgroundColor: "#0077b6" }}
            >
              {isAnalyzing ? "Analyzing..." : "Run AI Detection"}
            </button>
          </div>
        </div>

        {/* Right: Results Section */}
        <div>
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <h2 className="mb-6">Analysis Results</h2>

            {!result && !isAnalyzing && (
              <div className="text-center py-12 text-gray-500">
                <AlertCircle className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                <p>Upload an image and run detection to see results</p>
              </div>
            )}

            {isAnalyzing && (
              <div className="text-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-4 border-[#caf0f8] border-t-[#0077b6] mx-auto mb-4"></div>
                <p className="text-gray-600">AI is analyzing the image...</p>
              </div>
            )}

            {result && (
              <div className="space-y-6">
                <div className="flex items-start gap-3 p-4 rounded-lg bg-green-50 border border-green-200">
                  <CheckCircle className="w-6 h-6 text-green-600 mt-0.5" />
                  <div>
                    <p className="text-green-900">Analysis Complete</p>
                    <p className="text-sm text-green-700">AI detection has been completed successfully</p>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="p-4 rounded-lg" style={{ backgroundColor: "#caf0f8" }}>
                    <p className="text-sm text-gray-600 mb-1">Predicted Class</p>
                    <p className="text-xl" style={{ color: "#0077b6" }}>
                      {result.prediction}
                    </p>
                  </div>

                  <div className="p-4 rounded-lg" style={{ backgroundColor: "#caf0f8" }}>
                    <p className="text-sm text-gray-600 mb-1">Confidence Score</p>
                    <p className="text-xl" style={{ color: "#0077b6" }}>
                      {result.confidence}%
                    </p>
                    <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                      <div
                        className="h-2 rounded-full"
                        style={{
                          width: `${result.confidence}%`,
                          backgroundColor: "#0077b6",
                        }}
                      ></div>
                    </div>
                  </div>

                  <div className="p-4 rounded-lg" style={{ backgroundColor: "#caf0f8" }}>
                    <p className="text-sm text-gray-600 mb-2">Risk Level</p>
                    <span
                      className="inline-block px-4 py-2 rounded-full"
                      style={{
                        backgroundColor:
                          result.riskLevel === "Low"
                            ? "#caf0f8"
                            : result.riskLevel === "Medium"
                            ? "#00b4d8"
                            : "#0077b6",
                        color: result.riskLevel === "High" ? "white" : "#0077b6",
                      }}
                    >
                      {result.riskLevel} Risk
                    </span>
                  </div>
                </div>

                <div className="pt-4 space-y-3">
                  <button
                    className="w-full px-6 py-3 rounded-lg text-white transition-all hover:opacity-90"
                    style={{ backgroundColor: "#0077b6" }}
                  >
                    Generate Report
                  </button>
                  <button
                    className="w-full px-6 py-3 rounded-lg border border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors"
                  >
                    Save Results
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
