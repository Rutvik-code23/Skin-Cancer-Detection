import { createBrowserRouter, Navigate } from "react-router";
import { DashboardLayout } from "./components/DashboardLayout";
import { LoginPage } from "./pages/LoginPage";
import { DashboardOverview } from "./pages/DashboardOverview";
import { PatientList } from "./pages/PatientList";
import { CreatePatient } from "./pages/CreatePatient";
import { UploadScan } from "./pages/UploadScan";
import { PatientRecord } from "./pages/PatientRecord";
import { ReportGeneration } from "./pages/ReportGeneration";
import { Appointments } from "./pages/Appointments";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: LoginPage,
  },
  {
    path: "/dashboard",
    Component: DashboardLayout,
    children: [
      { index: true, Component: DashboardOverview },
      { path: "patients", Component: PatientList },
      { path: "patients/create", Component: CreatePatient },
      { path: "patients/:id", Component: PatientRecord },
      { path: "upload", Component: UploadScan },
      { path: "reports", Component: ReportGeneration },
      { path: "appointments", Component: Appointments },
      { path: "*", element: <Navigate to="/dashboard" replace /> },
    ],
  },
  {
    path: "*",
    element: <Navigate to="/" replace />,
  },
]);
