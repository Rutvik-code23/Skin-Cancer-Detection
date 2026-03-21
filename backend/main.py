from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import os
import shutil
from uuid import uuid4
from fpdf import FPDF
from datetime import datetime

import models
import schemas
from database import engine, SessionLocal
from model_service import predict_image

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Skin Cancer Detection API")

# Configure CORS for Vite
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev only, update for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
REPORTS_DIR = os.path.join(UPLOAD_DIR, "reports")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/api/patients/", response_model=schemas.PatientOut)
def create_patient(patient: schemas.PatientCreate, db: Session = Depends(get_db)):
    db_patient = models.Patient(**patient.dict())
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

@app.get("/api/patients/", response_model=List[schemas.PatientOut])
def read_patients(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    patients = db.query(models.Patient).offset(skip).limit(limit).all()
    return patients

@app.get("/api/patients/{patient_id}", response_model=schemas.PatientOut)
def read_patient(patient_id: int, db: Session = Depends(get_db)):
    db_patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if db_patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return db_patient

@app.delete("/api/patients/{patient_id}", status_code=204)
def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    db_patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if db_patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Note: SQLAlchemy cascade delete on Scan model (defined in models.py) will also delete related scans from DB
    db.delete(db_patient)
    db.commit()
    return None

@app.post("/api/scans/", response_model=schemas.ScanOut)
async def upload_scan(
    patient_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Verify patient exists
    db_patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not db_patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Save file
    file_ext = file.filename.split(".")[-1]
    unique_filename = f"{uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run Prediction
    prediction_result = predict_image(file_path)

    # Save to DB
    db_scan = models.Scan(
        patient_id=patient_id,
        image_path=f"/api/uploads/{unique_filename}", # Important so frontend can render
        diagnosis=prediction_result["prediction"],
        confidence=prediction_result["confidence"],
        risk_level=prediction_result["riskLevel"],
    )
    db.add(db_scan)
    db.commit()
    db.refresh(db_scan)

    return db_scan

@app.post("/api/reports/", response_model=schemas.ReportOut)
def generate_report(report: schemas.ReportCreate, db: Session = Depends(get_db)):
    # Verify patient exists
    db_patient = db.query(models.Patient).filter(models.Patient.id == report.patient_id).first()
    if not db_patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Fetch scan if provided
    scan = None
    if report.scan_id:
        scan = db.query(models.Scan).filter(models.Scan.id == report.scan_id, models.Scan.patient_id == report.patient_id).first()

    # Create PDF Report
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 119, 182) # #0077b6
    pdf.cell(0, 10, "Skin Cancer Screening Report", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "DermAI Screening Tool - Medical Report", ln=True, align='C')
    pdf.ln(10)
    
    # Patient Details
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Patient Details", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Name: {db_patient.name}", ln=True)
    pdf.cell(0, 8, f"Patient ID: P-{str(db_patient.id).zfill(3)}", ln=True)
    pdf.cell(0, 8, f"Age: {db_patient.age if db_patient.age else 'N/A'}", ln=True)
    pdf.cell(0, 8, f"Gender: {db_patient.gender if db_patient.gender else 'N/A'}", ln=True)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.ln(10)

    # Image
    if scan and scan.image_path:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Uploaded Scan", ln=True)
        # Note: image_path is like /api/uploads/filename.jpg, we need local path
        local_img_path = scan.image_path.replace("/api/uploads/", os.path.join(UPLOAD_DIR, ""))
        if os.path.exists(local_img_path):
            pdf.image(local_img_path, w=100)
        else:
            pdf.set_font("Arial", 'I', 12)
            pdf.cell(0, 10, "[Image File Not Found Local]", ln=True)
        pdf.ln(10)
    
    # Doctor Notes
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Doctor Notes / Assessment", ln=True)
    pdf.set_font("Arial", '', 12)
    notes_text = report.doctor_notes if report.doctor_notes else "No specific notes provided."
    pdf.multi_cell(0, 8, notes_text)

    # Disclaimer/Footer
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(128, 128, 128)

    # Save PDF
    unique_filename = f"report_{uuid4()}.pdf"
    file_path = os.path.join(REPORTS_DIR, unique_filename)
    pdf.output(file_path)

    # Save to DB
    db_report = models.Report(
        patient_id=report.patient_id,
        scan_id=report.scan_id,
        doctor_notes=report.doctor_notes,
        pdf_path=f"/api/uploads/reports/{unique_filename}"
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)

    return db_report

from fastapi.staticfiles import StaticFiles
app.mount("/api/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
