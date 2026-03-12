from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import os
import shutil
from uuid import uuid4

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
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

from fastapi.staticfiles import StaticFiles
app.mount("/api/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
