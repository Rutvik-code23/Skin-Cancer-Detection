from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ScanBase(BaseModel):
    pass

class ScanCreate(ScanBase):
    patient_id: int

class ScanOut(ScanBase):
    id: int
    patient_id: int
    image_path: str
    diagnosis: str
    confidence: float
    risk_level: str
    scan_date: datetime

    class Config:
        from_attributes = True

class ReportBase(BaseModel):
    doctor_notes: Optional[str] = None

class ReportCreate(ReportBase):
    patient_id: int
    scan_id: Optional[int] = None

class ReportOut(ReportBase):
    id: int
    patient_id: int
    scan_id: Optional[int] = None
    pdf_path: str
    created_at: datetime

    class Config:
        from_attributes = True
class PatientBase(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    notes: Optional[str] = None

class PatientCreate(PatientBase):
    pass

class PatientOut(PatientBase):
    id: int
    created_at: datetime
    scans: List[ScanOut] = []
    reports: List[ReportOut] = []

    class Config:
        from_attributes = True
