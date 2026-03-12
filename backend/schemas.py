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

    class Config:
        from_attributes = True
