# Skin Cancer Detection

A comprehensive web-based application for detecting skin cancer using Vision Transformer (ViT) model. The system features a patient management dashboard, medical image upload and analysis capabilities, and automated report generation.

## 🎯 Features

- **Patient Management**: Create and manage patient records with comprehensive medical history
- **Medical Image Analysis**: Upload and analyze skin lesion images using advanced AI models
- **AI-Powered Detection**: Vision Transformer-based skin cancer detection with high accuracy
- **Report Generation**: Automated medical report generation with analysis results
- **Doctor Dashboard**: Intuitive dashboard for healthcare professionals
- **Appointment Management**: Schedule and track patient appointments
- **Responsive Design**: Modern, mobile-friendly web interface

## 📁 Project Structure

```
Skin-Cancer-Detection/
├── backend/                    # Python FastAPI backend
│   ├── main.py                # FastAPI application entry point
│   ├── model_service.py       # ML model inference service
│   ├── database.py            # Database configuration and setup
│   ├── models.py              # SQLAlchemy ORM models
│   ├── schemas.py             # Pydantic request/response schemas
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile
│   ├── model/
│   │   └── skin_cancer_vit.pth  # Pre-trained ViT model
│   ├── database/
│   │   └── create_query.sql   # Database initialization scripts
│   ├── notebooks/             # Jupyter notebooks for development
│   └── uploads/               # Uploaded files and reports
├── frontend/                   # React + TypeScript frontend
│   ├── src/
│   │   ├── main.tsx
│   │   ├── app/
│   │   │   ├── App.tsx
│   │   │   ├── routes.tsx
│   │   │   ├── components/    # Reusable UI components
│   │   │   └── pages/         # Page components
│   │   └── styles/
│   ├── package.json
│   ├── vite.config.ts
│   ├── Dockerfile
│   └── index.html
└── docker-compose.yml         # Docker Compose configuration

```

## 🛠️ Prerequisites

- **Python 3.8+** (for backend)
- **Node.js 16+** (for frontend)
- **Docker & Docker Compose** (for containerized deployment)

## ⚙️ Installation & Setup

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the backend server:
   ```bash
   uvicorn main:app --reload
   ```
   The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
   The application will be available at `http://localhost:5173`

## 🐳 Docker Deployment

### Building Individual Images

**Backend Image:**
```bash
cd backend
docker build -t skin-backend .
docker run -p 8000:8000 -v skin_data:/app skin-backend
```

**Frontend Image:**
```bash
cd frontend
docker build -t skin-frontend .
docker run -p 5173:5173 skin-frontend:latest
```

### Docker Compose (Recommended)

Run the entire application stack with a single command:

```bash
docker compose up --build
```

This will:
- Start the backend service on port 8000
- Start the frontend service on port 5173
- Set up the database
- Configure networking between services

**Note:** When using Docker Compose, update frontend API calls from `http://localhost:8000` to `http://backend:8000`

## 📡 API Endpoints

The backend provides the following API endpoints:

- **Patients**: Manage patient records
- **Uploads**: Image upload and analysis
- **Reports**: Generate and retrieve medical reports
- **Appointments**: Manage appointments

Access API documentation at `http://localhost:8000/docs` (Swagger UI)

## 📧 Support

For questions or issues, please contact the development team.