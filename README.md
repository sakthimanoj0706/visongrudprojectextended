# 🛡️ VisionGuard AI

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![React](https://img.shields.io/badge/React-19-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-Latest-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)

**VisionGuard AI** is a state-of-the-art Security Operations Center (SOC) platform designed for advanced facial recognition, real-time video surveillance, and intelligent alerting. Built with a high-performance backend and a modern frontend, it offers comprehensive tools for monitoring, tracking, and analyzing subjects across CCTV networks.

## ✨ Key Features

### 🔍 Advanced AI Pipeline
- **High-Accuracy Facial Detection**: Powered by **YuNet** for ultra-fast, robust face detection in crowded scenes.
- **Deep Facial Recognition**: Leverages **SFace** for highly accurate facial embeddings.
- **Vector Search Engine**: Integrated **FAISS** vector database for sub-millisecond matching against massive watchlists.

### 🌐 Scalable Backend (FastAPI)
- **Real-Time Data Streaming**: WebSocket integration for instant alerts and live metrics.
- **Role-Based Access Control (RBAC)**: Secure JWT authentication with roles like Admin, Operator, Investigator, and Auditor.
- **RESTful API**: Comprehensive endpoints for system diagnostics, event querying, and target management.

### 💻 Security Operations Center (SOC) Dashboard
- **React 19 & Vite**: Blazing fast frontend built with TypeScript.
- **Material UI (MUI) & Framer Motion**: Sleek, fully responsive, glassmorphic dark-themed UI.
- **Live Metrics & Visualization**: Real-time charts powered by **ECharts** for system health, sightings, and risk analysis.
- **Zustand State Management**: Lightweight and lightning-fast state synchronization.

### 🛠️ Powerful CLI Tool
Enroll targets, analyze recorded videos, and generate detailed chronological timelines directly from your terminal.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Node.js & npm (v18+)

### 2. Backend Setup
1. Navigate to the project root.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download AI model weights:
   ```bash
   python models/download_weights.py
   ```
4. Start the VisionGuard API server:
   ```bash
   python main.py server
   ```
   *The server runs on http://127.0.0.1:8000.*

### 3. Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install npm dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   *The dashboard runs on http://localhost:5173.*

### 4. Default Credentials
| Username | Password | Role |
|---|---|---|
| `admin_user` | `admin_pass` | Admin |
| `operator_user` | `operator_pass` | Operator |
| `investigator_user` | `investigator_pass` | Investigator |
| `auditor_user` | `auditor_pass` | Auditor |

---

## 🖥️ Command Line Interface (CLI) Usage

**Enroll a Target:**
```bash
python main.py enroll --id target_001 --name "John Doe" --category Watchlist --risk-level High --image target_face.jpg
```

**Search a Video:**
```bash
python main.py search --video sample_cctv.mp4 --camera-id CAM001 --camera-location "Main Entrance"
```

**Generate Timeline:**
```bash
python main.py timeline --target-id target_001
```

---

## 🏗️ Tech Stack
- **AI/ML**: OpenCV, YuNet, SFace, FAISS
- **Backend**: Python, FastAPI, SQLite
- **Frontend**: React 19, TypeScript, Vite, Material UI (MUI), Zustand, Framer Motion, ECharts, Axios

## 📄 License
This project is licensed under the MIT License.
