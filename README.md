# Sistem Penerjemah Ucapan Bahasa Sunda

Aplikasi web untuk menerjemahkan ucapan bahasa Sunda ke bahasa Indonesia menggunakan deep learning dengan pendekatan audio-visual fusion.


---

## 🎯 Fitur Utama

| Mode | Deskripsi | Max Size |
|------|-----------|----------|
| **Upload Video** | Audio + visual fusion (gerakan bibir) | 100 MB |
| **Upload Audio** | Audio-only processing | 16 MB |

- ✅ Transkripsi otomatis bahasa Sunda (WER 3.66%)
- ✅ Terjemahan ke bahasa Indonesia
- ✅ Autentikasi pengguna (JWT + Google OAuth)
- ✅ Riwayat terjemahan
- ✅ Metrik evaluasi (WER, CER)

---

## 🤖 Model yang Tersedia

| Model | Deskripsi | WER | VRAM |
|-------|-----------|-----|------|
| `whisper-medium-sundanese` | **TERBAIK!** HuggingFace trained | 3.66% | ~5GB |
| `whisper-small-openslr` | Kaggle trained Small | ~7% | ~2GB |
| `whisper-base-4500steps` | Extended baseline | ~12% | ~1GB |
| `whisper-base-1500steps` | Original baseline | ~15% | ~1GB |

**Default:** `whisper-medium-sundanese` (WER terbaik!)

---

## 🛠 Teknologi

### Backend (Python)
- Flask + Flask-CORS
- **Whisper Medium** (fine-tuned untuk bahasa Sunda, WER 3.66%)
- NLLB-200 1.3B (terjemahan Sunda → Indonesia)
- CNN untuk ekstraksi fitur visual
- MediaPipe Face Mesh
- SQLite + SQLAlchemy
- PyTorch dengan GPU acceleration

### Frontend (TypeScript)
- Next.js 14
- Tailwind CSS + Radix UI
- Axios
- React Context API

---

## 📁 Struktur Project

```
.
├── backend/
│   ├── src/
│   │   ├── app.py              # Flask API server
│   │   ├── models.py           # AudioVisualModel, CNN, Fusion
│   │   ├── models_db.py        # User, TranslationHistory
│   │   ├── whisper_finetuned.py # Wrapper Whisper fine-tuned
│   │   ├── config.py           # Konfigurasi sistem
│   │   ├── metrics.py          # WER, CER calculation
│   │   └── utils.py            # Helper functions
│   ├── models/                 # Model checkpoints (gitignored)
│   │   ├── whisper-medium-sundanese/  # BEST MODEL!
│   │   ├── whisper-small-openslr/
│   │   ├── whisper-cp1000/
│   │   └── whisper-cp4500/
│   ├── tests/                  # Unit tests
│   ├── instance/               # SQLite database
│   ├── requirements.txt
│   └── .env
│
├── frontend/
│   ├── app/                    # Next.js pages
│   ├── components/             # React components
│   ├── contexts/               # AuthContext
│   ├── lib/                    # API client
│   └── package.json
│
├── scripts/                    # Training & evaluation scripts
│   ├── evaluate_model.py
│   ├── generate_thesis_metrics.py
│   └── ...
│
└── README.md
```

---

## 🚀 Quick Start

### 1. Backend

```bash
cd backend

# Virtual environment
python -m venv .venv

# Aktivasi (Windows)
.\.venv\Scripts\Activate.ps1

# Aktivasi (Linux/Mac)  
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env

# Jalankan server
python src/app.py
```

Server berjalan di `http://localhost:5001`

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Setup environment
cp .env.example .env.local

# Jalankan development server
npm run dev
```

Buka `http://localhost:3000`

---

## ⚙️ Environment Variables

### Backend (.env)

```env
# Flask & Auth
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret

# Google OAuth (optional)
GOOGLE_CLIENT_ID=your-google-client-id

# Model Selection (PILIH SALAH SATU)
# whisper-base-1500steps, whisper-base-4500steps, whisper-small-openslr, whisper-medium-sundanese
WHISPER_MODEL_VARIANT=whisper-medium-sundanese

# GPU Settings
WHISPER_DEVICE=cuda
NLLB_DEVICE=cuda
ENABLE_GPU_ACCELERATION=true

# Google Speech API (disabled by default)
USE_GOOGLE_SPEECH_API=false
```

### Frontend (.env.local)

```env
NEXT_PUBLIC_API_URL=http://localhost:5001
NEXT_PUBLIC_GOOGLE_CLIENT_ID=your-google-client-id
```

---

## 📡 API Endpoints

### Authentication
| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| POST | `/api/register` | Registrasi user baru |
| POST | `/api/login` | Login, return JWT token |
| POST | `/api/auth/google` | Google OAuth login |
| POST | `/api/logout` | Logout (protected) |
| GET | `/api/profile` | Get user profile (protected) |

### Translation
| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| POST | `/api/upload-audio` | Upload audio file |
| POST | `/api/upload-video` | Upload video file |
| GET | `/api/translation-history` | Get history (protected) |
| DELETE | `/api/translation-history/:id` | Delete history (protected) |

### Response Format
```json
{
  "success": true,
  "transcription": "hasil transkripsi (Sunda)",
  "translation": "hasil terjemahan (Indonesia)",
  "mode": "audio-only | audio-visual",
  "metrics": {
    "wer": 0.0366,
    "cer": 0.018,
    "processing_time": 3.5,
    "audio_duration": 10.2
  }
}
```

---

## 💻 System Requirements

### Hardware
- CPU: Intel Core i5 / AMD Ryzen 5+
- RAM: 16 GB minimum (32 GB recommended)
- GPU: NVIDIA dengan 8GB VRAM (RTX 3070 recommended)
- Storage: 15 GB free

### Software
- Python 3.9 - 3.11
- Node.js 18+
- CUDA Toolkit 11.8+
- ffmpeg

---

## 📊 Model Performance

### Whisper Medium Sundanese (BEST)
| Metric | Value |
|--------|-------|
| WER | **3.66%** |
| CER | ~1.8% |
| Dataset | OpenSLR-36 (3,367 samples) |
| Training | 5,000 steps |
| Platform | HuggingFace |

### Training History
| Model | Steps | WER | Notes |
|-------|-------|-----|-------|
| whisper-base-1500steps | 1,500 | ~15% | Initial baseline |
| whisper-base-4500steps | 4,500 | ~12% | Extended training |
| whisper-small-openslr | 5,000 | ~7% | Kaggle trained |
| **whisper-medium-sundanese** | 5,000 | **3.66%** | **Current best** |

---

## 🧪 Testing

```bash
cd backend

# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run evaluation on test dataset
python scripts/run_evaluation.py
```

---

## 📋 Changelog

### v2.0.0 (January 2026)
- ✅ Whisper Medium Sundanese (WER 3.66%)
- ✅ Improved model naming conventions
- ✅ Google OAuth integration
- ✅ Project cleanup & reorganization

### v1.0.0 (December 2025)
- ✅ Initial release
- ✅ Audio-visual fusion system
- ✅ Whisper Large v3 fine-tuned

---

## 📄 License

Academic project