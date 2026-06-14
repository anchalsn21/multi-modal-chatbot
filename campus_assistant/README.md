# Greenfield Campus Orientation Assistant

**Phase 1 — Text Intelligence**

An AI-powered campus chatbot for Greenfield University. Built as an MSc assignment
covering multimodal AI (Phase 1: text only; Phase 2: voice + image).

## Architecture

```
campus_assistant/
├── backend/    # Python FastAPI + DistilBERT intent classifier
└── frontend/   # Next.js 15 dark-theme chat UI
```

## Quick Start (Both Services)

**Terminal 1 — Backend:**
```bash
cd backend
pip install -r requirements.txt
python src/train.py          # train once (~3 min)
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm install
cp .env.example .env.local   # already set to http://localhost:8000
npm run dev
```

Open **http://localhost:3000** in your browser.

## What Phase 1 Includes

| Component | Status |
|-----------|--------|
| DistilBERT intent classifier (5 classes) | ✅ Complete |
| Campus knowledge base (20 records, JSON) | ✅ Complete |
| FAQ training dataset (150+ examples) | ✅ Complete |
| FastAPI REST API (3 endpoints) | ✅ Complete |
| Next.js landing page | ✅ Complete |
| Next.js chat UI (text + voice + image controls) | ✅ Complete |
| Whisper voice ASR | 🔜 Phase 2 |
| CLIP + FAISS image retrieval | 🔜 Phase 2 |
| Multimodal fusion layer | 🔜 Phase 2 |

## Supported Intents

- `find_location` — "Where is the main library?"
- `ask_hours` — "What time does the gym close?"
- `find_event` — "Tell me about the freshers fair"
- `find_department` — "Where is the CS department?"
- `find_study_area` — "Is there a 24-hour study room?"

See [backend/README.md](backend/README.md) and [frontend/README.md](frontend/README.md) for detailed setup.
