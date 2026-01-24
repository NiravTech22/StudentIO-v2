# StudentIO - Complete Production System

## 🎯 Overview

StudentIO is a production-ready adaptive learning system combining:
- **Julia Backend**: POMDP meta-learning + student modeling (RNN+CNN)
- **Python Model**: Multi-modal transformer for Q&A (text, PDF, images)
- **Privacy Filter**: Academic content classifier
- **React Frontend**: Modern pastel-themed UI

## 📁 Project Structure

```
StudentIO-v2/
├── julia_backend/          # Julia POMDP + meta-learning
│   ├── src/
│   │   ├── StudentIO.jl   # Main module
│   │   ├── core/          # POMDP components
│   │   ├── meta/          # Meta-learning
│   │   └── api_server.jl  # HTTP API
│   └── Project.toml
│
├── python_model/          # Multi-modal transformer
│   ├── transformer.py     # Main Q&A model
│   ├── embeddings.py      # Multi-modal embeddings
│   ├── privacy_filter.py  # Content classifier
│   ├── api_server.py      # FastAPI server
│   └── requirements.txt
│
├── frontend/              # React UI
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   └── theme.ts       # Pastel themes
│   └── package.json
│
└── docs/
    └── API.md             # API documentation
```

## 🚀 Quick Start

### 1. Start Julia Backend (Port 8080)
```powershell
cd julia_backend
julia --project -e 'using StudentIO; StudentIO.start_server()'
```

### 2. Start Python Model (Port 8000)
```powershell
cd python_model
pip install -r requirements.txt
python api_server.py
```

### 3. Start Frontend (Port 5173)
```powershell
cd frontend
npm install
npm run dev
```

### 4. Access Application
Open `http://localhost:5173`

## 🏗️ Architecture

### Data Flow

```
User Input (text/PDF/image)
    ↓
Privacy Filter (Python)
    ↓
Multi-Modal Embeddings (Python Transformer)
    ↓
Julia Meta-Learning Engine ←→ Student State (POMDP)
    ↓
Response Generation (Python)
    ↓
Frontend Display (React)
```

### Component Communication

- **Frontend ←→ Julia**: WebSocket for real-time updates
- **Julia ←→ Python**: REST API for embeddings & Q&A
- **All**: JSON data exchange

## 📊 Features

### Julia Backend
- ✅ POMDP-based student modeling
- ✅ GRU belief filtering
- ✅ Meta-learning with FOMAML
- ✅ Latent state tracking (mastery, misconceptions, abstractions)
- ✅ Real-time adaptation

### Python Model
- ✅ Multi-modal transformers (text, PDF, image)
- ✅ Cross-attention across modalities
- ✅ Embedding generation
- ✅ Academic content classification
- ✅ Diagram/table generation

### Frontend
- ✅ Pastel light/dark themes
- ✅ File upload (PDF, images)
- ✅ Query history sidebar
- ✅ Real-time responses
- ✅ Progress visualization
- ✅ Smooth animations

## 🔒 Privacy

The privacy filter ensures only academic content reaches the meta-learning system:
- Text classification (academic vs non-academic)
- Separate processing pipelines
- No sensitive data in training

## 📝 API Endpoints

### Julia Backend (Port 8080)
```
POST /api/session/create          # Create learning session
POST /api/session/{id}/step       # Process interaction
GET  /api/session/{id}/state      # Get student state
GET  /api/session/{id}/diagnostics # Get metrics
```

### Python Model (Port 8000)
```
POST /api/query                    # Answer question
POST /api/embeddings               # Generate embeddings
POST /api/privacy/filter           # Filter content
POST /api/multimodal               # Process mixed inputs
```

## 🎨 Theming

The frontend supports dual pastel themes:
- **Light**: Soft gradients (lavender, mint, peach)
- **Dark**: Deep pastels (midnight blue, charcoal, plum)

Toggle with the theme button in the header.

## 📈 Meta-Learning

The system continuously adapts:
1. Tracks student interactions
2. Updates belief states via GRU filtering
3. Adjusts difficulty dynamically
4. Recommends personalized content
5. Detects misconceptions and addresses them

## 🧪 Testing

```powershell
# Test Julia backend
julia --project test/runtests.jl

# Test Python model
pytest python_model/tests/

# Test frontend
cd frontend
npm test
```

## 📦 Deployment

### Development
```powershell
.\run_all.ps1  # Starts all services
```

### Production
- Julia: Deploy to AWS/GCP with HTTP.jl server
- Python: Deploy FastAPI to Render/Fly.io
- Frontend: Deploy to Vercel/Netlify

## 🔧 Configuration

Edit `.env` files in each module:

**julia_backend/.env**
```
PORT=8080
PYTHON_API_URL=http://localhost:8000
```

**python_model/.env**
```
PORT=8000
MODEL_NAME=facebook/bart-large
DEVICE=cuda  # or cpu
```

**frontend/.env**
```
VITE_JULIA_API=http://localhost:8080
VITE_PYTHON_API=http://localhost:8000
```

## 📚 Documentation

- [API Reference](docs/API.md)
- [Julia Module](julia_backend/README.md)
- [Python Model](python_model/README.md)
- [Frontend Guide](frontend/README.md)

## 🤝 Contributing

This is a research prototype. For production use:
1. Add authentication
2. Implement database persistence
3. Add rate limiting
4. Deploy with proper scaling

## 📄 License

MIT License - see LICENSE file

## 🙏 Credits

Built on:
- Julia Flux.jl for neural networks
- Hugging Face Transformers
- React + Vite
- FastAPI

---

**Version**: 2.0.0  
**Status**: Production-Ready Research Prototype
