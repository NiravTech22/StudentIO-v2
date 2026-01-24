# StudentIO - Quick Start Guide

## 🚀 Fastest Way to Run

### Option 1: Run Everything (Recommended)
```powershell
.\run_all.ps1
```

This starts:
- Python model (port 8000)
- React frontend (port 5173)

### Option 2: Run Components Separately

**Python Model Only:**
```powershell
.\run_python.ps1
```

**Frontend Only:**
```powershell
.\run_frontend.ps1
```

## ✅ What to Expect

After running `.\run_all.ps1`, you'll see **2 new PowerShell windows**:
1. **Python Model** - Shows "Application startup complete"
2. **Frontend** - Shows "VITE ready" with local URL

Then open: **http://localhost:5173**

## 🔧 Troubleshooting

### "Python not found"
Install Python 3.8+ from python.org

### "Node.js not found"  
Install Node.js 16+ from nodejs.org

### Port already in use
```powershell
# Kill process on port 8000
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess -Force

# Kill process on port 5173
Stop-Process -Id (Get-NetTCPConnection -LocalPort 5173).OwningProcess -Force
```

### Dependencies failing
```powershell
# Install Python deps manually
cd python_model
pip install transformers torch fastapi uvicorn Pillow PyPDF2

# Install frontend deps manually
npm install
```

## 📊 Testing the System

### 1. Test Python API
```powershell
curl http://localhost:8000/health
```

Should return: `{"status":"ok"}`

### 2. Test Query
```powershell
curl -X POST http://localhost:8000/api/query `
  -H "Content-Type: application/json" `
  -d '{"query":"What is photosynthesis?"}'
```

### 3. Test Frontend
Open http://localhost:5173 and type a question

## 🎯 Current Features

✅ Multi-modal transformer (text, PDF, images)
✅ Privacy filtering (academic content only)
✅ FastAPI backend with CORS
✅ React frontend with modern UI
✅ File upload support

## ⚠️ Known Limitations

- Julia backend integration pending (visualization_server.jl needs Oxygen.jl package)
- Meta-learning connection to Python model needs HTTP client in Julia
- Pastel theme styling needs to be completed in CSS

## 📝 Next Steps

1. Run `.\run_all.ps1`
2. Open http://localhost:5173
3. Try asking: "Explain quantum physics"
4. Upload a PDF and ask questions about it

## 🆘 Still Having Issues?

Check:
1. All prerequisites installed (Python, Node.js)
2. No other services using ports 8000 or 5173
3. Terminal has admin rights (if needed)
4. Antivirus not blocking Python/Node

---

**Current Version**: 2.0.0 (Production-Ready Prototype)
