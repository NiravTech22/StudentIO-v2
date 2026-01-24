/**
 * StudentIO Backend Server
 * Node.js/Express API Gateway
 * Routes requests between frontend and Python AI service
 */

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises;
const axios = require('axios');
const http = require('http');
const socketIo = require('socket.io');

// Initialize Express app
const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

// Configuration
const PORT = process.env.PORT || 3000;
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';
const UPLOAD_DIR = path.join(__dirname, 'uploads');
const MAX_FILE_SIZE = parseInt(process.env.MAX_FILE_SIZE) || 10 * 1024 * 1024; // 10MB

// Ensure upload directory exists
fs.mkdir(UPLOAD_DIR, { recursive: true }).catch(console.error);

// Middleware
app.use(cors({
  origin: process.env.CORS_ORIGIN || '*',
  credentials: true
}));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const studentId = req.body.studentId || 'default';
    const studentDir = path.join(UPLOAD_DIR, studentId);
    await fs.mkdir(studentDir, { recursive: true });
    cb(null, studentDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + '-' + file.originalname);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: MAX_FILE_SIZE },
  fileFilter: (req, file, cb) => {
    const allowedTypes = /pdf|docx|doc|pptx|ppt|png|jpg|jpeg|tiff|bmp/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);

    if (extname && mimetype) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Allowed: PDF, DOCX, PPTX, images'));
    }
  }
});

// ============================================================================
// Routes
// ============================================================================

// Health Check
app.get('/health', async (req, res) => {
  try {
    const aiHealth = await axios.get(`${AI_SERVICE_URL}/health`);
    res.json({
      status: 'ok',
      backend: 'online',
      ai_service: aiHealth.data.status,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(503).json({
      status: 'degraded',
      backend: 'online',
      ai_service: 'offline',
      error: error.message
    });
  }
});

// Chat Endpoint - Answer Questions
app.post('/api/chat', async (req, res) => {
  try {
    const { question, studentId = 'default', context = [], conversationHistory = [] } = req.body;

    if (!question) {
      return res.status(400).json({ error: 'Question is required' });
    }

    console.log(`📝 Question from student ${studentId}: ${question.substring(0, 50)}...`);

    // Forward to AI service
    const response = await axios.post(`${AI_SERVICE_URL}/answer`, {
      question,
      student_id: studentId,
      context,
      conversation_history: conversationHistory
    });

    res.json(response.data);

  } catch (error) {
    console.error('Chat error:', error.message);
    res.status(500).json({
      error: 'Failed to get answer',
      details: error.message
    });
  }
});

// Streaming Chat Endpoint
app.get('/api/chat/stream', async (req, res) => {
  const { question, studentId = 'default' } = req.query;

  if (!question) {
    return res.status(400).json({ error: 'Question is required' });
  }

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
  });

  try {
    // Stream from AI service
    const aiStream = await axios.get(`${AI_SERVICE_URL}/answer/stream`, {
      params: { question, student_id: studentId },
      responseType: 'stream'
    });

    aiStream.data.on('data', (chunk) => {
      res.write(chunk);
    });

    aiStream.data.on('end', () => {
      res.end();
    });

  } catch (error) {
    res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
    res.end();
  }
});

// File Upload
app.post('/api/files/upload', upload.array('files', 10), async (req, res) => {
  try {
    const { studentId = 'default' } = req.body;

    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: 'No files uploaded' });
    }

    console.log(`📤 Processing ${req.files.length} files for student ${studentId}`);

    const processedFiles = [];

    for (const file of req.files) {
      try {
        // Call Python document parser
        const formData = new FormData();
        const fileBuffer = await fs.readFile(file.path);
        formData.append('file', new Blob([fileBuffer]), file.originalname);
        formData.append('student_id', studentId);

        const parseResponse = await axios.post(
          `${AI_SERVICE_URL}/parse_document`,
          formData,
          { headers: formData.getHeaders() }
        );

        processedFiles.push({
          filename: file.originalname,
          size: file.size,
          type: file.mimetype,
          status: 'processed',
          chunks: parseResponse.data.chunks_added || 0
        });

      } catch (parseError) {
        console.error(`Error parsing ${file.originalname}:`, parseError.message);
        processedFiles.push({
          filename: file.originalname,
          size: file.size,
          type: file.mimetype,
          status: 'error',
          error: parseError.message
        });
      }
    }

    res.json({
      success: true,
      files: processedFiles,
      total: req.files.length
    });

  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({
      error: 'Upload failed',
      details: error.message
    });
  }
});

// Get Student Profile
app.get('/api/profile/:studentId', async (req, res) => {
  try {
    const { studentId } = req.params;

    const response = await axios.get(`${AI_SERVICE_URL}/profile/${studentId}`);
    res.json(response.data);

  } catch (error) {
    res.status(500).json({
      error: 'Failed to get profile',
      details: error.message
    });
  }
});

// Submit Feedback
app.post('/api/feedback', async (req, res) => {
  try {
    const { studentId, question, answer, helpful, rating } = req.body;

    await axios.post(`${AI_SERVICE_URL}/feedback`, {
      student_id: studentId,
      question,
      answer,
      helpful,
      rating
    });

    res.json({ success: true });

  } catch (error) {
    res.status(500).json({
      error: 'Failed to submit feedback',
      details: error.message
    });
  }
});

// ============================================================================
// WebSocket for Real-time Features
// ============================================================================

io.on('connection', (socket) => {
  console.log('🔌 Client connected:', socket.id);

  socket.on('join_student', (studentId) => {
    socket.join(`student_${studentId}`);
    console.log(`Student ${studentId} joined room`);
  });

  socket.on('typing', ({ studentId, isTyping }) => {
    io.to(`student_${studentId}`).emit('user_typing', { isTyping });
  });

  socket.on('disconnect', () => {
    console.log('🔌 Client disconnected:', socket.id);
  });
});

// ============================================================================
// Error Handling
// ============================================================================

app.use((err, req, res, next) => {
  console.error('Error:', err);

  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: 'File too large. Max size: 10MB' });
    }
    return res.status(400).json({ error: err.message });
  }

  res.status(500).json({
    error: 'Internal server error',
    details: process.env.NODE_ENV === 'development' ? err.message : undefined
  });
});

// ============================================================================
// Start Server
// ============================================================================

server.listen(PORT, () => {
  console.log('🚀 StudentIO Backend Server Started');
  console.log(`📡 Listening on port ${PORT}`);
  console.log(`🤖 AI Service URL: ${AI_SERVICE_URL}`);
  console.log(`📁 Upload directory: ${UPLOAD_DIR}`);
  console.log(`⚙️  Environment: ${process.env.NODE_ENV || 'development'}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('👋 SIGTERM received, shutting down gracefully...');
  server.close(() => {
    console.log('💤 Server closed');
    process.exit(0);
  });
});

module.exports = app;
