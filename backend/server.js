const express = require('express');
const cors = require('cors');
const WebSocket = require('ws');
const { v4: uuidv4 } = require('uuid');
const http = require('http');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Middleware
app.use(cors());
app.use(express.json());

// ============================================================================
// Meta-Learning Model Implementation
// ============================================================================

class MetaLearningModel {
  constructor() {
    this.learningRate = 0.01;
    this.metaLearningRate = 0.001;
    this.beliefState = new Array(50).fill(0.1); // Initialize with low confidence
    this.variance = new Array(50).fill(0.9); // High initial uncertainty
    this.knowledgeBase = new Map(); // Store learned patterns
    this.interactionHistory = [];
    this.personalizationFactors = new Map(); // Per-student adaptation
  }

  // Generate answer to any question using current knowledge state
  generateAnswer(question, studentId = null) {
    const personalization = this.personalizationFactors.get(studentId) || {
      difficulty: 0.5,
      learningStyle: 'balanced',
      confidence: 0.5
    };

    // Simulate reasoning process
    const reasoning = this.reasonAboutQuestion(question, personalization);
    const answer = this.generateResponse(reasoning, personalization);
    
    // Update belief state based on the reasoning process
    this.updateBeliefState(reasoning);
    
    return {
      answer,
      reasoning: reasoning.steps,
      confidence: reasoning.confidence,
      knowledgeDimensions: this.getKnowledgeDimensions(),
      timestamp: Date.now()
    };
  }

  reasonAboutQuestion(question, personalization) {
    // Extract key concepts from question
    const concepts = this.extractConcepts(question);
    const reasoning = {
      steps: [],
      confidence: 0.5,
      concepts: concepts
    };

    // Simulate multi-step reasoning
    reasoning.steps.push(`Analyzing question: "${question}"`);
    reasoning.steps.push(`Identified ${concepts.length} key concepts`);
    
    // Use belief state to inform reasoning
    const relevantKnowledge = concepts.map(concept => ({
      concept,
      activation: this.beliefState[concept.hash % 50] || 0.1
    }));

    const avgActivation = relevantKnowledge.reduce((sum, k) => sum + k.activation, 0) / relevantKnowledge.length;
    reasoning.confidence = Math.min(0.95, avgActivation * 1.5);

    reasoning.steps.push(`Knowledge activation: ${avgActivation.toFixed(2)}`);
    reasoning.steps.push(`Confidence level: ${reasoning.confidence.toFixed(2)}`);

    return reasoning;
  }

  generateResponse(reasoning, personalization) {
    const responses = [
      `Based on my analysis, I can help you understand this concept better.`,
      `Let me break this down for you step by step.`,
      `This is an interesting question that involves several key ideas.`,
      `I can explain this from multiple perspectives to help you learn.`
    ];

    const baseResponse = responses[Math.floor(Math.random() * responses.length)];
    const detail = reasoning.confidence > 0.7 ? 
      " I have high confidence in this area and can provide detailed explanations." :
      " I'm still learning about this topic, but I'll share what I know.";

    return baseResponse + detail;
  }

  extractConcepts(question) {
    // Simple concept extraction - in production would use NLP
    const words = question.toLowerCase().split(/\s+/);
    const concepts = words.filter(word => word.length > 4).map((word, i) => ({
      word,
      hash: this.simpleHash(word + i)
    }));
    return concepts;
  }

  simpleHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return Math.abs(hash) % 50;
  }

  updateBeliefState(reasoning) {
    // Update belief state based on reasoning confidence
    const updateAmount = reasoning.confidence * this.learningRate;
    
    reasoning.concepts.forEach(concept => {
      const index = concept.hash % 50;
      this.beliefState[index] = Math.min(1.0, this.beliefState[index] + updateAmount);
      this.variance[index] = Math.max(0.1, this.variance[index] - updateAmount * 0.5);
    });

    // Store interaction for meta-learning
    this.interactionHistory.push({
      timestamp: Date.now(),
      reasoning,
      beliefState: [...this.beliefState]
    });
  }

  learnFromFeedback(question, answer, feedback, studentId = null) {
    // Meta-learning: adapt based on student feedback
    const feedbackScore = feedback.correct ? 1.0 : -0.5;
    const adaptation = feedbackScore * this.metaLearningRate;

    // Update personalization factors
    const personalization = this.personalizationFactors.get(studentId) || {
      difficulty: 0.5,
      learningStyle: 'balanced',
      confidence: 0.5
    };

    personalization.confidence = Math.max(0.1, Math.min(1.0, 
      personalization.confidence + adaptation * 0.1));

    this.personalizationFactors.set(studentId, personalization);

    // Update belief state based on feedback
    const concepts = this.extractConcepts(question);
    concepts.forEach(concept => {
      const index = concept.hash % 50;
      this.beliefState[index] = Math.max(0.05, Math.min(1.0, 
        this.beliefState[index] + adaptation));
    });

    return {
      adapted: true,
      personalization,
      knowledgeGrowth: this.calculateKnowledgeGrowth()
    };
  }

  getKnowledgeDimensions() {
    return this.beliefState.map((belief, i) => ({
      dimension: i,
      belief: belief,
      variance: this.variance[i],
      active: belief > 0.1
    }));
  }

  calculateKnowledgeGrowth() {
    if (this.interactionHistory.length < 2) return 0;
    const recent = this.interactionHistory[this.interactionHistory.length - 1];
    const previous = this.interactionHistory[this.interactionHistory.length - 2];
    
    const recentAvg = recent.beliefState.reduce((a, b) => a + b, 0) / recent.beliefState.length;
    const previousAvg = previous.beliefState.reduce((a, b) => a + b, 0) / previous.beliefState.length;
    
    return recentAvg - previousAvg;
  }

  // Simulate multiple student instances for dashboard
  generateMultipleStudentStates(count = 6) {
    const students = [];
    for (let i = 0; i < count; i++) {
      const studentModel = new MetaLearningModel();
      // Simulate different learning states
      studentModel.beliefState = studentModel.beliefState.map(() => Math.random() * 0.8);
      studentModel.variance = studentModel.variance.map(() => Math.random() * 0.5 + 0.1);
      
      students.push({
        id: i + 1,
        belief: studentModel.beliefState,
        variance: studentModel.variance,
        timestep: Math.floor(Math.random() * 100)
      });
    }
    return students;
  }
}

// ============================================================================
// Initialize Model and API Routes
// ============================================================================

const model = new MetaLearningModel();
const activeSessions = new Map();

// API Routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: Date.now() });
});

app.get('/api/telemetry', (req, res) => {
  // Return simulated student states for dashboard
  const students = model.generateMultipleStudentStates();
  res.json({
    timestamp: Date.now(),
    students
  });
});

app.post('/api/ask', (req, res) => {
  const { question, studentId } = req.body;
  
  if (!question) {
    return res.status(400).json({ error: 'Question is required' });
  }

  const response = model.generateAnswer(question, studentId);
  res.json(response);
});

app.post('/api/feedback', (req, res) => {
  const { question, answer, feedback, studentId } = req.body;
  
  if (!question || !feedback) {
    return res.status(400).json({ error: 'Question and feedback are required' });
  }

  const result = model.learnFromFeedback(question, answer, feedback, studentId);
  res.json(result);
});

app.get('/api/student/:id/state', (req, res) => {
  const studentId = req.params.id;
  const personalization = model.personalizationFactors.get(studentId) || {
    difficulty: 0.5,
    learningStyle: 'balanced',
    confidence: 0.5
  };
  
  res.json({
    studentId,
    personalization,
    knowledgeState: model.getKnowledgeDimensions()
  });
});

// ============================================================================
// WebSocket for Real-time Communication
// ============================================================================

wss.on('connection', (ws) => {
  const sessionId = uuidv4();
  console.log(`New WebSocket connection: ${sessionId}`);

  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message);
      
      switch (data.type) {
        case 'ask':
          const response = model.generateAnswer(data.question, data.studentId);
          ws.send(JSON.stringify({
            type: 'answer',
            sessionId,
            ...response
          }));
          break;
          
        case 'feedback':
          const feedbackResult = model.learnFromFeedback(
            data.question, 
            data.answer, 
            data.feedback, 
            data.studentId
          );
          ws.send(JSON.stringify({
            type: 'learning_update',
            sessionId,
            ...feedbackResult
          }));
          break;
          
        default:
          ws.send(JSON.stringify({
            type: 'error',
            message: 'Unknown message type'
          }));
      }
    } catch (error) {
      ws.send(JSON.stringify({
        type: 'error',
        message: 'Invalid message format'
      }));
    }
  });

  ws.on('close', () => {
    console.log(`WebSocket connection closed: ${sessionId}`);
  });
});

// ============================================================================
// Start Server
// ============================================================================

const PORT = process.env.PORT || 8080;
server.listen(PORT, () => {
  console.log(`🚀 StudentIO Backend running on port ${PORT}`);
  console.log(`📊 Dashboard telemetry: http://localhost:${PORT}/api/telemetry`);
  console.log(`🔗 WebSocket server ready for connections`);
});
