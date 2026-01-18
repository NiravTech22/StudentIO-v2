# StudentIO Frontend API Specification

## Overview

This document defines the interface contract for integrating StudentIO with web-based frontends. The API supports both REST endpoints for discrete operations and WebSocket connections for real-time streaming.

## Base URL

```
Production: https://api.studentio.edu/v1
Development: http://localhost:8080/v1
```

---

## Authentication

All endpoints require JWT authentication:

```http
Authorization: Bearer <token>
```

Tokens are issued via `/auth/login` and expire after 24 hours.

---

## REST Endpoints

### Session Management

#### Create Session
```http
POST /sessions
```

**Request:**
```json
{
  "student_id": "string",
  "course_id": "string",
  "initial_assessment": {
    "prior_topics": ["algebra_basics", "fractions"],
    "estimated_level": 0.3
  }
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "created_at": "2026-01-18T12:00:00Z",
  "belief_state": {
    "uncertainty": 1.0,
    "estimated_mastery": 0.1
  }
}
```

#### Get Session State
```http
GET /sessions/{session_id}
```

**Response:**
```json
{
  "session_id": "uuid",
  "step_count": 42,
  "current_belief": {
    "uncertainty": 0.3,
    "estimated_mastery": 0.65,
    "top_mastered_topics": ["algebra_basics", "linear_equations"],
    "struggling_topics": ["quadratic_equations"]
  },
  "last_action": {
    "action_type": "PRESENT_PROBLEM",
    "problem_id": 123,
    "difficulty": 0.7
  }
}
```

#### Close Session
```http
DELETE /sessions/{session_id}
```

---

### Interaction

#### Submit Student Response
```http
POST /sessions/{session_id}/observations
```

**Request:**
```json
{
  "observation_type": "problem_response",
  "data": {
    "problem_id": 123,
    "correct": true,
    "response_time_ms": 45000,
    "confidence": 0.8,
    "partial_credit": 1.0
  }
}
```

**Response:**
```json
{
  "observation_id": "uuid",
  "processed_at": "2026-01-18T12:01:00Z",
  "belief_update": {
    "previous_uncertainty": 0.35,
    "new_uncertainty": 0.30,
    "mastery_change": 0.02
  },
  "next_action": {
    "action_type": "PRESENT_PROBLEM",
    "problem_id": 456,
    "topic_id": 12,
    "difficulty": 0.75,
    "rationale": "Building on recent success with linear equations"
  }
}
```

#### Get Next Action (without observation)
```http
GET /sessions/{session_id}/next-action
```

**Response:**
```json
{
  "action": {
    "action_type": "REVIEW_CONCEPT",
    "topic_id": 8,
    "difficulty": 0.5,
    "emphasis": 0.7,
    "pacing": 0.3
  },
  "alternatives": [
    {"action_type": "PRESENT_PROBLEM", "value": 0.82},
    {"action_type": "PROVIDE_HINT", "value": 0.75}
  ],
  "explanation": {
    "value_estimate": 0.85,
    "uncertainty": 0.25,
    "reasoning": "Student showing fatigue, recommending conceptual review"
  }
}
```

---

### Teacher Override

#### Override Next Action
```http
POST /sessions/{session_id}/override
```

**Request:**
```json
{
  "override_type": "force_action",
  "action": {
    "action_type": "SWITCH_TOPIC",
    "topic_id": 15
  },
  "reason": "Student requested to work on geometry"
}
```

**Response:**
```json
{
  "override_id": "uuid",
  "applied": true,
  "original_recommendation": {
    "action_type": "PRESENT_PROBLEM",
    "problem_id": 789
  }
}
```

#### Set Constraints
```http
PUT /sessions/{session_id}/constraints
```

**Request:**
```json
{
  "max_difficulty": 0.8,
  "excluded_topics": [5, 12],
  "time_limit_minutes": 30,
  "require_hints_before_solution": true
}
```

---

### Diagnostics

#### Get Belief Trajectory
```http
GET /sessions/{session_id}/diagnostics/belief-trajectory
```

**Response:**
```json
{
  "trajectory": [
    {"step": 1, "mastery": 0.1, "uncertainty": 1.0},
    {"step": 2, "mastery": 0.15, "uncertainty": 0.9},
    ...
  ],
  "drift_events": [
    {"step": 15, "magnitude": 0.3, "cause": "surprising_observation"}
  ]
}
```

#### Get Learning Progress
```http
GET /sessions/{session_id}/diagnostics/progress
```

**Response:**
```json
{
  "total_steps": 100,
  "topics_mastered": 5,
  "current_mastery_avg": 0.72,
  "retention_estimate": 0.85,
  "learning_velocity": 0.02,
  "time_to_goal_estimate_minutes": 45
}
```

---

## WebSocket API

### Connection
```
wss://api.studentio.edu/v1/ws/sessions/{session_id}
```

### Message Types

#### Client → Server

**Submit Observation:**
```json
{
  "type": "observation",
  "data": {
    "correct": true,
    "response_time_ms": 30000
  }
}
```

**Request State:**
```json
{
  "type": "get_state"
}
```

#### Server → Client

**Belief Update:**
```json
{
  "type": "belief_update",
  "data": {
    "uncertainty": 0.25,
    "mastery_estimate": 0.7,
    "timestamp": "2026-01-18T12:02:00Z"
  }
}
```

**Action Recommendation:**
```json
{
  "type": "action",
  "data": {
    "action_type": "PRESENT_PROBLEM",
    "problem_id": 567,
    "difficulty": 0.72
  }
}
```

**Diagnostic Alert:**
```json
{
  "type": "alert",
  "severity": "warning",
  "message": "Uncertainty collapse detected - consider varying problem types"
}
```

---

## Error Codes

| Code | Meaning |
|------|---------|
| 400 | Invalid request format |
| 401 | Authentication required |
| 403 | Insufficient permissions |
| 404 | Session not found |
| 409 | Session already closed |
| 422 | Invalid observation data |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| POST /observations | 60/minute |
| GET endpoints | 300/minute |
| WebSocket messages | 120/minute |

---

## Data Types

### ActionType Enum
```
PRESENT_PROBLEM, PROVIDE_HINT, PROVIDE_SOLUTION,
REVIEW_CONCEPT, ADJUST_DIFFICULTY, SWITCH_TOPIC,
ENCOURAGE, PAUSE
```

### ObservationType Enum
```
problem_response, confidence_report, time_spent,
help_requested, session_pause
```

---

## Ethical Guardrails

The API enforces the following:

1. **No demographic data**: Requests with demographic fields are rejected
2. **Uncertainty logging**: All actions include uncertainty estimates
3. **Teacher override**: Always available, logged for audit
4. **Explainability**: Every action includes machine-readable rationale
5. **Rate limiting**: Prevents rapid-fire interactions that could indicate automation

---

## Example Integration (JavaScript)

```javascript
const session = await fetch('/v1/sessions', {
  method: 'POST',
  headers: { 'Authorization': `Bearer ${token}` },
  body: JSON.stringify({ student_id: 'stu_123' })
}).then(r => r.json());

// WebSocket for real-time updates
const ws = new WebSocket(`wss://api.studentio.edu/v1/ws/sessions/${session.session_id}`);

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'action') {
    presentProblem(msg.data.problem_id);
  }
};

// Submit student response
ws.send(JSON.stringify({
  type: 'observation',
  data: { correct: true, response_time_ms: 25000 }
}));
```
