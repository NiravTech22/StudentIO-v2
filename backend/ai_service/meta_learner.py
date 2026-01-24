"""
Meta-Learning System
Tracks student interactions and adapts teaching style
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
from datetime import datetime
import json
import os


class MetaLearner:
    """Learns from student interactions to personalize responses"""
    
    def __init__(self, storage_path: str = "./meta_data"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # In-memory storage (will persist to disk)
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.interactions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Load existing data
        self._load_data()
        
        print("✅ Meta-learner initialized")
    
    def get_profile(self, student_id: str) -> Dict[str, Any]:
        """
        Get or create a student's learning profile
        
        Profile includes:
        - Learning style (visual, detailed, concise, example-driven)
        - Preferred topics
        - Difficulty level
        - Response preferences
        """
        if student_id not in self.profiles:
            self.profiles[student_id] = {
                "student_id": student_id,
                "style": "balanced",  # default
                "preferred_detail_level": "medium",
                "favorite_topics": [],
                "question_count": 0,
                "avg_confidence": 0.0,
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat()
            }
        
        return self.profiles[student_id]
    
    def record_interaction(
        self,
        student_id: str,
        question: str,
        answer: str,
        confidence: float
    ):
        """Record a question-answer interaction"""
        
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "question_length": len(question),
            "answer_length": len(answer),
            "topic": self._infer_topic(question)
        }
        
        self.interactions[student_id].append(interaction)
        
        # Update profile
        profile = self.get_profile(student_id)
        profile["question_count"] += 1
        profile["last_active"] = datetime.now().isoformat()
        
        # Update average confidence
        total_conf = profile["avg_confidence"] * (profile["question_count"] - 1) + confidence
        profile["avg_confidence"] = total_conf / profile["question_count"]
        
        # Update learning style based on patterns
        self._update_learning_style(student_id)
        
        # Persist periodically (every 10 interactions)
        if profile["question_count"] % 10 == 0:
            self._save_data()
    
    def record_feedback(
        self,
        student_id: str,
        question: str,
        answer: str,
        helpful: bool,
        rating: Optional[int] = None
    ):
        """Record explicit feedback from student"""
        
        # Find the most recent matching interaction
        recent_interactions = self.interactions[student_id][-10:]  # Check last 10
        
        for interaction in reversed(recent_interactions):
            if interaction["question"] == question:
                interaction["helpful"] = helpful
                interaction["rating"] = rating
                interaction["feedback_timestamp"] = datetime.now().isoformat()
                break
        
        # Adjust learning style based on feedback
        if not helpful:
            # Try a different approach next time
            profile = self.get_profile(student_id)
            self._adjust_style_on_negative_feedback(profile)
        
        self._save_data()
    
    def get_stats(self, student_id: str) -> Dict[str, Any]:
        """Get statistics for a student"""
        
        profile = self.get_profile(student_id)
        interactions = self.interactions.get(student_id, [])
        
        # Topic distribution
        topics = [i.get("topic", "Unknown") for i in interactions]
        topic_counts = Counter(topics)
        
        # Feedback stats
        helpful_count = sum(1 for i in interactions if i.get("helpful", False))
        total_feedback = sum(1 for i in interactions if "helpful" in i)
        
        # Average ratings
        ratings = [i.get("rating") for i in interactions if i.get("rating is not None")]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        return {
            "total_questions": profile["question_count"],
            "avg_confidence": profile["avg_confidence"],
            "topic_distribution": dict(topic_counts.most_common(10)),
            "learning_style": profile["style"],
            "helpful_percentage": (helpful_count / total_feedback * 100) if total_feedback > 0 else 0,
            "avg_rating": avg_rating,
            "last_active": profile["last_active"]
        }
    
    def _infer_topic(self, question: str) -> str:
        """Simple topic inference based on keywords"""
        
        question_lower = question.lower()
        
        topics = {
            "math": ["math", "calculate", "equation", "algebra", "geometry", "calculus"],
            "science": ["science", "physics", "chemistry", "biology", "atom", "molecule"],
            "history": ["history", "war", "ancient", "civilization", "century", "historical"],
            "programming": ["code", "program", "python", "java", "function", "algorithm"],
            "language": ["grammar", "sentence", "word", "language", "writing", "essay"],
            "geography": ["country", "capital", "mountain", "river", "continent", "ocean"]
        }
        
        for topic, keywords in topics.items():
            if any(keyword in question_lower for keyword in keywords):
                return topic
        
        return "general"
    
    def _update_learning_style(self, student_id: str):
        """
        Analyze interaction patterns to determine learning style
        
        Learning styles:
        - detailed: Long questions, wants comprehensive answers
        - concise: Short questions, quick learner
        - visual: Responds well to analogies and descriptions
        - example-driven: Asks for examples often
        """
        
        interactions = self.interactions[student_id]
        
        if len(interactions) < 5:
            return  # Need more data
        
        recent = interactions[-20:]  # Look at last 20 interactions
        
        # Analyze question patterns
        avg_question_length = sum(i["question_length"] for i in recent) / len(recent)
        
        # Count keywords
        example_requests = sum(
            1 for i in recent
            if any(word in i["question"].lower() for word in ["example", "show me", "demonstrate"])
        )
        
        explain_requests = sum(
            1 for i in recent
            if any(word in i["question"].lower() for word in ["explain", "how", "why", "what is"])
        )
        
        # Determine style
        profile = self.get_profile(student_id)
        
        if example_requests > len(recent) * 0.3:
            profile["style"] = "example-driven"
        elif avg_question_length > 100:
            profile["style"] = "detailed"
        elif avg_question_length < 30:
            profile["style"] = "concise"
        elif explain_requests > len(recent) * 0.5:
            profile["style"] = "visual"
        else:
            profile["style"] = "balanced"
    
    def _adjust_style_on_negative_feedback(self, profile: Dict[str, Any]):
        """Adjust learning style when feedback is negative"""
        
        current_style = profile["style"]
        
        # Rotate through styles
        style_rotation = {
            "balanced": "detailed",
            "detailed": "example-driven",
            "example-driven": "visual",
            "visual": "concise",
            "concise": "balanced"
        }
        
        profile["style"] = style_rotation.get(current_style, "balanced")
        print(f"🔄 Adjusted learning style to: {profile['style']}")
    
    def _save_data(self):
        """Persist data to disk"""
        # Save profiles
        profiles_path = os.path.join(self.storage_path, "profiles.json")
        with open(profiles_path, 'w') as f:
            json.dump(self.profiles, f, indent=2)
        
        # Save interactions (keep last 100 per student to prevent huge files)
        interactions_path = os.path.join(self.storage_path, "interactions.json")
        trimmed_interactions = {
            student_id: interactions[-100:]
            for student_id, interactions in self.interactions.items()
        }
        with open(interactions_path, 'w') as f:
            json.dump(trimmed_interactions, f, indent=2)
    
    def _load_data(self):
        """Load data from disk"""
        try:
            profiles_path = os.path.join(self.storage_path, "profiles.json")
            if os.path.exists(profiles_path):
                with open(profiles_path, 'r') as f:
                    self.profiles = json.load(f)
            
            interactions_path = os.path.join(self.storage_path, "interactions.json")
            if os.path.exists(interactions_path):
                with open(interactions_path, 'r') as f:
                    loaded = json.load(f)
                    self.interactions = defaultdict(list, loaded)
            
            print(f"📂 Loaded {len(self.profiles)} student profiles")
        
        except Exception as e:
            print(f"⚠️  Error loading meta-learning data: {e}")
