import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
from app import db
from models import CardAnalysis

class FeedbackManager:
    """Manages user feedback for model retraining"""
    
    def __init__(self):
        self.feedback_file = "training_feedback.json"
        self.feedback_data = self._load_feedback()
        logging.info("Feedback Manager initialized")
    
    def _load_feedback(self) -> List[Dict]:
        """Load existing feedback data"""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logging.error(f"Failed to load feedback: {e}")
            return []
    
    def _save_feedback(self):
        """Save feedback data to file"""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Failed to save feedback: {e}")
    
    def record_feedback(self, analysis_id: str, user_correction: str, confidence: float, 
                       reasoning: Optional[str] = None, user_id: Optional[str] = None) -> bool:
        """Record user feedback for model improvement"""
        try:
            # Get the original analysis
            analysis = CardAnalysis.query.get(analysis_id)
            if not analysis:
                logging.error(f"Analysis {analysis_id} not found")
                return False
            
            feedback_entry = {
                'feedback_id': f"fb_{int(datetime.now().timestamp())}",
                'analysis_id': analysis_id,
                'original_prediction': analysis.prediction,
                'original_confidence': float(analysis.confidence),
                'user_correction': user_correction,
                'user_confidence': confidence,
                'reasoning': reasoning,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'image_path': analysis.file_path,
                'ocr_text': analysis.ocr_text,
                'ai_features': analysis.ai_features,
                'status': 'pending_review'
            }
            
            self.feedback_data.append(feedback_entry)
            self._save_feedback()
            
            # Update analysis record with feedback
            analysis.user_feedback = user_correction
            analysis.feedback_confidence = confidence
            analysis.feedback_reasoning = reasoning
            db.session.commit()
            
            logging.info(f"Feedback recorded for analysis {analysis_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to record feedback: {e}")
            return False
    
    def get_feedback_statistics(self) -> Dict:
        """Get statistics about collected feedback"""
        total_feedback = len(self.feedback_data)
        corrections = sum(1 for f in self.feedback_data 
                         if f['original_prediction'] != f['user_correction'])
        
        accuracy_improvements = sum(1 for f in self.feedback_data 
                                  if f['user_confidence'] > f['original_confidence'])
        
        return {
            'total_feedback': total_feedback,
            'corrections': corrections,
            'correction_rate': corrections / total_feedback if total_feedback > 0 else 0,
            'accuracy_improvements': accuracy_improvements,
            'ready_for_training': self._count_ready_for_training()
        }
    
    def _count_ready_for_training(self) -> int:
        """Count feedback entries ready for retraining"""
        return sum(1 for f in self.feedback_data if f['status'] == 'pending_review')
    
    def get_training_data(self, min_feedback_count: int = 10) -> List[Dict]:
        """Get feedback data ready for model retraining"""
        if len(self.feedback_data) < min_feedback_count:
            return []
        
        # Return feedback with corrections only
        training_data = [f for f in self.feedback_data 
                        if f['original_prediction'] != f['user_correction']]
        
        return training_data
    
    def mark_feedback_used(self, feedback_ids: List[str]):
        """Mark feedback as used in training"""
        for feedback in self.feedback_data:
            if feedback['feedback_id'] in feedback_ids:
                feedback['status'] = 'used_in_training'
                feedback['training_date'] = datetime.now().isoformat()
        
        self._save_feedback()
    
    def get_recent_feedback(self, limit: int = 20) -> List[Dict]:
        """Get recent feedback entries"""
        sorted_feedback = sorted(self.feedback_data, 
                               key=lambda x: x['timestamp'], reverse=True)
        return sorted_feedback[:limit]