# rule_based_evaluator.py
import re
import numpy as np
from collections import Counter
from typing import Dict

# Install required packages if not available
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
except ImportError:
    print("NLTK not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "nltk"])
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize

try:
    from textstat import flesch_reading_ease
except ImportError:
    print("textstat not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "textstat"])
    from textstat import flesch_reading_ease

class RuleBasedEvaluator:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Define clinical patterns for depression and anxiety
        self.depression_indicators = {
            'emotional_words': ['sad', 'hopeless', 'empty', 'worthless', 'guilty', 'numb', 'depressed'],
            'cognitive_patterns': ['always', 'never', 'nothing', 'everything', 'worst', 'terrible', 'awful'],
            'behavioral_indicators': ['tired', 'exhausted', 'sleep', 'appetite', 'isolat', 'withdraw'],
            'temporal_patterns': ['lately', 'recently', 'past week', 'these days', 'anymore']
        }
        
        self.anxiety_indicators = {
            'emotional_words': ['worried', 'nervous', 'anxious', 'scared', 'panic', 'overwhelmed'],
            'physical_symptoms': ['racing', 'breathing', 'heart', 'sweating', 'tense', 'restless'],
            'cognitive_patterns': ['what if', 'catastrophe', 'disaster', 'control', 'uncertain'],
            'avoidance_behaviors': ['avoid', 'escape', 'cancel', 'postpone', 'procrastinate']
        }
    
    def evaluate_response(self, question, response, condition_type="depression"):
        scores = {}
        
        # 1. Clinical Realism Score
        scores['clinical_realism'] = self.calculate_clinical_realism(response, condition_type)
        
        # 2. Emotional Expression Score
        scores['emotional_expression'] = self.calculate_emotional_expression(response)
        
        # 3. Conversational Quality Score
        scores['conversational_quality'] = self.calculate_conversational_quality(response)
        
        # 4. Symptom Accuracy Score
        scores['symptom_accuracy'] = self.calculate_symptom_accuracy(response, condition_type)
        
        # 5. Response Appropriateness Score
        scores['response_appropriateness'] = self.calculate_response_appropriateness(question, response)
        
        # 6. Readability Score (simpler = more realistic)
        scores['readability'] = self.calculate_readability_score(response)
        
        # 7. Overall Score (weighted average)
        scores['overall'] = np.mean(list(scores.values()))
        
        return scores
    
    def calculate_clinical_realism(self, response, condition_type):
        """Score based on presence of clinically relevant indicators"""
        indicators = self.depression_indicators if condition_type == "depression" else self.anxiety_indicators
        
        response_lower = response.lower()
        total_score = 0
        category_count = 0
        
        for category, words in indicators.items():
            category_score = sum(1 for word in words if word in response_lower)
            # Normalize by category size and response length
            normalized_score = min(category_score / len(words) * 10, 10)
            total_score += normalized_score
            category_count += 1
        
        return total_score / category_count if category_count > 0 else 5
    
    def calculate_emotional_expression(self, response):
        """Score based on emotional language and first-person expressions"""
        emotion_words = ['feel', 'feeling', 'felt', 'emotion', 'mood', 'heart', 'mind']
        first_person = ['i', 'me', 'my', 'myself', 'mine']
        
        response_lower = response.lower()
        words = response_lower.split()
        
        emotion_score = sum(1 for word in emotion_words if word in response_lower)
        first_person_score = sum(1 for word in first_person if word in words)
        
        # Normalize scores
        emotion_normalized = min(emotion_score / len(response_lower.split()) * 50, 10)
        first_person_normalized = min(first_person_score / len(words) * 20, 10)
        
        return (emotion_normalized + first_person_normalized) / 2
    
    def calculate_conversational_quality(self, response):
        """Score based on conversational markers and natural flow"""
        conversational_markers = ['well', 'you know', 'i mean', 'like', 'um', 'uh', 'maybe', 'i guess']
        hesitation_markers = ['...', 'pause', 'silence', 'hmm']
        
        response_lower = response.lower()
        
        conv_score = sum(1 for marker in conversational_markers if marker in response_lower)
        hesitation_score = sum(1 for marker in hesitation_markers if marker in response_lower)
        
        # Natural length (not too short, not too long)
        length_score = 10 if 20 <= len(response.split()) <= 150 else 5
        
        total_score = (conv_score + hesitation_score + length_score/10) * 2
        return min(total_score, 10)
    
    def calculate_symptom_accuracy(self, response, condition_type):
        """Score based on accurate symptom presentation"""
        if condition_type == "depression":
            key_symptoms = ['sleep', 'appetite', 'energy', 'concentration', 'interest', 'guilt', 'worth']
        else:  # anxiety
            key_symptoms = ['worry', 'restless', 'fatigue', 'concentration', 'muscle', 'sleep', 'irritable']
        
        response_lower = response.lower()
        symptom_mentions = sum(1 for symptom in key_symptoms if symptom in response_lower)
        
        # Score based on symptom relevance
        return min(symptom_mentions / len(key_symptoms) * 10, 10)
    
    def calculate_response_appropriateness(self, question, response):
        """Score based on how well response addresses the question"""
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        question_words -= stop_words
        response_words -= stop_words
        
        # Calculate overlap
        overlap = len(question_words.intersection(response_words))
        relevance_score = min(overlap / len(question_words) * 10 if question_words else 5, 10)
        
        return relevance_score
    
    def calculate_readability_score(self, response):
        """Score based on readability (simpler = more realistic for patients)"""
        try:
            flesch_score = flesch_reading_ease(response)
            # Convert Flesch score to 1-10 scale (higher Flesch = easier reading = higher score)
            normalized_score = min(flesch_score / 10, 10)
            return max(normalized_score, 1)
        except:
            return 5  # Default score if calculation fails