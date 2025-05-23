# ensemble_evaluator.py
import numpy as np
from typing import Dict, List
import json

# Import the individual evaluators
from llama_evaluator import LlamaEvaluator
from mistral_evaluator import MistralEvaluator
from rule_based_evaluator import RuleBasedEvaluator

class EnsembleEvaluator:
    def __init__(self):
        print("Initializing ensemble evaluator...")
        self.llama_evaluator = LlamaEvaluator()
        print("Llama evaluator loaded")
        self.mistral_evaluator = MistralEvaluator()
        print("Mistral evaluator loaded")
        self.rule_based_evaluator = RuleBasedEvaluator()
        print("Rule-based evaluator loaded")
        
        # Weights for different evaluators (can be tuned)
        self.weights = {
            'llama': 0.4,
            'mistral': 0.3,
            'rule_based': 0.3
        }
    
    def evaluate_response(self, question: str, response: str, condition_type: str = "depression") -> Dict:
        """Evaluate using ensemble of methods"""
        
        print(f"Evaluating with ensemble...")
        
        # Get evaluations from all methods
        try:
            llama_scores = self.llama_evaluator.evaluate_response(question, response)
            print("Llama evaluation complete")
        except Exception as e:
            print(f"Llama evaluation failed: {e}")
            llama_scores = {}
        
        try:
            mistral_scores = self.mistral_evaluator.evaluate_response(question, response)
            print("Mistral evaluation complete")
        except Exception as e:
            print(f"Mistral evaluation failed: {e}")
            mistral_scores = {}
        
        try:
            rule_scores = self.rule_based_evaluator.evaluate_response(question, response, condition_type)
            print("Rule-based evaluation complete")
        except Exception as e:
            print(f"Rule-based evaluation failed: {e}")
            rule_scores = {}
        
        # Normalize metric names and combine
        combined_scores = {}
        
        # Map different metric names to standard ones
        metric_mapping = {
            'active_listening': ['active_listening', 'response_appropriateness'],
            'emotional_expression': ['emotional_expression', 'emotional_expression'],
            'clinical_realism': ['clinical_realism', 'realism', 'clinical_realism'],
            'conversational_quality': ['conversational_flow', 'conversational_quality', 'conversational_quality'],
            'symptom_accuracy': ['symptom_accuracy', 'clinical_accuracy', 'symptom_accuracy'],
            'overall_quality': ['overall_quality', 'training_value', 'overall']
        }
        
        for standard_metric, source_metrics in metric_mapping.items():
            scores = []
            
            # Get score from Llama if available
            if llama_scores and source_metrics[0] in llama_scores:
                scores.append(llama_scores[source_metrics[0]] * self.weights['llama'])
            
            # Get score from Mistral if available
            if mistral_scores and len(source_metrics) > 1 and source_metrics[1] in mistral_scores:
                scores.append(mistral_scores[source_metrics[1]] * self.weights['mistral'])
            
            # Get score from rule-based if available
            if rule_scores and len(source_metrics) > 2 and source_metrics[2] in rule_scores:
                scores.append(rule_scores[source_metrics[2]] * self.weights['rule_based'])
            
            # Calculate weighted average
            if scores:
                combined_scores[standard_metric] = sum(scores)
            else:
                combined_scores[standard_metric] = 5.0  # Default score
        
        # Add confidence score based on agreement between evaluators
        combined_scores['confidence'] = self.calculate_confidence(llama_scores, mistral_scores, rule_scores)
        
        return combined_scores
    
    def calculate_confidence(self, llama_scores: Dict, mistral_scores: Dict, rule_scores: Dict) -> float:
        """Calculate confidence based on agreement between evaluators"""
        agreements = []
        
        # Compare scores where metrics overlap
        for metric in ['overall', 'realism', 'clinical_realism']:
            scores = []
            if llama_scores and metric in llama_scores:
                scores.append(llama_scores[metric])
            if mistral_scores and 'realism' in mistral_scores and metric in ['realism', 'clinical_realism']:
                scores.append(mistral_scores['realism'])
            if rule_scores and 'overall' in rule_scores and metric == 'overall':
                scores.append(rule_scores['overall'])
            
            if len(scores) >= 2:
                # Calculate standard deviation (lower = higher agreement)
                std_dev = np.std(scores)
                agreement = max(0, 10 - std_dev * 2)  # Convert to 0-10 scale
                agreements.append(agreement)
        
        return np.mean(agreements) if agreements else 5.0