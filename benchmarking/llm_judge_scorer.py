"""
LLM-as-a-Judge scorer for evaluating VLM responses.
Uses GPT-4o-mini for cost-efficient semantic scoring.
"""

import os
from typing import Optional, Dict, Any
from openai import OpenAI


class LLMJudgeScorer:
    """Uses an LLM to judge the quality of VLM responses."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None):
        """
        Initialize LLM judge scorer.
        
        Args:
            model_name: OpenAI model to use for judging
            api_key: OpenAI API key (optional, uses env var if None)
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        print(f"[OK] Initialized LLM Judge with model: {model_name}")
    
    def score_response(self, 
                      question: str, 
                      response: str, 
                      ground_truth: Any,
                      question_type: str) -> float:
        """
        Score a response using LLM as judge.
        
        Args:
            question: The question asked
            response: The VLM's response
            ground_truth: Ground truth answer (can be dict, str, list, etc.)
            question_type: Type of question for context
            
        Returns:
            Score from 0.0 to 1.0
        """
        if ground_truth is None:
            return 0.0
        
        # Format ground truth based on type
        if isinstance(ground_truth, dict):
            gt_str = str(ground_truth)
        elif isinstance(ground_truth, list):
            gt_str = ", ".join(str(x) for x in ground_truth)
        else:
            gt_str = str(ground_truth)
        
        # Create judging prompt
        judge_prompt = f"""You are evaluating a chess AI's response to a question.

Question: {question}
AI Response: {response}
Ground Truth: {gt_str}

Score the AI's response on a scale from 0.0 to 1.0 based on:
- Correctness: Does it match the ground truth?
- Completeness: Does it answer the full question?
- Accuracy: Are the details correct?

Scoring guidelines:
- 1.0: Perfect match with ground truth
- 0.8-0.9: Mostly correct with minor errors
- 0.5-0.7: Partially correct
- 0.2-0.4: Mostly incorrect but some valid points
- 0.0: Completely wrong or irrelevant

Respond with ONLY a number between 0.0 and 1.0, nothing else."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise evaluator. Respond only with a number."},
                    {"role": "user", "content": judge_prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
        except Exception as e:
            print(f"[ERROR] LLM Judge scoring failed: {e}")
            return 0.0
    
    def score_fen_extraction(self, response: str, ground_truth_fen: str) -> float:
        """Score FEN extraction specifically."""
        if not ground_truth_fen:
            return 0.0
        
        # Extract FEN from response (might be embedded in text)
        response_clean = response.strip().split('\n')[0]  # First line usually has FEN
        
        # Normalize both FENs (remove move counters for comparison)
        def normalize_fen(fen):
            parts = fen.split()
            return ' '.join(parts[:4]) if len(parts) >= 4 else fen
        
        gt_normalized = normalize_fen(ground_truth_fen)
        
        # Check if response contains the FEN
        if gt_normalized in response:
            return 1.0
        
        # Use LLM judge for partial credit
        return self.score_response(
            "What is the FEN notation?",
            response,
            ground_truth_fen,
            "fen_extraction"
        )
