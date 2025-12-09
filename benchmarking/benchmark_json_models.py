"""
Benchmark script for JSON-based models (Exp 1A, 1B, 1C, 1D).

This script evaluates the reasoning capabilities of VLMs when provided with
FEN context from JSON-based predictors, comparing:
1. No FEN context (base VLM)
2. Predicted FEN (from JSON model predictions)
3. Ground truth FEN (for reference)

Usage:
    python benchmark_json_models.py \
        --predictions predictions_clip_exp1b.jsonl \
        --dataset_csv ../data/hf_chess_puzzles/test.csv \
        --vlm_model gpt-4o \
        --num_images 10 \
        --output_dir benchmark_results_json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from questions import QUESTIONS, get_scoring_questions
from ground_truth import GroundTruthExtractor
from vlm_integration import VLMInterface, MockVLMInterface, OpenAIVLMInterface
from llm_judge_scorer import LLMJudgeScorer
from scoring import ResponseScorer

# Import JSON to FEN converter
try:
    from Improved_representations.data_processing.converters import json_to_fen
except ImportError:
    # Fallback: define simple json_to_fen locally
    def json_to_fen(json_repr: Dict) -> str:
        """Convert JSON representation to FEN string."""
        # Reverse mapping: piece name to FEN character
        PIECE_NAME_TO_FEN = {
            'white_pawn': 'P', 'white_knight': 'N', 'white_bishop': 'B',
            'white_rook': 'R', 'white_queen': 'Q', 'white_king': 'K',
            'black_pawn': 'p', 'black_knight': 'n', 'black_bishop': 'b',
            'black_rook': 'r', 'black_queen': 'q', 'black_king': 'k'
        }
        
        board = [[None for _ in range(8)] for _ in range(8)]
        
        for piece_info in json_repr.get('pieces', []):
            piece_name = piece_info['piece']
            square = piece_info['square']
            file_idx = ord(square[0]) - ord('a')
            rank_idx = int(square[1]) - 1
            fen_char = PIECE_NAME_TO_FEN.get(piece_name)
            if fen_char:
                board[7 - rank_idx][file_idx] = fen_char
        
        fen_rows = []
        for row in board:
            fen_row = ""
            empty_count = 0
            for cell in row:
                if cell is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += cell
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)
        
        board_fen = "/".join(fen_rows)
        metadata = json_repr.get('metadata', {})
        to_move = metadata.get('to_move', 'white')
        active_color = 'w' if to_move == 'white' else 'b'
        
        return f"{board_fen} {active_color} - -"


class JSONModelBenchmark:
    """
    Benchmark class for evaluating VLMs with JSON-based FEN predictions.
    
    Compares three conditions:
    1. No FEN (base VLM)
    2. Predicted FEN (from JSON model)
    3. Ground truth FEN (oracle)
    """
    
    def __init__(
        self,
        vlm_model_name: str = "gpt-4o",
        use_mock_vlm: bool = False,
        device: str = None
    ):
        """Initialize benchmark."""
        self.device = device or "cuda"
        
        print("Initializing VLM...")
        if use_mock_vlm:
            self.vlm = MockVLMInterface()
            print("Using mock VLM for testing")
        elif vlm_model_name.startswith("gpt-"):
            try:
                self.vlm = OpenAIVLMInterface(model_name=vlm_model_name)
                print(f"Using OpenAI VLM: {vlm_model_name}")
            except Exception as e:
                print(f"Warning: Could not load OpenAI VLM: {e}")
                self.vlm = MockVLMInterface()
        else:
            try:
                self.vlm = VLMInterface(vlm_model_name, self.device)
                print(f"Using local VLM: {vlm_model_name}")
            except Exception as e:
                print(f"Warning: Could not load VLM, using mock: {e}")
                self.vlm = MockVLMInterface()
        
        self.ground_truth_extractor = GroundTruthExtractor()
        self.llm_scorer = LLMJudgeScorer()
        self.scorer = ResponseScorer()
        self.results = []
    
    def load_predictions(self, predictions_path: str) -> Dict[str, Dict]:
        """
        Load JSON predictions from JSONL file.
        
        Returns dict mapping image_path -> {predicted_json, predicted_fen}
        """
        predictions = {}
        
        with open(predictions_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                image_path = entry.get('image_path', '')
                # Normalize path
                image_path = image_path.replace('\\', '/')
                filename = os.path.basename(image_path)
                
                predictions[filename] = {
                    'predicted_json': entry.get('predicted_json', {}),
                    'predicted_fen': entry.get('predicted_fen', '')
                }
        
        print(f"Loaded {len(predictions)} predictions")
        return predictions
    
    def load_ground_truth_fens(self, dataset_csv: str) -> Dict[str, str]:
        """Load ground truth FENs from dataset CSV."""
        gt_fens = {}
        df = pd.read_csv(dataset_csv)
        
        for _, row in df.iterrows():
            image_path = row.get('image_path', '')
            fen = row.get('fen', '')
            if image_path and fen:
                filename = os.path.basename(image_path.replace('\\', '/'))
                gt_fens[filename] = fen
        
        print(f"Loaded {len(gt_fens)} ground truth FENs")
        return gt_fens
    
    def run_benchmark(
        self,
        image_paths: List[str],
        predictions: Dict[str, Dict],
        gt_fens: Dict[str, str],
        questions: Optional[List[Dict]] = None,
        output_dir: str = "benchmark_results_json"
    ) -> Dict:
        """
        Run benchmark comparing three conditions.
        
        Args:
            image_paths: List of image paths to test
            predictions: Dict of predictions from JSON model
            gt_fens: Dict of ground truth FENs
            questions: List of questions (default: all)
            output_dir: Directory to save results
        """
        if questions is None:
            questions = get_scoring_questions()
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"JSON Model Benchmark")
        print(f"Images: {len(image_paths)}")
        print(f"Questions: {len(questions)}")
        print(f"Conditions: No FEN | Predicted FEN | Ground Truth FEN")
        print(f"{'='*60}\n")
        
        all_results = []
        total_tests = len(image_paths) * len(questions)
        test_count = 0
        
        for img_idx, image_path in enumerate(image_paths):
            filename = os.path.basename(image_path.replace('\\', '/'))
            
            # Get predicted and ground truth FEN
            pred_entry = predictions.get(filename, {})
            pred_json = pred_entry.get('predicted_json', {})
            pred_fen = pred_entry.get('predicted_fen', '')
            gt_fen = gt_fens.get(filename, '')
            
            if not pred_fen and pred_json:
                # Convert JSON to FEN if not already available
                try:
                    pred_fen = json_to_fen(pred_json)
                except Exception as e:
                    print(f"  Warning: Could not convert JSON to FEN: {e}")
                    pred_fen = ''
            
            print(f"\n[{img_idx + 1}/{len(image_paths)}] {filename}")
            print(f"  Predicted FEN: {pred_fen[:40]}..." if pred_fen else "  Predicted FEN: N/A")
            print(f"  Ground Truth:  {gt_fen[:40]}..." if gt_fen else "  Ground Truth: N/A")
            
            # Get ground truths for scoring
            ground_truths = self._get_ground_truths(gt_fen, questions) if gt_fen else {}
            
            # Test each question
            for question in questions:
                test_count += 1
                q_id = question['id']
                q_type = question['type']
                
                print(f"  [{test_count}/{total_tests}] Q{q_id} ({q_type})")
                
                # 1. No FEN (base VLM)
                answer_no_fen = self.vlm.answer_question(
                    image_path, question['prompt'], fen_context=None
                )
                score_no_fen = self._score_response(
                    answer_no_fen, question, ground_truths.get(q_id)
                )
                
                # 2. With Predicted FEN
                answer_pred_fen = self.vlm.answer_question(
                    image_path, question['prompt'], fen_context=pred_fen if pred_fen else None
                )
                score_pred_fen = self._score_response(
                    answer_pred_fen, question, ground_truths.get(q_id)
                )
                
                # 3. With Ground Truth FEN
                answer_gt_fen = self.vlm.answer_question(
                    image_path, question['prompt'], fen_context=gt_fen if gt_fen else None
                )
                score_gt_fen = self._score_response(
                    answer_gt_fen, question, ground_truths.get(q_id)
                )
                
                # Store result
                result = {
                    'image_path': image_path,
                    'filename': filename,
                    'question_id': q_id,
                    'question_type': q_type,
                    'question_prompt': question['prompt'],
                    'predicted_fen': pred_fen,
                    'ground_truth_fen': gt_fen,
                    
                    # Answers
                    'answer_no_fen': answer_no_fen,
                    'answer_pred_fen': answer_pred_fen,
                    'answer_gt_fen': answer_gt_fen,
                    
                    # Scores
                    'score_no_fen': score_no_fen,
                    'score_pred_fen': score_pred_fen,
                    'score_gt_fen': score_gt_fen,
                    
                    # Improvements
                    'improvement_pred_vs_none': score_pred_fen - score_no_fen,
                    'improvement_gt_vs_none': score_gt_fen - score_no_fen,
                    'improvement_gt_vs_pred': score_gt_fen - score_pred_fen,
                    
                    'ground_truth_answer': ground_truths.get(q_id),
                    'weight': question['weight']
                }
                all_results.append(result)
                
                print(f"    No FEN: {score_no_fen:.2f} | Pred FEN: {score_pred_fen:.2f} | GT FEN: {score_gt_fen:.2f}")
        
        # Save results
        self.results = all_results
        self._save_results(all_results, output_dir)
        
        # Generate summary
        summary = self._generate_summary(all_results)
        self._save_summary(summary, output_dir)
        
        return summary
    
    def _get_ground_truths(self, fen_string: Optional[str], questions: List[Dict]) -> Dict[int, Any]:
        """Extract ground truth answers from FEN."""
        if not fen_string:
            return {}
        
        ground_truths = {}
        for question in questions:
            q_id = question['id']
            q_type = question['type']
            
            try:
                if q_type == "fen_extraction":
                    ground_truths[q_id] = fen_string
                elif q_type == "check_status":
                    ground_truths[q_id] = self.ground_truth_extractor.get_check_status(fen_string)
                elif q_type == "material_balance":
                    ground_truths[q_id] = self.ground_truth_extractor.get_material_balance(fen_string)
                elif q_type == "piece_count":
                    ground_truths[q_id] = self.ground_truth_extractor.get_piece_count(fen_string)
                elif q_type == "castling_available":
                    ground_truths[q_id] = self.ground_truth_extractor.get_castling_rights(fen_string)
                elif q_type == "best_move":
                    ground_truths[q_id] = self.ground_truth_extractor.get_best_move(fen_string)
            except Exception as e:
                pass
        
        return ground_truths
    
    def _score_response(self, response: str, question: Dict, ground_truth: Any) -> float:
        """Score a VLM response against ground truth."""
        if ground_truth is None:
            return 0.0
        
        q_type = question['type']
        
        try:
            if q_type == "fen_extraction":
                return self.llm_scorer.score_fen_extraction(response, ground_truth)
            elif q_type == "check_status":
                return self.scorer.score_check_status(response, ground_truth)
            elif q_type == "material_balance":
                return self.scorer.score_material_count(response, ground_truth)
            elif q_type == "piece_count":
                return self.scorer.score_material_count(response, ground_truth)
            elif q_type == "castling_available":
                return self.scorer.score_castling_rights(response, ground_truth)
            elif q_type == "best_move":
                return self.scorer.score_best_move(response, ground_truth)
            else:
                return self.llm_scorer.score_response(
                    question['prompt'], response, ground_truth, q_type
                )
        except Exception as e:
            print(f"    Warning: Scoring error: {e}")
            return 0.0
    
    def _save_results(self, results: List[Dict], output_dir: str):
        """Save detailed results."""
        json_path = os.path.join(output_dir, "detailed_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Saved detailed results to {json_path}")
        
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"[OK] Saved CSV to {csv_path}")
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics."""
        df = pd.DataFrame(results)
        
        summary = {
            'total_images': df['image_path'].nunique(),
            'total_questions': df['question_id'].nunique(),
            'total_tests': len(results),
            
            # Average scores
            'avg_score_no_fen': float(df['score_no_fen'].mean()),
            'avg_score_pred_fen': float(df['score_pred_fen'].mean()),
            'avg_score_gt_fen': float(df['score_gt_fen'].mean()),
            
            # Accuracy (score >= 0.9)
            'accuracy_no_fen': float((df['score_no_fen'] >= 0.9).mean() * 100),
            'accuracy_pred_fen': float((df['score_pred_fen'] >= 0.9).mean() * 100),
            'accuracy_gt_fen': float((df['score_gt_fen'] >= 0.9).mean() * 100),
            
            # Improvements
            'avg_improvement_pred_vs_none': float(df['improvement_pred_vs_none'].mean()),
            'avg_improvement_gt_vs_none': float(df['improvement_gt_vs_none'].mean()),
            'avg_improvement_gt_vs_pred': float(df['improvement_gt_vs_pred'].mean()),
            
            # Relative improvements
            'relative_improvement_pred_vs_none': float(
                (df['score_pred_fen'].mean() - df['score_no_fen'].mean()) / 
                max(df['score_no_fen'].mean(), 0.01) * 100
            ),
            'relative_improvement_gt_vs_none': float(
                (df['score_gt_fen'].mean() - df['score_no_fen'].mean()) / 
                max(df['score_no_fen'].mean(), 0.01) * 100
            ),
        }
        
        # Per-question statistics
        questions_summary = {}
        for q_id in df['question_id'].unique():
            qdf = df[df['question_id'] == q_id]
            q_type = qdf['question_type'].iloc[0]
            
            questions_summary[str(q_id)] = {
                'question_type': q_type,
                'avg_score_no_fen': float(qdf['score_no_fen'].mean()),
                'avg_score_pred_fen': float(qdf['score_pred_fen'].mean()),
                'avg_score_gt_fen': float(qdf['score_gt_fen'].mean()),
                'accuracy_no_fen': float((qdf['score_no_fen'] >= 0.9).mean() * 100),
                'accuracy_pred_fen': float((qdf['score_pred_fen'] >= 0.9).mean() * 100),
                'accuracy_gt_fen': float((qdf['score_gt_fen'] >= 0.9).mean() * 100),
                'improvement_pred_vs_none': float(qdf['improvement_pred_vs_none'].mean()),
                'improvement_gt_vs_none': float(qdf['improvement_gt_vs_none'].mean()),
                'total_tests': int(len(qdf))
            }
        
        summary['questions_summary'] = questions_summary
        summary['timestamp'] = datetime.now().isoformat()
        
        return summary
    
    def _save_summary(self, summary: Dict, output_dir: str):
        """Save summary to JSON."""
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark JSON-based models for chess VLM reasoning"
    )
    
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSONL file (e.g., predictions_clip_exp1b.jsonl)"
    )
    parser.add_argument(
        "--dataset_csv",
        type=str,
        required=True,
        help="Path to dataset CSV with ground truth FENs"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Directory containing images (default: derived from dataset)"
    )
    parser.add_argument(
        "--vlm_model",
        type=str,
        default="gpt-4o",
        help="VLM model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--use_mock_vlm",
        action="store_true",
        help="Use mock VLM for testing"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of images to benchmark"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results_json",
        help="Output directory for results"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="exp1b",
        help="Experiment name (e.g., exp1a, exp1b, exp1c, exp1d)"
    )
    
    args = parser.parse_args()
    
    # Update output dir with experiment name
    args.output_dir = f"{args.output_dir}_{args.experiment_name}"
    
    # Initialize benchmark
    benchmark = JSONModelBenchmark(
        vlm_model_name=args.vlm_model,
        use_mock_vlm=args.use_mock_vlm
    )
    
    # Load predictions
    predictions = benchmark.load_predictions(args.predictions)
    
    # Load ground truth FENs
    gt_fens = benchmark.load_ground_truth_fens(args.dataset_csv)
    
    # Get image paths from predictions
    if args.images_dir:
        images_dir = Path(args.images_dir)
    else:
        # Derive from dataset CSV
        csv_dir = Path(args.dataset_csv).parent
        images_dir = csv_dir / "images"
        if not images_dir.exists():
            images_dir = csv_dir.parent / "test" / "images"
    
    # Build image paths list
    image_paths = []
    for filename in predictions.keys():
        img_path = images_dir / filename
        if img_path.exists():
            image_paths.append(str(img_path))
        else:
            # Try alternate paths
            alt_paths = [
                Path(args.dataset_csv).parent.parent / "test" / "images" / filename,
                Path(args.dataset_csv).parent / filename,
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    image_paths.append(str(alt_path))
                    break
    
    # Limit to specified number
    if len(image_paths) > args.num_images:
        image_paths = image_paths[:args.num_images]
    
    print(f"Found {len(image_paths)} images to benchmark")
    
    if not image_paths:
        print("Error: No images found!")
        return
    
    # Run benchmark
    summary = benchmark.run_benchmark(
        image_paths=image_paths,
        predictions=predictions,
        gt_fens=gt_fens,
        output_dir=args.output_dir
    )
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Total tests: {summary['total_tests']}")
    print(f"\nAverage Scores:")
    print(f"  No FEN:       {summary['avg_score_no_fen']:.3f} (Acc: {summary['accuracy_no_fen']:.1f}%)")
    print(f"  Pred FEN:     {summary['avg_score_pred_fen']:.3f} (Acc: {summary['accuracy_pred_fen']:.1f}%)")
    print(f"  GT FEN:       {summary['avg_score_gt_fen']:.3f} (Acc: {summary['accuracy_gt_fen']:.1f}%)")
    print(f"\nImprovements:")
    print(f"  Pred vs None: {summary['avg_improvement_pred_vs_none']:+.3f} ({summary['relative_improvement_pred_vs_none']:+.1f}%)")
    print(f"  GT vs None:   {summary['avg_improvement_gt_vs_none']:+.3f} ({summary['relative_improvement_gt_vs_none']:+.1f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
