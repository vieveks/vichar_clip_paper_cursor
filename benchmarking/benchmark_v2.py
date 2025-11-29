"""
Fast benchmark script that uses ground truth FENs directly from dataset.
Skips CLIP FEN extraction since CLIP already has 99%+ accuracy.
This is much faster for large-scale benchmarking.
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
from datetime import datetime
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from questions import QUESTIONS, get_scoring_questions
from ground_truth import GroundTruthExtractor
from vlm_integration import VLMInterface, MockVLMInterface, OpenAIVLMInterface
from llm_judge_scorer import LLMJudgeScorer
from scoring import ResponseScorer


class FastChessVLMBenchmark:
    """Fast benchmark that uses ground truth FENs directly (no CLIP extraction)."""
    
    def __init__(
        self,
        vlm_model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        use_mock_vlm: bool = False,
        device: str = None,
        dataset_csv: str = None,
        local_vlm_path: str = None
    ):
        """
        Initialize the fast benchmark.
        
        Args:
            vlm_model_name: VLM model identifier
            use_mock_vlm: Use mock VLM for testing
            device: Device to run inference on
            dataset_csv: CSV file with ground truth FENs
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_csv = dataset_csv
        
        # Load image-to-FEN mapping from dataset
        self.image_to_fen = {}
        if dataset_csv:
            self._load_image_to_fen_mapping()
        
        print("Initializing ground truth extractor...")
        self.ground_truth_extractor = GroundTruthExtractor(dataset_csv=dataset_csv)
        
        print("Initializing VLM...")
        if use_mock_vlm:
            self.vlm = MockVLMInterface()
            print("Using mock VLM for testing")
        elif vlm_model_name.startswith("gpt-"):
            # Use OpenAI Interface
            from vlm_integration import OpenAIVLMInterface
            try:
                self.vlm = OpenAIVLMInterface(model_name=vlm_model_name)
            except Exception as e:
                print(f"Warning: Could not load OpenAI VLM: {e}")
                self.vlm = MockVLMInterface()
        else:
            try:
                self.vlm = VLMInterface(vlm_model_name, self.device, local_model_path=local_vlm_path)
            except Exception as e:
                print(f"Warning: Could not load VLM, using mock: {e}")
                self.vlm = MockVLMInterface()
        
        self.llm_scorer = LLMJudgeScorer()  # Use LLM judge for scoring
        self.scorer = ResponseScorer()  # Use ResponseScorer for specific methods
        
        # Results storage
        self.results = []
    
    def _load_image_to_fen_mapping(self):
        """Load mapping from image paths to ground truth FEN from dataset CSV."""
        if not self.dataset_csv:
            return
        
        try:
            df = pd.read_csv(self.dataset_csv)
            import os
            for _, row in df.iterrows():
                image_path = row.get('image_path', '')
                fen = row.get('fen', '')
                if image_path and fen:
                    # Normalize path (handle both Windows and Unix paths)
                    normalized_path = image_path.replace('\\', '/')
                    filename = os.path.basename(normalized_path)
                    
                    # Store multiple variations for matching
                    self.image_to_fen[normalized_path] = fen
                    self.image_to_fen[filename] = fen  # Store by filename
                    # Also store with test/images/ prefix
                    if 'test' in normalized_path.lower():
                        self.image_to_fen[f"test/images/{filename}"] = fen
                        self.image_to_fen[f"test\\images\\{filename}"] = fen
            print(f"[OK] Loaded {len(df)} image-to-FEN mappings from dataset")
        except Exception as e:
            print(f"Warning: Could not load image-to-FEN mapping: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_ground_truth_fen(self, image_path: str) -> Optional[str]:
        """Get ground truth FEN for an image from dataset CSV."""
        import os
        # Try multiple path formats
        normalized_path = image_path.replace('\\', '/')
        filename = os.path.basename(normalized_path)
        
        # Try exact match first
        if normalized_path in self.image_to_fen:
            return self.image_to_fen[normalized_path]
        
        # Try matching by filename (most reliable)
        if filename in self.image_to_fen:
            return self.image_to_fen[filename]
        
        # Try partial matches
        for key, fen in self.image_to_fen.items():
            if filename in key or key.endswith(filename) or filename in key:
                return fen
        
        return None
    
    def run_benchmark(
        self,
        image_paths: List[str],
        questions: Optional[List[Dict]] = None,
        output_dir: str = "benchmark_results",
        dataset_csv: Optional[str] = None
    ) -> Dict:
        """
        Run the fast benchmark on a set of images using ground truth FENs.
        
        Args:
            image_paths: List of paths to chess board images
            questions: Optional list of question dicts (uses all if None)
            output_dir: Directory to save results
            dataset_csv: CSV file with ground truth FENs
            
        Returns:
            Dict with benchmark results
        """
        if questions is None:
            questions = get_scoring_questions()  # Only questions with weight > 0
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Running FAST benchmark on {len(image_paths)} images")
        print(f"Testing {len(questions)} questions")
        print(f"Using GROUND TRUTH FENs (no CLIP extraction)")
        print(f"{'='*60}\n")
        
        # STEP 1: Load ground truth FENs for all images
        print("="*60)
        print("STEP 1: Loading ground truth FENs from dataset...")
        print("="*60)
        gt_fens = {}
        for img_idx, image_path in enumerate(image_paths):
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            gt_fen = self._get_ground_truth_fen(image_path)
            if gt_fen:
                gt_fens[image_path] = gt_fen
                if (img_idx + 1) % 10 == 0:
                    print(f"  [{img_idx + 1}/{len(image_paths)}] Loaded FENs...")
            else:
                print(f"  Warning: No FEN found for {os.path.basename(image_path)}")
        
        print(f"\n[OK] Loaded ground truth FENs for {len(gt_fens)} images")
        
        # STEP 2: Batch extract ground truths for all images
        print("\n" + "="*60)
        print("STEP 2: Extracting ground truth answers for all positions...")
        print("="*60)
        all_ground_truths = {}
        for img_idx, (image_path, gt_fen) in enumerate(gt_fens.items()):
            if gt_fen:
                all_ground_truths[image_path] = self._get_ground_truths(gt_fen, questions)
                if (img_idx + 1) % 10 == 0:
                    print(f"  [{img_idx + 1}/{len(gt_fens)}] Extracted ground truths...")
        
        print(f"[OK] Extracted ground truth for {len(all_ground_truths)} images")
        
        # STEP 3: Process all VLM questions
        print("\n" + "="*60)
        print("STEP 3: Running VLM on all images and questions...")
        print("="*60)
        all_results = []
        
        total_tests = len(gt_fens) * len(questions)
        test_count = 0
        
        for img_idx, image_path in enumerate(gt_fens.keys()):
            gt_fen = gt_fens[image_path]
            ground_truths = all_ground_truths.get(image_path, {})
            
            print(f"\n[{img_idx + 1}/{len(gt_fens)}] Processing: {os.path.basename(image_path)}")
            print(f"  Ground truth FEN: {gt_fen[:50]}...")
            
            # Test each question
            for question in questions:
                test_count += 1
                print(f"  [{test_count}/{total_tests}] Question {question['id']}: {question['prompt'][:50]}...")
                
                # Test without FEN context (base VLM)
                answer_without_fen = self.vlm.answer_question(
                    image_path, question['prompt'], fen_context=None
                )
                score_without_fen = self._score_response(
                    answer_without_fen, question, ground_truths.get(question['id'])
                )
                
                # Test with GROUND TRUTH FEN context (VLM + FEN)
                answer_with_fen = self.vlm.answer_question(
                    image_path, question['prompt'], fen_context=gt_fen
                )
                score_with_fen = self._score_response(
                    answer_with_fen, question, ground_truths.get(question['id'])
                )
                
                # Store results
                result = {
                    'image_path': image_path,
                    'question_id': question['id'],
                    'question_type': question['type'],
                    'question_prompt': question['prompt'],
                    'ground_truth_fen': gt_fen,  # Ground truth FEN (used as context)
                    'answer_without_fen': answer_without_fen,  # Base VLM answer
                    'answer_with_fen': answer_with_fen,  # VLM + FEN answer
                    'score_without_fen': score_without_fen,  # Base VLM score
                    'score_with_fen': score_with_fen,  # VLM + FEN score
                    'improvement': score_with_fen - score_without_fen,
                    'ground_truth': ground_truths.get(question['id']),  # From ground truth FEN
                    'weight': question['weight']
                }
                all_results.append(result)
                
                print(f"    Without FEN: {score_without_fen:.2f} | With FEN: {score_with_fen:.2f} | Improvement: {score_with_fen - score_without_fen:+.2f}")
        
        # Save results
        self.results = all_results
        self._save_results(all_results, output_dir)
        
        # Generate summary
        summary = self._generate_summary(all_results)
        self._save_summary(summary, output_dir)
        
        return summary
    
    def _get_ground_truths(self, fen_string: Optional[str], questions: List[Dict]) -> Dict[int, Any]:
        """Extract ground truth answers for all questions from a FEN string."""
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
                elif q_type == "piece_on_square":
                    # This requires a square parameter, skip for now
                    pass
                elif q_type == "tactical_pattern":
                    # Complex, skip for now
                    pass
            except Exception as e:
                print(f"    Warning: Could not extract ground truth for question {q_id}: {e}")
        
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
                # Use material count scorer for piece count
                return self.scorer.score_material_count(response, ground_truth)
            elif q_type == "castling_available":
                return self.scorer.score_castling_rights(response, ground_truth)
            elif q_type == "best_move":
                return self.scorer.score_best_move(response, ground_truth)
            else:
                # Generic scoring using LLM judge for other types
                return self.llm_scorer.score_response(
                    question['prompt'], response, ground_truth, q_type
                )
        except Exception as e:
            print(f"    Warning: Scoring error: {e}")
            return 0.0
    
    def _save_results(self, results: List[Dict], output_dir: str):
        """Save detailed results to JSON and CSV."""
        # Save JSON
        json_path = os.path.join(output_dir, "detailed_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Saved detailed results to {json_path}")
        
        # Save CSV
        import pandas as pd
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"[OK] Saved results CSV to {csv_path}")
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics from results."""
        import pandas as pd
        df = pd.DataFrame(results)
        
        # Overall statistics
        summary = {
            'total_images': df['image_path'].nunique(),
            'total_questions': df['question_id'].nunique(),
            'total_tests': len(results),
            'average_score_without_fen': float(df['score_without_fen'].mean()),
            'average_score_with_fen': float(df['score_with_fen'].mean()),
            'accuracy_without_fen': float((df['score_without_fen'] >= 0.9).mean() * 100),
            'accuracy_with_fen': float((df['score_with_fen'] >= 0.9).mean() * 100),
            'accuracy_improvement': float((df['score_with_fen'] >= 0.9).mean() * 100 - (df['score_without_fen'] >= 0.9).mean() * 100),
            'average_improvement': float(df['score_with_fen'].mean() - df['score_without_fen'].mean()),
            'improvement_percentage': float((df['score_with_fen'].mean() - df['score_without_fen'].mean()) / max(df['score_without_fen'].mean(), 0.01) * 100),
        }
        
        # Per-question statistics
        questions_summary = {}
        for q_id in df['question_id'].unique():
            qdf = df[df['question_id'] == q_id]
            q_type = qdf['question_type'].iloc[0]
            
            q_accuracy_without = (qdf['score_without_fen'] >= 0.9).mean() * 100
            q_accuracy_with = (qdf['score_with_fen'] >= 0.9).mean() * 100
            
            questions_summary[str(q_id)] = {
                'question_type': q_type,
                'average_score_without_fen': float(qdf['score_without_fen'].mean()),
                'average_score_with_fen': float(qdf['score_with_fen'].mean()),
                'accuracy_without_fen': float(q_accuracy_without),
                'accuracy_with_fen': float(q_accuracy_with),
                'average_improvement': float(qdf['score_with_fen'].mean() - qdf['score_without_fen'].mean()),
                'total_tests': int(len(qdf))
            }
        
        summary['questions_summary'] = questions_summary
        
        return summary
    
    def _save_summary(self, summary: Dict, output_dir: str):
        """Save summary to JSON."""
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] Saved summary to {summary_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fast Chess VLM Benchmark (uses ground truth FENs)")
    
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=False,
        help="Paths to chess board images to test"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Directory containing chess board images (alternative to --images)"
    )
    parser.add_argument(
        "--dataset_csv",
        type=str,
        required=True,
        help="CSV file with ground truth FENs (must have 'image_path' and 'fen' columns)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--vlm_model",
        type=str,
        default="llava-hf/llava-v1.6-mistral-7b-hf",
        help="VLM model identifier (default: LLaVA 1.6 Mistral)"
    )
    parser.add_argument(
        "--use_mock_vlm",
        action="store_true",
        help="Use mock VLM for testing (no actual model required)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (cuda/cpu)"
    )
    parser.add_argument(
        "--local_vlm_path",
        type=str,
        default=None,
        help="Path to local LLaVA model directory (e.g., for Mistral version)"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=100,
        help="Number of images to benchmark (default: 100, use 1 for quick test)"
    )
    
    args = parser.parse_args()
    
    # Collect image paths
    image_paths = []
    if args.images:
        image_paths.extend(args.images)
    if args.images_dir:
        image_dir = Path(args.images_dir)
        image_paths.extend([str(p) for p in image_dir.glob("*.png")])
        image_paths.extend([str(p) for p in image_dir.glob("*.jpg")])
        image_paths.extend([str(p) for p in image_dir.glob("*.jpeg")])
    
    if not image_paths:
        print("Error: No images found. Provide --images or --images_dir")
        return
    
    # Limit to specified number of images
    num_images = args.num_images
    if len(image_paths) > num_images:
        print(f"Found {len(image_paths)} images, limiting to first {num_images} for testing")
        image_paths = image_paths[:num_images]
    else:
        print(f"Found {len(image_paths)} images to process")
    
    # Initialize and run benchmark
    benchmark = FastChessVLMBenchmark(
        vlm_model_name=args.vlm_model,
        use_mock_vlm=args.use_mock_vlm,
        device=args.device,
        dataset_csv=args.dataset_csv,
        local_vlm_path=args.local_vlm_path
    )
    
    summary = benchmark.run_benchmark(
        image_paths=image_paths,
        output_dir=args.output_dir,
        dataset_csv=args.dataset_csv
    )
    
    print("\n[OK] Fast benchmark completed!")
    print(f"\nSummary:")
    print(f"  Average score without FEN: {summary['average_score_without_fen']:.3f}")
    print(f"  Average score with FEN: {summary['average_score_with_fen']:.3f}")
    print(f"  Improvement: {summary['average_improvement']:.3f} ({summary['improvement_percentage']:.1f}%)")


if __name__ == "__main__":
    main()

