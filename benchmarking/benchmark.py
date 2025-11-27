"""
Main benchmarking script for evaluating VLM performance with and without FEN context.
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
from clip_fen_extractor import CLIPFENExtractor
from ground_truth import GroundTruthExtractor
from vlm_integration import VLMInterface, MockVLMInterface, OpenAIVLMInterface
from llm_judge_scorer import LLMJudgeScorer


class ChessVLMBenchmark:
    """Main benchmark class for evaluating VLM performance."""
    
    def __init__(
        self,
        clip_checkpoint_path: str,
        clip_model_name: str = "ViT-B-32",
        vlm_model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",  # LLaVA 1.6 Mistral
        use_mock_vlm: bool = False,
        device: str = None,
        dataset_csv: str = None,
        local_vlm_path: str = None
    ):
        """
        Initialize the benchmark.
        
        Args:
            clip_checkpoint_path: Path to trained CLIP model checkpoint
            clip_model_name: CLIP model architecture name
            vlm_model_name: VLM model identifier
            use_mock_vlm: Use mock VLM for testing
            device: Device to run inference on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        print("Initializing CLIP FEN extractor...")
        self.clip_extractor = CLIPFENExtractor(clip_checkpoint_path, clip_model_name, self.device)
        
        print("Initializing ground truth extractor...")
        # Load dataset CSV to map images to ground truth FEN
        self.dataset_csv = dataset_csv
        self.image_to_gt_fen = {}
        if dataset_csv:
            self._load_image_to_fen_mapping()
        
        # Ground truth extractor uses dataset FEN (not predicted FEN)
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
        
        self.scorer = LLMJudgeScorer()  # Use LLM judge for scoring
        
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
                    self.image_to_gt_fen[normalized_path] = fen
                    self.image_to_gt_fen[filename] = fen  # Store by filename
                    # Also store with test/images/ prefix
                    if 'test' in normalized_path.lower():
                        self.image_to_gt_fen[f"test/images/{filename}"] = fen
                        self.image_to_gt_fen[f"test\\images\\{filename}"] = fen
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
        if normalized_path in self.image_to_gt_fen:
            return self.image_to_gt_fen[normalized_path]
        
        # Try matching by filename (most reliable)
        if filename in self.image_to_gt_fen:
            return self.image_to_gt_fen[filename]
        
        # Try partial matches
        for key, fen in self.image_to_gt_fen.items():
            if filename in key or key.endswith(filename) or filename in key:
                return fen
        
        return None
    
    def run_benchmark(
        self,
        image_paths: List[str],
        questions: Optional[List[Dict]] = None,
        output_dir: str = "benchmark_results",
        fen_candidates_csv: Optional[str] = None,
        dataset_csv: Optional[str] = None
    ) -> Dict:
        """
        Run the benchmark on a set of images.
        
        Args:
            image_paths: List of paths to chess board images
            questions: Optional list of question dicts (uses all if None)
            output_dir: Directory to save results
            fen_candidates_csv: Optional CSV with FEN candidates for CLIP matching
            
        Returns:
            Dict with benchmark results
        """
        if questions is None:
            questions = get_scoring_questions()  # Only questions with weight > 0
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Running benchmark on {len(image_paths)} images")
        print(f"Testing {len(questions)} questions")
        print(f"{'='*60}\n")
        
        # STEP 1: Batch extract all FENs first (more efficient)
        print("="*60)
        print("STEP 1: Batch extracting FENs from all images...")
        print("="*60)
        predicted_fens = {}
        gt_fens = {}
        
        for img_idx, image_path in enumerate(image_paths):
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            print(f"  [{img_idx + 1}/{len(image_paths)}] Extracting FEN from {os.path.basename(image_path)}...")
            
            # Extract PREDICTED FEN
            try:
                if fen_candidates_csv:
                    predicted_fen = self.clip_extractor.extract_fen_from_image_with_candidates_file(
                        image_path, fen_candidates_csv, top_k=1
                    )
                else:
                    predicted_fen = self.clip_extractor.extract_fen_from_image(image_path, top_k=1)
                predicted_fens[image_path] = predicted_fen
                # Clear GPU cache after CLIP inference to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"    Error extracting FEN: {e}")
                predicted_fens[image_path] = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Get GROUND TRUTH FEN
            gt_fen = self._get_ground_truth_fen(image_path)
            gt_fens[image_path] = gt_fen
        
        print(f"\n[OK] Extracted FENs from {len(predicted_fens)} images")
        
        # STEP 2: Batch extract ground truths for all images
        print("\n" + "="*60)
        print("STEP 2: Extracting ground truth for all positions...")
        print("="*60)
        all_ground_truths = {}
        for image_path in predicted_fens.keys():
            # Use ground truth FEN if available, otherwise use predicted FEN
            gt_fen = gt_fens.get(image_path)
            fen_for_gt = gt_fen if gt_fen else predicted_fens.get(image_path)
            
            if fen_for_gt:
                all_ground_truths[image_path] = self._get_ground_truths(fen_for_gt, questions)
                if not gt_fen:
                    print(f"  Note: Using predicted FEN as ground truth for {os.path.basename(image_path)}")
        
        print(f"[OK] Extracted ground truth for {len(all_ground_truths)} images")
        
        # STEP 3: Process all VLM questions
        print("\n" + "="*60)
        print("STEP 3: Running VLM on all images and questions...")
        print("="*60)
        all_results = []
        
        total_tests = len(image_paths) * len(questions)
        test_count = 0
        
        for img_idx, image_path in enumerate(image_paths):
            if image_path not in predicted_fens:
                continue
            
            predicted_fen = predicted_fens[image_path]
            gt_fen = gt_fens.get(image_path)
            ground_truths = all_ground_truths.get(image_path, {})
            
            print(f"\n[{img_idx + 1}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")
            if predicted_fen:
                print(f"  Predicted FEN: {predicted_fen[:50]}...")
            if gt_fen:
                print(f"  Ground truth FEN: {gt_fen[:50]}...")
            
            # Test each question
            for question in questions:
                test_count += 1
                print(f"  [{test_count}/{total_tests}] Question {question['id']}: {question['prompt'][:50]}...")
                
                # Test without FEN context (base LLaVA)
                answer_without_fen = self.vlm.answer_question(
                    image_path, question['prompt'], fen_context=None
                )
                score_without_fen = self._score_response(
                    answer_without_fen, question, ground_truths.get(question['id'])
                )
                
                # Test with PREDICTED FEN context (LLaVA + predicted FEN)
                answer_with_fen = self.vlm.answer_question(
                    image_path, question['prompt'], fen_context=predicted_fen
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
                    'predicted_fen': predicted_fen,  # CLIP-predicted FEN (used as context)
                    'ground_truth_fen': gt_fen,  # Ground truth FEN from dataset (used for ground truth)
                    'answer_without_fen': answer_without_fen,  # Base LLaVA answer
                    'answer_with_fen': answer_with_fen,  # LLaVA + predicted FEN answer
                    'score_without_fen': score_without_fen,  # Base LLaVA score
                    'score_with_fen': score_with_fen,  # LLaVA + predicted FEN score
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
        """Extract ground truth for all questions."""
        ground_truths = {}
        
        if not fen_string:
            return ground_truths
        
        for question in questions:
            qid = question['id']
            qtype = question['type']
            
            try:
                if qtype == "fen_extraction":
                    ground_truths[qid] = fen_string
                elif qtype == "piece_count":
                    ground_truths[qid] = self.ground_truth_extractor.get_piece_count(fen_string)
                elif qtype == "check_status":
                    ground_truths[qid] = self.ground_truth_extractor.get_check_status(fen_string)
                elif qtype == "material_balance":
                    ground_truths[qid] = self.ground_truth_extractor.get_material_balance(fen_string)
                elif qtype == "best_move":
                    ground_truths[qid] = self.ground_truth_extractor.get_best_move(fen_string)
                elif qtype == "tactical_pattern":
                    ground_truths[qid] = self.ground_truth_extractor.get_tactical_patterns(fen_string)
                elif qtype == "castling_available":
                    ground_truths[qid] = self.ground_truth_extractor.get_castling_rights(fen_string)
                elif qtype == "piece_on_square":
                    ground_truths[qid] = self.ground_truth_extractor.get_piece_on_square(fen_string, "e4")
            except Exception as e:
                print(f"    Warning: Could not get ground truth for question {qid}: {e}")
                ground_truths[qid] = None
        
        return ground_truths
    
    def _score_response(self, response: str, question: Dict, ground_truth: Any) -> float:
        """Score a VLM response against ground truth using LLM judge."""
        if ground_truth is None:
            return 0.0
        
        qtype = question['type']
        
        try:
            # Use LLM judge for all scoring
            return self.scorer.score_response(
                question=question['prompt'],
                response=response,
                ground_truth=ground_truth,
                question_type=qtype
            )
        except Exception as e:
            print(f"    Warning: Error scoring response: {e}")
            return 0.0
    
    def _save_results(self, results: List[Dict], output_dir: str):
        """Save detailed results to JSON and CSV."""
        # Save JSON
        json_path = os.path.join(output_dir, "detailed_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Saved detailed results to {json_path}")
        
        # Save CSV
        csv_path = os.path.join(output_dir, "results.csv")
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(f"[OK] Saved CSV results to {csv_path}")
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics."""
        df = pd.DataFrame(results)
        
        # Calculate accuracy (percentage of perfect scores, i.e., score >= 0.9)
        accuracy_without_fen = (df['score_without_fen'] >= 0.9).mean() * 100
        accuracy_with_fen = (df['score_with_fen'] >= 0.9).mean() * 100
        
        summary = {
            'total_images': int(df['image_path'].nunique()),
            'total_questions': int(df['question_id'].nunique()),
            'total_tests': int(len(results)),
            'average_score_without_fen': float(df['score_without_fen'].mean()),
            'average_score_with_fen': float(df['score_with_fen'].mean()),
            'accuracy_without_fen': float(accuracy_without_fen),
            'accuracy_with_fen': float(accuracy_with_fen),
            'accuracy_improvement': float(accuracy_with_fen - accuracy_without_fen),
            'average_improvement': float(df['improvement'].mean()),
            'improvement_percentage': float((df['score_with_fen'].mean() - df['score_without_fen'].mean()) / max(df['score_without_fen'].mean(), 0.01) * 100),
            'questions_summary': {}
        }
        
        # Per-question summary (convert numpy int64 to regular int for JSON)
        for qid in df['question_id'].unique():
            qid_int = int(qid)  # Convert numpy int64 to regular int
            qdf = df[df['question_id'] == qid]
            q_accuracy_without = (qdf['score_without_fen'] >= 0.9).mean() * 100
            q_accuracy_with = (qdf['score_with_fen'] >= 0.9).mean() * 100
            summary['questions_summary'][qid_int] = {
                'question_type': qdf['question_type'].iloc[0],
                'average_score_without_fen': float(qdf['score_without_fen'].mean()),
                'average_score_with_fen': float(qdf['score_with_fen'].mean()),
                'accuracy_without_fen': float(q_accuracy_without),
                'accuracy_with_fen': float(q_accuracy_with),
                'average_improvement': float(qdf['improvement'].mean()),
                'total_tests': int(len(qdf))
            }
        
        return summary
    
    def _save_summary(self, summary: Dict, output_dir: str):
        """Save summary to file."""
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also print summary
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Total images tested: {summary['total_images']}")
        print(f"Total questions: {summary['total_questions']}")
        print(f"Total tests: {summary['total_tests']}")
        print(f"\nOverall Performance:")
        print(f"  Average score without FEN: {summary['average_score_without_fen']:.3f}")
        print(f"  Average score with FEN: {summary['average_score_with_fen']:.3f}")
        print(f"  Accuracy without FEN: {summary['accuracy_without_fen']:.2f}%")
        print(f"  Accuracy with FEN: {summary['accuracy_with_fen']:.2f}%")
        print(f"  Accuracy improvement: {summary['accuracy_improvement']:+.2f}%")
        print(f"  Average improvement: {summary['average_improvement']:+.3f}")
        print(f"  Improvement percentage: {summary['improvement_percentage']:+.1f}%")
        print(f"\nPer-Question Performance:")
        for qid, qsummary in summary['questions_summary'].items():
            print(f"\n  Question {qid} ({qsummary['question_type']}):")
            print(f"    Without FEN: Score={qsummary['average_score_without_fen']:.3f}, Accuracy={qsummary['accuracy_without_fen']:.2f}%")
            print(f"    With FEN: Score={qsummary['average_score_with_fen']:.3f}, Accuracy={qsummary['accuracy_with_fen']:.2f}%")
            print(f"    Improvement: {qsummary['average_improvement']:+.3f}")
        print(f"\n[OK] Summary saved to {summary_path}")


def main():
    """Main entry point for the benchmark."""
    parser = argparse.ArgumentParser(description="Chess VLM Benchmark")
    parser.add_argument(
        "--clip_checkpoint",
        type=str,
        required=True,
        help="Path to trained CLIP model checkpoint"
    )
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
        "--fen_candidates",
        type=str,
        default=None,
        help="CSV file with FEN candidates for CLIP matching"
    )
    parser.add_argument(
        "--dataset_csv",
        type=str,
        default=None,
        help="CSV file with ground truth FEN and best moves (same as fen_candidates usually)"
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
        "--clip_model",
        type=str,
        default="ViT-B-32",
        help="CLIP model architecture name"
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
    benchmark = ChessVLMBenchmark(
        clip_checkpoint_path=args.clip_checkpoint,
        clip_model_name=args.clip_model,
        vlm_model_name=args.vlm_model,
        use_mock_vlm=args.use_mock_vlm,
        device=args.device,
        local_vlm_path=args.local_vlm_path
    )
    
    # Use dataset_csv if provided, otherwise use fen_candidates (same file usually)
    dataset_csv = args.dataset_csv if args.dataset_csv else args.fen_candidates
    
    summary = benchmark.run_benchmark(
        image_paths=image_paths,
        output_dir=args.output_dir,
        fen_candidates_csv=args.fen_candidates,
        dataset_csv=dataset_csv
    )
    
    print("\n[OK] Benchmark completed!")


if __name__ == "__main__":
    main()

