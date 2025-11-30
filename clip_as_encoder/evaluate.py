"""
Evaluation script for comparing baseline LLaVA vs ChessCLIP-LLaVA.
"""

import argparse
import torch
import json
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import Dict, List
import logging

from model import ChessLLaVA
from dataset import ChessQADataset, create_qa_dataset_from_benchmark

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import CLIP FEN extractor
import sys
sys.path.append(str(Path(__file__).parent.parent / "benchmarking"))
try:
    from clip_fen_extractor import CLIPFENExtractor
    CLIP_EXTRACTOR_AVAILABLE = True
except ImportError:
    CLIP_EXTRACTOR_AVAILABLE = False
    logger.warning("CLIPFENExtractor not available. Cannot use CLIP-predicted FEN.")


def evaluate_model(model, dataloader, device, questions, clip_predicted_fens=None):
    """Evaluate model on dataset.
    
    Args:
        model: ChessLLaVA model to evaluate
        dataloader: DataLoader with evaluation samples
        device: Device to run on
        questions: List of question definitions
        clip_predicted_fens: Optional dict mapping image_path -> predicted FEN
    """
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            questions_batch = batch["question"]
            answers_batch = batch["answer"]
            image_paths = batch["image_path"]
            ground_truth_fens = batch.get("fen", [None] * len(images))
            
            # Use CLIP-predicted FEN if available, otherwise use ground truth
            if clip_predicted_fens:
                fens = []
                for img_path in image_paths:
                    predicted_fen = clip_predicted_fens.get(img_path)
                    if predicted_fen:
                        fens.append(predicted_fen)
                    else:
                        # Fallback to ground truth
                        idx = image_paths.index(img_path) if img_path in image_paths else 0
                        fens.append(ground_truth_fens[idx] if idx < len(ground_truth_fens) else None)
            else:
                fens = ground_truth_fens
            
            # Generate responses - process one at a time to avoid batch issues
            responses = []
            for i in range(len(images)):
                try:
                    # Get single image and question
                    # Extract single image tensor (remove batch dim if needed)
                    if images.dim() == 4:  # [B, C, H, W]
                        single_image = images[i:i+1]  # Keep batch dimension for tensor
                    else:
                        single_image = images[i]
                    
                    single_question = questions_batch[i]
                    single_fen = fens[i] if i < len(fens) and fens[i] else None
                    
                    # Add FEN context to prompt if available
                    if single_fen:
                        fen_context = f"\n\nContext: The FEN (Forsyth-Edwards Notation) for this position is: {single_fen}"
                        enhanced_question = single_question + fen_context
                    else:
                        enhanced_question = single_question
                    
                    # Format prompt based on model type
                    if hasattr(model, 'llava_type') and model.llava_type == "next":
                        # LLaVA Next uses chat template - just pass the question
                        prompt = enhanced_question
                    else:
                        prompt = f"USER: <image>\n{enhanced_question}\nASSISTANT:"
                    
                    # Clear GPU cache before each generation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Generate for single image
                    single_responses = model.generate(
                        images=single_image,
                        prompts=[prompt],
                        max_new_tokens=128  # Reduced from 256 to save memory
                    )
                    responses.extend(single_responses)
                    
                    # Clear GPU cache after each generation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e) or "CUDA" in str(e):
                        logger.warning(f"OOM error processing image {i}, clearing cache and skipping")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        responses.append("Error: Out of memory")
                    else:
                        logger.warning(f"Error processing image {i} in batch: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        responses.append("Error generating response")
                except Exception as e:
                    logger.warning(f"Error processing image {i} in batch: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    responses.append("Error generating response")
            
            # Store results
            for i, (image_path, question, answer, response, fen) in enumerate(
                zip(image_paths, questions_batch, answers_batch, responses, fens)
            ):
                # Find question type
                question_type = None
                for q in questions:
                    if q['prompt'] == question:
                        question_type = q['type']
                        break
                
                results.append({
                    "image_path": image_path,
                    "question": question,
                    "question_type": question_type,
                    "ground_truth_answer": answer,
                    "model_response": response,
                    "fen": fen
                })
    
    return results


def score_responses(results: List[Dict], questions: List[Dict]) -> Dict:
    """Score model responses using LLM judge."""
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "benchmarking"))
    
    try:
        from llm_judge_scorer import LLMJudgeScorer
        scorer = LLMJudgeScorer()
    except ImportError:
        logger.warning("Could not import LLMJudgeScorer, using simple scoring")
        scorer = None
    
    scored_results = []
    
    for result in results:
        question_type = result["question_type"]
        ground_truth = result["ground_truth_answer"]
        response = result["model_response"]
        question = result["question"]
        
        if scorer:
            try:
                score = scorer.score_response(
                    question=question,
                    response=response,
                    ground_truth=ground_truth,
                    question_type=question_type
                )
            except Exception as e:
                logger.warning(f"Scoring error: {e}")
                score = 0.0
        else:
            # Simple exact match fallback
            score = 1.0 if response.strip().lower() == ground_truth.strip().lower() else 0.0
        
        result["score"] = score
        scored_results.append(result)
    
    return scored_results


def generate_summary(scored_results: List[Dict], questions: List[Dict]) -> Dict:
    """Generate summary statistics."""
    df = pd.DataFrame(scored_results)
    
    summary = {
        "total_samples": len(scored_results),
        "average_score": float(df["score"].mean()),
        "accuracy": float((df["score"] >= 0.9).mean() * 100),
        "per_question_type": {}
    }
    
    # Per question type
    for question in questions:
        qtype = question["type"]
        qdf = df[df["question_type"] == qtype]
        
        if len(qdf) > 0:
            summary["per_question_type"][qtype] = {
                "count": int(len(qdf)),
                "average_score": float(qdf["score"].mean()),
                "accuracy": float((qdf["score"] >= 0.9).mean() * 100)
            }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChessLLaVA models")
    
    # Model arguments
    parser.add_argument(
        "--baseline_model",
        type=str,
        default=None,
        help="Path to baseline (generic CLIP) model checkpoint (optional, will create if not exists)"
    )
    parser.add_argument(
        "--chess_model",
        type=str,
        default=None,
        help="Path to chess-CLIP model checkpoint (optional, will create if not exists)"
    )
    parser.add_argument(
        "--chess_clip_checkpoint",
        type=str,
        required=False,
        default=None,
        help="Path to chess-finetuned CLIP checkpoint (required if evaluating chess-CLIP model)"
    )
    parser.add_argument(
        "--language_model",
        type=str,
        default="llava-hf/llava-v1.6-mistral-7b-hf",
        help="LLaVA language model name"
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_csv",
        type=str,
        required=True,
        help="Path to dataset CSV"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing images"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--evaluate_baseline_only",
        action="store_true",
        help="Only evaluate baseline model (skip chess-CLIP)"
    )
    parser.add_argument(
        "--evaluate_chess_only",
        action="store_true",
        help="Only evaluate chess-CLIP model (skip baseline)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--use_clip_fen",
        action="store_true",
        help="Use CLIP-predicted FEN instead of ground truth FEN in context"
    )
    parser.add_argument(
        "--fen_candidates_csv",
        type=str,
        default=None,
        help="CSV file with FEN candidates for CLIP matching (defaults to dataset_csv if not provided)"
    )
    
    args = parser.parse_args()
    
    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load questions - use enhanced questions if available
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "benchmarking"))
    
    # Try to load enhanced questions first, fallback to standard
    try:
        from questions_enhanced import get_scoring_questions, get_material_questions
        use_enhanced = True
    except ImportError:
        from questions import get_scoring_questions
        use_enhanced = False
    
    # Use enhanced questions (with more material balance questions)
    if use_enhanced:
        questions = get_scoring_questions()
        logger.info(f"Loaded {len(questions)} question types (enhanced with material questions)")
    else:
        questions = get_scoring_questions()
        logger.info(f"Loaded {len(questions)} question types")
    
    # Create dataset
    logger.info("Creating evaluation dataset...")
    dataset = create_qa_dataset_from_benchmark(
        dataset_csv=args.dataset_csv,
        images_dir=args.images_dir,
        questions=questions,
        num_samples=args.num_samples
    )
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Pre-extract FENs using CLIP if needed (before loading LLaVA to save memory)
    clip_predicted_fens = None
    if args.use_clip_fen:
        if not CLIP_EXTRACTOR_AVAILABLE:
            logger.error("CLIPFENExtractor not available. Cannot use --use_clip_fen")
            return
        
        if not args.chess_clip_checkpoint:
            logger.error("--chess_clip_checkpoint is required when using --use_clip_fen")
            return
        
        logger.info("Pre-extracting FENs using CLIP (before loading LLaVA)...")
        try:
            clip_extractor = CLIPFENExtractor(
                checkpoint_path=args.chess_clip_checkpoint,
                model_name="ViT-B-32",
                device=device
            )
            
            # Get candidates CSV
            candidates_csv = args.fen_candidates_csv or args.dataset_csv
            
            # Pre-extract FENs for all unique images
            logger.info(f"Extracting FENs from {len(dataset)} samples...")
            clip_predicted_fens = {}
            unique_images = {}
            for i in range(len(dataset)):
                sample = dataset[i]
                img_path = sample["image_path"]
                if img_path not in unique_images:
                    unique_images[img_path] = True
                    try:
                        # Clear cache before extraction
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        predicted_fen = clip_extractor.extract_fen_from_image_with_candidates_file(
                            img_path,
                            candidates_csv=candidates_csv,
                            top_k=1
                        )
                        clip_predicted_fens[img_path] = predicted_fen
                        
                        # Clear cache after extraction
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e:
                        logger.warning(f"CLIP FEN extraction failed for {img_path}: {e}, will use ground truth")
                        clip_predicted_fens[img_path] = None
            
            # Clean up CLIP extractor to free memory
            del clip_extractor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            import gc
            gc.collect()
            
            logger.info(f"[OK] Pre-extracted FENs for {len(clip_predicted_fens)} unique images")
        except Exception as e:
            logger.error(f"Failed to pre-extract FENs: {e}")
            return
    
    # Evaluate models
    results = {}
    baseline_summary = None
    chess_summary = None
    
    # 1. Baseline (generic CLIP) - unless chess_only
    if not args.evaluate_chess_only:
        logger.info("\n" + "="*60)
        logger.info("Evaluating Baseline LLaVA (Generic CLIP)")
        logger.info("="*60)
        
        baseline_model = ChessLLaVA(
            vision_encoder_type="generic",
            chess_clip_checkpoint=None,
            language_model_name=args.language_model,
            device=device
        )
        
        if args.baseline_model and os.path.exists(args.baseline_model):
            logger.info(f"Loading baseline model from {args.baseline_model}")
            checkpoint = torch.load(args.baseline_model, map_location=device)
            baseline_model.load_state_dict(checkpoint["model_state_dict"])
        
        baseline_results = evaluate_model(
            baseline_model, dataloader, device, questions,
            clip_predicted_fens=clip_predicted_fens
        )
        baseline_scored = score_responses(baseline_results, questions)
        baseline_summary = generate_summary(baseline_scored, questions)
        
        results["baseline"] = {
            "results": baseline_scored,
            "summary": baseline_summary
        }
        
        logger.info(f"Baseline - Average Score: {baseline_summary['average_score']:.3f}, "
                    f"Accuracy: {baseline_summary['accuracy']:.2f}%")
        
        # Save baseline results immediately (in case script crashes later)
        baseline_results_path = os.path.join(args.output_dir, "baseline_results.json")
        with open(baseline_results_path, 'w') as f:
            json.dump({
                "baseline": {
                    "results": baseline_scored,
                    "summary": baseline_summary
                }
            }, f, indent=2)
        logger.info(f"Saved baseline results to {baseline_results_path}")
        
        # Clear GPU memory before loading next model
        logger.info("\nClearing GPU memory...")
        del baseline_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()
        logger.info("GPU memory cleared")
    
    # 2. Chess-CLIP LLaVA - unless baseline_only
    if not args.evaluate_baseline_only:
        if not args.chess_clip_checkpoint:
            logger.error("--chess_clip_checkpoint is required when evaluating chess-CLIP model")
            return
        
        logger.info("\n" + "="*60)
        logger.info("Evaluating ChessCLIP-LLaVA")
        logger.info("="*60)
        
        chess_model = ChessLLaVA(
            vision_encoder_type="chess_finetuned",
            chess_clip_checkpoint=args.chess_clip_checkpoint,
            language_model_name=args.language_model,
            device=device
        )
        
        if args.chess_model and os.path.exists(args.chess_model):
            logger.info(f"Loading chess model from {args.chess_model}")
            checkpoint = torch.load(args.chess_model, map_location=device)
            chess_model.load_state_dict(checkpoint["model_state_dict"])
        
        chess_results = evaluate_model(
            chess_model, dataloader, device, questions,
            clip_predicted_fens=clip_predicted_fens
        )
        chess_scored = score_responses(chess_results, questions)
        chess_summary = generate_summary(chess_scored, questions)
        
        results["chess_clip"] = {
            "results": chess_scored,
            "summary": chess_summary
        }
        
        logger.info(f"ChessCLIP - Average Score: {chess_summary['average_score']:.3f}, "
                    f"Accuracy: {chess_summary['accuracy']:.2f}%")
        
        # Save chess-CLIP results immediately (in case script crashes later)
        chess_results_path = os.path.join(args.output_dir, "chess_clip_results.json")
        with open(chess_results_path, 'w') as f:
            json.dump({
                "chess_clip": {
                    "results": chess_scored,
                    "summary": chess_summary
                }
            }, f, indent=2)
        logger.info(f"Saved chess-CLIP results to {chess_results_path}")
    
    # Comparison (only if both models evaluated)
    if baseline_summary and chess_summary:
        logger.info("\n" + "="*60)
        logger.info("COMPARISON")
        logger.info("="*60)
        
        score_improvement = chess_summary["average_score"] - baseline_summary["average_score"]
        accuracy_improvement = chess_summary["accuracy"] - baseline_summary["accuracy"]
        
        logger.info(f"Score Improvement: {score_improvement:+.3f} "
                    f"({score_improvement / max(baseline_summary['average_score'], 0.01) * 100:+.1f}%)")
        logger.info(f"Accuracy Improvement: {accuracy_improvement:+.2f}%")
        
        # Per-question comparison
        logger.info("\nPer-Question Type Comparison:")
        for qtype in baseline_summary["per_question_type"].keys():
            baseline_q = baseline_summary["per_question_type"][qtype]
            chess_q = chess_summary["per_question_type"].get(qtype, {})
            
            if chess_q:
                score_diff = chess_q["average_score"] - baseline_q["average_score"]
                acc_diff = chess_q["accuracy"] - baseline_q["accuracy"]
                logger.info(f"  {qtype}: Score {score_diff:+.3f}, Accuracy {acc_diff:+.2f}%")
        
        # Save comparison summary
        comparison = {
            "baseline_summary": baseline_summary,
            "chess_clip_summary": chess_summary,
            "improvements": {
                "score_improvement": float(score_improvement),
                "score_improvement_percent": float(score_improvement / max(baseline_summary['average_score'], 0.01) * 100),
                "accuracy_improvement": float(accuracy_improvement)
            }
        }
        
        comparison_path = os.path.join(args.output_dir, "comparison_summary.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Saved comparison to {comparison_path}")
    
    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to {results_path}")
    
    logger.info("\n[OK] Evaluation completed!")


if __name__ == "__main__":
    main()

