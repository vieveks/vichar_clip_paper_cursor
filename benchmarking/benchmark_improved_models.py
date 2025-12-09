"""
Benchmark the JSON-predictor CLIP encoders and fine-tuned Qwen2-VL on reasoning tasks.

This script evaluates:
1. LLaVA with CLIP encoders from JSON predictor models (Exp 1A, 1B, 1D)
2. Fine-tuned Qwen2-VL from Exp 1C

Both on the 10-question reasoning benchmark from the benchmarking folder.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from questions import get_scoring_questions
from ground_truth import GroundTruthExtractor
from llm_judge_scorer import LLMJudgeScorer
from scoring import ResponseScorer

# Import JSON predictor model to extract CLIP encoder
try:
    from Improved_representations.json_predictor.model import JSONPredictorModel
except ImportError:
    print("Warning: Could not import JSONPredictorModel")
    JSONPredictorModel = None

# Import VLM fine-tuning evaluation
try:
    from Improved_representations.vlm_finetuning.evaluate_qwen import load_model as load_qwen_model
except ImportError:
    print("Warning: Could not import Qwen evaluation utilities")
    load_qwen_model = None


class OpenCLIPWrapper(torch.nn.Module):
    """Wrapper for open_clip VisualTransformer to match transformers interface."""
    
    def __init__(self, vision_tower, hidden_dim=768):
        super().__init__()
        self.vision_tower = vision_tower
        self.config = type('Config', (), {
            'hidden_size': hidden_dim, 
            'image_size': 224, 
            'patch_size': 32,
            'model_type': 'clip_vision_model'
        })()
        self.dtype = next(vision_tower.parameters()).dtype
        self.device = next(vision_tower.parameters()).device
        self.dtype = next(vision_tower.parameters()).dtype
        self.device = next(vision_tower.parameters()).device
        # LLaVA expects 336px images and 14px patches -> 24x24 = 576 patches
        # But we are feeding 224px images.
        # If we set patch_size=14, LLaVA calculates 224/14 = 16 -> 16x16 = 256 patches
        # Our CLIP (ViT-B/32) produces 7x7 = 49 patches + 1 CLS = 50 tokens
        # Wait, CLIP ViT-B/32 on 224px produces 7x7 grid (32px stride)
        # We need to upsample our features to match what LLaVA expects?
        # Or we can just resize the input image to 336px and let CLIP process it?
        # CLIP ViT-B/32 at 336px -> 336/32 = 10.5 -> 10x10 = 100 patches.
        
        # Let's try to match the feature map size LLaVA expects.
        # LLaVA expects 576 tokens (24x24) for 336px image.
        # We have 50 tokens (7x7 + 1).
        # We should probably interpolate our features to match LLaVA's expected grid.
        
        self.patch_size = 14 # LLaVA default
        self.image_size = 336 # LLaVA default

    def forward(self, pixel_values, output_hidden_states=None, return_dict=True, **kwargs):
        # Handle 5D input [B, N, C, H, W]
        # LLaVA expects 576 tokens (from forced image_sizes=[336,336]).
        # We must provide exactly 576 features.
        
        bs = 1
        if pixel_values.dim() == 5:
            # [B, N, C, H, W]
            bs, n, c, h, w = pixel_values.shape
            # Take first crop (global view)
            pixel_values = pixel_values[:, 0, :, :, :]
        
        # Resize to 224x224 if needed (CLIP ViT-B/32 expects 224)
        if pixel_values.shape[-1] != 224 or pixel_values.shape[-2] != 224:
            pixel_values = torch.nn.functional.interpolate(
                pixel_values, size=(224, 224), mode='bicubic', align_corners=False
            )
        
        # Ensure input dtype matches model dtype
        target_dtype = self.vision_tower.conv1.weight.dtype
        if pixel_values.dtype != target_dtype:
            pixel_values = pixel_values.to(dtype=target_dtype)
        
        x = self.vision_tower.conv1(pixel_values)
        # x is [B*N, 768, 7, 7]
        
        # Interpolate to 24x24 (576 patches)
        x = torch.nn.functional.interpolate(x, size=(24, 24), mode='bicubic', align_corners=False)
        
        x = x.reshape(x.shape[0], x.shape[1], -1) # [B, 768, 576]
        x = x.permute(0, 2, 1) # [B, 576, 768]
        
        # Add CLS token back (LLaVA strips index 0)
        cls_token = self.vision_tower.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)
        
        # Add position embedding features
        # CLIP has 1 CLS + 49 Spatial pos_embeds.
        pos_embed = self.vision_tower.positional_embedding.to(x.dtype)
        cls_pos = pos_embed[0:1] # Keep CLS pos
        spatial_pos = pos_embed[1:]
        
        # Reshape spatial pos to 7x7 and interpolate to 24x24
        spatial_pos = spatial_pos.reshape(1, 7, 7, -1).permute(0, 3, 1, 2)
        spatial_pos = torch.nn.functional.interpolate(spatial_pos, size=(24, 24), mode='bicubic', align_corners=False)
        spatial_pos = spatial_pos.permute(0, 2, 3, 1).reshape(576, -1)
        
        # Combine pos embeds
        pos_embed = torch.cat([cls_pos, spatial_pos], dim=0)
        
        # Add to x
        # Note: broadcast across batch
        x = x + pos_embed.unsqueeze(0)
        
        # Pre-norm
        x = self.vision_tower.ln_pre(x)
        
        # Transformer
        x = x.permute(1, 0, 2)
        x = self.vision_tower.transformer(x)
        x = x.permute(1, 0, 2)
        
        # Post-norm
        x = self.vision_tower.ln_post(x)
        
        # Reshape to flatten batch and crops: [B, N*Patches, Hidden]
        # x is currently [B*N, 577, 768] (with CLS)
        x = x.reshape(bs, -1, x.shape[-1])
        
        if not return_dict:
            return (x, x[:, 0])
            
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        return BaseModelOutputWithPooling(
            last_hidden_state=x,
            pooler_output=x[:, 0],
            hidden_states=(x, x) if output_hidden_states else None
        )


class ImprovedModelsBenchmark:
    """Benchmark for JSON-predictor CLIP encoders + fine-tuned Qwen2-VL."""
    
    def __init__(
        self,
        device: str = None,
        use_llm_judge: bool = True
    ):
        """Initialize benchmark."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ground_truth_extractor = GroundTruthExtractor()
        self.use_llm_judge = use_llm_judge
        
        if use_llm_judge:
            try:
                self.llm_scorer = LLMJudgeScorer()
                print("LLM judge scorer initialized")
            except Exception as e:
                print(f"Warning: Could not initialize LLM judge: {e}")
                print("Falling back to rule-based scoring only")
                self.llm_scorer = None
                self.use_llm_judge = False
        else:
            self.llm_scorer = None
            print("Using rule-based scoring only (no LLM judge)")
        
        self.scorer = ResponseScorer()
    
    def load_json_predictor_clip(self, checkpoint_path: str):
        """
        Load CLIP encoder from JSON predictor checkpoint.
        
        Returns the clip_model component.
        """
        print(f"Loading JSON predictor from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        model_args = checkpoint.get('args', {})
        model = JSONPredictorModel(
            hidden_dim=model_args.get('hidden_dim', 512),
            freeze_encoder=True
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Extract CLIP encoder
        clip_encoder = model.visual_encoder
        return OpenCLIPWrapper(clip_encoder)

    
    def create_llava_with_clip(self, clip_encoder):
        """
        Create LLaVA model with custom CLIP encoder.
        
        This follows the pattern from clip_as_encoder/model.py
        """
        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            
            # Load base LLaVA model
            model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
            print(f"Loading LLaVA from {model_name}")
            
            llava_model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            
            # Replace vision tower with custom CLIP
            print(f"Original vision tower type: {type(llava_model.vision_tower)}")
            llava_model.vision_tower = clip_encoder
            print(f"New vision tower type: {type(llava_model.vision_tower)}")
            
            # Handle dimension mismatch (768 vs 1024)
            # LLaVA-v1.6-Mistral projector expects 1024 dim input
            if hasattr(llava_model, 'multi_modal_projector'):
                projector = llava_model.multi_modal_projector
                if hasattr(projector, 'linear_1'):
                    linear_1 = projector.linear_1
                    if linear_1.in_features != clip_encoder.config.hidden_size:
                        print(f"Resizing projector input from {linear_1.in_features} to {clip_encoder.config.hidden_size}")
                        
                        # Create new linear layer
                        new_linear = torch.nn.Linear(
                            clip_encoder.config.hidden_size, 
                            linear_1.out_features, 
                            bias=linear_1.bias is not None
                        )
                        
                        # Slice weights (taking first 768 columns)
                        with torch.no_grad():
                            new_linear.weight.copy_(linear_1.weight[:, :clip_encoder.config.hidden_size])
                            if linear_1.bias is not None:
                                new_linear.bias.copy_(linear_1.bias)
                        
                        # Move to device and dtype
                        new_linear = new_linear.to(device=linear_1.weight.device, dtype=linear_1.weight.dtype)
                        print(f"New linear layer device: {new_linear.weight.device}, dtype: {new_linear.weight.dtype}")
                        
                        # Replace layer
                        projector.linear_1 = new_linear
                        llava_model.multi_modal_projector.linear_1 = new_linear
                        print("Projector resized successfully")
                        
                        # Ensure CLIP encoder is also on correct dtype
                        clip_encoder.to(dtype=linear_1.weight.dtype)
                        print(f"CLIP encoder moved to {linear_1.weight.dtype}")

            # Also try assigning to model.vision_tower if it exists, as some implementations use that
            if hasattr(llava_model, 'model') and hasattr(llava_model.model, 'vision_tower'):
                print("Assigning to llava_model.model.vision_tower as well")
                llava_model.model.vision_tower = clip_encoder
            
            processor = AutoProcessor.from_pretrained(model_name)
            # Force single crop configuration to match our OpenCLIPWrapper
            if hasattr(processor, 'image_processor'):
                processor.image_processor.do_image_splitting = False
                processor.image_processor.do_resize = True
                processor.image_processor.size = {"height": 336, "width": 336}
                processor.image_processor.crop_size = {"height": 336, "width": 336}
            
            return llava_model, processor
        
        except Exception as e:
            print(f"Error creating LLaVA model: {e}")
            return None, None
    
    def evaluate_llava_clip(
        self,
        clip_checkpoint: str,
        image_paths: List[str],
        questions: List[Dict],
        gt_fens: Dict[str, str]
    ) -> List[Dict]:
        """Evaluate LLaVA with JSON-predictor CLIP encoder."""
        print(f"\n{'='*60}")
        print(f"Evaluating LLaVA with CLIP from: {os.path.basename(clip_checkpoint)}")
        print(f"{'='*60}\n")
        
        # Load CLIP encoder
        clip_encoder = self.load_json_predictor_clip(clip_checkpoint)
        
        # Create LLaVA with custom CLIP
        llava_model, processor = self.create_llava_with_clip(clip_encoder)
        
        if llava_model is None:
            print("Failed to create LLaVA model, skipping...")
            return []
        
        results = []
        
        for img_idx, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            filename = os.path.basename(image_path)
            gt_fen = gt_fens.get(filename, '')
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            # Force resize image to 336x336 to ensure single crop
            image = image.resize((336, 336), Image.BICUBIC)
            
            # Get ground truths
            ground_truths = self._get_ground_truths(gt_fen, questions)
            
            # Test each question
            for question in questions:
                q_id = question['id']
                q_type = question['type']
                prompt = question['prompt']
                
                # Generate answer
                # Ensure <image> token is present
                prompt_with_image = f"[INST] <image>\n{prompt} [/INST]"
                
                # Explicitly disable image splitting in the call
                inputs = processor(
                    text=prompt_with_image,
                    images=image,
                    return_tensors="pt",
                    do_image_splitting=False 
                ).to(self.device)
                
                # Force image_sizes to match our single crop (336, 336)
                if 'image_sizes' in inputs:
                   inputs['image_sizes'] = torch.tensor([[336, 336]], device=self.device)
                
                # Manual Truncation of Image Tokens to 576 (Single Tile)
                # Processor persists in generating 1176 tokens (2 tiles + newlines).
                # We forced Wrapper to 576 features (1 tile).
                # So we must truncate inputs to 576 image tokens.
                
                input_ids = inputs['input_ids']
                image_token_index = getattr(llava_model.config, 'image_token_index', 32000)
                
                # Find occurrences
                img_mask = input_ids == image_token_index
                num_img_tokens = img_mask.sum()
                
                if num_img_tokens > 576:
                    print(f"Truncating image tokens from {num_img_tokens} to 576")
                    # We assume image tokens are somewhat contiguous or we just keep the first 576
                    # To be safe, we reconstruct input_ids keeping non-image tokens and first 576 image tokens
                    
                    # Flatten to 1D
                    ids_flat = input_ids[0]
                    new_ids = []
                    img_count = 0
                    
                    for token in ids_flat:
                        if token == image_token_index:
                            if img_count < 576:
                                new_ids.append(token)
                                img_count += 1
                        else:
                            new_ids.append(token)
                    
                    inputs['input_ids'] = torch.tensor([new_ids], device=self.device)
                    # Adjust attention_mask accordingly
                    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'], device=self.device)

                with torch.no_grad():
                    outputs = llava_model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=False
                    )
                
                answer = processor.decode(outputs[0], skip_special_tokens=True)
                
                # Score answer
                score = self._score_response(
                    answer, question, ground_truths.get(q_id)
                )
                
                results.append({
                    'image_path': image_path,
                    'model': f"LLaVA+CLIP({os.path.basename(clip_checkpoint)})",
                    'question_id': q_id,
                    'question_type': q_type,
                    'answer': answer,
                    'score': score,
                    'ground_truth': ground_truths.get(q_id)
                })
        
        return results
    
    def evaluate_qwen_finetuned(
        self,
        model_path: str,
        image_paths: List[str],
        questions: List[Dict],
        gt_fens: Dict[str, str],
        image_base_dir: Path
    ) -> List[Dict]:
        """Evaluate fine-tuned Qwen2-VL model."""
        print(f"\n{'='*60}")
        print(f"Evaluating fine-tuned Qwen2-VL from: {model_path}")
        print(f"{'='*60}\n")
        
        if load_qwen_model is None:
            print("Could not import Qwen evaluation utilities, skipping...")
            return []
        
        # Load fine-tuned Qwen2-VL
        model, processor = load_qwen_model(model_path, use_lora=True)
        
        results = []
        
        for img_idx, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            filename = os.path.basename(image_path)
            gt_fen = gt_fens.get(filename, '')
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Get ground truths
            ground_truths = self._get_ground_truths(gt_fen, questions)
            
            # Test each question
            for question in questions:
                q_id = question['id']
                q_type = question['type']
                prompt = question['prompt']
                
                # Prepare messages
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                
                # Process and generate
                try:
                    inputs = processor(
                        messages,
                        padding=True,
                        truncation=True,
                        max_length=2048,
                        return_tensors="pt"
                    )
                except Exception as e:
                    # Fallback
                    text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    inputs = processor(
                        text=[text],
                        images=[image],
                        padding=True,
                        truncation=True,
                        max_length=2048,
                        return_tensors="pt"
                    )
                
                inputs = {k: v.to(self.device) if torch.is_tensor(v) else v 
                         for k, v in inputs.items()}
                
                # Ensure image_grid_thw
                if 'image_grid_thw' not in inputs or inputs.get('image_grid_thw') is None:
                    if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                        _, _, h, w = inputs['pixel_values'].shape
                        patch_size = 14
                        grid_h = (h + patch_size - 1) // patch_size
                        grid_w = (w + patch_size - 1) // patch_size
                        inputs['image_grid_thw'] = torch.tensor([[1, grid_h, grid_w]], device=self.device)
                    else:
                        inputs['image_grid_thw'] = torch.tensor([[1, 32, 32]], device=self.device)
                
                # Generate
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=False
                    )
                
                # Decode
                input_length = inputs['input_ids'].shape[1]
                generated_ids_trimmed = generated_ids[:, input_length:]
                answer = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                # Score answer
                score = self._score_response(
                    answer, question, ground_truths.get(q_id)
                )
                
                results.append({
                    'image_path': image_path,
                    'model': 'Qwen2-VL-Finetuned',
                    'question_id': q_id,
                    'question_type': q_type,
                    'answer': answer,
                    'score': score,
                    'ground_truth': ground_truths.get(q_id)
                })
        
        return results
    
    def _get_ground_truths(self, fen_string: str, questions: List[Dict]) -> Dict:
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
    
    def _score_response(self, response: str, question: Dict, ground_truth) -> float:
        """Score a response against ground truth."""
        if ground_truth is None:
            return 0.0
        
        q_type = question['type']
        
        try:
            if q_type == "fen_extraction":
                if self.llm_scorer:
                    return self.llm_scorer.score_fen_extraction(response, ground_truth)
                else:
                    # Simple exact match for FEN
                    return 1.0 if response.strip() == str(ground_truth).strip() else 0.0
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
                if self.llm_scorer:
                    return self.llm_scorer.score_response(
                        question['prompt'], response, ground_truth, q_type
                    )
                else:
                    # Fallback: simple string matching
                    return 1.0 if str(ground_truth).lower() in response.lower() else 0.0
        except Exception as e:
            print(f"Scoring error: {e}")
            return 0.0
    
    def save_results(self, results: List[Dict], output_path: str):
        """Save results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Saved results to {output_path}")
    
    def generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics."""
        df = pd.DataFrame(results)
        
        summary = {
            'overall': {
                'total_tests': len(results),
                'avg_score': float(df['score'].mean()),
                'accuracy': float((df['score'] >= 0.9).mean() * 100)
            },
            'by_model': {},
            'by_question_type': {}
        }
        
        # Per model
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            summary['by_model'][model] = {
                'avg_score': float(model_df['score'].mean()),
                'accuracy': float((model_df['score'] >= 0.9).mean() * 100),
                'total_tests': len(model_df)
            }
        
        # Per question type
        for q_type in df['question_type'].unique():
            type_df = df[df['question_type'] == q_type]
            summary['by_question_type'][q_type] = {
                'avg_score': float(type_df['score'].mean()),
                'accuracy': float((type_df['score'] >= 0.9).mean() * 100),
                'total_tests': len(type_df)
            }
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark JSON-predictor CLIP encoders and fine-tuned Qwen2-VL"
    )
    
    # Model paths
    parser.add_argument(
        "--clip_checkpoints",
        type=str,
        nargs='+',
        help="Paths to JSON predictor checkpoints (Exp 1A, 1B, 1D)"
    )
    parser.add_argument(
        "--qwen_model_path",
        type=str,
        help="Path to fine-tuned Qwen2-VL model (Exp 1C)"
    )
    
    # Data paths
    parser.add_argument(
        "--dataset_csv",
        type=str,
        required=True,
        help="Path to test dataset CSV"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing images"
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default="data/hf_chess_puzzles",
        help="Base directory for images"
    )
    
    # Options
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of images to test"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results_improved",
        help="Output directory"
    )
    parser.add_argument(
        "--skip_llava",
        action="store_true",
        help="Skip LLaVA evaluation"
    )
    parser.add_argument(
        "--skip_qwen",
        action="store_true",
        help="Skip Qwen evaluation"
    )
    parser.add_argument(
        "--no_llm_judge",
        action="store_true",
        help="Disable LLM judge (use rule-based scoring only)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize benchmark
    benchmark = ImprovedModelsBenchmark(
        use_llm_judge=not args.no_llm_judge
    )
    
    # Load ground truth FENs
    df = pd.read_csv(args.dataset_csv)
    gt_fens = {}
    for _, row in df.iterrows():
        filename = os.path.basename(row['image_path'])
        gt_fens[filename] = row['fen']
    
    # Get image paths
    images_dir = Path(args.images_dir)
    image_paths = list(images_dir.glob("*.png"))[:args.num_images]
    
    print(f"Found {len(image_paths)} images to test")
    
    # Get questions
    questions = get_scoring_questions()
    
    # Collect all results
    all_results = []
    
    # Evaluate LLaVA with CLIP encoders
    if not args.skip_llava and args.clip_checkpoints:
        for clip_checkpoint in args.clip_checkpoints:
            if Path(clip_checkpoint).exists():
                results = benchmark.evaluate_llava_clip(
                    clip_checkpoint,
                    [str(p) for p in image_paths],
                    questions,
                    gt_fens
                )
                all_results.extend(results)
            else:
                print(f"Checkpoint not found: {clip_checkpoint}")
    
    # Evaluate fine-tuned Qwen2-VL
    if not args.skip_qwen and args.qwen_model_path:
        if Path(args.qwen_model_path).exists():
            results = benchmark.evaluate_qwen_finetuned(
                args.qwen_model_path,
                [str(p) for p in image_paths],
                questions,
                gt_fens,
                Path(args.image_base_dir)
            )
            all_results.extend(results)
        else:
            print(f"Model not found: {args.qwen_model_path}")
    
    # Save results
    if all_results:
        benchmark.save_results(
            all_results,
            os.path.join(args.output_dir, "detailed_results.json")
        )
        
        # Generate and save summary
        summary = benchmark.generate_summary(all_results)
        with open(os.path.join(args.output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(json.dumps(summary, indent=2))
    else:
        print("No results to save!")


if __name__ == "__main__":
    main()
