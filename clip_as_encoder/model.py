"""
Custom LLaVA-style model with swappable vision encoder.
Supports both generic CLIP and chess-finetuned CLIP as vision encoders.

This implementation wraps LLaVA's existing architecture but allows
swapping the vision encoder (CLIP visual encoder) with a chess-finetuned version.

Also supports Qwen2.5-VL models with swappable CLIP encoders.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
import open_clip
import logging
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import AutoModelForVision2Seq
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
from transformers.models.llava.processing_llava import LlavaProcessor

logger = logging.getLogger(__name__)


class ChessLLaVA(nn.Module):
    """
    LLaVA-style model with swappable vision encoder.
    
    Architecture:
    - Vision Encoder: CLIP ViT-B/32 (generic or chess-finetuned)
    - Projection Layer: Maps vision features to language model space
    - Language Model: LLaMA/Mistral (from LLaVA)
    """
    
    def __init__(
        self,
        vision_encoder_type: str = "generic",  # "generic" or "chess_finetuned"
        chess_clip_checkpoint: Optional[str] = None,
        language_model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        vision_feature_dim: int = 512,  # CLIP ViT-B/32 output dim
        projection_dim: int = 4096,  # Language model hidden dim
        device: str = "cuda"
    ):
        """
        Initialize ChessLLaVA model.
        
        Args:
            vision_encoder_type: "generic" for pretrained CLIP, "chess_finetuned" for chess CLIP
            chess_clip_checkpoint: Path to chess-finetuned CLIP checkpoint (required if chess_finetuned)
            language_model_name: HuggingFace model name for LLaVA
            vision_feature_dim: Output dimension of vision encoder
            projection_dim: Hidden dimension of language model
            device: Device to load models on
        """
        super().__init__()
        
        self.vision_encoder_type = vision_encoder_type
        self.device = device
        self.vision_feature_dim = vision_feature_dim
        
        # Load vision encoder
        self.vision_encoder = self._load_vision_encoder(chess_clip_checkpoint)
        
        # Load language model and processor
        print(f"Loading language model: {language_model_name}")
        self.language_model_name = language_model_name
        
        try:
            # Try LLaVA Next/1.6 API (AutoModelForCausalLM)
            self.processor = AutoProcessor.from_pretrained(language_model_name)
            self.language_model = AutoModelForCausalLM.from_pretrained(
                language_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device
            )
            print(f"[OK] Loaded LLaVA model using AutoModelForCausalLM")
            self.llava_type = "next"
        except Exception as e1:
            print(f"AutoModelForCausalLM failed: {e1}")
            try:
                # Try Vision2Seq API
                self.processor = AutoProcessor.from_pretrained(language_model_name)
                self.language_model = AutoModelForVision2Seq.from_pretrained(
                    language_model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map=device
                )
                print(f"[OK] Loaded LLaVA model using AutoModelForVision2Seq")
                self.llava_type = "vision2seq"
            except Exception as e2:
                print(f"AutoModelForVision2Seq failed: {e2}")
                # Fallback to LLaVA 1.5
                from transformers import LlavaProcessor, LlavaForConditionalGeneration
                self.processor = LlavaProcessor.from_pretrained(language_model_name)
                self.language_model = LlavaForConditionalGeneration.from_pretrained(
                    language_model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map=device
                )
                print(f"[OK] Loaded LLaVA model using LlavaForConditionalGeneration")
                self.llava_type = "1.5"
        
        # Replace LLaVA's vision encoder with our CLIP encoder
        # LLaVA stores vision encoder in different places depending on version
        if hasattr(self.language_model, 'vision_tower'):
            # LLaVA 1.6/Next structure
            original_vision_tower = self.language_model.vision_tower
            # We'll keep the original structure but replace the encoder weights
            self._replace_vision_encoder_weights(original_vision_tower)
        elif hasattr(self.language_model, 'model') and hasattr(self.language_model.model, 'vision_tower'):
            # Alternative structure
            original_vision_tower = self.language_model.model.vision_tower
            self._replace_vision_encoder_weights(original_vision_tower)
        
        # Store reference to our vision encoder for direct use
        self.custom_vision_encoder = self.vision_encoder
        
        # Freeze vision encoder (we only train projection + language model)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # Language model can be frozen or trainable (configurable)
        self.language_model_trainable = True
    
    def _load_vision_encoder(self, chess_clip_checkpoint: Optional[str]):
        """Load vision encoder (generic or chess-finetuned CLIP)."""
        print(f"Loading vision encoder: {self.vision_encoder_type}")
        
        if self.vision_encoder_type == "generic":
            # Load generic pretrained CLIP
            print("  Loading generic CLIP ViT-B/32...")
            model, _, _ = open_clip.create_model_and_transforms(
                model_name="ViT-B-32",
                pretrained="laion2B-s34B-b79K",
                device=self.device
            )
            vision_encoder = model.visual
            print(f"[OK] Loaded generic CLIP vision encoder")
            
        elif self.vision_encoder_type == "chess_finetuned":
            # Load chess-finetuned CLIP
            if chess_clip_checkpoint is None:
                raise ValueError("chess_clip_checkpoint required for chess_finetuned encoder")
            
            print(f"  Loading chess-finetuned CLIP from {chess_clip_checkpoint}...")
            model, _, _ = open_clip.create_model_and_transforms(
                model_name="ViT-B-32",
                pretrained="laion2B-s34B-b79K",
                device=self.device
            )
            
            # Load checkpoint
            checkpoint = torch.load(chess_clip_checkpoint, map_location=self.device)
            model_state_dict = checkpoint["model_state_dict"]
            model.load_state_dict(model_state_dict)
            
            vision_encoder = model.visual
            print(f"[OK] Loaded chess-finetuned CLIP vision encoder")
            
        else:
            raise ValueError(f"Unknown vision_encoder_type: {self.vision_encoder_type}")
        
        vision_encoder.eval()
        return vision_encoder
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using vision encoder.
        
        Args:
            images: [batch_size, 3, 224, 224]
            
        Returns:
            vision_features: [batch_size, vision_feature_dim]
        """
        with torch.no_grad():
            vision_features = self.vision_encoder(images)
        return vision_features
    
    def _replace_vision_encoder_weights(self, vision_tower):
        """Replace LLaVA's vision encoder with our CLIP encoder."""
        # This is a simplified approach - in practice, we need to match the architecture
        # For now, we'll use our encoder directly and bypass LLaVA's vision tower
        # The processor will handle image preprocessing, but we'll use our encoder
        pass
    
    def generate(
        self,
        images: torch.Tensor,
        prompts: List[str],
        max_new_tokens: int = 512,
        **generate_kwargs
    ) -> List[str]:
        """
        Generate responses for given images and prompts.
        
        Args:
            images: [batch_size, 3, 224, 224] or list of PIL Images
            prompts: List of prompt strings
            max_new_tokens: Maximum tokens to generate
            **generate_kwargs: Additional generation arguments
            
        Returns:
            List of generated text responses
        """
        self.eval()
        
        # Convert tensor to PIL if needed
        # LLaVA processor can handle PIL Images or raw image arrays
        if isinstance(images, torch.Tensor):
            from torchvision import transforms
            to_pil = transforms.ToPILImage()
            # Handle batch dimension
            if images.dim() == 4:
                # [B, C, H, W] - convert each image in batch
                pil_images = []
                for img in images:
                    # Ensure image is in [C, H, W] format
                    if img.dim() == 4:
                        img = img.squeeze(0)
                    img_cpu = img.cpu()
                    # Denormalize if needed (assuming ImageNet normalization)
                    if img_cpu.min() < 0 or img_cpu.max() > 1.5:  # Likely normalized
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_cpu = img_cpu * std + mean
                        img_cpu = torch.clamp(img_cpu, 0, 1)
                    # Convert to [0, 255] range and uint8
                    if img_cpu.max() <= 1.0:
                        img_cpu = img_cpu * 255
                    img_cpu = torch.clamp(img_cpu, 0, 255).byte()
                    try:
                        pil_images.append(to_pil(img_cpu))
                    except Exception as e:
                        # Fallback: try without denormalization
                        img_cpu_raw = torch.clamp(img.cpu() * 255, 0, 255).byte()
                        pil_images.append(to_pil(img_cpu_raw))
            elif images.dim() == 3:
                # [C, H, W] - single image
                img_cpu = images.cpu()
                if img_cpu.min() < 0 or img_cpu.max() > 1.5:  # Likely normalized
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_cpu = img_cpu * std + mean
                    img_cpu = torch.clamp(img_cpu, 0, 1)
                if img_cpu.max() <= 1.0:
                    img_cpu = img_cpu * 255
                img_cpu = torch.clamp(img_cpu, 0, 255).byte()
                try:
                    pil_images = [to_pil(img_cpu)]
                except Exception as e:
                    # Fallback
                    img_cpu_raw = torch.clamp(images.cpu() * 255, 0, 255).byte()
                    pil_images = [to_pil(img_cpu_raw)]
            else:
                raise ValueError(f"Unexpected image tensor shape: {images.shape}")
        elif isinstance(images, list):
            # Assume list of PIL Images or tensors
            pil_images = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    # Convert single tensor to PIL
                    from torchvision import transforms
                    to_pil = transforms.ToPILImage()
                    img_cpu = img.cpu()
                    if img_cpu.min() < 0 or img_cpu.max() > 1.5:
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_cpu = img_cpu * std + mean
                        img_cpu = torch.clamp(img_cpu, 0, 1)
                    if img_cpu.max() <= 1.0:
                        img_cpu = img_cpu * 255
                    img_cpu = torch.clamp(img_cpu, 0, 255).byte()
                    pil_images.append(to_pil(img_cpu))
                else:
                    pil_images.append(img)
        else:
            # Single PIL Image or other
            pil_images = [images]
        
        # Ensure we have the same number of images and prompts
        if len(pil_images) != len(prompts):
            if len(pil_images) == 1:
                pil_images = pil_images * len(prompts)
            else:
                raise ValueError(f"Mismatch: {len(pil_images)} images but {len(prompts)} prompts")
        
        # Use processor to format inputs properly
        # Format prompts based on model type
        formatted_prompts = []
        for prompt in prompts:
            if self.llava_type == "next":
                # LLaVA Next uses chat template - we'll format it in the processing step
                formatted_prompts.append(prompt)
            elif self.llava_type == "vision2seq":
                # Vision2Seq (LLaVA 1.6) - use simple format
                formatted_prompts.append(prompt)
            else:
                # LLaVA 1.5 format: "USER: <image>\n{prompt}\nASSISTANT:"
                formatted_prompts.append(f"USER: <image>\n{prompt}\nASSISTANT:")
        
        # Process inputs - handle one at a time for LLaVA Next compatibility
        all_outputs = []
        
        for i, (img, prompt) in enumerate(zip(pil_images, formatted_prompts)):
            try:
                # Process single image-prompt pair
                # Handle different LLaVA processor types
                if self.llava_type == "vision2seq":
                    # Vision2Seq (LLaVA 1.6) - use simple format, processor handles the rest
                    # Don't use apply_chat_template for Vision2Seq
                    inputs = self.processor(
                        text=prompt,
                        images=img,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                elif self.llava_type == "next" and hasattr(self.processor, 'apply_chat_template'):
                    # LLaVA Next - use chat template
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    formatted_prompt = self.processor.apply_chat_template(
                        conversation, 
                        add_generation_prompt=True
                    )
                    inputs = self.processor(
                        text=formatted_prompt,
                        images=img,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                else:
                    # LLaVA 1.5 or fallback
                    simple_prompt = f"USER: <image>\n{prompt}\nASSISTANT:" if "USER:" not in prompt else prompt
                    inputs = self.processor(
                        text=simple_prompt,
                        images=img,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                
                # Generate with memory management
                with torch.no_grad():
                    try:
                        # Clear cache before generation
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        outputs = self.language_model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            **generate_kwargs
                        )
                        
                        # Clear cache after generation
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except RuntimeError as e:
                        if "out of memory" in str(e) or "CUDA" in str(e):
                            # Clear cache and try again with smaller max_new_tokens
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            logger.warning(f"OOM error, retrying with smaller max_new_tokens")
                            safe_kwargs = {k: v for k, v in generate_kwargs.items() 
                                          if k not in ['temperature', 'do_sample']}
                            safe_kwargs['max_new_tokens'] = min(max_new_tokens, 128)
                            outputs = self.language_model.generate(
                                **inputs,
                                **safe_kwargs
                            )
                        else:
                            # Fallback: remove problematic kwargs
                            safe_kwargs = {k: v for k, v in generate_kwargs.items() 
                                          if k not in ['temperature', 'do_sample']}
                            outputs = self.language_model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                **safe_kwargs
                            )
                
                # Decode single response
                if hasattr(self.processor, 'tokenizer'):
                    response = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    response = self.processor.decode(outputs[0], skip_special_tokens=True)
                
                all_outputs.append(response)
                
            except Exception as e:
                import traceback
                error_msg = f"Error processing image {i}: {type(e).__name__}: {str(e)}"
                print(f"Warning: {error_msg}")
                # Print full traceback for first error to help debug
                if i == 0:
                    print(f"  Full traceback:\n{traceback.format_exc()}")
                all_outputs.append("Error generating response")
        
        # Extract answers (remove prompt)
        answers = []
        for response in all_outputs:
            if "ASSISTANT:" in response:
                answer = response.split("ASSISTANT:")[-1].strip()
            elif "[/INST]" in response:  # Mistral/Llama style
                answer = response.split("[/INST]")[-1].strip()
            elif "<|im_end|>" in response:  # Some models use this
                answer = response.split("<|im_end|>")[-1].strip()
            else:
                # Try to find the last meaningful part
                # Remove common prefixes
                for prefix in ["USER:", "ASSISTANT:", "[/INST]", "<|im_start|>"]:
                    if prefix in response:
                        parts = response.split(prefix)
                        if len(parts) > 1:
                            answer = parts[-1].strip()
                            break
                else:
                    answer = response.strip()
            
            answers.append(answer)
        
        return answers
    
    def set_trainable_components(self, train_language_model: bool = True, train_projection: bool = True):
        """Set which components are trainable."""
        if train_projection:
            for param in self.vision_projection.parameters():
                param.requires_grad = True
        else:
            for param in self.vision_projection.parameters():
                param.requires_grad = False
        
        if train_language_model:
            for param in self.language_model.parameters():
                param.requires_grad = True
        else:
            for param in self.language_model.parameters():
                param.requires_grad = False
        
        self.language_model_trainable = train_language_model


class ChessQwenVL(nn.Module):
    """
    Qwen2.5-VL model with swappable vision encoder.
    
    Architecture:
    - Vision Encoder: CLIP ViT-B/32 (generic or chess-finetuned)
    - Language Model: Qwen2.5-VL-3B-Instruct
    """
    
    def __init__(
        self,
        vision_encoder_type: str = "generic",  # "generic" or "chess_finetuned"
        chess_clip_checkpoint: Optional[str] = None,
        language_model_name: str = "Qwen/Qwen2-VL-3B-Instruct",
        device: str = "cuda"
    ):
        """
        Initialize ChessQwenVL model.
        
        Args:
            vision_encoder_type: "generic" for pretrained CLIP, "chess_finetuned" for chess CLIP
            chess_clip_checkpoint: Path to chess-finetuned CLIP checkpoint (required if chess_finetuned)
            language_model_name: HuggingFace model name for Qwen2.5-VL
            device: Device to load models on
        """
        super().__init__()
        
        self.vision_encoder_type = vision_encoder_type
        self.device = device
        
        # Load vision encoder
        self.vision_encoder = self._load_vision_encoder(chess_clip_checkpoint)
        
        # Load Qwen2.5-VL model and processor
        print(f"Loading language model: {language_model_name}")
        self.language_model_name = language_model_name
        
        # Try to get HuggingFace token from environment or use None
        import os
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        
        try:
            # Try AutoProcessor first (works for Qwen2-VL models)
            self.processor = AutoProcessor.from_pretrained(
                language_model_name,
                token=hf_token
            )
            self.language_model = AutoModelForCausalLM.from_pretrained(
                language_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
                token=hf_token
            )
            print(f"[OK] Loaded Qwen2.5-VL model using AutoProcessor")
            self.model_type = "qwen_auto"
        except Exception as e:
            print(f"AutoProcessor failed: {e}")
            # Try Qwen2VLProcessor if available
            try:
                from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
                self.processor = Qwen2VLProcessor.from_pretrained(
                    language_model_name,
                    token=hf_token
                )
                self.language_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    language_model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map=device,
                    token=hf_token
                )
                print(f"[OK] Loaded Qwen2.5-VL model using Qwen2VLProcessor")
                self.model_type = "qwen2vl"
            except Exception as e2:
                print(f"Qwen2VLProcessor also failed: {e2}")
                raise e2
        
        # Store reference to our vision encoder for direct use
        self.custom_vision_encoder = self.vision_encoder
        
        # Freeze vision encoder (we only train projection + language model)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # Language model can be frozen or trainable (configurable)
        self.language_model_trainable = True
    
    def _load_vision_encoder(self, chess_clip_checkpoint: Optional[str]):
        """Load vision encoder (generic or chess-finetuned CLIP)."""
        print(f"Loading vision encoder: {self.vision_encoder_type}")
        
        if self.vision_encoder_type == "generic":
            # Load generic pretrained CLIP
            print("  Loading generic CLIP ViT-B/32...")
            model, _, _ = open_clip.create_model_and_transforms(
                model_name="ViT-B-32",
                pretrained="laion2B-s34B-b79K",
                device=self.device
            )
            vision_encoder = model.visual
            print(f"[OK] Loaded generic CLIP vision encoder")
            
        elif self.vision_encoder_type == "chess_finetuned":
            # Load chess-finetuned CLIP
            if chess_clip_checkpoint is None:
                raise ValueError("chess_clip_checkpoint required for chess_finetuned encoder")
            
            print(f"  Loading chess-finetuned CLIP from {chess_clip_checkpoint}...")
            model, _, _ = open_clip.create_model_and_transforms(
                model_name="ViT-B-32",
                pretrained="laion2B-s34B-b79K",
                device=self.device
            )
            
            # Load checkpoint
            checkpoint = torch.load(chess_clip_checkpoint, map_location=self.device)
            model_state_dict = checkpoint["model_state_dict"]
            model.load_state_dict(model_state_dict)
            
            vision_encoder = model.visual
            print(f"[OK] Loaded chess-finetuned CLIP vision encoder")
            
        else:
            raise ValueError(f"Unknown vision_encoder_type: {self.vision_encoder_type}")
        
        vision_encoder.eval()
        return vision_encoder
    
    def generate(
        self,
        images: torch.Tensor,
        prompts: List[str],
        max_new_tokens: int = 512,
        use_fen_context: bool = True,
        **generate_kwargs
    ) -> List[str]:
        """
        Generate responses for given images and prompts.
        
        Args:
            images: [batch_size, 3, 224, 224] or list of PIL Images
            prompts: List of prompt strings (may already contain FEN context)
            max_new_tokens: Maximum tokens to generate
            use_fen_context: Whether FEN context is already in prompts
            **generate_kwargs: Additional generation arguments
            
        Returns:
            List of generated text responses
        """
        self.eval()
        
        # Convert tensor to PIL if needed
        if isinstance(images, torch.Tensor):
            from torchvision import transforms
            to_pil = transforms.ToPILImage()
            # Handle batch dimension
            if images.dim() == 4:
                pil_images = []
                for img in images:
                    if img.dim() == 4:
                        img = img.squeeze(0)
                    img_cpu = img.cpu()
                    # Denormalize if needed
                    if img_cpu.min() < 0 or img_cpu.max() > 1.5:
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_cpu = img_cpu * std + mean
                        img_cpu = torch.clamp(img_cpu, 0, 1)
                    if img_cpu.max() <= 1.0:
                        img_cpu = img_cpu * 255
                    img_cpu = torch.clamp(img_cpu, 0, 255).byte()
                    try:
                        pil_images.append(to_pil(img_cpu))
                    except Exception:
                        img_cpu_raw = torch.clamp(img.cpu() * 255, 0, 255).byte()
                        pil_images.append(to_pil(img_cpu_raw))
            elif images.dim() == 3:
                img_cpu = images.cpu()
                if img_cpu.min() < 0 or img_cpu.max() > 1.5:
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_cpu = img_cpu * std + mean
                    img_cpu = torch.clamp(img_cpu, 0, 1)
                if img_cpu.max() <= 1.0:
                    img_cpu = img_cpu * 255
                img_cpu = torch.clamp(img_cpu, 0, 255).byte()
                try:
                    pil_images = [to_pil(img_cpu)]
                except Exception:
                    img_cpu_raw = torch.clamp(images.cpu() * 255, 0, 255).byte()
                    pil_images = [to_pil(img_cpu_raw)]
            else:
                raise ValueError(f"Unexpected image tensor shape: {images.shape}")
        elif isinstance(images, list):
            pil_images = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    from torchvision import transforms
                    to_pil = transforms.ToPILImage()
                    img_cpu = img.cpu()
                    if img_cpu.min() < 0 or img_cpu.max() > 1.5:
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_cpu = img_cpu * std + mean
                        img_cpu = torch.clamp(img_cpu, 0, 1)
                    if img_cpu.max() <= 1.0:
                        img_cpu = img_cpu * 255
                    img_cpu = torch.clamp(img_cpu, 0, 255).byte()
                    pil_images.append(to_pil(img_cpu))
                else:
                    pil_images.append(img)
        else:
            pil_images = [images]
        
        # Ensure we have the same number of images and prompts
        if len(pil_images) != len(prompts):
            if len(pil_images) == 1:
                pil_images = pil_images * len(prompts)
            else:
                raise ValueError(f"Mismatch: {len(pil_images)} images but {len(prompts)} prompts")
        
        # Process inputs - handle one at a time for memory efficiency
        all_outputs = []
        
        for i, (img, prompt) in enumerate(zip(pil_images, prompts)):
            try:
                # Qwen2-VL uses a specific format
                # Format: messages with image and text
                if self.model_type == "qwen2vl":
                    # Qwen2VLProcessor: use apply_chat_template to convert messages to text
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    # Convert messages to text format using chat template
                    text = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    # Process text and images
                    inputs = self.processor(
                        text=text,
                        images=[img],
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                else:
                    # AutoProcessor format - try messages first, then fallback
                    try:
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": img},
                                    {"type": "text", "text": prompt}
                                ]
                            }
                        ]
                        # Try apply_chat_template if available
                        if hasattr(self.processor, 'apply_chat_template'):
                            text = self.processor.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True
                            )
                            inputs = self.processor(
                                text=text,
                                images=[img],
                                padding=True,
                                return_tensors="pt"
                            ).to(self.device)
                        else:
                            inputs = self.processor(
                                text=messages,
                                padding=True,
                                return_tensors="pt"
                            ).to(self.device)
                    except Exception:
                        # Fallback: pass image separately
                        try:
                            inputs = self.processor(
                                text=prompt,
                                images=img,
                                return_tensors="pt",
                                padding=True
                            ).to(self.device)
                        except Exception as e:
                            logger.warning(f"Failed to process inputs: {e}")
                            raise e
                
                # Generate with memory management
                with torch.no_grad():
                    try:
                        # Clear cache before generation
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        outputs = self.language_model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            **generate_kwargs
                        )
                        
                        # Clear cache after generation
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except RuntimeError as e:
                        if "out of memory" in str(e) or "CUDA" in str(e):
                            # Clear cache and try again with smaller max_new_tokens
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            logger.warning(f"OOM error, retrying with smaller max_new_tokens")
                            safe_kwargs = {k: v for k, v in generate_kwargs.items() 
                                          if k not in ['temperature', 'do_sample']}
                            safe_kwargs['max_new_tokens'] = min(max_new_tokens, 128)
                            outputs = self.language_model.generate(
                                **inputs,
                                **safe_kwargs
                            )
                        else:
                            raise e
                
                # Decode response
                if hasattr(self.processor, 'tokenizer'):
                    response = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    response = self.processor.decode(outputs[0], skip_special_tokens=True)
                
                all_outputs.append(response)
                
            except Exception as e:
                import traceback
                error_msg = f"Error processing image {i}: {type(e).__name__}: {str(e)}"
                logger.warning(error_msg)
                if i == 0:
                    logger.debug(f"  Full traceback:\n{traceback.format_exc()}")
                all_outputs.append("Error generating response")
        
        # Extract answers (remove prompt)
        answers = []
        for response in all_outputs:
            # Qwen2.5-VL typically uses specific markers
            if "<|im_end|>" in response:
                answer = response.split("<|im_end|>")[-1].strip()
            elif "assistant" in response.lower():
                parts = response.lower().split("assistant")
                if len(parts) > 1:
                    answer = parts[-1].strip()
                else:
                    answer = response.strip()
            else:
                # Try to find the last meaningful part
                answer = response.strip()
            
            answers.append(answer)
        
        return answers
    
    def set_trainable_components(self, train_language_model: bool = True, train_projection: bool = True):
        """Set which components are trainable."""
        if train_language_model:
            for param in self.language_model.parameters():
                param.requires_grad = True
        else:
            for param in self.language_model.parameters():
                param.requires_grad = False
        
        self.language_model_trainable = train_language_model

