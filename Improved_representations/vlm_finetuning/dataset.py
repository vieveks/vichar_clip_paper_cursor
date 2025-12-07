"""
Dataset loader for Qwen2-VL-2B fine-tuning.

Loads VLM dataset format and prepares it for training.
"""

import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional
import os


class QwenVLDataset(Dataset):
    """
    Dataset for Qwen2-VL fine-tuning.
    
    Loads images and conversation pairs for instruction fine-tuning.
    """
    
    def __init__(
        self,
        data_path: str,
        image_base_dir: Optional[str] = None,
        processor=None,
        max_length: int = 2048
    ):
        """
        Args:
            data_path: Path to JSON dataset file (VLM format)
            image_base_dir: Base directory for resolving image paths
            processor: Qwen2-VL processor for image/text processing
            max_length: Maximum sequence length
        """
        self.data_path = Path(data_path)
        self.image_base_dir = Path(image_base_dir) if image_base_dir else self.data_path.parent
        self.processor = processor
        self.max_length = max_length
        
        # Load dataset
        print(f"Loading dataset from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        item = self.data[idx]
        
        # Load image
        image_path = item['image']
        if not Path(image_path).is_absolute():
            full_image_path = self.image_base_dir / image_path
        else:
            full_image_path = Path(image_path)
        
        if not full_image_path.exists():
            # Try alternative paths
            alt_paths = [
                self.data_path.parent.parent / 'data' / 'hf_chess_puzzles' / image_path,
                Path(image_path),
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    full_image_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(full_image_path).convert('RGB')
        
        # Extract conversations
        conversations = item['conversations']
        user_message = conversations[0]['value']  # Contains <image>\n{instruction}
        assistant_message = conversations[1]['value']  # JSON response
        
        # Process with processor if available
        if self.processor:
            # Extract text from user message (remove <image> tag)
            user_text = user_message.replace("<image>\n", "").strip()
            
            # Format messages for Qwen2-VL (following pattern from clip_as_encoder/model.py)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_text}
                    ]
                },
                {
                    "role": "assistant",
                    "content": assistant_message
                }
            ]
            
            # Convert messages to text using chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Process using messages format - this should include image_grid_thw
            # Qwen2-VL processor needs messages format to compute grid_thw correctly
            try:
                inputs = self.processor(
                    messages,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
            except Exception as e:
                # Fallback: process text and images separately
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
            
            # Always ensure image_grid_thw is present
            # Compute grid_thw from pixel_values if not provided by processor
            if 'image_grid_thw' not in inputs or inputs.get('image_grid_thw') is None:
                if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                    # Get image dimensions from pixel_values
                    # pixel_values shape: [batch, channels, height, width]
                    _, _, h, w = inputs['pixel_values'].shape
                    # Qwen2-VL uses 14x14 patches, compute grid
                    # Grid format: [[t, h, w]] where t=1 for single image
                    patch_size = 14
                    grid_h = (h + patch_size - 1) // patch_size
                    grid_w = (w + patch_size - 1) // patch_size
                    inputs['image_grid_thw'] = [[1, grid_h, grid_w]]
                else:
                    # Default fallback
                    inputs['image_grid_thw'] = [[1, 32, 32]]
            
            # Extract labels (only assistant tokens)
            labels = inputs['input_ids'].clone()
            # Mask out user tokens (set to -100)
            # This is a simplified version - in practice, you'd need to properly mask
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
            # Prepare return dict with all required fields
            # IMPORTANT: Squeeze the batch dimension (0) added by the processor
            return_dict = {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': labels.squeeze(0),
            }
            
            # Handle pixel_values: squeeze if it has batch dimension
            if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                return_dict['pixel_values'] = inputs['pixel_values'].squeeze(0)
            
            # Handle image_grid_thw: Ensure it's a tensor and squeeze batch dimension
            if 'image_grid_thw' in inputs:
                grid = inputs['image_grid_thw']
                
                # Convert list to tensor if necessary
                if not torch.is_tensor(grid):
                    grid = torch.tensor(grid)
                
                # Squeeze shape from (1, 3) to (3,) so batching creates (Batch, 3)
                if grid.dim() == 2 and grid.shape[0] == 1:
                    grid = grid.squeeze(0)
                    
                return_dict['image_grid_thw'] = grid
            
            return return_dict
        else:
            # Return raw data if no processor
            return {
                'image': image,
                'user_message': user_message,
                'assistant_message': assistant_message,
                'conversations': conversations
            }

