"""
VLM (Vision Language Model) integration for answering chess questions.
Supports LLaVA and other VLMs via transformers.
"""

import torch
from PIL import Image
from typing import Optional, Dict
import os


class VLMInterface:
    """Interface for Vision Language Models to answer chess questions."""
    
    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf", device: str = None, local_model_path: str = None):
        """
        Initialize VLM interface.
        
        Args:
            model_name: HuggingFace model identifier for LLaVA (used if local_model_path is None)
            device: Device to run inference on
            local_model_path: Path to local model directory (e.g., for Mistral version)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.local_model_path = local_model_path
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the VLM model and processor."""
        try:
            # Use local model path if provided
            model_path = self.local_model_path if self.local_model_path else self.model_name
            
            if self.local_model_path:
                print(f"Loading local VLM model from: {self.local_model_path}")
            else:
                print(f"Loading VLM model from HuggingFace: {self.model_name}")
            
            # Try LLaVA Next/1.6 API first (uses AutoProcessor)
            try:
                from transformers import AutoProcessor, AutoModelForCausalLM
                
                print("  Trying LLaVA Next API (AutoModelForCausalLM)...")
                self.processor = AutoProcessor.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map=self.device
                )
                print(f"✅ VLM model loaded on {self.device} (LLaVA Next API)")
                return
            except Exception as e1:
                print(f"  LLaVA Next API failed: {e1}")
                # Try LLaVA 1.6 Vision2Seq API
                try:
                    from transformers import AutoProcessor, AutoModelForVision2Seq
                    
                    print("  Trying LLaVA 1.6 Vision2Seq API...")
                    self.processor = AutoProcessor.from_pretrained(model_path)
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map=self.device
                    )
                    print(f"✅ VLM model loaded on {self.device} (LLaVA 1.6 Vision2Seq API)")
                    return
                except Exception as e2:
                    print(f"  LLaVA 1.6 Vision2Seq API failed: {e2}")
                    # Fallback to LLaVA 1.5 API
                    try:
                        from transformers import LlavaProcessor, LlavaForConditionalGeneration
                        
                        print("  Trying LLaVA 1.5 API...")
                        self.processor = LlavaProcessor.from_pretrained(model_path)
                        self.model = LlavaForConditionalGeneration.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                            device_map=self.device
                        )
                        print(f"✅ VLM model loaded on {self.device} (LLaVA 1.5 API)")
                        return
                    except Exception as e3:
                        print(f"  LLaVA 1.5 API failed: {e3}")
                        raise e1
        except ImportError:
            raise ImportError(
                "transformers library not found. Install with: pip install transformers"
            )
        except Exception as e:
            print(f"❌ Error: Could not load LLaVA model: {e}")
            print("Falling back to mock mode for testing.")
            self.model = None
            self.processor = None
    
    def answer_question(self, image_path: str, question: str, fen_context: Optional[str] = None) -> str:
        """
        Answer a question about a chess board image.
        
        Args:
            image_path: Path to chess board image
            question: Question to ask about the position
            fen_context: Optional FEN string to concatenate to the prompt
            
        Returns:
            Model's answer as a string
        """
        if self.model is None:
            # Mock mode for testing
            return f"Mock answer for question: {question}"
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Construct prompt
            if fen_context:
                prompt = f"{question}\n\nFEN representation: {fen_context}"
            else:
                prompt = question
            
            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.7,
                )
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer (remove prompt from response)
            if prompt in response:
                answer = response.split(prompt)[-1].strip()
            else:
                answer = response.strip()
            
            return answer
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Error: {str(e)}"
    
    def answer_question_batch(self, image_paths: list, questions: list, fen_contexts: Optional[list] = None) -> list:
        """
        Answer multiple questions in batch.
        
        Args:
            image_paths: List of image paths
            questions: List of questions
            fen_contexts: Optional list of FEN contexts
            
        Returns:
            List of answers
        """
        if fen_contexts is None:
            fen_contexts = [None] * len(image_paths)
        
        answers = []
        for img_path, question, fen_ctx in zip(image_paths, questions, fen_contexts):
            answer = self.answer_question(img_path, question, fen_ctx)
            answers.append(answer)
        
        return answers


class MockVLMInterface:
    """Mock VLM interface for testing without actual model."""
    
    def __init__(self):
        self.device = "cpu"
    
    def answer_question(self, image_path: str, question: str, fen_context: Optional[str] = None) -> str:
        """Mock answer generation."""
        if fen_context:
            return f"Mock answer with FEN context: {fen_context[:20]}..."
        return f"Mock answer for: {question}"
    
    def answer_question_batch(self, image_paths: list, questions: list, fen_contexts: Optional[list] = None) -> list:
        """Mock batch answer generation."""
        return [self.answer_question(img, q, fen) for img, q, fen in zip(image_paths, questions, fen_contexts or [None]*len(questions))]

