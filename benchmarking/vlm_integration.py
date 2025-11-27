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
        self.use_imagetext_api = False  # Default to False
        self._load_model()
    
    def _load_model(self):
        """Load the VLM model and processor."""
        # Check for OpenAI models
        if self.model_name.startswith("gpt-"):
            print(f"Initializing OpenAI model: {self.model_name}")
            # We'll replace self with OpenAIVLMInterface instance logic or just delegate
            # Since we can't easily swap 'self', we'll use a delegate pattern or just handle it here
            # But a cleaner way for this codebase is to handle it in the __init__ or factory
            return

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
                print(f"[OK] VLM model loaded on {self.device} (LLaVA Next API)")
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
                    print(f"[OK] VLM model loaded on {self.device} (LLaVA 1.6 Vision2Seq API)")
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
                        print(f"[OK] VLM model loaded on {self.device} (LLaVA 1.5 API)")
                        return
                    except Exception as e3:
                        print(f"  LLaVA 1.5 API failed: {e3}")
                        raise e1
        except ImportError:
            raise ImportError(
                "transformers library not found. Install with: pip install transformers"
            )
        except Exception as e:
            print(f"[ERROR] Could not load LLaVA model: {e}")
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
            content_text = question
            if fen_context:
                content_text = f"{question}\n\nFEN representation: {fen_context}"
            
            # Use chat template if available (Standard for LLaVA 1.6/Next)
            if hasattr(self.processor, "apply_chat_template"):
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": content_text},
                        ],
                    },
                ]
                prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            else:
                # Fallback manual prompt construction
                # LLaVA 1.5 style
                prompt = f"USER: <image>\n{content_text}\nASSISTANT:"
            
            # Process inputs - handle different LLaVA API versions
            if self.use_imagetext_api:
                # ImageTextToText API (newer)
                inputs = self.processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)
            else:
                # Standard format (Vision2Seq or LLaVA 1.5)
                try:
                    inputs = self.processor(
                        text=prompt,
                        images=image,
                        return_tensors="pt"
                    ).to(self.device)
                except Exception as e1:
                    # Try alternative format for Vision2Seq
                    if hasattr(self.processor, 'image_processor') and hasattr(self.processor, 'tokenizer'):
                        image_inputs = self.processor.image_processor(image, return_tensors="pt")
                        text_inputs = self.processor.tokenizer(prompt, return_tensors="pt", padding=True)
                        inputs = {**image_inputs, **text_inputs}
                        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    else:
                        raise e1
            
            # Generate response
            with torch.no_grad():
                generate_kwargs = {
                    "max_new_tokens": 512,
                    "do_sample": False,
                }
                
                try:
                    outputs = self.model.generate(**inputs, **generate_kwargs)
                except Exception as e:
                    # Remove temperature if do_sample is False
                    if "temperature" in str(e).lower() or "unexpected keyword" in str(e).lower():
                        generate_kwargs.pop("temperature", None)
                    try:
                        outputs = self.model.generate(**inputs, **generate_kwargs)
                    except Exception as e2:
                        # Last resort: try with minimal kwargs
                        outputs = self.model.generate(**inputs, max_new_tokens=512)
            
            # Decode response
            try:
                if hasattr(self.processor, 'tokenizer'):
                    response = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    response = self.processor.decode(outputs[0], skip_special_tokens=True)
            except:
                # Fallback decoding
                response = str(outputs[0])
            
            # Extract answer (remove prompt from response)
            # With chat templates, the prompt might be complex, so we might need better cleaning
            # But usually decode(skip_special_tokens=True) handles it if the model outputs only the answer
            # However, LLaVA often includes the prompt in the output
            
            # Simple heuristic: if the prompt is in the response, split it
            # Note: prompt variable contains the full formatted prompt
            
            # For now, just return the whole thing if we can't easily split, 
            # or try to find the last "ASSISTANT:" or equivalent
            
            if "ASSISTANT:" in response:
                answer = response.split("ASSISTANT:")[-1].strip()
            elif "[/INST]" in response: # Mistral/Llama style
                answer = response.split("[/INST]")[-1].strip()
            else:
                # If we can't find a delimiter, just return the whole response
                # It might contain the prompt, but it's better than nothing
                answer = response.strip()
                
                # Try to remove the prompt if it's a direct prefix
                # This is tricky with chat templates as the decoded prompt might differ slightly
                pass
            
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



class OpenAIVLMInterface:
    """Interface for OpenAI Vision Models (GPT-4o, etc)."""
    
    def __init__(self, model_name: str = "gpt-4o", api_key: str = None):
        """
        Initialize OpenAI VLM interface.
        
        Args:
            model_name: OpenAI model identifier
            api_key: OpenAI API key (optional, uses env var if None)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai library not found. Install with: pip install openai")
            
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        print(f"[OK] Initialized OpenAI VLM with model: {model_name}")

    def answer_question(self, image_path: str, question: str, fen_context: Optional[str] = None) -> str:
        """Answer question using OpenAI API."""
        import base64
        
        # Encode image
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
        # Construct prompt
        content_text = question
        if fen_context:
            content_text = f"{question}\n\nFEN representation: {fen_context}"
            
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": content_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] OpenAI API error: {e}")
            return f"Error: {str(e)}"

    def answer_question_batch(self, image_paths: list, questions: list, fen_contexts: Optional[list] = None) -> list:
        """Batch answer (sequential for API)."""
        if fen_contexts is None:
            fen_contexts = [None] * len(image_paths)
        return [self.answer_question(img, q, fen) for img, q, fen in zip(image_paths, questions, fen_contexts)]


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

