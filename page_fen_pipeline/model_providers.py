"""
Multi-Model Provider Support
Unified interface for different AI model providers (OpenAI, Anthropic, Google).
"""

import os
import base64
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class ModelProvider:
    """Base class for model providers."""
    
    def generate_fen(self, data_uri: str, prompt: str) -> Dict:
        """Generate FEN from image. Must be implemented by subclasses."""
        raise NotImplementedError


class OpenAIProvider(ModelProvider):
    """OpenAI GPT models (GPT-4, GPT-4o, GPT-4.1, etc.)"""
    
    def __init__(self, model="gpt-4o"):
        self.model = model
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=api_key)
    
    def generate_fen(self, data_uri: str, prompt: str) -> Dict:
        """Generate FEN using OpenAI's chat completions API."""
        
        # For GPT-4.1, GPT-4o, use standard chat.completions
        if any(x in self.model.lower() for x in ['gpt-4o', 'gpt-4.1', 'gpt-4-turbo']):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_uri}
                            }
                        ]
                    }
                ],
                max_tokens=800
            )
            
            text = response.choices[0].message.content
            
        # For GPT-5 and newer models, use responses.create API
        elif 'gpt-5' in self.model.lower():
            user_block = {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_uri},
                ],
            }
            
            response = self.client.responses.create(
                model=self.model,
                input=[user_block],
                max_output_tokens=4000,  # Increased even more for GPT-5's heavy reasoning
                reasoning={
                    "effort": "low"  # Reduce reasoning effort to save tokens for output
                }
            )
            
            # Extract text from GPT-5 response - try multiple methods
            text = None
            
            # Method 1: Direct output_text attribute
            text = getattr(response, "output_text", None)
            
            # Method 2: Try to get from output blocks (look for text items, skip reasoning)
            if not text:
                chunks = []
                output = getattr(response, "output", None)
                if output:
                    # If output is a list
                    if isinstance(output, list):
                        for block in output:
                            # GPT-5 outputs have a 'type' field - we want 'text' not 'reasoning'
                            block_type = getattr(block, "type", None)
                            
                            # Skip reasoning blocks, only process text blocks
                            if block_type == "reasoning":
                                continue
                            
                            # Look for text content
                            content = getattr(block, "content", None)
                            if content:
                                if isinstance(content, list):
                                    for c in content:
                                        c_type = getattr(c, "type", None)
                                        if c_type in ("output_text", "text"):
                                            c_text = getattr(c, "text", "")
                                            if c_text:
                                                chunks.append(c_text)
                                elif isinstance(content, str):
                                    chunks.append(content)
                            
                            # Also try direct text attribute on the block
                            block_text = getattr(block, "text", None)
                            if block_text and isinstance(block_text, str):
                                chunks.append(block_text)
                    
                    # If output is a dict-like object
                    elif hasattr(output, "content"):
                        content = output.content
                        if isinstance(content, str):
                            chunks.append(content)
                
                if chunks:
                    text = "\n".join(chunks).strip()
            
            # Method 3: Try response.text (some APIs use this)
            if not text:
                text = getattr(response, "text", None)
            
            # Method 4: Try response.content
            if not text:
                content = getattr(response, "content", None)
                if content and isinstance(content, str):
                    text = content
            
            # Method 5: Try to convert response to dict and extract
            if not text:
                try:
                    response_dict = response.model_dump() if hasattr(response, "model_dump") else {}
                    if "output" in response_dict:
                        output_data = response_dict["output"]
                        if isinstance(output_data, list) and len(output_data) > 0:
                            first_output = output_data[0]
                            if isinstance(first_output, dict) and "content" in first_output:
                                content_data = first_output["content"]
                                if isinstance(content_data, list) and len(content_data) > 0:
                                    for item in content_data:
                                        if isinstance(item, dict) and item.get("type") in ("output_text", "text"):
                                            text = item.get("text", "")
                                            if text:
                                                break
                except:
                    pass
            
            # Check for incomplete status
            status = getattr(response, "status", None)
            if status == "incomplete":
                incomplete_details = getattr(response, "incomplete_details", None)
                if incomplete_details:
                    reason = getattr(incomplete_details, "reason", "unknown")
                    print(f"    ⚠️  Warning: GPT-5 response incomplete (reason: {reason})")
                    if reason == "max_output_tokens":
                        print(f"    💡 GPT-5 used all tokens for reasoning. Trying with more tokens...")
            
            # If still no text, provide detailed error
            if not text:
                print(f"    ⚠️  Warning: Could not extract text from GPT-5 response")
                print(f"    Response status: {status}")
                print(f"    Response type: {type(response)}")
                
                # Check output types
                output = getattr(response, "output", None)
                if output and isinstance(output, list):
                    output_types = [getattr(item, "type", "unknown") for item in output]
                    print(f"    Output types: {output_types}")
                    if "reasoning" in output_types and "text" not in output_types:
                        print(f"    💡 Only reasoning output found, no text output")
                        print(f"    💡 This means GPT-5 spent all tokens thinking and didn't output text")
                
                if hasattr(response, "model_dump"):
                    import json
                    try:
                        dump = response.model_dump()
                        print(f"    Response dump (truncated):")
                        print(f"    {json.dumps(dump, indent=2)[:500]}...")
                    except:
                        print(f"    Response: {str(response)[:500]}...")
                
                text = "[No text returned - GPT-5 spent all tokens on reasoning]"
        
        else:
            raise ValueError(f"Unsupported OpenAI model: {self.model}")
        
        return {
            'fen': text.strip(),
            'raw_response': text,
            'provider': 'openai',
            'model': self.model
        }


class AnthropicProvider(ModelProvider):
    """Anthropic Claude models."""
    
    def __init__(self, model="claude-3-5-sonnet-20241022"):
        self.model = model
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate_fen(self, data_uri: str, prompt: str) -> Dict:
        """Generate FEN using Anthropic's API."""
        
        # Extract base64 data from data URI
        if data_uri.startswith('data:image'):
            # Format: data:image/png;base64,<base64_data>
            media_type = data_uri.split(';')[0].split(':')[1]
            base64_data = data_uri.split(',')[1]
        else:
            media_type = "image/png"
            base64_data = data_uri
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )
        
        text = response.content[0].text
        
        return {
            'fen': text.strip(),
            'raw_response': text,
            'provider': 'anthropic',
            'model': self.model
        }


class GoogleProvider(ModelProvider):
    """Google Gemini models."""
    
    def __init__(self, model="gemini-2.0-flash-exp"):
        self.model = model
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model)
    
    def generate_fen(self, data_uri: str, prompt: str) -> Dict:
        """Generate FEN using Google's Gemini API."""
        
        # Extract base64 data from data URI
        if data_uri.startswith('data:image'):
            media_type = data_uri.split(';')[0].split(':')[1].split('/')[1]  # e.g., 'png'
            base64_data = data_uri.split(',')[1]
        else:
            media_type = "png"
            base64_data = data_uri
        
        # Decode base64 to bytes
        import base64 as b64
        image_bytes = b64.b64decode(base64_data)
        
        # Create PIL Image
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes))
        
        # Generate content
        response = self.client.generate_content([prompt, img])
        
        text = response.text
        
        return {
            'fen': text.strip(),
            'raw_response': text,
            'provider': 'google',
            'model': self.model
        }


# Model registry
MODEL_PROVIDERS = {
    # OpenAI models
    'gpt-4o': ('openai', 'gpt-4o'),
    'gpt-4o-mini': ('openai', 'gpt-4o-mini'),
    'gpt-4-turbo': ('openai', 'gpt-4-turbo'),
    'gpt-4.1-mini': ('openai', 'gpt-4.1-mini'),
    'gpt-4.1-nano': ('openai', 'gpt-4.1-nano'),
    'gpt-5': ('openai', 'gpt-5'),
    'gpt-5-mini': ('openai', 'gpt-5-mini'),
    'gpt-5-nano': ('openai', 'gpt-5-nano'),
    
    # Anthropic models
    'claude-3-5-sonnet-20241022': ('anthropic', 'claude-3-5-sonnet-20241022'),
    'claude-3-5-sonnet': ('anthropic', 'claude-3-5-sonnet-20241022'),
    'claude-3-5-haiku': ('anthropic', 'claude-3-5-haiku-20241022'),
    'claude-3-opus': ('anthropic', 'claude-3-opus-20240229'),
    'claude-3-sonnet': ('anthropic', 'claude-3-sonnet-20240229'),
    'claude-3-haiku': ('anthropic', 'claude-3-haiku-20240307'),
    'claude-4.5-sonnet': ('anthropic', 'claude-sonnet-4-5-20250929'),
    'claude-4.5-haiku': ('anthropic', 'claude-haiku-4-5-20251001'),
    'claude-4.1-opus': ('anthropic', 'claude-opus-4-1-20250805'),
    
    # Google models
    'gemini-2.0-flash-exp': ('google', 'gemini-2.0-flash-exp'),
    'gemini-2.0-flash': ('google', 'gemini-2.0-flash-exp'),
    'gemini-1.5-pro': ('google', 'gemini-1.5-pro'),
    'gemini-1.5-flash': ('google', 'gemini-1.5-flash'),
    'gemini-pro': ('google', 'gemini-pro'),
    'gemini-2.5-flash': ('google', 'gemini-2.5-flash'),
    'gemini-2.5-flash-exp': ('google', 'gemini-2.5-flash-exp'),
    'gemini-2.5-flash-lite': ('google', 'gemini-2.5-flash-lite'),
    'gemini-2.5-pro': ('google', 'gemini-2.5-pro'),
}


def get_model_provider(model: str) -> ModelProvider:
    """
    Get the appropriate provider for a model.
    
    Args:
        model: Model identifier
    
    Returns:
        ModelProvider instance
    """
    model_lower = model.lower()
    
    # Check registry
    if model_lower in MODEL_PROVIDERS:
        provider_name, actual_model = MODEL_PROVIDERS[model_lower]
    else:
        # Try to infer provider from model name
        if 'gpt' in model_lower:
            provider_name = 'openai'
            actual_model = model
        elif 'claude' in model_lower:
            provider_name = 'anthropic'
            actual_model = model
        elif 'gemini' in model_lower:
            provider_name = 'google'
            actual_model = model
        else:
            raise ValueError(f"Unknown model: {model}. Cannot determine provider.")
    
    # Create provider
    if provider_name == 'openai':
        return OpenAIProvider(actual_model)
    elif provider_name == 'anthropic':
        return AnthropicProvider(actual_model)
    elif provider_name == 'google':
        return GoogleProvider(actual_model)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def list_available_models() -> Dict[str, list]:
    """List all available models by provider."""
    models_by_provider = {
        'openai': [],
        'anthropic': [],
        'google': []
    }
    
    for model_name, (provider, _) in MODEL_PROVIDERS.items():
        if model_name not in models_by_provider[provider]:
            models_by_provider[provider].append(model_name)
    
    return models_by_provider


def generate_fen_universal(img_array, model: str, prompt: Optional[str] = None) -> Dict:
    """
    Generate FEN using any supported model provider.
    
    Args:
        img_array: numpy array of the chess board image
        model: Model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet', 'gemini-2.0-flash')
        prompt: Optional custom prompt (uses default if None)
    
    Returns:
        Dict with 'fen', 'raw_response', 'provider', 'model'
    """
    from fen_generator import numpy_array_to_data_uri
    
    # Default prompt
    if prompt is None:
        prompt = """Analyze this chess board image carefully and return ONLY the FEN (Forsyth–Edwards Notation) of the position.

The FEN should look like this: r3k2r/ppb1pppp/2n5/4B3/1b6/8/PPP2PPP/RN1QK2R w KQkq - 0 1

Please examine the board carefully, count all pieces including pawns, and construct the FEN accurately.
If you cannot determine the position with confidence, respond with 'UNCERTAIN'."""
    
    # Get data URI
    data_uri = numpy_array_to_data_uri(img_array)
    
    # Get provider and generate
    provider = get_model_provider(model)
    result = provider.generate_fen(data_uri, prompt)
    
    return result

