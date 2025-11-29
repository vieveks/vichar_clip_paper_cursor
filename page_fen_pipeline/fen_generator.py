"""
FEN Generator Module
Uses OpenAI GPT models to analyze chess board images and generate FEN notation.
"""

import os
import base64
import mimetypes
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from PIL import Image
import io

# Load environment variables
load_dotenv()


def get_openai_client():
    """Initialize and return OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)


def numpy_array_to_data_uri(img_array):
    """
    Convert a numpy array (BGR or RGB) to a base64 data URI.
    
    Args:
        img_array: numpy array of the image (can be BGR from OpenCV)
    
    Returns:
        str: base64 data URI
    """
    # Convert BGR to RGB if needed (OpenCV uses BGR)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_rgb = img_array[:, :, ::-1]  # BGR to RGB
    else:
        img_rgb = img_array
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img_rgb)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    
    # Encode to base64
    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def file_to_data_uri(image_path: str) -> str:
    """Convert local image file to base64 data URI."""
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/png"
    
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def generate_fen_from_image_array(img_array, model="gpt-4o", client=None):
    """
    Generate FEN notation from a chess board image array.
    
    Args:
        img_array: numpy array of the chess board image
        model: Model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet', 'gemini-2.0-flash')
        client: Deprecated - kept for backward compatibility
    
    Returns:
        dict with 'fen' and 'raw_response' keys
    """
    # Use universal model provider
    from model_providers import generate_fen_universal
    
    prompt = (
        "Analyze this chess board image carefully and return ONLY the FEN (Forsyth–Edwards Notation) of the position.\n"
        "The FEN should look like this: r3k2r/ppb1pppp/2n5/4B3/1b6/8/PPP2PPP/RN1QK2R w KQkq - 0 1\n"
        "Please examine the board carefully, count all pieces including pawns, and construct the FEN accurately.\n"
        "If you cannot determine the position with confidence, respond with 'UNCERTAIN'."
    )
    
    result = generate_fen_universal(img_array, model, prompt)
    
    return {
        'fen': result['fen'],
        'raw_response': result['raw_response'],
        'provider': result.get('provider'),
        'model': result.get('model')
    }


def generate_fen_from_file(image_path: str, model="gpt-4o", client=None):
    """
    Generate FEN notation from a chess board image file.
    
    Args:
        image_path: Path to the chess board image
        model: Model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet', 'gemini-2.0-flash')
        client: Deprecated - kept for backward compatibility
    
    Returns:
        dict with 'fen' and 'raw_response' keys
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image as numpy array
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Use the array-based function
    return generate_fen_from_image_array(img, model=model, client=client)

