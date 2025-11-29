"""
Enhanced FEN Generator Module
Advanced techniques for improving FEN generation accuracy.
"""

import os
import base64
import mimetypes
from typing import Dict, List, Optional
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from PIL import Image, ImageEnhance
import io
import cv2
import re

# Load environment variables
load_dotenv()


def get_openai_client():
    """Initialize and return OpenAI client."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)


def enhance_board_image(img_array):
    """
    Apply image enhancements to improve board clarity.
    
    Args:
        img_array: numpy array of the board image (BGR or RGB)
    
    Returns:
        Enhanced numpy array
    """
    # Convert to PIL for easy enhancement
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_rgb = img_array[:, :, ::-1]  # BGR to RGB if needed
    else:
        img_rgb = img_array
    
    pil_img = Image.fromarray(img_rgb.astype(np.uint8))
    
    # 1. Increase contrast
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.3)
    
    # 2. Increase sharpness
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.5)
    
    # 3. Slight brightness adjustment
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.1)
    
    return np.array(pil_img)


def upscale_board_image(img_array, target_size=1024):
    """
    Upscale board image to higher resolution for better recognition.
    
    Args:
        img_array: numpy array of the board image
        target_size: target size for the longest edge
    
    Returns:
        Upscaled numpy array
    """
    h, w = img_array.shape[:2]
    if max(h, w) >= target_size:
        return img_array
    
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Use LANCZOS for high-quality upscaling
    if len(img_array.shape) == 3:
        img_rgb = img_array[:, :, ::-1]  # BGR to RGB
    else:
        img_rgb = img_array
    
    pil_img = Image.fromarray(img_rgb.astype(np.uint8))
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return np.array(pil_img)


def numpy_array_to_data_uri(img_array):
    """Convert numpy array to base64 data URI."""
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_rgb = img_array[:, :, ::-1]  # BGR to RGB
    else:
        img_rgb = img_array
    
    pil_img = Image.fromarray(img_rgb.astype(np.uint8))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def get_detailed_prompt():
    """Get an enhanced, more detailed prompt for FEN generation."""
    return """You are a chess position analyzer. Analyze this chess board image and provide the FEN (Forsyth-Edwards Notation).

IMPORTANT INSTRUCTIONS:
1. Examine the board carefully, square by square, from rank 8 (top) to rank 1 (bottom)
2. For each rank, go from file a (left) to file h (right)
3. Count pieces carefully - a standard game has 16 pieces per side initially
4. Use proper notation:
   - Uppercase for White pieces: K, Q, R, B, N, P
   - Lowercase for Black pieces: k, q, r, b, n, p
   - Numbers for consecutive empty squares (1-8)
   - Forward slash (/) between ranks

5. FEN format: [position] [turn] [castling] [en passant] [halfmove] [fullmove]
   - Position: The piece placement (required)
   - Turn: 'w' for white, 'b' for black (use 'w' if unsure)
   - Castling: Combination of KQkq or '-' (use '-' if unsure)
   - En passant: Target square or '-' (use '-' if unsure)
   - Halfmove clock: Number (use '0' if unsure)
   - Fullmove number: Number (use '1' if unsure)

6. Verify your FEN:
   - Each rank should account for all 8 squares
   - Sum of empty squares and pieces in each rank = 8
   - Kings (K and k) should be present

EXAMPLE FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

Now analyze the provided chess board and return ONLY the FEN notation, nothing else."""


def get_step_by_step_prompt():
    """Get a chain-of-thought prompt for more careful analysis."""
    return """Analyze this chess board image step by step.

STEP 1: Describe what you see
- Identify the board orientation (is White on bottom or top?)
- Note the piece colors and positions you can clearly see
- Count the total pieces visible

STEP 2: Analyze rank by rank (from rank 8 to rank 1)
For each rank, list what pieces you see from left to right (files a-h)

STEP 3: Generate the FEN notation
Based on your analysis, provide the complete FEN in this format:
[position] [turn] [castling] [en passant] [halfmove] [fullmove]

STEP 4: Verification
- Check that each rank has exactly 8 squares
- Verify both kings are present
- Confirm piece counts are reasonable

Finally, provide your final FEN on a line starting with "FINAL FEN:"."""


def validate_fen(fen: str) -> Dict:
    """
    Validate FEN notation and return validation results.
    
    Args:
        fen: FEN string to validate
    
    Returns:
        Dict with 'valid' (bool), 'errors' (list), and 'cleaned_fen' (str)
    """
    errors = []
    
    # Extract just the FEN if there's extra text
    fen_match = re.search(r'[rnbqkpRNBQKP1-8/]+ [wb] [KQkq-]+ [a-h36-]+ \d+ \d+', fen)
    if fen_match:
        fen = fen_match.group(0)
    else:
        # Try simpler pattern (position only)
        fen_match = re.search(r'[rnbqkpRNBQKP1-8/]+', fen)
        if fen_match:
            fen = fen_match.group(0) + " w KQkq - 0 1"
    
    parts = fen.strip().split()
    
    if len(parts) < 1:
        return {'valid': False, 'errors': ['Empty FEN'], 'cleaned_fen': ''}
    
    position = parts[0]
    ranks = position.split('/')
    
    # Check number of ranks
    if len(ranks) != 8:
        errors.append(f'Expected 8 ranks, found {len(ranks)}')
    
    # Check each rank
    for i, rank in enumerate(ranks, 1):
        count = 0
        for char in rank:
            if char.isdigit():
                count += int(char)
            elif char in 'rnbqkpRNBQKP':
                count += 1
            else:
                errors.append(f'Invalid character in rank {i}: {char}')
        
        if count != 8:
            errors.append(f'Rank {i} has {count} squares instead of 8')
    
    # Check for kings
    if 'K' not in position:
        errors.append('Missing white king (K)')
    if 'k' not in position:
        errors.append('Missing black king (k)')
    
    # Count kings (should be exactly 1 of each)
    if position.count('K') != 1:
        errors.append(f'Found {position.count("K")} white kings, expected 1')
    if position.count('k') != 1:
        errors.append(f'Found {position.count("k")} black kings, expected 1')
    
    # Ensure proper FEN format
    if len(parts) < 6:
        # Add missing parts with defaults
        while len(parts) < 6:
            if len(parts) == 1:
                parts.append('w')  # turn
            elif len(parts) == 2:
                parts.append('-')  # castling
            elif len(parts) == 3:
                parts.append('-')  # en passant
            elif len(parts) == 4:
                parts.append('0')  # halfmove
            elif len(parts) == 5:
                parts.append('1')  # fullmove
        fen = ' '.join(parts)
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'cleaned_fen': fen
    }


def generate_fen_enhanced(
    img_array,
    model="gpt-4o",
    client=None,
    use_enhancement=True,
    use_upscaling=True,
    detailed_prompt=True
) -> Dict:
    """
    Generate FEN with enhanced techniques.
    
    Args:
        img_array: numpy array of the board image
        model: Model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet', 'gemini-2.0-flash')
        client: Deprecated - kept for backward compatibility
        use_enhancement: Apply image enhancement
        use_upscaling: Upscale image to higher resolution
        detailed_prompt: Use detailed prompt instead of simple one
    
    Returns:
        Dict with 'fen', 'raw_response', 'validation', 'enhanced'
    """
    # Preprocess image
    processed_img = img_array.copy()
    
    if use_upscaling:
        processed_img = upscale_board_image(processed_img)
    
    if use_enhancement:
        processed_img = enhance_board_image(processed_img)
    
    # Choose prompt
    prompt_text = get_detailed_prompt() if detailed_prompt else get_step_by_step_prompt()
    
    # Use universal model provider
    from model_providers import generate_fen_universal
    result = generate_fen_universal(processed_img, model, prompt_text)
    
    fen = result['fen']
    text = result['raw_response']
    
    # Try to find "FINAL FEN:" if using step-by-step
    if not detailed_prompt and "FINAL FEN:" in text:
        lines = text.split('\n')
        for line in lines:
            if line.startswith("FINAL FEN:"):
                fen = line.replace("FINAL FEN:", "").strip()
                break
    
    # Validate FEN
    validation = validate_fen(fen)
    
    return {
        'fen': validation['cleaned_fen'] if validation['valid'] else fen,
        'raw_response': text,
        'validation': validation,
        'enhanced': use_enhancement or use_upscaling,
        'provider': result.get('provider'),
        'model': result.get('model')
    }


def generate_fen_with_consensus(
    img_array,
    model="gpt-4o",
    client=None,
    num_attempts=3,
    use_enhancement=True
) -> Dict:
    """
    Generate FEN multiple times and use consensus for better accuracy.
    
    Args:
        img_array: numpy array of the board image
        model: Model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet', 'gemini-2.0-flash')
        client: Deprecated - kept for backward compatibility
        num_attempts: Number of attempts to make (default: 3)
        use_enhancement: Apply image enhancement
    
    Returns:
        Dict with 'fen', 'confidence', 'all_attempts', 'consensus_method'
    """
    print(f"    Generating FEN with consensus ({num_attempts} attempts)...")
    
    attempts = []
    for i in range(num_attempts):
        result = generate_fen_enhanced(
            img_array,
            model=model,
            client=None,  # Not used anymore
            use_enhancement=use_enhancement,
            use_upscaling=(i == 0),  # Only upscale first attempt
            detailed_prompt=(i % 2 == 0)  # Alternate prompt styles
        )
        attempts.append(result['fen'])
        print(f"      Attempt {i+1}: {result['fen'][:60]}...")
    
    # Find most common FEN
    fen_counts = Counter(attempts)
    most_common_fen, count = fen_counts.most_common(1)[0]
    
    confidence = count / num_attempts
    
    return {
        'fen': most_common_fen,
        'confidence': confidence,
        'all_attempts': attempts,
        'consensus_method': 'majority_vote',
        'agreement_count': f"{count}/{num_attempts}",
        'model': model
    }


def generate_fen_best_effort(
    img_array,
    model="gpt-4o",
    client=None,
    strategy="enhanced"
) -> Dict:
    """
    Generate FEN using best available strategy.
    
    Args:
        img_array: numpy array of the board image
        model: OpenAI model to use
        client: OpenAI client instance
        strategy: Strategy to use:
            - "simple": Basic generation (fastest, cheapest)
            - "enhanced": Enhanced with preprocessing (good balance)
            - "consensus": Multiple attempts with voting (most accurate, expensive)
    
    Returns:
        Dict with FEN and metadata
    """
    if client is None:
        client = get_openai_client()
    
    if strategy == "simple":
        # Use original simple method
        from fen_generator import generate_fen_from_image_array
        return generate_fen_from_image_array(img_array, model=model, client=client)
    
    elif strategy == "enhanced":
        return generate_fen_enhanced(
            img_array,
            model=model,
            client=client,
            use_enhancement=True,
            use_upscaling=True,
            detailed_prompt=True
        )
    
    elif strategy == "consensus":
        return generate_fen_with_consensus(
            img_array,
            model=model,
            client=client,
            num_attempts=3,
            use_enhancement=True
        )
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

