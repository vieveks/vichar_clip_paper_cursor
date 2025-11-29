"""
FEN Strategy Comparison Tool
Compare different FEN generation strategies side-by-side.
"""

import sys
import cv2
from pathlib import Path
from fen_generator import generate_fen_from_image_array as generate_simple
from fen_generator_enhanced import (
    generate_fen_enhanced,
    generate_fen_with_consensus,
    get_openai_client
)


def load_image(image_path):
    """Load image from file."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img


def print_result(strategy_name, result, timing=None):
    """Print results in a formatted way."""
    print(f"\n{'='*70}")
    print(f" {strategy_name.upper()}")
    print(f"{'='*70}")
    
    if timing:
        print(f"⏱️  Time: {timing:.2f}s")
    
    print(f"\n📋 FEN: {result.get('fen', 'N/A')}")
    
    if 'validation' in result:
        val = result['validation']
        if val['valid']:
            print(f"✅ Validation: PASSED")
        else:
            print(f"❌ Validation: FAILED")
            print(f"   Errors:")
            for error in val['errors']:
                print(f"   - {error}")
    
    if 'confidence' in result:
        conf_pct = result['confidence'] * 100
        print(f"🎯 Confidence: {conf_pct:.0f}% ({result.get('agreement_count', 'N/A')} agreement)")
        
        if 'all_attempts' in result:
            print(f"\n📊 All attempts:")
            for i, attempt in enumerate(result['all_attempts'], 1):
                marker = "✓" if attempt == result['fen'] else " "
                print(f"   {marker} {i}. {attempt[:70]}...")
    
    if 'enhanced' in result and result['enhanced']:
        print(f"🎨 Image preprocessing: Applied")
    
    # Show partial raw response if available
    if 'raw_response' in result and result['raw_response']:
        response = result['raw_response']
        if len(response) > 150:
            print(f"\n💬 Response excerpt: {response[:150]}...")
        else:
            print(f"\n💬 Response: {response}")


def compare_strategies(image_path, model="gpt-4o"):
    """Compare all FEN generation strategies on a single image."""
    print(f"\n{'='*70}")
    print(f" FEN STRATEGY COMPARISON")
    print(f"{'='*70}")
    print(f"\n📁 Image: {image_path}")
    print(f"🤖 Model: {model}")
    
    # Load image
    print(f"\n📸 Loading image...")
    img = load_image(image_path)
    print(f"   Size: {img.shape[1]}x{img.shape[0]}px")
    
    # Initialize client
    client = get_openai_client()
    
    # Strategy 1: Simple (baseline)
    print(f"\n\n🔄 Running SIMPLE strategy...")
    import time
    start = time.time()
    try:
        result_simple = generate_simple(img, model=model, client=client)
        time_simple = time.time() - start
        print_result("Simple (Baseline)", result_simple, time_simple)
    except Exception as e:
        print(f"❌ Error: {e}")
        result_simple = None
        time_simple = 0
    
    # Strategy 2: Enhanced
    print(f"\n\n🔄 Running ENHANCED strategy...")
    start = time.time()
    try:
        result_enhanced = generate_fen_enhanced(
            img,
            model=model,
            client=client,
            use_enhancement=True,
            use_upscaling=True,
            detailed_prompt=True
        )
        time_enhanced = time.time() - start
        print_result("Enhanced", result_enhanced, time_enhanced)
    except Exception as e:
        print(f"❌ Error: {e}")
        result_enhanced = None
        time_enhanced = 0
    
    # Strategy 3: Consensus
    print(f"\n\n🔄 Running CONSENSUS strategy (3 attempts)...")
    start = time.time()
    try:
        result_consensus = generate_fen_with_consensus(
            img,
            model=model,
            client=client,
            num_attempts=3,
            use_enhancement=True
        )
        time_consensus = time.time() - start
        print_result("Consensus (3x)", result_consensus, time_consensus)
    except Exception as e:
        print(f"❌ Error: {e}")
        result_consensus = None
        time_consensus = 0
    
    # Summary comparison
    print(f"\n\n{'='*70}")
    print(f" SUMMARY COMPARISON")
    print(f"{'='*70}")
    
    results = [
        ("Simple", result_simple, time_simple),
        ("Enhanced", result_enhanced, time_enhanced),
        ("Consensus", result_consensus, time_consensus)
    ]
    
    print(f"\n{'Strategy':<15} {'Time':<10} {'Valid':<10} {'FEN':<40}")
    print(f"{'-'*70}")
    
    for name, result, timing in results:
        if result:
            fen = result.get('fen', 'ERROR')[:35]
            valid = "✅ Yes" if result.get('validation', {}).get('valid', False) else "❌ No"
            time_str = f"{timing:.2f}s"
        else:
            fen = "ERROR"
            valid = "N/A"
            time_str = "N/A"
        
        print(f"{name:<15} {time_str:<10} {valid:<10} {fen}...")
    
    # Check for consensus
    print(f"\n📊 Analysis:")
    
    fens = []
    for name, result, _ in results:
        if result and 'fen' in result:
            fens.append(result['fen'])
    
    if len(fens) >= 2:
        if fens[0] == fens[1] == (fens[2] if len(fens) > 2 else fens[1]):
            print(f"   ✅ All strategies agree!")
        elif fens[1] == (fens[2] if len(fens) > 2 else fens[1]):
            print(f"   ⚠️  Enhanced and Consensus agree, Simple differs")
        else:
            print(f"   ⚠️  Strategies produced different results")
            print(f"   💡 Recommendation: Use Consensus result or manual review")
    
    # Cost estimate
    print(f"\n💰 Estimated API Cost (with gpt-4o):")
    print(f"   Simple:    ~$0.015")
    print(f"   Enhanced:  ~$0.015")
    print(f"   Consensus: ~$0.045 (3x calls)")
    
    print(f"\n{'='*70}\n")
    
    return {
        'simple': result_simple,
        'enhanced': result_enhanced,
        'consensus': result_consensus
    }


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python compare_fen_strategies.py <image_path> [model]")
        print("\nExample:")
        print("  python compare_fen_strategies.py board.png")
        print("  python compare_fen_strategies.py board.png gpt-4.1-mini")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4o"
    
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    
    try:
        compare_strategies(image_path, model)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

