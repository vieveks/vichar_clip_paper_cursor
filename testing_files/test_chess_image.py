#!/usr/bin/env python3
"""
Test script to verify the PIL-based chess board image generation works on Windows.
"""

import chess
from dataset_prep_simple import create_chess_board_image
from PIL import Image
import io

def test_chess_image_generation():
    """Test the chess board image generation function."""
    print("Testing chess board image generation...")
    
    # Test with starting position
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print(f"Generating image for starting position: {starting_fen}")
    
    try:
        # Generate image
        png_bytes = create_chess_board_image(starting_fen, size=350)
        print(f"✅ Successfully generated {len(png_bytes)} bytes of PNG data")
        
        # Save to file for visual inspection
        with open("test_starting_position.png", "wb") as f:
            f.write(png_bytes)
        print("✅ Saved test image to 'test_starting_position.png'")
        
        # Test another position (after e4)
        after_e4_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        print(f"Generating image for position after 1.e4: {after_e4_fen}")
        
        png_bytes_2 = create_chess_board_image(after_e4_fen, size=350)
        print(f"✅ Successfully generated {len(png_bytes_2)} bytes of PNG data")
        
        with open("test_after_e4.png", "wb") as f:
            f.write(png_bytes_2)
        print("✅ Saved test image to 'test_after_e4.png'")
        
        # Test a complex position
        complex_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4"
        print(f"Generating image for complex position: {complex_fen}")
        
        png_bytes_3 = create_chess_board_image(complex_fen, size=350)
        print(f"✅ Successfully generated {len(png_bytes_3)} bytes of PNG data")
        
        with open("test_complex_position.png", "wb") as f:
            f.write(png_bytes_3)
        print("✅ Saved test image to 'test_complex_position.png'")
        
        print("\n🎉 All tests passed! The PIL-based chess board generation is working correctly.")
        print("You can now use 'dataset_prep_simple.py' to generate your dataset without Cairo DLL issues.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during image generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chess_image_generation()
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
