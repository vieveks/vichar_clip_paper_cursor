#!/usr/bin/env python3
"""
Test script to verify the improved chess piece rendering.
"""

from dataset_prep_simple import create_chess_board_image

def test_improved_rendering():
    """Test the improved chess piece rendering."""
    print("Testing improved chess piece rendering...")
    
    # Test with starting position
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print(f"Generating improved image for starting position...")
    
    try:
        # Generate image with improved rendering
        png_bytes = create_chess_board_image(starting_fen, size=400)  # Slightly larger for better visibility
        print(f"✅ Successfully generated {len(png_bytes)} bytes of PNG data")
        
        # Save to file for visual inspection
        with open("improved_starting_position.png", "wb") as f:
            f.write(png_bytes)
        print("✅ Saved improved image to 'improved_starting_position.png'")
        
        # Test a mid-game position with all piece types
        midgame_fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4"
        print(f"Generating image for mid-game position...")
        
        png_bytes_2 = create_chess_board_image(midgame_fen, size=400)
        print(f"✅ Successfully generated {len(png_bytes_2)} bytes of PNG data")
        
        with open("improved_midgame_position.png", "wb") as f:
            f.write(png_bytes_2)
        print("✅ Saved improved image to 'improved_midgame_position.png'")
        
        print("\n🎉 Improved rendering test completed!")
        print("Check the generated PNG files to see if the chess pieces are now visible and distinct.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during improved rendering: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_improved_rendering()
    if success:
        print("\n✅ Improved rendering test completed successfully!")
    else:
        print("\n❌ Improved rendering test failed!")
