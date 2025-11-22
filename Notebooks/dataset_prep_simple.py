import chess.pgn
from pathlib import Path
from tqdm import tqdm
import logging
import argparse
from PIL import Image, ImageDraw, ImageFont
import io

# ======================================================================
# Setup more verbose logging to see all levels of messages
# ======================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_chess_board_image(fen: str, size: int = 350) -> bytes:
    """
    Create a chess board image using PIL only.
    This is the most reliable Windows-compatible solution.
    """
    # Create a new image with white background
    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)
    
    # Calculate square size
    square_size = size // 8
    
    # Draw the chess board squares
    for row in range(8):
        for col in range(8):
            x1 = col * square_size
            y1 = row * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size
            
            # Alternate colors (light and dark squares)
            if (row + col) % 2 == 0:
                color = '#F0D9B5'  # Light squares (beige)
            else:
                color = '#B58863'  # Dark squares (brown)
            
            draw.rectangle([x1, y1, x2, y2], fill=color)
    
    # Parse FEN to get piece positions
    board = chess.Board(fen)
    
    # Better chess piece symbols - using filled Unicode symbols that render better
    piece_symbols = {
        'P': '♟', 'N': '♞', 'B': '♝', 'R': '♜', 'Q': '♛', 'K': '♚',  # White pieces (using filled symbols)
        'p': '♙', 'n': '♘', 'b': '♗', 'r': '♖', 'q': '♕', 'k': '♔'   # Black pieces (using outlined symbols)
    }
    
    # Alternative ASCII pieces with better symbols
    piece_symbols_ascii = {
        'P': 'WP', 'N': 'WN', 'B': 'WB', 'R': 'WR', 'Q': 'WQ', 'K': 'WK',  # White pieces
        'p': 'BP', 'n': 'BN', 'b': 'BB', 'r': 'BR', 'q': 'BQ', 'k': 'BK'   # Black pieces
    }
    
    # Try to load a font that supports Unicode chess symbols
    font = None
    use_unicode = False
    
    try:
        font_size = max(square_size // 2, 24)  # Larger font for better visibility
        
        # Try fonts that support Unicode chess symbols
        unicode_fonts = [
            "seguisym.ttf",  # Segoe UI Symbol (Windows)
            "C:/Windows/Fonts/seguisym.ttf",
            "C:/Windows/Fonts/seguiemj.ttf",  # Segoe UI Emoji
            "C:/Windows/Fonts/segoeuil.ttf",  # Segoe UI Light
            "DejaVuSans.ttf",  # DejaVu Sans
            "arial.ttf",
            "C:/Windows/Fonts/arial.ttf"
        ]
        
        for font_path in unicode_fonts:
            try:
                test_font = ImageFont.truetype(font_path, font_size)
                # Test if font can render chess symbols
                test_img = Image.new('RGB', (10, 10), 'white')
                test_draw = ImageDraw.Draw(test_img)
                test_draw.text((0, 0), '♛', font=test_font, fill='black')
                font = test_font
                use_unicode = True
                break
            except:
                continue
        
        # If Unicode fonts fail, try regular fonts for ASCII
        if font is None:
            for font_path in ["arial.ttf", "C:/Windows/Fonts/arial.ttf", "calibri.ttf", "C:/Windows/Fonts/calibri.ttf"]:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue
        
        # Final fallback
        if font is None:
            font = ImageFont.load_default()
            
    except Exception as e:
        logging.warning(f"Font loading failed: {e}. Using fallback rendering.")
        font = None
    
    # Place pieces on the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Convert square to row, col (chess uses different coordinate system)
            col = chess.square_file(square)
            row = 7 - chess.square_rank(square)  # Flip row for display (rank 1 at bottom)
            
            x = col * square_size + square_size // 2
            y = row * square_size + square_size // 2
            
            # Get piece symbol based on font capabilities
            if use_unicode and font:
                piece_char = piece_symbols.get(piece.symbol(), piece.symbol())
            else:
                piece_char = piece_symbols_ascii.get(piece.symbol(), piece.symbol())
            
            # Determine piece color for better visibility
            if piece.color == chess.WHITE:
                text_color = '#FFFFFF' if use_unicode else '#000000'  # White for Unicode, black for ASCII
                outline_color = '#000000' if use_unicode else '#FFFFFF'
            else:
                text_color = '#000000' if use_unicode else '#FFFFFF'  # Black for Unicode, white for ASCII  
                outline_color = '#FFFFFF' if use_unicode else '#000000'
            
            if font:
                # Get text bounding box for centering
                try:
                    bbox = draw.textbbox((0, 0), piece_char, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Center the text
                    text_x = x - text_width // 2
                    text_y = y - text_height // 2
                    
                    # Draw with better outline for visibility
                    outline_thickness = 2 if use_unicode else 1
                    for adj_x in range(-outline_thickness, outline_thickness + 1):
                        for adj_y in range(-outline_thickness, outline_thickness + 1):
                            if adj_x != 0 or adj_y != 0:
                                draw.text((text_x + adj_x, text_y + adj_y), piece_char, fill=outline_color, font=font)
                    
                    # Draw the main text
                    draw.text((text_x, text_y), piece_char, fill=text_color, font=font)
                except:
                    # Fallback if textbbox is not available (older PIL versions)
                    draw.text((x-15, y-15), piece_char, fill=text_color, font=font)
            else:
                # Fallback without font
                draw.text((x-10, y-10), piece_char, fill=text_color)
    
    # Add coordinates (optional, for better visualization)
    coord_font_size = max(12, square_size // 6)
    try:
        coord_font = ImageFont.truetype("arial.ttf", coord_font_size)
    except:
        coord_font = font if font else None
    
    if coord_font:
        # Add file letters (a-h) at bottom
        for i, letter in enumerate('abcdefgh'):
            x = i * square_size + square_size - 15
            y = size - 15
            draw.text((x, y), letter, fill='black', font=coord_font)
        
        # Add rank numbers (1-8) at left
        for i, number in enumerate('87654321'):  # Reversed because we flip the board
            x = 5
            y = i * square_size + 5
            draw.text((x, y), number, fill='black', font=coord_font)
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()

def pgn_to_examples(pgn_path: str, out_dir: str, max_games: int):
    """
    Processes a PGN file to generate image-text pairs for training a CLIP model on chess positions.
    Uses PIL-only approach for maximum Windows compatibility.
    """
    logging.info("--- Starting Dataset Preparation ---")
    logging.info("Using PIL-only method for Windows compatibility")

    # --- Step 1: Validate PGN file existence ---
    pgn_file = Path(pgn_path)
    if not pgn_file.exists():
        logging.error(f"FATAL ERROR: The input PGN file was not found at the specified path: '{pgn_path}'")
        logging.error("Please check if the file name and path are correct.")
        return # Exit the function immediately

    logging.info(f"Found PGN file: '{pgn_path}'")

    # --- Step 2: Set up output directories ---
    out_dir = Path(out_dir)
    fen_only_dir = out_dir / "fen_only"
    fen_move_dir = out_dir / "fen_move"

    try:
        logging.info(f"Creating output directory: '{fen_only_dir}'")
        (fen_only_dir / "images").mkdir(parents=True, exist_ok=True)
        (fen_only_dir / "texts").mkdir(parents=True, exist_ok=True)

        logging.info(f"Creating output directory: '{fen_move_dir}'")
        (fen_move_dir / "images").mkdir(parents=True, exist_ok=True)
        (fen_move_dir / "texts").mkdir(parents=True, exist_ok=True)
        logging.info("Successfully created/verified output directories.")
    except PermissionError:
        logging.error(f"FATAL ERROR: Permission denied to create directories in '{out_dir}'.")
        logging.error("Please check your write permissions for this location.")
        return
    except Exception as e:
        logging.error(f"FATAL ERROR: An unexpected error occurred while creating directories: {e}")
        return

    # --- Step 3: Process the PGN file ---
    try:
        with open(pgn_path, "r", encoding="utf-8") as f:
            game_count = 0
            total_examples = 0

            # Use tqdm to show progress over the number of games
            with tqdm(total=max_games, desc="Processing games") as pbar:
                while True:
                    # Check if we have reached the desired number of games
                    if game_count >= max_games:
                        logging.info(f"Reached the max_games limit of {max_games}.")
                        break

                    game = chess.pgn.read_game(f)

                    if game is None:
                        logging.warning("Finished reading all games from the PGN file before reaching max_games limit.")
                        break

                    board = game.board()
                    # Iterate through moves and create examples
                    for move in game.mainline_moves():
                        fen = board.fen()
                        move_uci = move.uci()

                        try:
                            # Generate board image using PIL
                            png_bytes = create_chess_board_image(fen, size=350)

                            # Save image for FEN only dataset
                            img_path_fen_only = fen_only_dir / "images" / f"{total_examples}.png"
                            with open(img_path_fen_only, "wb") as img_file:
                                img_file.write(png_bytes)

                            # Save image for FEN + move dataset
                            img_path_fen_move = fen_move_dir / "images" / f"{total_examples}.png"
                            with open(img_path_fen_move, "wb") as img_file:
                                img_file.write(png_bytes)

                        except Exception as e:
                            logging.error(f"Failed to generate image for position at index {total_examples}: {e}")
                            board.push(move) # Still push the move to continue correctly
                            continue

                        # Version A: Save FEN only
                        with open(fen_only_dir / "texts" / f"{total_examples}.txt", "w", encoding="utf-8") as ftxt:
                            ftxt.write(fen)

                        # Version B: Save FEN + Move
                        with open(fen_move_dir / "texts" / f"{total_examples}.txt", "w", encoding="utf-8") as ftxt:
                            ftxt.write(f"{fen} | Next move: {move_uci}")

                        board.push(move)
                        total_examples += 1

                    game_count += 1
                    pbar.update(1) # Update progress bar for each game processed

            logging.info(f"--- Dataset Preparation Complete ---")
            logging.info(f"Processed {game_count} games.")
            logging.info(f"✅ Successfully created {total_examples} image-text pair examples in '{out_dir}'")

    except Exception as e:
        logging.error(f"An unexpected error occurred during PGN processing: {e}", exc_info=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate image-text datasets from a PGN chess file (Windows-compatible PIL version).")
    parser.add_argument("pgn_path", type=str, help="Path to the input PGN file.")
    parser.add_argument("out_dir", type=str, help="The root directory to save the generated datasets.")
    parser.add_argument("--max_games", type=int, default=1000, help="Maximum number of games to process from the PGN file.")
    args = parser.parse_args()

    pgn_to_examples(args.pgn_path, args.out_dir, args.max_games)
