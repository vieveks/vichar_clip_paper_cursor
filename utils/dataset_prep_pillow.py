import chess.pgn
import chess.svg
from pathlib import Path
from tqdm import tqdm
import logging
import argparse
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# ======================================================================
# Setup more verbose logging to see all levels of messages
# ======================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def svg_to_png_selenium(svg_content: str, size: int = 350) -> bytes:
    """
    Convert SVG to PNG using Selenium and Chrome browser.
    This is a Windows-compatible alternative to cairosvg.
    """
    # Setup Chrome options for headless operation
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"--window-size={size},{size}")
    
    # Create HTML content with the SVG
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 0; }}
            svg {{ display: block; }}
        </style>
    </head>
    <body>
        {svg_content}
    </body>
    </html>
    """
    
    try:
        # Setup Chrome driver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Load the HTML content
        driver.get("data:text/html;charset=utf-8," + html_content)
        time.sleep(0.5)  # Wait for rendering
        
        # Take screenshot
        png_bytes = driver.get_screenshot_as_png()
        driver.quit()
        
        return png_bytes
        
    except Exception as e:
        logging.error(f"Selenium SVG conversion failed: {e}")
        raise

def svg_to_png_wand(svg_content: str, size: int = 350) -> bytes:
    """
    Convert SVG to PNG using Wand (ImageMagick).
    Alternative method if Selenium is not available.
    """
    try:
        from wand.image import Image as WandImage
        from wand.color import Color
        
        with WandImage() as img:
            img.format = 'svg'
            img.read(blob=svg_content.encode('utf-8'))
            img.format = 'png'
            img.resize(size, size)
            return img.make_blob()
            
    except ImportError:
        logging.error("Wand library not available. Please install: pip install Wand")
        raise
    except Exception as e:
        logging.error(f"Wand SVG conversion failed: {e}")
        raise

def create_simple_chess_board_image(fen: str, size: int = 350) -> bytes:
    """
    Create a simple chess board image using PIL only.
    This is the most reliable Windows-compatible solution.
    """
    # Create a new image with white background
    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)
    
    # Calculate square size
    square_size = size // 8
    
    # Draw the chess board
    for row in range(8):
        for col in range(8):
            x1 = col * square_size
            y1 = row * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size
            
            # Alternate colors (dark squares on a1, c1, e1, g1, b2, d2, f2, h2, etc.)
            if (row + col) % 2 == 0:
                color = '#F0D9B5'  # Light squares
            else:
                color = '#B58863'  # Dark squares
            
            draw.rectangle([x1, y1, x2, y2], fill=color)
    
    # Parse FEN to get piece positions
    board = chess.Board(fen)
    
    # Unicode chess pieces
    piece_symbols = {
        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',  # White pieces
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'   # Black pieces
    }
    
    # Try to load a font (fallback to default if not available)
    try:
        font_size = square_size // 2
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Place pieces on the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Convert square to row, col (chess uses different coordinate system)
            col = chess.square_file(square)
            row = 7 - chess.square_rank(square)  # Flip row for display
            
            x = col * square_size + square_size // 2
            y = row * square_size + square_size // 2
            
            piece_char = piece_symbols.get(piece.symbol(), piece.symbol())
            
            if font:
                # Get text bounding box for centering
                bbox = draw.textbbox((0, 0), piece_char, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Center the text
                text_x = x - text_width // 2
                text_y = y - text_height // 2
                
                draw.text((text_x, text_y), piece_char, fill='black', font=font)
            else:
                # Fallback without font
                draw.text((x-10, y-10), piece_char, fill='black')
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()

def pgn_to_examples(pgn_path: str, out_dir: str, max_games: int, conversion_method: str = "pil"):
    """
    Processes a PGN file to generate image-text pairs for training a CLIP model on chess positions.
    
    Args:
        pgn_path: Path to the input PGN file
        out_dir: Root directory to save generated datasets
        max_games: Maximum number of games to process
        conversion_method: Method for image generation ("pil", "selenium", or "wand")
    """
    logging.info("--- Starting Dataset Preparation ---")
    logging.info(f"Using {conversion_method} method for image generation")

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
                            # Generate board image using selected method
                            if conversion_method == "pil":
                                png_bytes = create_simple_chess_board_image(fen, size=350)
                            elif conversion_method == "selenium":
                                svg_data = chess.svg.board(board=board, size=350)
                                png_bytes = svg_to_png_selenium(svg_data, size=350)
                            elif conversion_method == "wand":
                                svg_data = chess.svg.board(board=board, size=350)
                                png_bytes = svg_to_png_wand(svg_data, size=350)
                            else:
                                raise ValueError(f"Unknown conversion method: {conversion_method}")

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
    parser = argparse.ArgumentParser(description="Generate image-text datasets from a PGN chess file.")
    parser.add_argument("pgn_path", type=str, help="Path to the input PGN file.")
    parser.add_argument("out_dir", type=str, help="The root directory to save the generated datasets.")
    parser.add_argument("--max_games", type=int, default=1000, help="Maximum number of games to process from the PGN file.")
    parser.add_argument("--method", type=str, choices=["pil", "selenium", "wand"], default="pil", 
                       help="Method for image generation: 'pil' (simple PIL-based), 'selenium' (browser-based), or 'wand' (ImageMagick)")
    args = parser.parse_args()

    pgn_to_examples(args.pgn_path, args.out_dir, args.max_games, args.method)
