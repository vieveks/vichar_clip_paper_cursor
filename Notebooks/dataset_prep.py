import chess.pgn
import chess.svg
import cairosvg
from pathlib import Path
from tqdm import tqdm
import logging
import argparse

# ======================================================================
# Setup more verbose logging to see all levels of messages
# ======================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def pgn_to_examples(pgn_path: str, out_dir: str, max_games: int):
    """
    Processes a PGN file to generate image-text pairs for training a CLIP model on chess positions.
    """
    logging.info("--- Starting Dataset Preparation ---")

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

                        # Generate and save the board image
                        svg_data = chess.svg.board(board=board, size=350)
                        try:
                            png_bytes = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))

                            # Save image for FEN only dataset
                            img_path_fen_only = fen_only_dir / "images" / f"{total_examples}.png"
                            with open(img_path_fen_only, "wb") as img_file:
                                img_file.write(png_bytes)

                            # Save image for FEN + move dataset
                            img_path_fen_move = fen_move_dir / "images" / f"{total_examples}.png"
                            with open(img_path_fen_move, "wb") as img_file:
                                img_file.write(png_bytes)

                        except Exception as e:
                            logging.error(f"Failed to convert SVG to PNG for position at index {total_examples}: {e}")
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
    args = parser.parse_args()

    pgn_to_examples(args.pgn_path, args.out_dir, args.max_games)
