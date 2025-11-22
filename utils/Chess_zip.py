import zipfile
import chess.pgn
import chess.svg
import cairosvg
import os
import csv

# ---------- SETTINGS ----------
ZIP_FILE = "Anand.zip"
OUTPUT_DIR = "anand_fen_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 1. Extract PGN from ZIP ----------
with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
    zip_ref.extractall("anand_pgns")

# Find PGN file inside extracted folder
pgn_path = [f for f in os.listdir("anand_pgns") if f.endswith(".pgn")][0]
pgn_file = open(os.path.join("anand_pgns", pgn_path), encoding="utf-8")

# ---------- 2. Parse First 10 Games ----------
game_count = 0
fen_image_pairs = []

while game_count < 10:
    game = chess.pgn.read_game(pgn_file)
    if game is None:
        break  # No more games
    board = game.board()
    move_number = 0
    for move in game.mainline_moves():
        board.push(move)
        move_number += 1
        fen = board.fen()

        # Create SVG from board
        svg_data = chess.svg.board(board=board, size=350)
        
        # Save PNG
        img_filename = f"game{game_count}_move{move_number}.png"
        img_path = os.path.join(OUTPUT_DIR, img_filename)
        cairosvg.svg2png(bytestring=svg_data.encode("utf-8"), write_to=img_path)

        # Store FEN + image path
        fen_image_pairs.append((fen, img_path))
    game_count += 1

print(f"Processed {game_count} games.")
print(f"Generated {len(fen_image_pairs)} (FEN, image) pairs.")

# ---------- 3. Save pairs to CSV ----------
with open("fen_image_pairs.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["fen", "image_path"])
    writer.writerows(fen_image_pairs)

print("Pipeline complete. Data saved to fen_image_pairs.csv")
