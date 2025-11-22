import requests
import zstandard as zstd
import os
from tqdm import tqdm
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_lichess_pgn(year: int, month: int, save_path: str = "."):
    """
    Downloads and decompresses a monthly PGN file from the Lichess Open Database.
    This version handles the modern .zst compression format.

    Args:
        year (int): The year of the games to download (e.g., 2022).
        month (int): The month of the games to download (1-12).
        save_path (str): The directory where the final PGN file will be saved.
    """
    # Format the month to be two digits (e.g., 7 -> 07)
    month_str = f"{month:02d}"

    # Construct the URL for the .zst compressed PGN file
    url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{year}-{month_str}.pgn.zst"

    compressed_file_name = os.path.join(save_path, f"lichess_games_{year}-{month_str}.pgn.zst")
    decompressed_file_name = os.path.join(save_path, f"lichess_games_{year}-{month_str}.pgn")

    logging.info(f"Target URL: {url}")

    try:
        # --- Step 1: Download the compressed file ---
        logging.info(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes (like 404)

        total_size = int(response.headers.get('content-length', 0))

        with open(compressed_file_name, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(compressed_file_name)}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        logging.info(f"Successfully downloaded to {compressed_file_name}")

        # --- Step 2: Decompress the .zst file ---
        logging.info(f"Decompressing {compressed_file_name}...")

        dctx = zstd.ZstdDecompressor()
        with open(compressed_file_name, 'rb') as in_file, open(decompressed_file_name, 'wb') as out_file:
            # Use a stream reader for efficient, chunked decompression
            reader = dctx.stream_reader(in_file)
            # Get the file size for the progress bar if possible (requires seeking)
            in_file.seek(0, os.SEEK_END)
            file_size = in_file.tell()
            in_file.seek(0)

            with tqdm(total=file_size, desc=f"Decompressing {os.path.basename(decompressed_file_name)}", unit='iB', unit_scale=True) as pbar:
                while True:
                    chunk = reader.read(16384) # Read in 16KB chunks
                    if not chunk:
                        break
                    out_file.write(chunk)
                    pbar.update(in_file.tell() - pbar.n) # Update progress based on input file read

        logging.info(f"✅ Successfully decompressed to {decompressed_file_name}")

        # --- Step 3: Clean up the compressed file ---
        os.remove(compressed_file_name)
        logging.info(f"Removed temporary file: {compressed_file_name}")

    except requests.exceptions.HTTPError as e:
        logging.error(f"Error: Could not download the file. Status code: {e.response.status_code}")
        logging.error("Please check that the year and month are valid on the Lichess Database page: https://database.lichess.org/")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # --- Example Usage ---
    # Let's download a recent, smaller file for testing, e.g., from the current year.
    # We'll use January 2024 as an example.
    target_year = 2013
    target_month = 1

    output_directory = "." # Save to the current directory

    # Create the directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    download_lichess_pgn(target_year, target_month, output_directory)
