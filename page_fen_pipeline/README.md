# Page-wise FEN Extraction Pipeline

This folder contains a complete pipeline for extracting chess board diagrams from PDF files or images and generating FEN (Forsyth–Edwards Notation) for each board using OpenAI's GPT Vision models.

## 🚀 Quick Start

**New users**: See [GETTING_STARTED.md](GETTING_STARTED.md) for a 5-minute quick start guide!

**Command-line users**: See [CLI_GUIDE.md](CLI_GUIDE.md) for complete CLI documentation.

## Features

- 📄 Extract chess boards from PDF files (page by page)
- 🖼️ Extract chess boards from image files
- 🤖 **NEW:** Multi-model support (OpenAI GPT, Anthropic Claude, Google Gemini)
- 🎯 Multiple accuracy strategies (simple, enhanced, consensus)
- 💾 Save board crops for inspection (organized by page/image)
- 📊 Export results to JSON format
- 📍 Track board positions with bounding boxes
- ✅ Built-in FEN validation and error detection
- ⚙️ Flexible page range selection

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. **(Optional)** Install additional model providers:
```bash
# For Anthropic Claude models
pip install anthropic

# For Google Gemini models
pip install google-generativeai
```

3. Set up your API keys in a `.env` file:
```
# OpenAI (required by default)
OPENAI_API_KEY=sk-your_key_here

# Anthropic (optional - for Claude models)
ANTHROPIC_API_KEY=sk-ant-your_key_here

# Google (optional - for Gemini models)
GOOGLE_API_KEY=your_key_here
```

See [MULTI_MODEL_GUIDE.md](MULTI_MODEL_GUIDE.md) for detailed multi-model setup.

## File Structure

- `cli.py` - **Command-line interface** (recommended for most users)
- `board_extractor.py` - Computer vision module for extracting chess boards from images/PDFs
- `fen_generator.py` - OpenAI GPT integration for FEN generation from board images
- `page_fen_processor.py` - Main orchestrator that combines extraction and FEN generation
- `example_usage.py` - Python API usage examples
- `requirements.txt` - Python dependencies
- `README.md` - This file
- `CLI_GUIDE.md` - Complete CLI documentation
- `QUICK_REFERENCE.md` - Quick reference for page limiting features
- `FOLDER_STRUCTURE.md` - Guide to organized board crops folder structure
- `ACCURACY_IMPROVEMENTS.md` - **NEW:** Guide to improving FEN accuracy
- `MULTI_MODEL_GUIDE.md` - **NEW:** Multi-model provider documentation
- `GPT5_DEBUG_GUIDE.md` - **NEW:** GPT-5 debugging and troubleshooting
- `debug_gpt5.py` - **NEW:** GPT-5 response debugging script
- `CHANGELOG.md` - Version history

## Usage

### Command-Line Interface (Recommended)

The easiest way to use the pipeline is through the command-line interface:

```bash
cd page_fen_pipeline

# Process entire PDF
python cli.py ../book2.pdf

# Process first 5 pages
python cli.py ../book2.pdf --max-pages 5

# Use different AI models
python cli.py ../book2.pdf --max-pages 3 --model claude-3-5-sonnet
python cli.py ../book2.pdf --max-pages 3 --model gemini-2.0-flash

# List all available models
python cli.py --list-models

# Process pages 10-20 and save crops
python cli.py ../book2.pdf --start-page 10 --end-page 20 --save-crops --output-dir crops

# Process image
python cli.py ../chess_board.png
```

See [CLI_GUIDE.md](CLI_GUIDE.md) for complete documentation.

### Python API

You can also use the pipeline programmatically in your Python code:

#### Process a PDF File

```python
from page_fen_processor import process_pdf_to_page_fens, save_results_to_json

# Process PDF and get page-wise FEN results
results = process_pdf_to_page_fens(
    pdf_path="../book2.pdf",
    output_dir="output_crops",  # Optional: where to save board crops
    save_crops=True,            # Optional: save extracted boards as images
    model="gpt-4o",            # OpenAI model to use
    dpi=240                     # DPI for PDF rendering
)

# Save results to JSON
save_results_to_json(results, "output/pdf_fens.json")

# Access results
for page in results:
    print(f"Page {page['page_num']}: {page['boards_count']} boards")
    for board in page['boards']:
        print(f"  Board {board['board_index']}: {board['fen']}")
```

### Process Limited Number of Pages (NEW!)

```python
# Process only the first 5 pages
results = process_pdf_to_page_fens(
    pdf_path="../book2.pdf",
    max_pages=5  # Only process first 5 pages
)

# Process a specific page range (pages 10-20)
results = process_pdf_to_page_fens(
    pdf_path="../book2.pdf",
    start_page=10,  # Start from page 10
    end_page=20     # End at page 20
)

# Process 10 pages starting from page 5 (pages 5-14)
results = process_pdf_to_page_fens(
    pdf_path="../book2.pdf",
    start_page=5,   # Start from page 5
    max_pages=10    # Process 10 pages
)
```

### Process a Single Image

```python
from page_fen_processor import process_image_to_fens, save_results_to_json

# Process image and get FEN results
result = process_image_to_fens(
    image_path="../example_chess_image.png",
    output_dir="output_crops",
    save_crops=True,
    model="gpt-4o"
)

# Save results to JSON
save_results_to_json(result, "output/image_fens.json")

# Access results
print(f"Found {result['boards_count']} boards")
for board in result['boards']:
    print(f"Board {board['board_index']}: {board['fen']}")
```

### Run the Example

```bash
# Navigate to the pipeline folder
cd page_fen_pipeline

# Run the main processor (processes book2.pdf by default)
python page_fen_processor.py
```

## Output Format

### PDF Processing Output
```json
[
  {
    "page_num": 1,
    "boards_count": 2,
    "boards": [
      {
        "board_index": 1,
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "bbox": [100, 200, 300, 300],
        "crop_path": "output_crops/page_001/board_1.png"
      },
      {
        "board_index": 2,
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "bbox": [100, 600, 300, 300],
        "crop_path": "output_crops/page_001/board_2.png"
      }
    ]
  }
]
```

### Image Processing Output
```json
{
  "image_path": "../example_chess_image.png",
  "boards_count": 1,
  "boards": [
    {
      "board_index": 1,
      "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "bbox": [50, 50, 400, 400],
      "crop_path": "output_crops/example_chess_image/board_1.png"
    }
  ]
}
```

## API Reference

### `process_pdf_to_page_fens(pdf_path, output_dir=None, save_crops=False, model="gpt-4o", dpi=240, start_page=1, end_page=None, max_pages=None)`

Process a PDF file and extract FEN notation for all chess boards on each page.

**Parameters:**
- `pdf_path` (str): Path to the PDF file
- `output_dir` (str, optional): Directory to save board crops
- `save_crops` (bool): Whether to save extracted board images
- `model` (str): OpenAI model to use for FEN generation
- `dpi` (int): DPI for PDF rendering
- `start_page` (int): First page to process (1-indexed, default=1)
- `end_page` (int, optional): Last page to process (1-indexed, None=all pages)
- `max_pages` (int, optional): Maximum number of pages to process (None=no limit). If both `end_page` and `max_pages` are set, `max_pages` takes precedence

**Returns:** List of page results with boards and FENs

### `process_image_to_fens(image_path, output_dir=None, save_crops=False, model="gpt-4o")`

Process a single image file and extract FEN notation for all chess boards found.

**Parameters:**
- `image_path` (str): Path to the image file
- `output_dir` (str, optional): Directory to save board crops
- `save_crops` (bool): Whether to save extracted board images
- `model` (str): OpenAI model to use for FEN generation

**Returns:** Dict with image results containing boards and FENs

### `save_results_to_json(results, output_path)`

Save processing results to a JSON file.

**Parameters:**
- `results`: Results from `process_pdf_to_page_fens` or `process_image_to_fens`
- `output_path` (str): Path where to save the JSON file

## Notes

- The board extraction uses computer vision techniques with a checker pattern correlation to identify chess boards
- FEN generation uses OpenAI's GPT-4o model by default (requires API key)
- You can change the model to other vision-capable models (e.g., "gpt-4.1-mini", "gpt-5-mini")
- Board crops are organized in page-specific folders:
  - PDFs: `output_dir/page_XXX/board_Y.png`
  - Images: `output_dir/imagename/board_Y.png`
- The pipeline processes boards in order (top-to-bottom, left-to-right)

## Improving FEN Accuracy 🎯

Several strategies are available to improve FEN generation accuracy:

### 1. Compare Strategies
Test different approaches on a single board:
```bash
cd page_fen_pipeline
python compare_fen_strategies.py path/to/board.png
```

### 2. Use Enhanced Generation
For better accuracy with image preprocessing:
```python
from fen_generator_enhanced import generate_fen_best_effort

# Enhanced strategy (recommended)
result = generate_fen_best_effort(
    board_image,
    strategy="enhanced"  # Applies image enhancement + better prompts
)

# Consensus strategy (highest accuracy)
result = generate_fen_best_effort(
    board_image,
    strategy="consensus"  # Multiple attempts with voting (3x cost)
)
```

### 3. Adjust Processing Parameters
- **Increase DPI**: `--dpi 300` or `--dpi 400` for better quality
- **Use better model**: `--model gpt-4o` for best results
- **Save crops for review**: `--save-crops` to manually verify

See [ACCURACY_IMPROVEMENTS.md](ACCURACY_IMPROVEMENTS.md) for detailed information.

## Troubleshooting

**Issue:** "OPENAI_API_KEY not found"
- **Solution:** Create a `.env` file in the parent directory with your OpenAI API key

**Issue:** No boards detected
- **Solution:** Try adjusting the `dpi` parameter (higher values = better quality but slower processing)

**Issue:** Incorrect FEN returned
- **Solution:** 
  1. Check saved board crops to verify image quality
  2. Try higher DPI (300-400)
  3. Use enhanced or consensus strategy
  4. See [ACCURACY_IMPROVEMENTS.md](ACCURACY_IMPROVEMENTS.md)

## License

This code combines logic from the parent directory's `crop_pipeline.py` and `main_2.py` files.

