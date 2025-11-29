# Command-Line Interface Guide

The `cli.py` script provides a simple command-line interface for extracting FEN notation from chess boards in PDF files or images.

## Basic Usage

```bash
cd page_fen_pipeline
python cli.py <input_file> [options]
```

## Quick Examples

### Process Entire PDF
```bash
python cli.py ../book2.pdf
```

### Process First 5 Pages
```bash
python cli.py ../book2.pdf --max-pages 5
```

### Process Pages 10-20
```bash
python cli.py ../book2.pdf --start-page 10 --end-page 20
```

### Process 10 Pages Starting from Page 5
```bash
python cli.py ../book2.pdf --start-page 5 --max-pages 10
```

### Process and Save Board Crops
```bash
python cli.py ../book2.pdf --max-pages 3 --output-dir crops --save-crops
```

### Process an Image File
```bash
python cli.py ../example_chess_image.png --output-json image_result.json
```

### Use Different Model
```bash
python cli.py ../book2.pdf --max-pages 5 --model gpt-4.1-mini
```

## All Command-Line Options

### Required Arguments

| Argument | Description |
|----------|-------------|
| `input_file` | Path to the PDF file or image to process |

### Page Control Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--start-page N` | int | 1 | First page to process (1-indexed) |
| `--end-page N` | int | None | Last page to process (1-indexed) |
| `--max-pages N` | int | None | Maximum number of pages to process |

### Output Options

| Option | Description |
|--------|-------------|
| `--output-dir DIR` | Directory to save board crop images |
| `--save-crops` | Save extracted board images to output directory |
| `--output-json FILE` | Path to save JSON results (default: output/results.json) |
| `--no-json` | Skip saving JSON output |

### Model Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model MODEL` | str | gpt-4o | OpenAI model to use |
| `--dpi N` | int | 240 | DPI for PDF rendering |

Available models:
- `gpt-4o` (default, best quality)
- `gpt-4.1-mini` (faster, cheaper)
- `gpt-4.1-nano` (fastest, cheapest)
- `gpt-5` (if you have access)
- `gpt-5-mini` (if you have access)
- `gpt-5-nano` (if you have access)

### Other Options

| Option | Description |
|--------|-------------|
| `--verbose` | Print detailed information during processing |

## Common Use Cases

### 1. Quick Test (First Page Only)
```bash
python cli.py ../book2.pdf --max-pages 1
```

### 2. Process Specific Chapter (Pages 50-75)
```bash
python cli.py ../book2.pdf --start-page 50 --end-page 75 --save-crops --output-dir chapter3
```

This will save boards in organized folders: `chapter3/page_050/board_1.png`, `chapter3/page_051/board_1.png`, etc.

### 3. Batch Processing (First 20 Pages)
```bash
python cli.py ../book2.pdf --max-pages 20 --output-json batch1.json
```

### 4. High-Quality Processing
```bash
python cli.py ../book2.pdf --dpi 300 --model gpt-4o --save-crops --output-dir high_quality
```

### 5. Fast Processing (Lower Quality)
```bash
python cli.py ../book2.pdf --dpi 150 --model gpt-4.1-mini --max-pages 10
```

### 6. Process Multiple PDFs in Sequence
```bash
# Batch 1
python cli.py ../book1.pdf --max-pages 50 --output-json book1_batch1.json

# Batch 2
python cli.py ../book1.pdf --start-page 51 --max-pages 50 --output-json book1_batch2.json

# Different book
python cli.py ../book2.pdf --output-json book2_all.json
```

### 7. Verbose Mode for Debugging
```bash
python cli.py ../book2.pdf --max-pages 1 --verbose
```

## Output

### Console Output

The script prints:
- Progress information for each page
- Number of boards found per page
- FEN notation for each board
- Summary of results

Example output:
```
📄 Processing PDF: ../book2.pdf

PDF has 150 pages. Processing pages 1 to 5

============================================================
Processing Page 1 - 2 board(s) found
============================================================

  Board 1/2 on Page 1:
    Generating FEN...
    FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

  Board 2/2 on Page 1:
    Generating FEN...
    FEN: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3

======================================================================
 SUMMARY
======================================================================

✅ Processed 5 pages
✅ Found 12 chess boards total

  Page 1: 2 board(s)
    Board 1: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
    Board 2: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R...

💾 Results saved to: output/results.json

======================================================================
```

### JSON Output

Results are saved to a JSON file with the following structure:

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
        "crop_path": "crops/page_001_board_1.png"
      }
    ]
  }
]
```

## Error Handling

The script handles various error conditions:

### File Not Found
```bash
python cli.py missing.pdf
# ❌ Error: File not found: missing.pdf
```

### Unsupported File Type
```bash
python cli.py document.docx
# ❌ Error: Unsupported file type: .docx
# Supported types: .pdf, .png, .jpg, .jpeg, .bmp, .tiff, .webp
```

### Missing Output Directory with Save Crops
```bash
python cli.py book.pdf --save-crops
# ❌ Error: --output-dir is required when using --save-crops
```

### Interrupted Processing
Press Ctrl+C during processing:
```
⚠️  Processing interrupted by user
```

## Tips

1. **Start small**: Test with `--max-pages 1` first to ensure everything works
2. **Use verbose mode**: Add `--verbose` when troubleshooting
3. **Save crops for review**: Use `--save-crops` to visually verify board detection
4. **Adjust DPI**: Lower DPI (150-200) for faster processing, higher (300+) for better quality
5. **Choose the right model**: Use `gpt-4o` for best results, `gpt-4.1-mini` for faster/cheaper processing
6. **Process in batches**: For large PDFs, process in smaller batches to avoid timeouts

## Integration with Scripts

You can also call the CLI from other scripts:

```bash
# In a bash script
for book in book1.pdf book2.pdf book3.pdf; do
    python cli.py "../$book" --max-pages 10 --output-json "results_${book%.pdf}.json"
done
```

```python
# In Python using subprocess
import subprocess

result = subprocess.run([
    'python', 'cli.py', 
    '../book2.pdf',
    '--max-pages', '5',
    '--output-json', 'output.json'
], capture_output=True, text=True)

print(result.stdout)
```

## Help

View all options:
```bash
python cli.py --help
```

Or see the examples in the help:
```bash
python cli.py -h
```

