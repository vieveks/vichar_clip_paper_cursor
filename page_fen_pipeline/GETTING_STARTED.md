# Getting Started - 5 Minute Guide

Welcome! This guide will help you start extracting FEN notation from your chess PDFs or images in just a few minutes.

## Step 1: Install Dependencies (One Time)

```bash
cd page_fen_pipeline
pip install -r requirements.txt
```

## Step 2: Set Up API Key (One Time)

Create a `.env` file in the **parent directory** (one level up from `page_fen_pipeline`):

```bash
# Navigate to parent directory
cd ..

# Create .env file (Windows PowerShell)
echo "OPENAI_API_KEY=your_api_key_here" > .env

# OR on Linux/Mac
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

Replace `your_api_key_here` with your actual OpenAI API key.

## Step 3: Run Your First Extraction

Navigate back to the pipeline folder:

```bash
cd page_fen_pipeline
```

### Process a PDF (First 3 Pages)

```bash
python cli.py ../book2.pdf --max-pages 3
```

### Process an Image

```bash
python cli.py ../chess_board.png
```

## Common Commands

### 1. Test with One Page First

```bash
python cli.py ../book2.pdf --max-pages 1
```

This is the fastest way to test everything is working!

### 2. Process Specific Pages

```bash
# Process pages 5 through 10
python cli.py ../book2.pdf --start-page 5 --end-page 10
```

### 3. Save Board Images

```bash
python cli.py ../book2.pdf --max-pages 5 --save-crops --output-dir my_boards
```

This saves the extracted chess board images to the `my_boards` folder, organized by page:
```
my_boards/
├── page_001/
│   ├── board_1.png
│   └── board_2.png
├── page_002/
│   └── board_1.png
...
```

See [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) for details on folder organization.

### 4. Process Entire PDF

```bash
python cli.py ../book2.pdf
```

⚠️ This processes ALL pages - may take a while and cost more API credits!

## View All Options

```bash
python cli.py --help
```

## What You'll Get

After processing, you'll receive:

1. **Console Output**: Progress and FEN notation printed in the terminal
2. **JSON File**: Structured results saved to `output/results.json` by default
3. **Board Images** (optional): Extracted chess board images if you use `--save-crops`

### Example Output

```
📄 Processing PDF: ../book2.pdf

PDF has 150 pages. Processing pages 1 to 3

============================================================
Processing Page 1 - 2 board(s) found
============================================================

  Board 1/2 on Page 1:
    Generating FEN...
    FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

======================================================================
 SUMMARY
======================================================================

✅ Processed 3 pages
✅ Found 6 chess boards total

💾 Results saved to: output/results.json
```

## Troubleshooting

### "OPENAI_API_KEY not found"

Make sure:
- The `.env` file is in the **parent directory** (not in `page_fen_pipeline`)
- The file contains: `OPENAI_API_KEY=sk-...`
- No extra spaces or quotes

### "File not found"

Make sure you're running from inside the `page_fen_pipeline` folder and use `../` to reference files in the parent directory:

```bash
cd page_fen_pipeline
python cli.py ../your_file.pdf
```

### No Boards Detected

Try increasing the DPI:

```bash
python cli.py ../book2.pdf --max-pages 1 --dpi 300
```

## Next Steps

Once you're comfortable with the basics:

- **Read the full documentation**: [CLI_GUIDE.md](CLI_GUIDE.md)
- **Learn about page limiting**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Use Python API**: [example_usage.py](example_usage.py)

## Quick Reference Card

```bash
# Test (1 page)
python cli.py ../book.pdf --max-pages 1

# First N pages
python cli.py ../book.pdf --max-pages 5

# Page range
python cli.py ../book.pdf --start-page 10 --end-page 20

# With board images
python cli.py ../book.pdf --max-pages 3 --save-crops --output-dir boards

# Process image
python cli.py ../image.png

# Fast/cheap model
python cli.py ../book.pdf --max-pages 5 --model gpt-4.1-mini

# High quality
python cli.py ../book.pdf --max-pages 3 --dpi 300

# Verbose output
python cli.py ../book.pdf --max-pages 1 --verbose
```

## Cost Considerations

- Each board image analyzed costs API credits
- Using `--max-pages` helps control costs during testing
- Consider using `gpt-4.1-mini` for cheaper processing
- Test with 1-3 pages first before processing entire books

## Support

For more detailed information:
- Full CLI documentation: [CLI_GUIDE.md](CLI_GUIDE.md)
- Main README: [README.md](README.md)
- Page limiting features: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

