# Quick Reference - Page Limiting Features

## Overview

You can now control which pages to process from your PDF files using three parameters:
- `start_page` - Where to start processing (default: 1)
- `end_page` - Where to stop processing (default: None = all pages)
- `max_pages` - How many pages to process (default: None = no limit)

## Examples

### 1. Process Only First N Pages

```python
# Process only the first 3 pages
results = process_pdf_to_page_fens(
    pdf_path="book.pdf",
    max_pages=3
)
```

### 2. Process Specific Page Range

```python
# Process pages 5 through 10
results = process_pdf_to_page_fens(
    pdf_path="book.pdf",
    start_page=5,
    end_page=10
)
```

### 3. Start from Page and Limit Count

```python
# Process 5 pages starting from page 10 (pages 10-14)
results = process_pdf_to_page_fens(
    pdf_path="book.pdf",
    start_page=10,
    max_pages=5
)
```

### 4. Process Single Page

```python
# Process only page 7
results = process_pdf_to_page_fens(
    pdf_path="book.pdf",
    start_page=7,
    end_page=7
)
# OR
results = process_pdf_to_page_fens(
    pdf_path="book.pdf",
    start_page=7,
    max_pages=1
)
```

### 5. Process All Pages (Default Behavior)

```python
# Process all pages (same as before)
results = process_pdf_to_page_fens(
    pdf_path="book.pdf"
)
```

## Parameter Priority

If both `end_page` and `max_pages` are specified, `max_pages` takes precedence:

```python
# This will process 3 pages starting from page 1 (pages 1-3)
# end_page=10 is ignored because max_pages is set
results = process_pdf_to_page_fens(
    pdf_path="book.pdf",
    end_page=10,    # Ignored
    max_pages=3     # This takes priority
)
```

## Use Cases

### Testing/Development
```python
# Quick test on first page only
results = process_pdf_to_page_fens("book.pdf", max_pages=1)
```

### Processing Large PDFs in Batches
```python
# Batch 1: Process pages 1-50
batch1 = process_pdf_to_page_fens("book.pdf", start_page=1, max_pages=50)

# Batch 2: Process pages 51-100
batch2 = process_pdf_to_page_fens("book.pdf", start_page=51, max_pages=50)

# Batch 3: Process pages 101-150
batch3 = process_pdf_to_page_fens("book.pdf", start_page=101, max_pages=50)
```

### Analyzing Specific Sections
```python
# Only analyze the opening chapter (pages 10-25)
results = process_pdf_to_page_fens(
    pdf_path="chess_book.pdf",
    start_page=10,
    end_page=25,
    save_crops=True
)
```

## Complete Example with All Options

```python
from page_fen_processor import process_pdf_to_page_fens, save_results_to_json

# Process 10 pages starting from page 5 with full configuration
results = process_pdf_to_page_fens(
    pdf_path="../book2.pdf",
    output_dir="output_crops",
    save_crops=True,
    model="gpt-4o",
    dpi=240,
    start_page=5,    # Start from page 5
    max_pages=10     # Process 10 pages (5-14)
)

save_results_to_json(results, "output/pages_5_to_14.json")

print(f"Processed {len(results)} pages")
for page in results:
    print(f"Page {page['page_num']}: {page['boards_count']} boards found")
```

## Running Examples

All examples are available in `example_usage.py`:

```bash
cd page_fen_pipeline
python example_usage.py
```

Edit the `main()` function to uncomment the example you want to run:
- `example_5_limit_pages()` - Process first N pages
- `example_6_page_range()` - Process specific page range
- `example_7_start_with_limit()` - Start from page with limit

