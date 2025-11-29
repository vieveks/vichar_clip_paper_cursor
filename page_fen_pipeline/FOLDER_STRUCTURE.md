# Board Crops Folder Structure

## Overview

Board crops are now saved in an organized folder structure with dedicated folders for each page (PDFs) or image (single images). This makes it easy to navigate, find, and process boards systematically.

## PDF Processing Structure

When processing a PDF with `--save-crops`, boards are organized by page:

```
output_crops/
├── page_001/
│   ├── board_1.png
│   ├── board_2.png
│   └── board_3.png
├── page_002/
│   └── board_1.png
├── page_003/
│   ├── board_1.png
│   └── board_2.png
├── page_004/
│   └── board_1.png
└── page_005/
    ├── board_1.png
    ├── board_2.png
    ├── board_3.png
    └── board_4.png
```

### Naming Convention (PDF)

- **Page folders**: `page_XXX/` where XXX is the 3-digit page number (e.g., `page_001`, `page_042`, `page_123`)
- **Board files**: `board_Y.png` where Y is the board number on that page (e.g., `board_1.png`, `board_2.png`)
- Boards are numbered in the order they appear on the page (top-to-bottom, left-to-right)

### Example Command

```bash
python cli.py ../book2.pdf --start-page 10 --end-page 15 --save-crops --output-dir my_boards
```

**Result:**
```
my_boards/
├── page_010/
│   ├── board_1.png
│   └── board_2.png
├── page_011/
│   └── board_1.png
├── page_012/
│   ├── board_1.png
│   ├── board_2.png
│   └── board_3.png
├── page_013/
│   └── board_1.png
├── page_014/
│   ├── board_1.png
│   └── board_2.png
└── page_015/
    └── board_1.png
```

## Image Processing Structure

When processing a single image file with `--save-crops`, boards are organized in a folder named after the image:

```
output_crops/
├── chess_board/
│   ├── board_1.png
│   └── board_2.png
├── example_chess_image/
│   └── board_1.png
└── puzzle_diagram/
    ├── board_1.png
    ├── board_2.png
    └── board_3.png
```

### Naming Convention (Images)

- **Image folders**: Named after the source image file (without extension)
- **Board files**: `board_Y.png` where Y is the board number found in that image

### Example Command

```bash
python cli.py ../example_chess_image.png --save-crops --output-dir extracted_boards
```

**Result:**
```
extracted_boards/
└── example_chess_image/
    ├── board_1.png
    └── board_2.png
```

## Benefits of This Structure

### 1. Easy Navigation
```bash
# Find all boards from page 15
ls output_crops/page_015/

# Count boards on each page
for dir in output_crops/page_*/; do echo "$dir: $(ls $dir | wc -l)"; done
```

### 2. Programmatic Access
```python
from pathlib import Path

# Get all boards from page 10
page_10_boards = list(Path("output_crops/page_010").glob("*.png"))

# Process each page folder
for page_folder in Path("output_crops").glob("page_*"):
    page_num = page_folder.name
    boards = list(page_folder.glob("board_*.png"))
    print(f"{page_num}: {len(boards)} boards")
```

### 3. Selective Processing
```bash
# Copy all boards from pages 10-20
cp -r output_crops/page_01[0-9] output_crops/page_020 selected_pages/

# Delete boards from specific pages
rm -rf output_crops/page_005 output_crops/page_006
```

### 4. Better Organization
- Each page's boards are grouped together
- Easy to see how many boards per page at a glance
- Simpler to manage large collections
- No confusion with long flat filenames

## JSON Output Integration

The JSON output reflects this folder structure:

```json
{
  "page_num": 15,
  "boards_count": 2,
  "boards": [
    {
      "board_index": 1,
      "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "crop_path": "output_crops/page_015/board_1.png"
    },
    {
      "board_index": 2,
      "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
      "crop_path": "output_crops/page_015/board_2.png"
    }
  ]
}
```

## Migration from Old Structure

If you have existing crops saved with the old flat structure, you can reorganize them:

### Python Script to Migrate

```python
from pathlib import Path
import shutil
import re

def migrate_crops(old_dir, new_dir):
    """Migrate from old flat structure to new folder structure."""
    old_path = Path(old_dir)
    new_path = Path(new_dir)
    new_path.mkdir(exist_ok=True)
    
    # Pattern: page_XXX_board_Y.png
    pattern = re.compile(r'page_(\d{3})_board_(\d+)\.png')
    
    for old_file in old_path.glob('page_*_board_*.png'):
        match = pattern.match(old_file.name)
        if match:
            page_num, board_num = match.groups()
            
            # Create new structure
            page_folder = new_path / f"page_{page_num}"
            page_folder.mkdir(exist_ok=True)
            
            new_file = page_folder / f"board_{board_num}.png"
            shutil.copy2(old_file, new_file)
            print(f"Migrated: {old_file.name} -> {new_file}")

# Usage
migrate_crops('old_output_crops', 'new_output_crops')
```

### Bash Script to Migrate

```bash
#!/bin/bash
# migrate_crops.sh

OLD_DIR="old_output_crops"
NEW_DIR="new_output_crops"

mkdir -p "$NEW_DIR"

for file in "$OLD_DIR"/page_*_board_*.png; do
    filename=$(basename "$file")
    
    # Extract page and board numbers
    page=$(echo "$filename" | sed -E 's/page_([0-9]{3})_board_[0-9]+\.png/\1/')
    board=$(echo "$filename" | sed -E 's/page_[0-9]{3}_board_([0-9]+)\.png/\1/')
    
    # Create page folder
    mkdir -p "$NEW_DIR/page_$page"
    
    # Copy file
    cp "$file" "$NEW_DIR/page_$page/board_$board.png"
    echo "Migrated: $filename -> page_$page/board_$board.png"
done
```

## Tips

1. **Consistent naming**: Always use the same `--output-dir` for related processing runs
2. **Batch processing**: Each batch can have its own output folder for easy management
3. **Backup**: Keep original PDFs and generated crops in separate locations
4. **Cleanup**: You can easily delete entire page folders if you need to reprocess specific pages

## Examples in the Wild

### Process entire book with organized output
```bash
python cli.py ../chess_tactics_book.pdf --save-crops --output-dir tactics_book_boards
```

### Process chapters separately
```bash
python cli.py ../book.pdf --start-page 1 --end-page 50 --save-crops --output-dir chapter1
python cli.py ../book.pdf --start-page 51 --end-page 100 --save-crops --output-dir chapter2
python cli.py ../book.pdf --start-page 101 --end-page 150 --save-crops --output-dir chapter3
```

### Process specific pages with detailed organization
```bash
python cli.py ../book.pdf --start-page 75 --max-pages 10 --save-crops --output-dir interesting_positions
```

Result:
```
interesting_positions/
├── page_075/
├── page_076/
├── page_077/
└── ...
```

