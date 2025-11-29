"""
Board Extractor Module
Extracts chess board diagrams from PDF files and images using computer vision.
"""

from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import fitz  # PyMuPDF


def verify_checker_correlation(roi_bgr, corr_threshold=0.30):
    """Is ROI an 8x8 board? Correlate block-averages with an ideal checker pattern."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    gray = (gray - gray.mean()) / (gray.std() + 1e-6)
    block = gray.reshape(8, 32, 8, 32).mean(axis=(1, 3))              # 8x8 block means
    ideal = (np.indices((8, 8)).sum(axis=0) % 2).astype(float)
    ideal = (ideal - ideal.mean()) / (ideal.std() + 1e-6)
    corr = abs(np.corrcoef(block.ravel(), ideal.ravel())[0, 1])
    return corr > corr_threshold


def extract_boards_from_image(img_bgr):
    """Return list of (crop, bbox) for boards found in a raster image page."""
    H, W = img_bgr.shape[:2]
    # Upscale helps on small/low-res pages
    scale = 2.0 if max(H, W) < 2000 else 1.0
    if scale != 1.0:
        img = cv2.resize(img_bgr, (int(W*scale), int(H*scale)), cv2.INTER_CUBIC)
    else:
        img = img_bgr.copy()
    H, W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), 2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 0.002*W*H:        # ignore tiny blobs
            continue
        ar = w/(h+1e-6)
        if not (0.7 <= ar <= 1.4): # near-square
            continue
        pad = int(0.02*max(w,h))
        x0,y0 = max(0,x-pad), max(0,y-pad)
        x1,y1 = min(W,x+w+pad), min(H,y+h+pad)
        roi = img[y0:y1, x0:x1]
        if verify_checker_correlation(roi):
            cand.append((roi, (x0,y0,x1-x0,y1-y0)))

    # sort top-to-bottom, then left-to-right
    cand.sort(key=lambda it: (it[1][1], it[1][0]))
    return cand


def save_board_crop(roi, out_path: Path):
    """Save a single board crop to a file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).save(out_path)
    return out_path


def extract_boards_from_pdf_pages(pdf_path: str, dpi=240, start_page=1, end_page=None, max_pages=None):
    """
    Extract boards from PDF and return structured data by page.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: DPI for rendering PDF pages
        start_page: First page to process (1-indexed, default=1)
        end_page: Last page to process (1-indexed, None=all pages)
        max_pages: Maximum number of pages to process (None=no limit)
                  Note: If both end_page and max_pages are set, max_pages takes precedence
    
    Returns:
        List of dicts with structure:
        [
            {
                'page_num': 1,
                'boards': [(board_image_array, bbox), ...]
            },
            ...
        ]
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # Determine the actual range of pages to process
    actual_start = max(1, start_page)
    
    if max_pages is not None:
        actual_end = min(total_pages, actual_start + max_pages - 1)
    elif end_page is not None:
        actual_end = min(total_pages, end_page)
    else:
        actual_end = total_pages
    
    print(f"PDF has {total_pages} pages. Processing pages {actual_start} to {actual_end}")
    
    pages_data = []
    
    for i in range(actual_start - 1, actual_end):  # Convert to 0-indexed
        page = doc[i]
        page_num = i + 1  # Convert back to 1-indexed for display
        
        zoom = dpi/72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4: 
            img = img[:, :, :3]
        
        boards = extract_boards_from_image(img)
        
        pages_data.append({
            'page_num': page_num,
            'boards': boards
        })
        
        print(f"Page {page_num}: Found {len(boards)} board(s)")
    
    return pages_data


def extract_boards_from_single_image(image_path: str):
    """
    Extract boards from a single image file.
    
    Returns:
        List of (board_image_array, bbox) tuples
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    boards = extract_boards_from_image(img)
    print(f"Found {len(boards)} board(s) in image")
    return boards

