"""
Page-wise FEN Processor
Main module that combines board extraction and FEN generation to process entire PDFs or images.
"""

from pathlib import Path
import json
from typing import List, Dict, Optional
from board_extractor import extract_boards_from_pdf_pages, extract_boards_from_single_image, save_board_crop
from fen_generator import generate_fen_from_image_array, get_openai_client


def process_pdf_to_page_fens(
    pdf_path: str,
    output_dir: Optional[str] = None,
    save_crops: bool = False,
    model: str = "gpt-4o",
    dpi: int = 240,
    start_page: int = 1,
    end_page: Optional[int] = None,
    max_pages: Optional[int] = None
) -> List[Dict]:
    """
    Process a PDF file and extract FEN notation for all chess boards on each page.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Optional directory to save board crops (if save_crops=True)
        save_crops: Whether to save extracted board images
        model: OpenAI model to use for FEN generation
        dpi: DPI for PDF rendering
        start_page: First page to process (1-indexed, default=1)
        end_page: Last page to process (1-indexed, None=all pages)
        max_pages: Maximum number of pages to process (None=no limit)
                   Note: If both end_page and max_pages are set, max_pages takes precedence
    
    Returns:
        List of dicts with structure:
        [
            {
                'page_num': 1,
                'boards_count': 2,
                'boards': [
                    {
                        'board_index': 1,
                        'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                        'crop_path': 'path/to/crop.png' (if save_crops=True)
                    },
                    ...
                ]
            },
            ...
        ]
    """
    print(f"Processing PDF: {pdf_path}")
    print(f"Model: {model}\n")
    
    # Initialize OpenAI client once
    client = get_openai_client()
    
    # Extract boards from PDF pages
    pages_data = extract_boards_from_pdf_pages(
        pdf_path, 
        dpi=dpi, 
        start_page=start_page, 
        end_page=end_page, 
        max_pages=max_pages
    )
    
    # Setup output directory if needed
    out_dir = None
    if save_crops and output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each page
    results = []
    for page_data in pages_data:
        page_num = page_data['page_num']
        boards = page_data['boards']
        
        print(f"\n{'='*60}")
        print(f"Processing Page {page_num} - {len(boards)} board(s) found")
        print(f"{'='*60}")
        
        page_result = {
            'page_num': page_num,
            'boards_count': len(boards),
            'boards': []
        }
        
        for idx, (board_img, bbox) in enumerate(boards, 1):
            print(f"\n  Board {idx}/{len(boards)} on Page {page_num}:")
            
            # Save crop if requested
            crop_path = None
            if save_crops and out_dir:
                # Create page-specific folder
                page_folder = out_dir / f"page_{page_num:03d}"
                crop_path = page_folder / f"board_{idx}.png"
                save_board_crop(board_img, crop_path)
                print(f"    Saved crop: {crop_path}")
            
            # Generate FEN
            print(f"    Generating FEN...")
            try:
                fen_result = generate_fen_from_image_array(board_img, model=model, client=client)
                fen = fen_result['fen']
                print(f"    FEN: {fen}")
                
                board_result = {
                    'board_index': idx,
                    'fen': fen,
                    'bbox': bbox
                }
                
                if crop_path:
                    board_result['crop_path'] = str(crop_path)
                
                page_result['boards'].append(board_result)
                
            except Exception as e:
                print(f"    Error generating FEN: {str(e)}")
                page_result['boards'].append({
                    'board_index': idx,
                    'fen': 'ERROR',
                    'error': str(e),
                    'bbox': bbox
                })
        
        results.append(page_result)
    
    return results


def process_image_to_fens(
    image_path: str,
    output_dir: Optional[str] = None,
    save_crops: bool = False,
    model: str = "gpt-4o"
) -> Dict:
    """
    Process a single image file and extract FEN notation for all chess boards found.
    
    Args:
        image_path: Path to the image file
        output_dir: Optional directory to save board crops (if save_crops=True)
        save_crops: Whether to save extracted board images
        model: OpenAI model to use for FEN generation
    
    Returns:
        Dict with structure:
        {
            'image_path': 'path/to/image.png',
            'boards_count': 2,
            'boards': [
                {
                    'board_index': 1,
                    'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                    'crop_path': 'path/to/crop.png' (if save_crops=True)
                },
                ...
            ]
        }
    """
    print(f"Processing Image: {image_path}")
    print(f"Model: {model}\n")
    
    # Initialize OpenAI client once
    client = get_openai_client()
    
    # Extract boards from image
    boards = extract_boards_from_single_image(image_path)
    
    # Setup output directory if needed
    out_dir = None
    if save_crops and output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        'image_path': image_path,
        'boards_count': len(boards),
        'boards': []
    }
    
    print(f"\n{'='*60}")
    print(f"Processing {len(boards)} board(s)")
    print(f"{'='*60}")
    
    for idx, (board_img, bbox) in enumerate(boards, 1):
        print(f"\n  Board {idx}/{len(boards)}:")
        
        # Save crop if requested
        crop_path = None
        if save_crops and out_dir:
            image_name = Path(image_path).stem
            # Create image-specific folder
            image_folder = out_dir / image_name
            crop_path = image_folder / f"board_{idx}.png"
            save_board_crop(board_img, crop_path)
            print(f"    Saved crop: {crop_path}")
        
        # Generate FEN
        print(f"    Generating FEN...")
        try:
            fen_result = generate_fen_from_image_array(board_img, model=model, client=client)
            fen = fen_result['fen']
            print(f"    FEN: {fen}")
            
            board_result = {
                'board_index': idx,
                'fen': fen,
                'bbox': bbox
            }
            
            if crop_path:
                board_result['crop_path'] = str(crop_path)
            
            result['boards'].append(board_result)
            
        except Exception as e:
            print(f"    Error generating FEN: {str(e)}")
            result['boards'].append({
                'board_index': idx,
                'fen': 'ERROR',
                'error': str(e),
                'bbox': bbox
            })
    
    return result


def save_results_to_json(results, output_path: str):
    """Save processing results to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n\nResults saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Example 1: Process a PDF file
    pdf_results = process_pdf_to_page_fens(
        pdf_path="../book2.pdf",
        output_dir="output_crops",
        save_crops=True,
        model="gpt-4o",
        dpi=240
    )
    
    # Save results to JSON
    save_results_to_json(pdf_results, "output/pdf_fens.json")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for page in pdf_results:
        print(f"Page {page['page_num']}: {page['boards_count']} board(s)")
        for board in page['boards']:
            print(f"  Board {board['board_index']}: {board['fen'][:50]}...")
    
    # Example 2: Process a single image
    # image_result = process_image_to_fens(
    #     image_path="../example_chess_image.png",
    #     output_dir="output_crops",
    #     save_crops=True,
    #     model="gpt-4o"
    # )
    # save_results_to_json(image_result, "output/image_fens.json")

