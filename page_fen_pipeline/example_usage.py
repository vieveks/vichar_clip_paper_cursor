"""
Example Usage Script
Simple examples demonstrating how to use the page-wise FEN extraction pipeline.
"""

from page_fen_processor import (
    process_pdf_to_page_fens,
    process_image_to_fens,
    save_results_to_json
)


def example_1_process_pdf():
    """Example: Process a PDF file and save results."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Processing PDF File")
    print("="*60 + "\n")
    
    results = process_pdf_to_page_fens(
        pdf_path="../book2.pdf",        # Path to your PDF
        output_dir="output_crops",       # Where to save board images
        save_crops=True,                 # Save the extracted boards
        model="gpt-4o",                 # OpenAI model
        dpi=240                          # Quality of PDF rendering
    )
    
    # Save to JSON
    save_results_to_json(results, "output/book2_fens.json")
    
    # Print summary
    print("\n" + "="*60)
    print("PDF Processing Summary")
    print("="*60)
    total_boards = sum(page['boards_count'] for page in results)
    print(f"Total pages: {len(results)}")
    print(f"Total boards: {total_boards}")
    
    for page in results:
        print(f"\nPage {page['page_num']}:")
        for board in page['boards']:
            fen = board.get('fen', 'ERROR')
            print(f"  Board {board['board_index']}: {fen[:60]}...")


def example_2_process_image():
    """Example: Process a single image file."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Processing Single Image")
    print("="*60 + "\n")
    
    result = process_image_to_fens(
        image_path="../example_chess_image.png",
        output_dir="output_crops",
        save_crops=True,
        model="gpt-4o"
    )
    
    # Save to JSON
    save_results_to_json(result, "output/example_image_fens.json")
    
    # Print summary
    print("\n" + "="*60)
    print("Image Processing Summary")
    print("="*60)
    print(f"Image: {result['image_path']}")
    print(f"Boards found: {result['boards_count']}")
    
    for board in result['boards']:
        fen = board.get('fen', 'ERROR')
        print(f"\nBoard {board['board_index']}:")
        print(f"  FEN: {fen}")
        if 'crop_path' in board:
            print(f"  Saved to: {board['crop_path']}")


def example_3_process_without_saving_crops():
    """Example: Process without saving board images (faster, less disk space)."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Processing Without Saving Crops")
    print("="*60 + "\n")
    
    results = process_pdf_to_page_fens(
        pdf_path="../book2.pdf",
        save_crops=False,  # Don't save board images
        model="gpt-4o"
    )
    
    # Just save JSON results
    save_results_to_json(results, "output/book2_fens_only.json")
    
    print(f"\nProcessed {len(results)} pages")
    print("Results saved to JSON (no image crops saved)")


def example_4_custom_model():
    """Example: Use a different OpenAI model."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Using Different Model")
    print("="*60 + "\n")
    
    # You can try different models like:
    # - "gpt-4o" (default, best quality)
    # - "gpt-4.1-mini" (faster, cheaper)
    # - "gpt-5-mini" (if you have access)
    
    results = process_pdf_to_page_fens(
        pdf_path="../book2.pdf",
        save_crops=False,
        model="gpt-4.1-mini",  # Different model
        dpi=200                 # Lower DPI for faster processing
    )
    
    save_results_to_json(results, "output/book2_fens_mini.json")
    print(f"\nProcessed with model: gpt-4.1-mini")


def example_5_limit_pages():
    """Example: Process only a limited number of pages."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Processing Limited Number of Pages")
    print("="*60 + "\n")
    
    # Process only the first 3 pages
    results = process_pdf_to_page_fens(
        pdf_path="../book2.pdf",
        output_dir="output_crops",
        save_crops=True,
        model="gpt-4o",
        max_pages=3  # Only process first 3 pages
    )
    
    save_results_to_json(results, "output/book2_first_3_pages.json")
    print(f"\nProcessed only {len(results)} pages")


def example_6_page_range():
    """Example: Process a specific range of pages."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Processing Specific Page Range")
    print("="*60 + "\n")
    
    # Process pages 5 through 10
    results = process_pdf_to_page_fens(
        pdf_path="../book2.pdf",
        output_dir="output_crops",
        save_crops=True,
        model="gpt-4o",
        start_page=5,   # Start from page 5
        end_page=10     # End at page 10
    )
    
    save_results_to_json(results, "output/book2_pages_5_to_10.json")
    print(f"\nProcessed pages 5-10: {len(results)} pages total")


def example_7_start_with_limit():
    """Example: Start from a specific page and limit the count."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Start Page + Max Pages")
    print("="*60 + "\n")
    
    # Process 5 pages starting from page 10
    results = process_pdf_to_page_fens(
        pdf_path="../book2.pdf",
        output_dir="output_crops",
        save_crops=True,
        model="gpt-4o",
        start_page=10,  # Start from page 10
        max_pages=5     # Process 5 pages (pages 10-14)
    )
    
    save_results_to_json(results, "output/book2_pages_10_to_14.json")
    print(f"\nProcessed 5 pages starting from page 10")


def main():
    """Run the examples."""
    print("\n" + "="*70)
    print(" PAGE-WISE FEN EXTRACTION - EXAMPLES")
    print("="*70)
    
    # Uncomment the example you want to run:
    
    # Example 1: Process PDF with full features
    # example_1_process_pdf()
    
    # Example 2: Process a single image
    # example_2_process_image()
    
    # Example 3: Process without saving crops (faster)
    # example_3_process_without_saving_crops()
    
    # Example 4: Use a different model
    # example_4_custom_model()
    
    # Example 5: Process only first N pages (NEW!)
    example_5_limit_pages()
    
    # Example 6: Process a specific page range (NEW!)
    # example_6_page_range()
    
    # Example 7: Start from a page and limit count (NEW!)
    # example_7_start_with_limit()
    
    print("\n" + "="*70)
    print(" Done! Check the 'output' folder for results.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

