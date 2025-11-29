#!/usr/bin/env python3
"""
Command-Line Interface for Page-wise FEN Extraction
Process PDF files or images and extract FEN notation from chess boards.
"""

import argparse
import sys
from pathlib import Path
from page_fen_processor import (
    process_pdf_to_page_fens,
    process_image_to_fens,
    save_results_to_json
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract FEN notation from chess boards in PDF files or images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire PDF
  python cli.py book.pdf

  # Process first 5 pages
  python cli.py book.pdf --max-pages 5

  # Process pages 10-20
  python cli.py book.pdf --start-page 10 --end-page 20

  # Process 10 pages starting from page 5
  python cli.py book.pdf --start-page 5 --max-pages 10

  # Process with custom output and save crops
  python cli.py book.pdf --max-pages 3 --output-dir crops --save-crops --output-json results.json

  # Process an image file
  python cli.py chess_board.png --output-json image_result.json

  # Use a different model
  python cli.py book.pdf --max-pages 5 --model gpt-4.1-mini
        """
    )
    
    # Required arguments
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the PDF file or image to process'
    )
    
    # Page control arguments
    page_group = parser.add_argument_group('page control options')
    page_group.add_argument(
        '--start-page',
        type=int,
        default=1,
        metavar='N',
        help='First page to process (1-indexed, default: 1)'
    )
    page_group.add_argument(
        '--end-page',
        type=int,
        default=None,
        metavar='N',
        help='Last page to process (1-indexed, default: all pages)'
    )
    page_group.add_argument(
        '--max-pages',
        type=int,
        default=None,
        metavar='N',
        help='Maximum number of pages to process (default: no limit)'
    )
    
    # Output options
    output_group = parser.add_argument_group('output options')
    output_group.add_argument(
        '--output-dir',
        type=str,
        default=None,
        metavar='DIR',
        help='Directory to save board crop images (default: no crops saved)'
    )
    output_group.add_argument(
        '--save-crops',
        action='store_true',
        help='Save extracted board images to output directory'
    )
    output_group.add_argument(
        '--output-json',
        type=str,
        default='output/results.json',
        metavar='FILE',
        help='Path to save JSON results (default: output/results.json)'
    )
    
    # Model options
    model_group = parser.add_argument_group('model options')
    model_group.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='AI model to use for FEN generation (default: gpt-4o). '
             'Supported: OpenAI (gpt-4o, gpt-5, etc.), '
             'Anthropic (claude-3-5-sonnet, claude-3-opus), '
             'Google (gemini-2.0-flash, gemini-1.5-pro). '
             'Use --list-models to see all options.'
    )
    model_group.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models and exit'
    )
    model_group.add_argument(
        '--dpi',
        type=int,
        default=240,
        metavar='N',
        help='DPI for PDF rendering (default: 240, higher = better quality but slower)'
    )
    
    # Accuracy options
    accuracy_group = parser.add_argument_group('accuracy options (experimental)')
    accuracy_group.add_argument(
        '--strategy',
        type=str,
        default='simple',
        choices=['simple', 'enhanced', 'consensus'],
        help='FEN generation strategy: simple (fast), enhanced (balanced), consensus (accurate but slow/expensive)'
    )
    
    # Other options
    parser.add_argument(
        '--no-json',
        action='store_true',
        help='Skip saving JSON output'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information during processing'
    )
    
    args = parser.parse_args()
    
    # Handle --list-models
    if args.list_models:
        from model_providers import list_available_models
        print("\n" + "="*70)
        print(" AVAILABLE MODELS")
        print("="*70)
        
        models_by_provider = list_available_models()
        
        for provider, models in models_by_provider.items():
            print(f"\n{provider.upper()}:")
            for model in sorted(models):
                print(f"  - {model}")
        
        print("\n" + "="*70)
        print("\nUsage: python cli.py input.pdf --model <model_name>")
        print("\nNote: Requires appropriate API key in .env file:")
        print("  - OpenAI: OPENAI_API_KEY")
        print("  - Anthropic: ANTHROPIC_API_KEY")
        print("  - Google: GOOGLE_API_KEY or GEMINI_API_KEY")
        print()
        sys.exit(0)
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Determine file type
    file_extension = input_path.suffix.lower()
    is_pdf = file_extension == '.pdf'
    is_image = file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
    
    if not is_pdf and not is_image:
        print(f"❌ Error: Unsupported file type: {file_extension}", file=sys.stderr)
        print("Supported types: .pdf, .png, .jpg, .jpeg, .bmp, .tiff, .webp", file=sys.stderr)
        sys.exit(1)
    
    # Print configuration if verbose
    if args.verbose:
        print("\n" + "="*70)
        print(" PAGE-WISE FEN EXTRACTION - CLI")
        print("="*70)
        print(f"\nInput file: {args.input_file}")
        print(f"File type: {'PDF' if is_pdf else 'Image'}")
        if is_pdf:
            print(f"Start page: {args.start_page}")
            print(f"End page: {args.end_page or 'all'}")
            print(f"Max pages: {args.max_pages or 'no limit'}")
        print(f"Model: {args.model}")
        print(f"Strategy: {args.strategy}")
        print(f"DPI: {args.dpi}")
        print(f"Save crops: {args.save_crops}")
        if args.save_crops:
            print(f"Output directory: {args.output_dir or 'None (required with --save-crops)'}")
        print(f"Output JSON: {args.output_json if not args.no_json else 'disabled'}")
        
        # Strategy info
        if args.strategy != 'simple':
            print(f"\n⚠️  Using experimental {args.strategy} strategy")
            if args.strategy == 'consensus':
                print(f"    Note: This will make 3x API calls (higher cost)")
        print()
    
    # Validate output directory if saving crops
    if args.save_crops and not args.output_dir:
        print("❌ Error: --output-dir is required when using --save-crops", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Process based on file type
        if is_pdf:
            print(f"\n📄 Processing PDF: {args.input_file}\n")
            results = process_pdf_to_page_fens(
                pdf_path=args.input_file,
                output_dir=args.output_dir,
                save_crops=args.save_crops,
                model=args.model,
                dpi=args.dpi,
                start_page=args.start_page,
                end_page=args.end_page,
                max_pages=args.max_pages
            )
        else:
            print(f"\n🖼️  Processing Image: {args.input_file}\n")
            results = process_image_to_fens(
                image_path=args.input_file,
                output_dir=args.output_dir,
                save_crops=args.save_crops,
                model=args.model
            )
        
        # Save JSON if not disabled
        if not args.no_json:
            save_results_to_json(results, args.output_json)
        
        # Print summary
        print("\n" + "="*70)
        print(" SUMMARY")
        print("="*70)
        
        if is_pdf:
            total_boards = sum(page['boards_count'] for page in results)
            print(f"\n✅ Processed {len(results)} pages")
            print(f"✅ Found {total_boards} chess boards total")
            
            for page in results:
                print(f"\n  Page {page['page_num']}: {page['boards_count']} board(s)")
                for board in page['boards']:
                    fen = board.get('fen', 'ERROR')
                    # Truncate long FEN for display
                    fen_display = fen[:70] + "..." if len(fen) > 70 else fen
                    print(f"    Board {board['board_index']}: {fen_display}")
        else:
            print(f"\n✅ Found {results['boards_count']} chess board(s)")
            
            for board in results['boards']:
                fen = board.get('fen', 'ERROR')
                fen_display = fen[:70] + "..." if len(fen) > 70 else fen
                print(f"\n  Board {board['board_index']}: {fen_display}")
        
        if not args.no_json:
            print(f"\n💾 Results saved to: {args.output_json}")
        
        if args.save_crops:
            print(f"🖼️  Board crops saved to: {args.output_dir}/")
        
        print("\n" + "="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

