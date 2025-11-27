"""
Example usage of the chess VLM benchmarking system.
"""

from benchmark import ChessVLMBenchmark
from questions import get_scoring_questions
import os


def example_single_image():
    """Example: Benchmark a single image."""
    
    # Path to your trained CLIP model
    clip_checkpoint = "../runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt"
    
    # Initialize benchmark
    benchmark = ChessVLMBenchmark(
        clip_checkpoint_path=clip_checkpoint,
        clip_model_name="ViT-B-32",
        vlm_model_name="llava-hf/llava-1.5-7b-hf",
        use_mock_vlm=True  # Set to False to use real LLaVA
    )
    
    # Run benchmark on a single image
    image_path = "../data/hf_chess_puzzles/test/images/0.png"
    
    if os.path.exists(image_path):
        summary = benchmark.run_benchmark(
            image_paths=[image_path],
            output_dir="example_results"
        )
        print("\nExample completed!")
    else:
        print(f"Image not found: {image_path}")
        print("Please update the image_path variable with a valid chess board image.")


def example_multiple_images():
    """Example: Benchmark multiple images."""
    
    clip_checkpoint = "../runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt"
    
    benchmark = ChessVLMBenchmark(
        clip_checkpoint_path=clip_checkpoint,
        use_mock_vlm=True
    )
    
    # Get images from test directory
    test_dir = "../data/hf_chess_puzzles/test/images"
    if os.path.exists(test_dir):
        import glob
        image_paths = glob.glob(os.path.join(test_dir, "*.png"))[:5]  # First 5 images
        
        summary = benchmark.run_benchmark(
            image_paths=image_paths,
            output_dir="example_results_multiple"
        )
        print("\nExample completed!")
    else:
        print(f"Test directory not found: {test_dir}")


def example_with_fen_candidates():
    """Example: Benchmark with FEN candidates CSV."""
    
    clip_checkpoint = "../runs/clip_hf_chess_100k_20epochs_fixed/best_model.pt"
    fen_candidates_csv = "../data/hf_chess_puzzles/test.csv"
    
    benchmark = ChessVLMBenchmark(
        clip_checkpoint_path=clip_checkpoint,
        use_mock_vlm=True
    )
    
    image_path = "../data/hf_chess_puzzles/test/images/0.png"
    
    if os.path.exists(image_path) and os.path.exists(fen_candidates_csv):
        summary = benchmark.run_benchmark(
            image_paths=[image_path],
            fen_candidates_csv=fen_candidates_csv,
            output_dir="example_results_with_candidates"
        )
        print("\nExample completed!")
    else:
        print("Please ensure both image and FEN candidates CSV exist.")


if __name__ == "__main__":
    print("Chess VLM Benchmarking - Example Usage")
    print("=" * 60)
    print("\nChoose an example to run:")
    print("1. Single image benchmark")
    print("2. Multiple images benchmark")
    print("3. Benchmark with FEN candidates")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        example_single_image()
    elif choice == "2":
        example_multiple_images()
    elif choice == "3":
        example_with_fen_candidates()
    else:
        print("Invalid choice. Running single image example...")
        example_single_image()

