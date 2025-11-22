#!/usr/bin/env python3
"""
Test script to demonstrate the complete benchmarking suite.
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and print results."""
    print(f"\n🚀 {description}")
    print("="*60)
    print(f"Command: {' '.join(command)}")
    print("-"*60)
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {command[0]}")
        return False

def test_benchmarking_suite():
    """Test the complete benchmarking suite."""
    print("🎯 Testing Chess CLIP Benchmarking Suite")
    print("="*80)
    
    # Check if required files exist
    required_files = [
        "benchmark_individual.py",
        "benchmark_comparative.py", 
        "inference.py",
        "dataset_loader.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Check if we have trained models
    model_paths = [
        "checkpoints/fen_only_model",
        "checkpoints/fen_move_model"
    ]
    
    existing_models = []
    for model_dir in model_paths:
        model_path = Path(model_dir)
        if model_path.exists():
            # Look for .pt files
            pt_files = list(model_path.glob("*.pt"))
            if pt_files:
                existing_models.append(str(pt_files[-1]))  # Use the last epoch
    
    if len(existing_models) < 2:
        print(f"⚠️  Need at least 2 trained models for comparison.")
        print(f"Found models: {existing_models}")
        
        # Try to find models in different locations
        all_pt_files = list(Path(".").rglob("*.pt"))
        print(f"All .pt files found: {[str(p) for p in all_pt_files]}")
        
        if len(all_pt_files) >= 2:
            existing_models = [str(all_pt_files[0]), str(all_pt_files[1])]
            print(f"Using models: {existing_models}")
        else:
            print("❌ Not enough models for comparison testing")
            return False
    
    # Check if we have test data
    test_data_dirs = [
        "test_datasets/fen_only",
        "test_datasets/fen_move"
    ]
    
    for data_dir in test_data_dirs:
        if not Path(data_dir).exists():
            print(f"❌ Test data directory not found: {data_dir}")
            return False
    
    print(f"✅ All required files and data found!")
    print(f"Models to test: {existing_models}")
    print(f"Test data: {test_data_dirs}")
    
    # Test 1: Individual model evaluation
    print(f"\n" + "="*80)
    print("TEST 1: Individual Model Evaluation")
    print("="*80)
    
    cmd1 = ["python", "benchmark_individual.py", existing_models[0], test_data_dirs[0]]
    success1 = run_command(cmd1, "Evaluating FEN-only model")
    
    # Test 2: Comparative evaluation
    print(f"\n" + "="*80)
    print("TEST 2: Comparative Model Evaluation")
    print("="*80)
    
    cmd2 = ["python", "benchmark_comparative.py", 
            existing_models[0], test_data_dirs[0],
            existing_models[1], test_data_dirs[1]]
    success2 = run_command(cmd2, "Comparing FEN-only vs FEN+move models")
    
    # Test 3: Inference testing
    print(f"\n" + "="*80)
    print("TEST 3: Inference Testing")
    print("="*80)
    
    # Find a test image
    test_images = list(Path("test_datasets/fen_only/images").glob("*.png"))
    if test_images:
        test_image = str(test_images[0])
        cmd3 = ["python", "inference.py", existing_models[0], test_image, "--create_sample"]
        success3 = run_command(cmd3, f"Testing inference on {test_image}")
    else:
        print("❌ No test images found for inference testing")
        success3 = False
    
    # Summary
    print(f"\n" + "="*80)
    print("🎯 BENCHMARKING SUITE TEST SUMMARY")
    print("="*80)
    print(f"Individual Evaluation: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Comparative Evaluation: {'✅ PASSED' if success2 else '❌ FAILED'}")
    print(f"Inference Testing: {'✅ PASSED' if success3 else '❌ FAILED'}")
    
    overall_success = success1 and success2 and success3
    print(f"\nOverall: {'🎉 ALL TESTS PASSED!' if overall_success else '⚠️ SOME TESTS FAILED'}")
    
    if overall_success:
        print(f"\n📚 Usage Examples:")
        print(f"# Evaluate individual model:")
        print(f"python benchmark_individual.py {existing_models[0]} {test_data_dirs[0]}")
        print(f"\n# Compare two models:")
        print(f"python benchmark_comparative.py {existing_models[0]} {test_data_dirs[0]} {existing_models[1]} {test_data_dirs[1]}")
        print(f"\n# Run inference:")
        print(f"python inference.py {existing_models[0]} {test_image if 'test_image' in locals() else 'path/to/image.png'}")
    
    return overall_success

if __name__ == "__main__":
    success = test_benchmarking_suite()
    sys.exit(0 if success else 1)
