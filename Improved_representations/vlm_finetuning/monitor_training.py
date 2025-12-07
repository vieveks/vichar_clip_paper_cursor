"""
Simple script to monitor Qwen2-VL training progress.
"""

import json
import os
from pathlib import Path
from datetime import datetime


def check_training_status(checkpoint_dir: str):
    """Check training status from checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print("❌ Checkpoint directory does not exist yet.")
        print(f"   Expected: {checkpoint_dir}")
        return
    
    print("="*60)
    print("TRAINING STATUS CHECK")
    print("="*60)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check for training state
    trainer_state_file = checkpoint_path / "trainer_state.json"
    if trainer_state_file.exists():
        with open(trainer_state_file, 'r') as f:
            state = json.load(f)
        
        print("✅ Training is active!")
        print(f"   Current epoch: {state.get('epoch', 'N/A')}")
        print(f"   Current step: {state.get('global_step', 'N/A')}")
        print(f"   Training loss: {state.get('log_history', [{}])[-1].get('train_loss', 'N/A')}")
        print(f"   Evaluation loss: {state.get('log_history', [{}])[-1].get('eval_loss', 'N/A')}")
    else:
        print("⏳ Training may still be initializing...")
    
    # Check for checkpoints
    checkpoints = list(checkpoint_path.glob("checkpoint-*"))
    if checkpoints:
        print(f"\n✅ Found {len(checkpoints)} checkpoint(s):")
        for ckpt in sorted(checkpoints):
            size = sum(f.stat().st_size for f in ckpt.rglob('*') if f.is_file()) / (1024**2)
            print(f"   - {ckpt.name} ({size:.1f} MB)")
    else:
        print("\n⏳ No checkpoints saved yet...")
    
    # Check for logs
    log_files = list(checkpoint_path.glob("*.log"))
    if log_files:
        print(f"\n📝 Found {len(log_files)} log file(s)")
        for log_file in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
            print(f"   - {log_file.name} (modified: {datetime.fromtimestamp(log_file.stat().st_mtime).strftime('%H:%M:%S')})")
    
    print("="*60)


if __name__ == '__main__':
    import sys
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else 'Improved_representations/checkpoints/qwen2vl_json'
    check_training_status(checkpoint_dir)

