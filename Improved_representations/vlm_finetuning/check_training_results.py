"""Check training results from trainer_state.json."""
import json
from pathlib import Path

checkpoint_dir = Path('Improved_representations/checkpoints/qwen2vl_json')
# Check both root and checkpoint subdirectories
state_file = checkpoint_dir / 'trainer_state.json'
if not state_file.exists():
    # Try checkpoint subdirectories
    checkpoints = list(checkpoint_dir.glob('checkpoint-*/trainer_state.json'))
    if checkpoints:
        state_file = checkpoints[-1]  # Use latest checkpoint

if state_file.exists():
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    print("="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Total steps: {state.get('global_step', 'N/A')}")
    print(f"Epochs completed: {state.get('epoch', 'N/A'):.2f}")
    print(f"Best metric: {state.get('best_metric', 'N/A')}")
    print(f"Best model checkpoint: {state.get('best_model_checkpoint', 'N/A')}")
    print()
    
    logs = state.get('log_history', [])
    if logs:
        print("Training logs:")
        print("-" * 60)
        for log in logs:
            step = log.get('step', '?')
            loss = log.get('loss', '?')
            eval_loss = log.get('eval_loss', None)
            lr = log.get('learning_rate', '?')
            epoch = log.get('epoch', '?')
            
            loss_str = f"{loss:.4f}" if isinstance(loss, float) else str(loss)
            eval_str = f"{eval_loss:.4f}" if eval_loss is not None and isinstance(eval_loss, float) else (str(eval_loss) if eval_loss is not None else 'N/A')
            lr_str = f"{lr:.2e}" if isinstance(lr, float) else str(lr)
            epoch_str = f"{epoch:.3f}" if isinstance(epoch, float) else str(epoch)
            
            print(f"Step {step} (Epoch {epoch_str}): loss={loss_str}, eval_loss={eval_str}, lr={lr_str}")
    print("="*60)
    
    # Summary
    if logs:
        first_loss = logs[0].get('loss')
        last_loss = logs[-1].get('loss')
        if isinstance(first_loss, float) and isinstance(last_loss, float):
            improvement = first_loss - last_loss
            print(f"\nLoss improvement: {first_loss:.4f} → {last_loss:.4f} (Δ {improvement:.4f})")
else:
    print(f"Training state file not found: {state_file}")
