# Setting Up Local LLaVA Model

## Quick Setup

1. **Find your local LLaVA model path** (e.g., `/path/to/llava-1.6-mistral`)

2. **Set the environment variable** before running:
   ```bash
   export LOCAL_VLM_PATH="/path/to/your/llava-1.6-mistral"
   ./run_benchmark.sh
   ```

3. **Or edit `run_benchmark.sh`** and uncomment/set the path:
   ```bash
   LOCAL_VLM_PATH="/path/to/your/llava-1.6-mistral"
   ```

## Finding Your Model Path

Your LLaVA model should be in a directory that contains:
- `config.json`
- `model files` (`.bin` or `.safetensors`)
- `tokenizer files`

Common locations:
- `~/.cache/huggingface/hub/models--llava-hf--llava-1.6-mistral-7b-hf/`
- Or wherever you downloaded/extracted the model

## Testing

To verify the path is correct:
```bash
ls "$LOCAL_VLM_PATH"
# Should show config.json and model files
```

