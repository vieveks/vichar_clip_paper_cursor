# Multi-Model Support Guide

The pipeline now supports multiple AI model providers for FEN generation, including OpenAI, Anthropic Claude, and Google Gemini models.

## 🎯 Supported Providers

### OpenAI (GPT Models)
- ✅ GPT-4o (default - recommended)
- ✅ GPT-4o-mini (faster, cheaper)
- ✅ GPT-4-turbo
- ✅ GPT-4.1-mini
- ✅ GPT-4.1-nano
- ✅ GPT-5 (if you have access - **fixed response parsing**)
- ✅ GPT-5-mini
- ✅ GPT-5-nano

### Anthropic (Claude Models)
- ✅ Claude 3.5 Sonnet (excellent quality)
- ✅ Claude 3.5 Haiku (fast)
- ✅ Claude 3 Opus (highest quality)
- ✅ Claude 3 Sonnet
- ✅ Claude 3 Haiku

### Google (Gemini Models)
- ✅ Gemini 2.0 Flash (fast, latest)
- ✅ Gemini 1.5 Pro (high quality)
- ✅ Gemini 1.5 Flash (balanced)
- ✅ Gemini Pro

## 🔧 Setup

### 1. Install Required Packages

**OpenAI (included by default):**
```bash
pip install openai
```

**Anthropic Claude (optional):**
```bash
pip install anthropic
```

**Google Gemini (optional):**
```bash
pip install google-generativeai
```

Or install all at once:
```bash
pip install openai anthropic google-generativeai
```

### 2. Configure API Keys

Add your API keys to the `.env` file in the parent directory:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic (if using Claude)
ANTHROPIC_API_KEY=sk-ant-...

# Google (if using Gemini)
GOOGLE_API_KEY=AIza...
# OR
GEMINI_API_KEY=AIza...
```

## 📋 List Available Models

```bash
cd page_fen_pipeline
python cli.py --list-models
```

Output:
```
======================================================================
 AVAILABLE MODELS
======================================================================

OPENAI:
  - gpt-4.1-mini
  - gpt-4.1-nano
  - gpt-4o
  - gpt-4o-mini
  - gpt-5
  - gpt-5-mini
  ...

ANTHROPIC:
  - claude-3-5-haiku
  - claude-3-5-sonnet
  - claude-3-haiku
  - claude-3-opus
  ...

GOOGLE:
  - gemini-1.5-flash
  - gemini-1.5-pro
  - gemini-2.0-flash
  ...
```

## 🚀 Usage

### Command Line Interface

```bash
# OpenAI GPT-4o (default)
python cli.py book.pdf --max-pages 3

# OpenAI GPT-5
python cli.py book.pdf --max-pages 3 --model gpt-5

# Anthropic Claude
python cli.py book.pdf --max-pages 3 --model claude-3-5-sonnet

# Google Gemini
python cli.py book.pdf --max-pages 3 --model gemini-2.0-flash
```

### Python API

```python
from page_fen_processor import process_pdf_to_page_fens

# OpenAI GPT-4o
results = process_pdf_to_page_fens(
    "book.pdf",
    model="gpt-4o"
)

# Anthropic Claude
results = process_pdf_to_page_fens(
    "book.pdf",
    model="claude-3-5-sonnet"
)

# Google Gemini
results = process_pdf_to_page_fens(
    "book.pdf",
    model="gemini-2.0-flash"
)
```

### Direct Model Provider Usage

```python
from model_providers import generate_fen_universal
import cv2

# Load image
img = cv2.imread("board.png")

# Use any model
result = generate_fen_universal(img, model="claude-3-5-sonnet")
print(f"FEN: {result['fen']}")
print(f"Provider: {result['provider']}")
print(f"Model: {result['model']}")
```

## 📊 Model Comparison

### Quality Comparison (Approximate)

| Model | Accuracy | Speed | Cost | Notes |
|-------|----------|-------|------|-------|
| **gpt-4o** | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | 💰💰 | Best balance (default) |
| **gpt-4o-mini** | ⭐⭐⭐⭐ | ⚡⚡⚡⚡ | 💰 | Fast & cheap |
| **gpt-5** | ⭐⭐⭐⭐⭐ | ⚡⚡ | 💰💰💰 | Latest, needs access |
| **claude-3-5-sonnet** | ⭐⭐⭐⭐⭐ | ⚡⚡⚡ | 💰💰 | Excellent quality |
| **claude-3-opus** | ⭐⭐⭐⭐⭐ | ⚡⚡ | 💰💰💰 | Highest quality |
| **claude-3-5-haiku** | ⭐⭐⭐⭐ | ⚡⚡⚡⚡⚡ | 💰 | Very fast |
| **gemini-2.0-flash** | ⭐⭐⭐⭐ | ⚡⚡⚡⚡ | 💰 | Fast & good |
| **gemini-1.5-pro** | ⭐⭐⭐⭐⭐ | ⚡⚡ | 💰💰 | High quality |

### Cost Comparison (Approximate per 1000 boards)

| Provider | Model | Cost | Notes |
|----------|-------|------|-------|
| OpenAI | gpt-4o-mini | ~$15 | Most economical OpenAI |
| OpenAI | gpt-4o | ~$20 | Best OpenAI balance |
| OpenAI | gpt-5 | ~$30 | Latest technology |
| Anthropic | claude-3-5-haiku | ~$10 | Most economical overall |
| Anthropic | claude-3-5-sonnet | ~$20 | Excellent value |
| Anthropic | claude-3-opus | ~$40 | Premium quality |
| Google | gemini-2.0-flash | ~$5 | Most economical |
| Google | gemini-1.5-pro | ~$15 | Great value |

*Note: Actual costs vary based on image size, prompt length, and API pricing changes*

## 🎓 Recommendations by Use Case

### Budget Processing
```bash
# Gemini 2.0 Flash - cheapest
python cli.py book.pdf --model gemini-2.0-flash --dpi 200
```

### Balanced Quality & Cost
```bash
# GPT-4o (default) or Claude 3.5 Sonnet
python cli.py book.pdf --model gpt-4o
python cli.py book.pdf --model claude-3-5-sonnet
```

### Maximum Quality
```bash
# Claude 3 Opus or GPT-5
python cli.py book.pdf --model claude-3-opus --dpi 300
python cli.py book.pdf --model gpt-5 --dpi 300
```

### Speed Priority
```bash
# Claude 3.5 Haiku or Gemini 2.0 Flash
python cli.py book.pdf --model claude-3-5-haiku
python cli.py book.pdf --model gemini-2.0-flash
```

## 🔍 Testing Different Models

Use the comparison tool to test multiple models:

```bash
# Compare strategies with different models
python compare_fen_strategies.py board.png gpt-4o
python compare_fen_strategies.py board.png claude-3-5-sonnet
python compare_fen_strategies.py board.png gemini-2.0-flash
```

Or create a custom comparison script:

```python
from model_providers import generate_fen_universal
import cv2

img = cv2.imread("board.png")

models = [
    "gpt-4o",
    "claude-3-5-sonnet",
    "gemini-2.0-flash"
]

for model in models:
    result = generate_fen_universal(img, model)
    print(f"\n{model}:")
    print(f"  FEN: {result['fen']}")
```

## 🐛 Troubleshooting

### "API key not found"

**Problem:** Provider API key is missing
**Solution:** Add the appropriate key to your `.env` file:
- OpenAI: `OPENAI_API_KEY=sk-...`
- Anthropic: `ANTHROPIC_API_KEY=sk-ant-...`
- Google: `GOOGLE_API_KEY=AIza...`

### "Package not installed"

**Problem:** Provider library is not installed
**Solution:** Install the required package:
```bash
pip install anthropic  # For Claude
pip install google-generativeai  # For Gemini
```

### "Invalid API key"

**Problem:** API key is incorrect or expired
**Solution:** 
1. Check your API key at the provider's dashboard
2. Ensure no extra spaces or quotes in `.env` file
3. Restart your terminal/shell after updating `.env`

### "Rate limit exceeded"

**Problem:** Too many requests to the API
**Solution:**
1. Add delays between requests
2. Use cheaper models (e.g., gemini-2.0-flash)
3. Process in smaller batches
4. Upgrade your API plan

### GPT-5 Not Working

**Problem:** GPT-5 returns "[No text returned]" or empty responses

**Solution:** Use the debugging script to see the actual response structure:
```bash
cd page_fen_pipeline
python debug_gpt5.py test_boards/page_008/board_1.png
```

This will show:
- Whether GPT-5 is accessible
- The exact response structure
- Which extraction method works
- Detailed error messages

**Common causes:**
1. **No GPT-5 access**: Check your OpenAI account tier
2. **API format changed**: OpenAI may have updated the API
3. **Old openai package**: Update with `pip install --upgrade openai`
4. **Wrong model name**: Try exact name from OpenAI docs

**Workarounds:**
1. Use `gpt-4o` instead (very similar quality)
2. Use `claude-3-5-sonnet` or `claude-opus-4-1` for comparable quality
3. Report the issue with debug script output

**If the debug script works but CLI doesn't:**
- The response parsing has been improved with 5 fallback methods
- Run with `--verbose` flag to see detailed extraction attempts
- Check the console output for extraction method used

## 💡 Tips

1. **Start with GPT-4o** - It's the default and well-tested
2. **Try Claude for variety** - Often gives different (sometimes better) results
3. **Use Gemini for bulk** - Most cost-effective for large batches
4. **Test before committing** - Use `--max-pages 1` to test a model first
5. **Mix and match** - Use different models for different books/quality levels
6. **Check results** - Always use `--save-crops` to verify accuracy
7. **Consider consensus** - Use multiple models with consensus for critical positions

## 📈 Future Enhancements

Planned features:
- [ ] Automatic model selection based on image quality
- [ ] Ensemble methods (combine multiple models)
- [ ] Model-specific prompt optimization
- [ ] Cost tracking and reporting
- [ ] Performance benchmarking tools

## 🔗 Related Documentation

- [Main README](README.md) - General usage
- [Accuracy Improvements](ACCURACY_IMPROVEMENTS.md) - Improving FEN quality
- [CLI Guide](CLI_GUIDE.md) - Command-line usage
- [Getting Started](GETTING_STARTED.md) - Quick start guide

