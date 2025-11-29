# GPT-5 Debugging Guide

## Issue

GPT-5 returns `"[No text returned]"` instead of FEN notation when processing boards.

## What We've Implemented

### 1. Enhanced Response Parsing (5 Methods)

Updated `model_providers.py` with multiple fallback extraction methods:

1. **Direct `output_text` attribute** - Standard method
2. **Parse `output` blocks** - Iterate through response structure
3. **Try `response.text`** - Alternative attribute
4. **Try `response.content`** - Content attribute
5. **Parse `model_dump()`** - Full response dictionary parsing

### 2. Detailed Debug Output

When extraction fails, you'll now see:
- Warning message
- Response type and attributes
- Response dump (first 500 chars)
- Which extraction methods were attempted

### 3. Debugging Script

Created `debug_gpt5.py` to inspect GPT-5 responses in detail.

## How to Debug

### Step 1: Run the Debug Script

```bash
cd page_fen_pipeline

# Test with a board image
python debug_gpt5.py book2/page_006/board_1.png

# Or with test boards
python debug_gpt5.py test_boards/page_008/board_1.png
```

**This will show:**
- ✅ Whether API key is valid
- ✅ Response structure details
- ✅ All available attributes
- ✅ What extraction methods return
- ✅ The actual response content

### Step 2: Check the Output

The debug script will show exactly what GPT-5 returns. Look for:

**Success indicators:**
```
✅ Extracted via output_text:
   rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```

**Failure indicators:**
```
❌ Could not extract text from response
```

### Step 3: Try the CLI Again

```bash
# Run with verbose to see extraction attempts
python cli.py ../book2.pdf --max-pages 1 --model gpt-5 --verbose
```

Look for new debug output like:
```
⚠️  Warning: Could not extract text from GPT-5 response
Response type: <class 'openai.types...'>
Response attributes: [...list of attributes...]
Response dump: {...response structure...}
```

## Common Issues & Solutions

### Issue 1: "No GPT-5 Access"

**Symptom:** Error about model not available or access denied

**Solution:**
1. Check your OpenAI account tier at platform.openai.com
2. GPT-5 may require a specific access level
3. **Workaround:** Use `gpt-4o` instead:
   ```bash
   python cli.py ../book2.pdf --max-pages 6 --model gpt-4o
   ```

### Issue 2: "API Format Changed"

**Symptom:** Response structure doesn't match any extraction method

**Solution:**
1. Update OpenAI package:
   ```bash
   pip install --upgrade openai
   ```
2. Share the debug script output so we can add the new format
3. **Workaround:** Use alternative models:
   ```bash
   # Anthropic Claude (excellent quality)
   python cli.py ../book2.pdf --max-pages 6 --model claude-3-5-sonnet
   
   # Claude 4.1 Opus (highest quality)
   python cli.py ../book2.pdf --max-pages 6 --model claude-4.1-opus
   ```

### Issue 3: "Empty Response"

**Symptom:** Response received but contains no text

**Possible causes:**
1. Image too large or corrupted
2. API timeout
3. Rate limiting

**Solution:**
1. Try with a smaller page range:
   ```bash
   python cli.py ../book2.pdf --max-pages 1 --model gpt-5
   ```
2. Try different DPI:
   ```bash
   python cli.py ../book2.pdf --max-pages 1 --model gpt-5 --dpi 200
   ```
3. Check API status at status.openai.com

## Alternative Models (Recommended)

While we debug GPT-5, these models work excellently:

### OpenAI GPT-4o (Recommended)
```bash
python cli.py ../book2.pdf --max-pages 6 --model gpt-4o \
  --output-dir book2 --save-crops --output-json results_gpt4o.json
```
- ✅ Proven stable
- ✅ Excellent quality
- ✅ Similar to GPT-5

### Anthropic Claude 3.5 Sonnet
```bash
python cli.py ../book2.pdf --max-pages 6 --model claude-3-5-sonnet \
  --output-dir book2 --save-crops --output-json results_claude.json
```
- ✅ Often better than GPT-4o
- ✅ Very reliable
- ✅ Competitive pricing

### Anthropic Claude 4.1 Opus (Premium)
```bash
python cli.py ../book2.pdf --max-pages 6 --model claude-4.1-opus \
  --output-dir book2 --save-crops --output-json results_claude_opus.json
```
- ✅ Highest quality available
- ✅ Best for critical positions
- 💰 More expensive

### Google Gemini 2.5 Pro
```bash
python cli.py ../book2.pdf --max-pages 6 --model gemini-2.5-pro \
  --output-dir book2 --save-crops --output-json results_gemini.json
```
- ✅ Good quality
- ✅ Cost-effective
- ✅ Fast processing

## Next Steps

1. **Run the debug script** to see what GPT-5 actually returns
2. **Share the output** if you need help (paste debug script results)
3. **Try alternative models** for immediate results
4. **Check for updates** - We'll fix GPT-5 as soon as we understand the response format

## Debug Script Options

```bash
# Basic usage
python debug_gpt5.py <image_path>

# Examples
python debug_gpt5.py book2/page_006/board_1.png
python debug_gpt5.py test_boards/page_008/board_1.png
python debug_gpt5.py ../chess_board.png
```

The script is safe and will:
- Show all response details
- Not modify any files
- Help us understand the GPT-5 response format

## Contributing

If you find the issue:
1. Run `python debug_gpt5.py <image>`
2. Copy the output
3. Share it in an issue/message
4. We'll add the correct parsing method

## Success Stories

Report back if you get GPT-5 working! We want to know:
- Which extraction method worked
- Your OpenAI package version (`pip show openai`)
- Any special configuration

This helps improve the tool for everyone! 🎯

