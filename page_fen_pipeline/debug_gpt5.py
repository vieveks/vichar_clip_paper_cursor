"""
Debug script for GPT-5 response parsing
Use this to test GPT-5 and see the exact response structure.
"""

import sys
import os
import cv2
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fen_generator import numpy_array_to_data_uri
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def test_gpt5_response(image_path):
    """Test GPT-5 response structure with a chess board image."""
    
    print("\n" + "="*70)
    print(" GPT-5 RESPONSE DEBUGGING")
    print("="*70)
    
    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment")
        return
    
    print(f"\n✅ API Key found: {api_key[:10]}...")
    
    # Load image
    print(f"\n📸 Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load image")
        return
    
    print(f"✅ Image loaded: {img.shape[1]}x{img.shape[0]}px")
    
    # Convert to data URI
    print(f"\n🔄 Converting to data URI...")
    data_uri = numpy_array_to_data_uri(img)
    print(f"✅ Data URI created ({len(data_uri)} bytes)")
    
    # Create client
    print(f"\n🤖 Creating OpenAI client...")
    client = OpenAI(api_key=api_key)
    print(f"✅ Client created")
    
    # Prepare prompt
    prompt = "Analyze this chess board and return the FEN notation."
    
    print(f"\n📤 Sending request to GPT-5...")
    print(f"   Model: gpt-5")
    print(f"   Prompt: {prompt}")
    
    # Create request
    user_block = {
        "role": "user",
        "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": data_uri},
        ],
    }
    
    try:
        response = client.responses.create(
            model="gpt-5",
            input=[user_block],
            max_output_tokens=4000,  # Increased even more for GPT-5's heavy reasoning
            reasoning={
                "effort": "low"  # Reduce reasoning effort to save tokens for output
            }
        )
        
        print(f"\n✅ Response received!")
        
        # Check status
        status = getattr(response, "status", "unknown")
        print(f"\n📊 Response Status: {status}")
        
        # Check reasoning effort
        reasoning = getattr(response, "reasoning", None)
        if reasoning:
            effort = getattr(reasoning, "effort", "unknown")
            print(f"🧠 Reasoning Effort: {effort}")
        
        if status == "incomplete":
            incomplete_details = getattr(response, "incomplete_details", None)
            if incomplete_details:
                reason = getattr(incomplete_details, "reason", "unknown")
                print(f"⚠️  Incomplete reason: {reason}")
                if reason == "max_output_tokens":
                    print(f"💡 GPT-5 used all {4000} tokens for reasoning!")
                    print(f"💡 Even with 'low' effort setting, GPT-5 is thinking too much!")
        
        # Inspect response structure
        print(f"\n" + "="*70)
        print(" RESPONSE STRUCTURE ANALYSIS")
        print("="*70)
        
        print(f"\n1. Response Type: {type(response)}")
        print(f"2. Response Class: {response.__class__.__name__}")
        print(f"3. Response Status: {status}")
        
        print(f"\n3. Available Attributes:")
        attrs = [a for a in dir(response) if not a.startswith('_')]
        for attr in attrs:
            print(f"   - {attr}")
        
        print(f"\n4. Testing Extraction Methods:")
        
        # Method 1: output_text
        output_text = getattr(response, "output_text", None)
        print(f"   Method 1 (output_text): {output_text if output_text else 'None'}")
        
        # Method 2: output (check for text vs reasoning blocks)
        output = getattr(response, "output", None)
        print(f"   Method 2 (output): {type(output) if output else 'None'}")
        if output:
            if isinstance(output, list):
                print(f"      - Is list with {len(output)} items")
                output_types = []
                for i, item in enumerate(output):
                    item_type = getattr(item, "type", "unknown")
                    output_types.append(item_type)
                    print(f"      - Item {i}: type={item_type}, class={type(item).__name__}")
                    
                    # Check for text content in each item
                    if item_type == "text":
                        content = getattr(item, "content", None)
                        text_attr = getattr(item, "text", None)
                        print(f"        - Has content: {content is not None}")
                        print(f"        - Has text attr: {text_attr is not None}")
                        if text_attr:
                            print(f"        - Text: {str(text_attr)[:100]}...")
                    elif item_type == "reasoning":
                        print(f"        - (Reasoning block - no text output)")
                
                print(f"      - Output types found: {output_types}")
                if "reasoning" in output_types and "text" not in output_types:
                    print(f"      ⚠️  Only reasoning blocks, no text blocks!")
            else:
                print(f"      - Value: {output}")
        
        # Method 3: text
        text = getattr(response, "text", None)
        print(f"   Method 3 (text): {text if text else 'None'}")
        
        # Method 4: content
        content = getattr(response, "content", None)
        print(f"   Method 4 (content): {type(content) if content else 'None'}")
        if content:
            print(f"      - Value: {content}")
        
        # Method 5: model_dump
        if hasattr(response, "model_dump"):
            print(f"\n5. Response Dump (model_dump):")
            try:
                dump = response.model_dump()
                print(json.dumps(dump, indent=2)[:1000])
                if len(json.dumps(dump)) > 1000:
                    print("   ... (truncated)")
            except Exception as e:
                print(f"   Error dumping: {e}")
        
        # Method 6: Try to print raw response
        print(f"\n6. Raw Response (str):")
        print(f"   {str(response)[:500]}")
        
        # Try to extract text using all methods
        print(f"\n" + "="*70)
        print(" ATTEMPTING TEXT EXTRACTION")
        print("="*70)
        
        extracted_text = None
        
        # Try all our methods
        if hasattr(response, "output_text") and response.output_text:
            extracted_text = response.output_text
            print(f"\n✅ Extracted via output_text:")
            print(f"   {extracted_text}")
        
        if not extracted_text and hasattr(response, "output"):
            output = response.output
            if isinstance(output, list) and len(output) > 0:
                for block in output:
                    if hasattr(block, "content"):
                        content = block.content
                        if isinstance(content, list):
                            for c in content:
                                if hasattr(c, "type") and c.type in ("output_text", "text"):
                                    if hasattr(c, "text"):
                                        extracted_text = c.text
                                        print(f"\n✅ Extracted via output[].content[]:")
                                        print(f"   {extracted_text}")
                                        break
        
        if not extracted_text:
            print(f"\n❌ Could not extract text from response")
            print(f"\n💡 Suggestions:")
            print(f"   1. Check if you have access to GPT-5")
            print(f"   2. Try updating openai package: pip install --upgrade openai")
            print(f"   3. Check OpenAI API documentation for GPT-5 response format")
            print(f"   4. File an issue with the response structure shown above")
        
        print(f"\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error calling GPT-5: {e}")
        import traceback
        traceback.print_exc()


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_gpt5.py <image_path>")
        print("\nExample:")
        print("  python debug_gpt5.py test_boards/page_008/board_1.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    
    test_gpt5_response(image_path)


if __name__ == "__main__":
    main()

