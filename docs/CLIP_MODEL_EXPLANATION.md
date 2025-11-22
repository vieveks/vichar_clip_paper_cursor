# CLIP Model Architecture Explanation

## What is CLIP?

**CLIP (Contrastive Language-Image Pre-Training)** is a neural network model that learns to understand images and text together. It was developed by OpenAI and can match images with their corresponding text descriptions.

## CLIP Architecture Components

CLIP consists of **two main parts**:

### 1. **Image Encoder (Vision Encoder)**
- **What we're using**: **ViT-B-32** (Vision Transformer Base with 32×32 patches)
- **What it does**: Takes a chess board image and converts it into a 512-dimensional vector (embedding)
- **ViT-B-32 means**:
  - **ViT** = Vision Transformer (a type of neural network architecture)
  - **B** = Base size (medium-sized model, good balance of speed and accuracy)
  - **32** = 32×32 pixel patches (how the image is divided for processing)

### 2. **Text Encoder**
- **What it does**: Takes a FEN string (text) and converts it into a 512-dimensional vector (embedding)
- **Architecture**: Transformer-based text encoder
- **Output**: Same 512-dimensional space as the image encoder

## How CLIP Works

1. **Image → Vector**: Chess board image → 512-dim vector
2. **Text → Vector**: FEN string → 512-dim vector  
3. **Compare**: Calculate similarity between image and text vectors
4. **Match**: The correct FEN should have high similarity with its corresponding image

## What We're Training

**We're fine-tuning the ENTIRE CLIP model**, which means:
- ✅ Training the **image encoder** (ViT-B-32) to better understand chess positions
- ✅ Training the **text encoder** to better understand FEN notation
- ✅ Training both to work together in the same embedding space

**We're NOT training from scratch** - we start with:
- **Pretrained weights**: `laion2B-s34B-b79K` (trained on 2 billion image-text pairs from LAION dataset)
- **Then fine-tune**: On our chess-specific dataset (125k chess positions)

## Model Specifications

- **Model Name**: `ViT-B-32`
- **Pretrained**: `laion2B-s34B-b79K` (LAION-2B dataset)
- **Embedding Dimension**: 512 (both image and text)
- **Image Input Size**: 224×224 pixels (resized from original)
- **Text Input**: FEN strings (up to 77 tokens)

## Why ViT-B-32?

- **Good balance**: Not too small (would underperform), not too large (would be slow)
- **Efficient**: Works well on consumer GPUs like RTX 5070 Ti
- **Proven**: Widely used and well-tested architecture
- **Suitable for chess**: Base model is sufficient for chess position recognition

## RTX 5070 Ti Compatibility

✅ **Yes, your RTX 5070 Ti will work perfectly!**

- **Memory**: RTX 5070 Ti has 16GB VRAM (more than enough)
- **CUDA**: Full CUDA support for PyTorch
- **Performance**: Can handle batch size 128 with FP16 mixed precision
- **Training Speed**: Should train at good speed with mixed precision enabled

## Training Configuration

- **Batch Size**: 128 (fits comfortably in 16GB VRAM)
- **Mixed Precision (FP16)**: Enabled for faster training and less memory usage
- **Model Size**: ~150M parameters (ViT-B-32 is relatively small)
- **Memory Usage**: ~4-6GB VRAM during training

## Summary

- **ViT-B-32** = The vision/image encoder part of CLIP
- **CLIP** = Both image encoder + text encoder working together
- **We're training**: The entire CLIP model (both encoders) on chess data
- **Your GPU**: RTX 5070 Ti is perfect for this task!

