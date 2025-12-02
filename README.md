# KNIFT-GPU: Mobile-Optimized Feature Descriptor

GPU-compatible reimplementation of MediaPipe's KNIFT (Keypoint Neural Invariant Feature Transform) feature descriptor, designed for efficient on-device inference with TensorFlow Lite GPU delegates.

## Overview

This project rebuilds the KNIFT feature descriptor from scratch to eliminate GPU delegate incompatibilities present in the original MediaPipe implementation. While maintaining robust matching performance across viewpoint and illumination changes, KNIFT-GPU removes problematic operations (L2 normalization, dynamic reshapes) that prevent GPU acceleration on mobile devices.

**Original KNIFT**: [MediaPipe KNIFT](https://developers.googleblog.com/mediapipe-knift-template-based-feature-matching/) by Google AI Edge team
**MediaPipe Framework**: Lugaresi et al., "MediaPipe: A Framework for Building Perception Pipelines" ([arXiv:1906.08172](https://arxiv.org/abs/1906.08172))

## Key Differences from Original KNIFT

| Feature | Original KNIFT | KNIFT-GPU |
|---------|---------------|-----------|
| **Architecture** | Inception-style CNN | Inception-style CNN (same) |
| **Output** | 40-dim embeddings | 40-dim embeddings |
| **L2 Normalization** | ✅ Yes (in model) | ❌ No (deferred to matching) |
| **Reshape Operations** | Dynamic (-1 dims) | Fixed dimensions only |
| **GPU Delegate** | ❌ Falls back to CPU | ✅ Full GPU support |
| **Training Data** | Proprietary | Custom dataset (20 images) |
| **Batch Sizes** | 200, 400, 1000 | 200, 400, 1000 |

## Architecture

Lightweight Inception-based CNN (~100K parameters):
- **Input**: 32×32 grayscale patches
- **Output**: 40-dimensional float32 embeddings (unnormalized)
- **Operations**: CONV_2D, MAX_POOL_2D, AVERAGE_POOL_2D, CONCATENATION, BATCH_NORM, DENSE, RELU
- **GPU Compatible**: All operations supported by TFLite GPU delegate

## Features

- Real-time loss and accuracy tracking during training
- Alignment task visualization during training
- Automatic model checkpointing
- Final TFLite export with batch sizes optimized for mobile GPU inference
- GPU-compatible architecture (no L2 normalization, fixed shapes only)
- Batch-hard triplet loss with in-batch semi-hard negative mining
- Multi-scale alignment testing suite

## Requirements

- Python 3.7+
- TensorFlow
- OpenCV
- NumPy
- Pillow
- Matplotlib (for visualization)
- Tkinter (for alignment test suite)

## Usage

### Training
```bash
python main.py
```

### Alignment Testing
```bash
python align_test.py
```

## Project Structure

- `main.py` - Main training script with GPU-compatible architecture
- `align_test.py` - Alignment testing suite for evaluating model performance
- `README.md` - Project documentation
- `LICENSE` - License information
- `Set14/` - Training dataset directory (user-provided)

## Training Configuration

- Training images: Set14 (13 images excluding baboon)
- Validation image: baboon.jpeg from Set14
- Epochs: 250
- Batch size: 32
- Steps per epoch: 20
- Total patches: ~480k

## Model Export

The trained model is exported to TensorFlow Lite format with multiple batch sizes:
- `knift_gpu_float.tflite` (batch size 200)
- `knift_gpu_float_400.tflite` (batch size 400)
- `knift_gpu_float_1k.tflite` (batch size 1000)
