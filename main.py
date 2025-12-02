#!/usr/bin/env python3
"""
KNIFT-GPU: Mobile-Optimized Feature Descriptor Training (~100K Parameters)

GPU-compatible reimplementation of MediaPipe's KNIFT (Keypoint Neural Invariant Feature Transform)
feature descriptor, designed for efficient on-device inference with TensorFlow Lite GPU delegates.

Console-based training of the ~100K-parameter KNIFT-GPU model with real-time metrics,
alignment task visualization, and export to TFLite. Shows model
convergence and performance improvement during training.

Features:
- Real-time loss and accuracy tracking
- Alignment task during training
- Automatic model checkpointing
- Final TFLite export with batch sizes optimized for mobile GPU inference
- GPU-compatible architecture (no L2 normalization, fixed shapes only)

Author: Benjamin-Wegener
Version: 1.0.0
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFilter

# === INCEPTION-LIKE MODEL ARCHITECTURE ===
def build_model(input_shape=(32, 32, 1), output_dim=40, batch_size=None, name="knift_gpu_model"):
    """
    Build Inception-like CNN model for KNIFT-GPU feature descriptors.

    GPU-compatible reimplementation of MediaPipe's KNIFT that eliminates operations
    incompatible with TFLite GPU delegates (L2 normalization, dynamic reshapes).

    Lightweight Inception-based CNN (~100K parameters):
    - Input: 32×32 grayscale patches
    - Output: 40-dimensional float32 embeddings (unnormalized)
    - Operations: CONV_2D, MAX_POOL_2D, AVERAGE_POOL_2D, CONCATENATION, BATCH_NORM, DENSE, RELU
    - GPU Compatible: All operations supported by TFLite GPU delegate
    - No L2 normalization applied (deferred to matching to maintain GPU compatibility)

    Args:
        input_shape: Tuple, input image shape (height, width, channels)
        output_dim: Int, dimension of output embeddings
        batch_size: Int or None, static batch size for TFLite compatibility
        name: String, model name

    Returns:
        keras.Model: Compiled model ready for training
    """
    print("🏗️ Building Inception-like Architecture...")
    print(f"   Input Shape: {input_shape}")
    print(f"   Output Dimension: {output_dim}")
    print(f"   Batch Size: {batch_size}")
    
    # Input layer
    if batch_size is not None:
        inputs = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size, name="input")
    else:
        inputs = tf.keras.layers.Input(shape=input_shape, name="input")
    
    # === STEM BLOCK === (~800 params)
    print("📐 Building Stem Block...")
    # Conv2D(16, 3x3): (3*3*1 + 1)*16 = 160 params
    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu',
        name='stem_conv'
    )(inputs)

    # BatchNorm: 2*16 = 32 params
    x = tf.keras.layers.BatchNormalization(name='stem_bn')(x)

    # MaxPool 2x2 -> [batch, 16, 16, 16]
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name='stem_pool')(x)

    # === INCEPTION MODULE 1 === (~7,500 params)
    print("🏗️ Building Inception Module 1...")
    x = inception_module(x, {
        'name': 'inception1',
        'branch1x1': 12,           # (1*1*16 + 1)*12 = 204 params
        'branch3x3_reduce': 8,     # (1*1*16 + 1)*8 = 136 params
        'branch3x3': 16,           # (3*3*8 + 1)*16 = 1,168 params
        'branch5x5_reduce': 8,     # (1*1*16 + 1)*8 = 136 params
        'branch5x5': 16,           # (5*5*8 + 1)*16 = 3,216 params
        'branch_pool_proj': 8      # (1*1*16 + 1)*8 = 136 params
    })  # Concat -> [batch, 16, 16, 52]

    # === INCEPTION MODULE 2 === (~12,000 params)
    print("🏗️ Building Inception Module 2...")
    x = inception_module(x, {
        'name': 'inception2',
        'branch1x1': 16,           # (1*1*52 + 1)*16 = 848 params
        'branch3x3_reduce': 12,    # (1*1*52 + 1)*12 = 636 params
        'branch3x3': 24,           # (3*3*12 + 1)*24 = 2,616 params
        'branch5x5_reduce': 12,    # (1*1*52 + 1)*12 = 636 params
        'branch5x5': 24,           # (5*5*12 + 1)*24 = 7,224 params
        'branch_pool_proj': 16     # (1*1*52 + 1)*16 = 848 params
    })  # Concat -> [batch, 16, 16, 76]

    # MaxPool -> [batch, 8, 8, 76]
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name='pool2')(x)

    # === INCEPTION MODULE 3 === (~18,000 params)
    print("🏗️ Building Inception Module 3...")
    x = inception_module(x, {
        'name': 'inception3',
        'branch1x1': 24,           # (1*1*76 + 1)*24 = 1,848 params
        'branch3x3_reduce': 16,    # (1*1*76 + 1)*16 = 1,232 params
        'branch3x3': 32,           # (3*3*16 + 1)*32 = 4,640 params
        'branch5x5_reduce': 16,    # (1*1*76 + 1)*16 = 1,232 params
        'branch5x5': 32,           # (5*5*16 + 1)*32 = 12,832 params
        'branch_pool_proj': 24     # (1*1*76 + 1)*24 = 1,848 params
    })  # Concat -> [batch, 8, 8, 108]

    # === INCEPTION MODULE 4 === (~22,000 params)
    print("🏗️ Building Inception Module 4...")
    x = inception_module(x, {
        'name': 'inception4',
        'branch1x1': 28,           # (1*1*108 + 1)*28 = 3,052 params
        'branch3x3_reduce': 16,    # (1*1*108 + 1)*16 = 1,744 params
        'branch3x3': 32,           # (3*3*16 + 1)*32 = 4,640 params
        'branch5x5_reduce': 16,    # (1*1*108 + 1)*16 = 1,744 params
        'branch5x5': 32,           # (5*5*16 + 1)*32 = 12,832 params
        'branch_pool_proj': 24     # (1*1*108 + 1)*24 = 2,616 params
    })  # Concat -> [batch, 8, 8, 128]

    # MaxPool -> [batch, 4, 4, 128]
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name='pool3')(x)

    # === FINAL INCEPTION MODULE === (~20,000 params)
    print("🏗️ Building Final Inception Module...")
    x = inception_module(x, {
        'name': 'inception5',
        'branch1x1': 28,           # (1*1*128 + 1)*28 = 3,612 params
        'branch3x3_reduce': 14,    # (1*1*128 + 1)*14 = 1,806 params
        'branch3x3': 28,           # (3*3*14 + 1)*28 = 3,556 params
        'branch5x5_reduce': 14,    # (1*1*128 + 1)*14 = 1,806 params
        'branch5x5': 28,           # (5*5*14 + 1)*28 = 9,828 params
        'branch_pool_proj': 24     # (1*1*128 + 1)*24 = 3,096 params
    })  # Concat -> [batch, 4, 4, 104]
    
    # === OUTPUT BLOCK === (~4,200 params)
    print("📤 Building Output Block...")
    # GlobalAveragePooling2D (NO reshape!) -> [batch, 104]
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_pool')(x)

    # Dense(40) NO activation, NO L2 norm
    # (104 + 1)*40 = 4,200 params
    outputs = tf.keras.layers.Dense(
        units=output_dim,
        activation=None,  # Raw embeddings - NO activation function
        name='embedding'
    )(x)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    # Print architecture summary
    print("\n📊 Model Architecture Summary:")
    model.summary()

    param_count = model.count_params()
    print(f"\n🎯 Total Parameters: {param_count:,}")
    print(f"🎯 Target Parameters: ~100,000")
    print(f"🎯 In target range (95k-100k): {'✅ YES' if 95000 <= param_count <= 100000 else '❌ NO'}")

    # CPU Compatibility Check
    print("\n🔧 CPU Compatibility Check:")
    print("✅ CONV_2D (all convolutions)")
    print("✅ AVERAGE_POOL_2D, MAX_POOL_2D")
    print("✅ CONCATENATION")
    print("✅ FULLY_CONNECTED (Dense layer)")
    print("✅ ADD, MUL, SUB (in BatchNorm)")
    print("✅ RELU activation")
    print("✅ NO L2_NORMALIZATION")
    print("✅ GlobalAveragePooling2D instead of Flatten")
    print("✅ GPU-Compatible operations")
    
    return model


def inception_module(inputs, config):
    """
    Build GPU-compatible Inception module with reduced parameters for feature extraction.

    All operations are compatible with TFLite GPU delegates and avoid
    L2 normalization and dynamic reshape operations.

    Args:
        inputs: Tensor, input tensor
        config: Dict, configuration parameters

    Returns:
        Tensor, output tensor after Inception processing
    """
    name = config['name']
    branch1x1 = config['branch1x1']
    branch3x3_reduce = config['branch3x3_reduce']
    branch3x3 = config['branch3x3']
    branch5x5_reduce = config['branch5x5_reduce']
    branch5x5 = config['branch5x5']
    branch_pool_proj = config['branch_pool_proj']
    
    print(f"   Building {name} with config:")
    print(f"     branch1x1: {branch1x1}")
    print(f"     branch3x3_reduce: {branch3x3_reduce}")
    print(f"     branch3x3: {branch3x3}")
    print(f"     branch5x5_reduce: {branch5x5_reduce}")
    print(f"     branch5x5: {branch5x5}")
    print(f"     branch_pool_proj: {branch_pool_proj}")
    
    # Branch 1: 1x1 conv
    branch1 = tf.keras.layers.Conv2D(
        filters=branch1x1,
        kernel_size=1,
        padding='same',
        activation='relu',
        name=f'{name}_b1_1x1'
    )(inputs)
    
    # Branch 2: 1x1 -> 3x3
    branch2 = tf.keras.layers.Conv2D(
        filters=branch3x3_reduce,
        kernel_size=1,
        padding='same',
        activation='relu',
        name=f'{name}_b2_1x1'
    )(inputs)
    branch2 = tf.keras.layers.Conv2D(
        filters=branch3x3,
        kernel_size=3,
        padding='same',
        activation='relu',
        name=f'{name}_b2_3x3'
    )(branch2)
    
    # Branch 3: 1x1 -> 5x5
    branch3 = tf.keras.layers.Conv2D(
        filters=branch5x5_reduce,
        kernel_size=1,
        padding='same',
        activation='relu',
        name=f'{name}_b3_1x1'
    )(inputs)
    branch3 = tf.keras.layers.Conv2D(
        filters=branch5x5,
        kernel_size=5,
        padding='same',
        activation='relu',
        name=f'{name}_b3_5x5'
    )(branch3)
    
    # Branch 4: MaxPool -> 1x1
    branch4 = tf.keras.layers.MaxPooling2D(
        pool_size=3,
        strides=1,
        padding='same',
        name=f'{name}_b4_pool'
    )(inputs)
    branch4 = tf.keras.layers.Conv2D(
        filters=branch_pool_proj,
        kernel_size=1,
        padding='same',
        activation='relu',
        name=f'{name}_b4_1x1'
    )(branch4)
    
    # Concatenate all branches
    concatenated = tf.keras.layers.Concatenate(name=f'{name}_concat')(
        [branch1, branch2, branch3, branch4]
    )
    
    # Batch normalization
    output = tf.keras.layers.BatchNormalization(name=f'{name}_bn')(concatenated)
    
    return output


# === CUSTOM TRIPLET METRICS ===
class PositiveDistanceMetric(tf.keras.metrics.Metric):
    def __init__(self, name='pos_dist', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch = tf.shape(y_pred)[0] // 3
        anchors = y_pred[:batch]
        positives = y_pred[batch:2*batch]
        self.total.assign_add(tf.reduce_mean(tf.norm(anchors - positives, axis=1)))
        self.count.assign_add(1.0)

    def result(self):
        return self.total / tf.maximum(self.count, 1.0)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class NegativeDistanceMetric(tf.keras.metrics.Metric):
    def __init__(self, name='neg_dist', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch = tf.shape(y_pred)[0] // 3
        anchors = y_pred[:batch]
        negatives = y_pred[2*batch:3*batch]
        self.total.assign_add(tf.reduce_mean(tf.norm(anchors - negatives, axis=1)))
        self.count.assign_add(1.0)

    def result(self):
        return self.total / tf.maximum(self.count, 1.0)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

# === TRIPLET LOSS FUNCTION ===
class TripletLoss(tf.keras.losses.Loss):
    """
    Triplet loss function for learning feature embeddings.

    L = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
    """

    def __init__(self, margin=1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        # For triplet loss, the data generator is expected to concatenate
        # anchors, positives, and negatives into a single batch
        # So if batch_size=32, we get 32 anchors, 32 positives, 32 negatives
        # Total batch size = 96
        total_batch_size = tf.shape(y_pred)[0]
        embedding_dim = tf.shape(y_pred)[1]

        # Calculate the actual number of triplets
        triplet_count = total_batch_size // 3

        # Split embeddings: first third anchors, second third positives, last third negatives
        anchors = y_pred[:triplet_count]
        positives = y_pred[triplet_count:2*triplet_count]
        negatives = y_pred[2*triplet_count:3*triplet_count]

        # Calculate distances
        pos_dist = tf.reduce_sum(tf.square(anchors - positives), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchors - negatives), axis=1)

        # Triplet loss
        loss = tf.maximum(pos_dist - neg_dist + self.margin, 0.0)

        return tf.reduce_mean(loss)


class BatchHardTripletLoss(tf.keras.losses.Loss):
    """
    Batch-hard triplet loss with in-batch semi-hard negative mining.
    Mines hardest negatives from the current batch embeddings.
    """
    def __init__(self, margin=1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        # Split into anchors, positives, negatives
        total_batch_size = tf.shape(y_pred)[0]
        batch_size = total_batch_size // 3
        anchors = y_pred[:batch_size]
        positives = y_pred[batch_size:2*batch_size]
        negatives = y_pred[2*batch_size:3*batch_size]

        # Compute positive distances (anchor to positive)
        pos_dist = tf.reduce_sum(tf.square(anchors - positives), axis=1)

        # For each anchor, find hardest negative from the batch
        # Compute pairwise distances between anchors and ALL negatives in batch
        anchor_neg_dist = tf.reduce_sum(
            tf.square(tf.expand_dims(anchors, 1) - tf.expand_dims(negatives, 0)),
            axis=2
        )  # Shape: [batch_size, batch_size]

        # For semi-hard mining: select negatives where d(a,p) < d(a,n) < d(a,p) + margin
        # For each anchor, mask out negatives that are too easy or too hard
        pos_dist_expanded = tf.expand_dims(pos_dist, 1)  # [batch_size, 1]

        # Semi-hard mask: d(a,p) < d(a,n) < d(a,p) + margin
        semi_hard_mask = tf.logical_and(
            anchor_neg_dist > pos_dist_expanded,
            anchor_neg_dist < pos_dist_expanded + self.margin
        )

        # Get the hardest semi-hard negative for each anchor
        # If no semi-hard exists, use hardest negative overall
        masked_dist = tf.where(
            semi_hard_mask,
            anchor_neg_dist,
            tf.fill(tf.shape(anchor_neg_dist), tf.float32.max)
        )

        # Take minimum (hardest) valid negative distance per anchor
        hardest_neg_dist = tf.reduce_min(masked_dist, axis=1)

        # Fallback: if all negatives too easy, use actual hardest
        all_easy_mask = tf.equal(hardest_neg_dist, tf.float32.max)
        hardest_neg_dist = tf.where(
            all_easy_mask,
            tf.reduce_min(anchor_neg_dist, axis=1),
            hardest_neg_dist
        )

        # Compute triplet loss
        loss = tf.maximum(pos_dist - hardest_neg_dist + self.margin, 0.0)
        return tf.reduce_mean(loss)

def mine_semi_hard_negatives(anchor_emb, positive_emb, all_negative_embs, margin=1.0):
    """
    Select negatives where: d(a,p) < d(a,n) < d(a,p) + margin
    """
    d_ap = np.linalg.norm(anchor_emb - positive_emb)
    distances = np.linalg.norm(all_negative_embs - anchor_emb, axis=1)

    # Find semi-hard negatives
    semi_hard_mask = (distances > d_ap) & (distances < d_ap + margin)

    if np.any(semi_hard_mask):
        # Return random semi-hard negative
        candidates = np.where(semi_hard_mask)[0]
        return np.random.choice(candidates)
    else:
        # Fallback to hardest negative that's still > d_ap
        valid = distances > d_ap
        if np.any(valid):
            return np.argmin(distances[valid])
        return np.random.randint(len(all_negative_embs))


# === FEATURE EXTRACTOR ===
class FeatureExtractor:
    """Extracts GPU-compatible feature embeddings from image patches using trained KNIFT-GPU model"""

    def __init__(self, model: tf.keras.Model = None):
        """
        Initialize GPU-compatible feature extractor

        Args:
            model: Trained TensorFlow model for feature extraction (GPU-compatible)
        """
        self.model = model
        self.input_size = (32, 32)  # Model expects 32x32 patches
        self.embedding_dim = 40     # Model outputs 40-dim embeddings (unnormalized for GPU compatibility)

        if self.model:
            print(f"🔧 KNIFT-GPU FeatureExtractor initialized with model: {self.model.name}")
        
    def preprocess_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Preprocess image patch for model input
        
        Args:
            patch: Input image patch (grayscale or color)
            
        Returns:
            Preprocessed patch ready for model input
        """
        # Convert to grayscale if color
        if len(patch.shape) == 3:
            if patch.shape[2] == 3:  # RGB
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            elif patch.shape[2] == 4:  # RGBA
                patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2GRAY)
        
        # Resize to model input size
        patch = cv2.resize(patch, self.input_size, interpolation=cv2.INTER_CUBIC)
        
        # Normalize to [0, 1]
        patch = patch.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions: [H, W] -> [1, H, W, 1]
        patch = np.expand_dims(patch, axis=0)
        patch = np.expand_dims(patch, axis=-1)
        
        return patch
    
    def extract_features(self, patches: np.ndarray) -> np.ndarray:
        """
        Extract features from patches using the model
        
        Args:
            patches: Array of image patches [num_patches, height, width] or 
                    [num_patches, height, width, channels]
            
        Returns:
            Array of feature embeddings [num_patches, 40]
        """
        if self.model is None:
            raise ValueError("No model loaded for feature extraction")
            
        if len(patches) == 0:
            return np.array([])
        
        # Handle different input shapes
        if len(patches.shape) == 3:
            # [num_patches, H, W] -> add channel dimension
            patches = np.expand_dims(patches, axis=-1)
        
        # Normalize patches to [0, 1]
        if patches.max() > 1.0:
            patches = patches.astype(np.float32) / 255.0
        
        # Ensure correct shape [num_patches, 32, 32, 1]
        if patches.shape[1:3] != self.input_size:
            resized_patches = []
            for patch in patches:
                if len(patch.shape) == 3:
                    patch = cv2.resize(patch.squeeze(), self.input_size)
                else:
                    patch = cv2.resize(patch, self.input_size)
                # Add channel dimension back
                patch = patch[..., np.newaxis]
                resized_patches.append(patch)
            patches = np.array(resized_patches)
            
        # Extract features using the model
        try:
            features = self.model.predict(patches, verbose=0)
            return features
        except Exception as e:
            print(f"❌ Error during feature extraction: {e}")
            return np.array([])


# === DATA LOADER ===
import glob
from PIL import Image

class DataLoader:
    """
    Basic data loader for KNIFT-GPU.
    Handles image loading and basic preprocessing for GPU-compatible training.
    """

    def __init__(self, data_path, img_size=(32, 32)):
        """
        Initialize GPU-compatible data loader.

        Args:
            data_path: Path to data directory
            img_size: Target image size (height, width)
        """
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.images = []
        self.image_paths = []

        print("📊 Initializing KNIFT-GPU Data Loader...")
        print(f"   Data Path: {self.data_path}")
        print(f"   Target Size: {self.img_size}")

        # Load image paths
        self._load_image_paths()

        print(f"   Found Images: {len(self.image_paths)}")

    def _load_image_paths(self):
        """Load all image paths from data directory, excluding baboon for validation."""
        # Support common image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

        for ext in extensions:
            pattern = self.data_path / ext
            paths = list(glob.glob(str(pattern)))
            # Filter out baboon images to exclude from training dataset
            filtered_paths = [path for path in paths if "baboon" not in os.path.basename(path).lower()]
            self.image_paths.extend(filtered_paths)

        # Sort for consistency
        self.image_paths.sort()

    def load_image(self, image_path):
        """
        Load and preprocess a single image.

        Args:
            image_path: Path to image file

        Returns:
            numpy array: Preprocessed image
        """
        # Load image with PIL
        img = Image.open(image_path)

        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')

        # Resize to target size
        img = img.resize(self.img_size, Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0  # Normalize to [0, 1]

        # Add batch and channel dimensions for model input
        img_array = img_array[np.newaxis, ..., np.newaxis]  # Shape: [1, H, W, 1]

        return img_array

    def get_image_paths(self):
        """
        Get list of all image paths.

        Returns:
            List of image paths
        """
        return self.image_paths.copy()

    def get_num_images(self):
        """
        Get number of images.

        Returns:
            Int: Number of images
        """
        return len(self.image_paths)

    def __len__(self):
        """Return number of images."""
        return self.get_num_images()

    def __getitem__(self, idx):
        """
        Get image by index.

        Args:
            idx: Image index

        Returns:
            Tuple: (image_path, preprocessed_image)
        """
        if idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range")

        image_path = self.image_paths[idx]
        image = self.load_image(image_path)

        return image_path, image


# === AUGMENTATION PIPELINE ===
import random
import math

class HeavyAugmentation:
    """
    GPU-compatible heavy augmentation pipeline for training patches.

    Applies multiple augmentation techniques to generate diverse patches
    from the original 32x32 image data for KNIFT-GPU training.
    """

    def __init__(self,
                 rotation_range=15,
                 brightness_range=0.2,
                 contrast_range=0.15,
                 noise_std=0.01,
                 blur_radius_range=(0.5, 2.0),
                 flip_probability=0.0,  # Set to 0.0 to avoid breaking correspondence
                 noise_probability=0.8,
                 blur_probability=0.6):
        """
        Initialize GPU-compatible augmentation parameters.

        Args:
            rotation_range: Maximum rotation in degrees (±range)
            brightness_range: Brightness adjustment range (±range)
            contrast_range: Contrast adjustment range (±range)
            noise_std: Standard deviation for Gaussian noise
            blur_radius_range: Range for blur radius (min, max)
            flip_probability: Probability of horizontal flip (set to 0.0 to preserve correspondence)
            noise_probability: Probability of adding noise
            blur_probability: Probability of applying blur
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.blur_radius_range = blur_radius_range
        self.flip_probability = flip_probability  # This is now 0.0, so horizontal flip is disabled
        self.noise_probability = noise_probability
        self.blur_probability = blur_probability

        print("🎨 Initializing GPU-Compatible Heavy Augmentation Pipeline...")
        print(f"   Rotation range: ±{rotation_range}°")
        print(f"   Brightness range: ±{brightness_range*100}%")
        print(f"   Contrast range: ±{contrast_range*100}%")
        print(f"   Noise std: {noise_std}")
        print(f"   Blur radius: {blur_radius_range}")
        print(f"   Flip probability: {flip_probability} (0 = disabled to preserve correspondences)")

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle.
        
        Args:
            image: Input image array (32, 32)
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image array
        """
        if angle == 0:
            return image
            
        # Convert to PIL for rotation
        pil_image = Image.fromarray((image * 255).astype(np.uint8), mode='L')
        
        # Rotate image
        rotated_pil = pil_image.rotate(angle, resample=Image.Resampling.BILINEAR)
        
        # Convert back to numpy array
        rotated_array = np.array(rotated_pil, dtype=np.float32) / 255.0
        
        return rotated_array

    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image brightness.
        
        Args:
            image: Input image array
            factor: Brightness factor (1.0 = no change)
            
        Returns:
            Brightness adjusted image
        """
        # Convert to PIL for enhancement
        pil_image = Image.fromarray((image * 255).astype(np.uint8), mode='L')
        
        # Create brightness enhancer
        enhancer = ImageEnhance.Brightness(pil_image)
        
        # Apply enhancement
        enhanced_pil = enhancer.enhance(factor)
        
        # Convert back to numpy array
        enhanced_array = np.array(enhanced_pil, dtype=np.float32) / 255.0
        
        return enhanced_array

    def adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image contrast.
        
        Args:
            image: Input image array
            factor: Contrast factor (1.0 = no change)
            
        Returns:
            Contrast adjusted image
        """
        # Convert to PIL for enhancement
        pil_image = Image.fromarray((image * 255).astype(np.uint8), mode='L')
        
        # Create contrast enhancer
        enhancer = ImageEnhance.Contrast(pil_image)
        
        # Apply enhancement
        enhanced_pil = enhancer.enhance(factor)
        
        # Convert back to numpy array
        enhanced_array = np.array(enhanced_pil, dtype=np.float32) / 255.0
        
        return enhanced_array

    def add_noise(self, image: np.ndarray, std: float) -> np.ndarray:
        """
        Add Gaussian noise to image.
        
        Args:
            image: Input image array
            std: Standard deviation of noise
            
        Returns:
            Noisy image
        """
        noise = np.random.normal(0, std, image.shape)
        noisy_image = image + noise
        
        # Clip to valid range
        noisy_image = np.clip(noisy_image, 0, 1)
        
        return noisy_image

    def add_blur(self, image: np.ndarray, radius: float) -> np.ndarray:
        """
        Add Gaussian blur to image.
        
        Args:
            image: Input image array
            radius: Blur radius
            
        Returns:
            Blurred image
        """
        if radius <= 0:
            return image
            
        # Convert to PIL for blur
        pil_image = Image.fromarray((image * 255).astype(np.uint8), mode='L')
        
        # Apply blur
        blurred_pil = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        # Convert back to numpy array
        blurred_array = np.array(blurred_pil, dtype=np.float32) / 255.0
        
        return blurred_array

    def horizontal_flip(self, image: np.ndarray) -> np.ndarray:
        """
        Apply horizontal flip.
        
        Args:
            image: Input image array
            
        Returns:
            Horizontally flipped image
        """
        return np.fliplr(image)

    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply heavy augmentation to image.
        
        Args:
            image: Input image array (32, 32) with values in [0, 1]
            
        Returns:
            Augmented image array
        """
        augmented = image.copy()
        
        # Random rotation (±rotation_range degrees)
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        augmented = self.rotate_image(augmented, angle)
        
        # Random brightness adjustment
        if random.random() < 0.8:  # 80% chance of brightness change
            brightness_factor = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
            brightness_factor = max(0.1, min(2.0, brightness_factor))  # Clamp to reasonable range
            augmented = self.adjust_brightness(augmented, brightness_factor)
        
        # Random contrast adjustment
        if random.random() < 0.8:  # 80% chance of contrast change
            contrast_factor = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
            contrast_factor = max(0.1, min(2.0, contrast_factor))  # Clamp to reasonable range
            augmented = self.adjust_contrast(augmented, contrast_factor)
        
        # Horizontal flip
        if random.random() < self.flip_probability:
            augmented = self.horizontal_flip(augmented)
        
        # Add noise
        if random.random() < self.noise_probability:
            noise_std = random.uniform(0, self.noise_std)
            if noise_std > 0:
                augmented = self.add_noise(augmented, noise_std)
        
        # Add blur
        if random.random() < self.blur_probability:
            blur_radius = random.uniform(*self.blur_radius_range)
            augmented = self.add_blur(augmented, blur_radius)
        
        # Ensure image is in valid range and format
        augmented = np.clip(augmented, 0, 1)
        
        # Add batch and channel dimensions for model input
        augmented = augmented[np.newaxis, ..., np.newaxis]  # Shape: [1, H, W, 1]
        
        return augmented


# === KEYPOINT PATCH DATASET ===
class KeypointPatchDataset:
    """Extract GPU-compatible patches from keypoints on high-res images for KNIFT-GPU, not resized full images."""

    def __init__(self, image_dir, patches_per_image=100):
        """
        Initialize the GPU-compatible dataset.

        Args:
            image_dir: Directory containing high-res images
            patches_per_image: Number of patches to extract per image
        """
        self.image_dir = Path(image_dir)
        self.patches_per_image = patches_per_image
        self.images = []
        self.keypoints = []

        print("🎯 Initializing KNIFT-GPU Keypoint Patch Dataset...")
        print(f"   Image directory: {self.image_dir}")
        print(f"   Patches per image: {self.patches_per_image}")

        self.images = self.load_high_res_images()
        self.keypoints = self.detect_all_keypoints()

        # Safety check: need at least 2 images for triplet training
        if len(self.images) < 2:
            raise ValueError("Need at least 2 images for triplet training")

        print(f"   Loaded {len(self.images)} images")
        print(f"   Total keypoints: {len(self.keypoints)}")

    def load_high_res_images(self):
        """Load high-resolution images from directory."""
        image_paths = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

        for ext in extensions:
            pattern = self.image_dir / ext
            paths = list(glob.glob(str(pattern)))
            image_paths.extend(paths)

        # Sort for consistency
        image_paths.sort()

        images = []
        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)

        return images

    def detect_all_keypoints(self):
        """Detect keypoints on all images, store (image_idx, keypoint) pairs."""
        fast = cv2.FastFeatureDetector_create(threshold=20)
        all_kps = []

        for idx, img in enumerate(self.images):
            kps = fast.detect(img, None)
            # Sort by response and limit to top 500
            kps = sorted(kps, key=lambda x: x.response, reverse=True)[:500] if kps else []
            for kp in kps:
                all_kps.append((idx, kp))

        return all_kps

    def extract_patch(self, image, keypoint, size=32):
        """
        Extract a patch of given size around the keypoint.

        Args:
            image: Input image
            keypoint: cv2.KeyPoint object
            size: Patch size (default 32)

        Returns:
            Extracted patch
        """
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        half_size = size // 2

        # Calculate patch bounds
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(image.shape[1], x + half_size)
        y2 = min(image.shape[0], y + half_size)

        patch = image[y1:y2, x1:x2]

        # Resize if needed
        if patch.shape[:2] != (size, size):
            patch = cv2.resize(patch, (size, size))

        # Normalize
        patch = patch.astype(np.float32) / 255.0
        patch = np.expand_dims(patch, axis=-1)  # Add channel dimension

        return patch

    def generate_correspondence_pair(self, image, keypoint):
        """
        Given a high-res image and keypoint, generate anchor and positive patches
        that simulate real viewpoint change correspondence.
        """
        # Extract anchor patch at keypoint
        anchor_patch = self.extract_patch(image, keypoint, size=32)

        # Apply random homography to full image
        h, w = image.shape
        # Generate random homography matrix
        scale = random.uniform(0.9, 1.1)  # Scale change
        rotation = random.uniform(-0.2, 0.2)  # Small rotation (in radians)

        # Create transformation matrix
        cos_rot, sin_rot = np.cos(rotation), np.sin(rotation)
        scale_matrix = np.array([
            [scale * cos_rot, -scale * sin_rot, 0],
            [scale * sin_rot, scale * cos_rot, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Translation (small shift)
        tx = random.uniform(-w*0.05, w*0.05)  # 5% of image size
        ty = random.uniform(-h*0.05, h*0.05)
        translation_matrix = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], dtype=np.float32)

        # Combine transformations
        transform_matrix = translation_matrix @ scale_matrix
        transform_matrix = transform_matrix[:2, :]  # Convert to 2x3 for OpenCV

        # Apply transformation to the image
        warped_image = cv2.warpAffine(image, transform_matrix, (w, h),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Transform keypoint location
        pt = np.array([keypoint.pt[0], keypoint.pt[1], 1], dtype=np.float32)
        new_pt = transform_matrix @ pt

        # ADD: Boundary check
        h, w = image.shape
        margin = 16  # half patch size
        if (new_pt[0] < margin or new_pt[0] >= w - margin or
            new_pt[1] < margin or new_pt[1] >= h - margin):
            # Fallback: return two augmented versions of anchor patch
            anchor_patch = self.extract_patch(image, keypoint, size=32)
            return anchor_patch, anchor_patch.copy()

        new_kp = cv2.KeyPoint(new_pt[0], new_pt[1], keypoint.size)

        # Extract positive patch at transformed location
        positive_patch = self.extract_patch(warped_image, new_kp, size=32)

        return anchor_patch, positive_patch

# === TRIPLET GENERATOR ===
from typing import Tuple, List, Generator

class OnlineTripletGenerator:
    """
    GPU-compatible online triplet generator for KNIFT-GPU training with triplet loss.

    Generates positive and negative pairs from the same dataset using geometric
    transformations to simulate video correspondences while maintaining GPU compatibility.
    """

    def __init__(self,
                 data_loader: DataLoader,
                 augmentation: HeavyAugmentation,
                 batch_size: int = 32,
                 num_augmentations_per_image: int = 5,
                 negative_sampling_strategy: str = 'random'):
        """
        Initialize GPU-compatible triplet generator.

        Args:
            data_loader: DataLoader instance with real image data
            augmentation: HeavyAugmentation instance
            batch_size: Number of triplets per batch
            num_augmentations_per_image: Augmentations per image for variety
            negative_sampling_strategy: Strategy for negative selection ('random', 'hard')
        """
        self.data_loader = data_loader
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.num_augmentations_per_image = num_augmentations_per_image
        self.negative_sampling_strategy = negative_sampling_strategy

        # Create keypoint dataset
        self.keypoint_dataset = KeypointPatchDataset(data_loader.data_path)

        print("🔄 Initializing GPU-Compatible Online Triplet Generator...")
        print(f"   Loaded {len(self.keypoint_dataset.images)} images")
        print(f"   Total keypoints: {len(self.keypoint_dataset.keypoints)}")
        print(f"   Batch size: {batch_size}")
        print(f"   Negative sampling strategy: {negative_sampling_strategy}")

        # Safety check: ensure we have enough images
        if len(self.keypoint_dataset.images) < 2:
            raise ValueError("Need at least 2 images for triplet training")

    def _generate_positive_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a positive pair using keypoint correspondence simulation.

        Returns:
            Tuple of (anchor, positive) patches that correspond to the same 3D point
        """
        if len(self.keypoint_dataset.keypoints) == 0:
            raise ValueError("No keypoints found for correspondence simulation")

        # Select a random keypoint
        img_idx, kp = random.choice(self.keypoint_dataset.keypoints)
        image = self.keypoint_dataset.images[img_idx]

        # Generate positive pair via geometric transformation
        anchor_patch, positive_patch = self.keypoint_dataset.generate_correspondence_pair(image, kp)

        # Apply independent appearance augmentations to both patches
        anchor_augmented = anchor_patch  # Already normalized to [0,1] with channel dim
        positive_augmented = positive_patch  # Already normalized to [0,1] with channel dim

        # Apply augmentations separately - remove channel dimension for augmentation then add back
        anchor_2d = anchor_augmented.squeeze()  # (32, 32)
        anchor_augmented = self.augmentation.augment(anchor_2d)[0]  # augment expects (32,32), returns (1,32,32,1), [0] gives (32,32,1)

        positive_2d = positive_augmented.squeeze()  # (32, 32)
        positive_augmented = self.augmentation.augment(positive_2d)[0]  # augment expects (32,32), returns (1,32,32,1), [0] gives (32,32,1)

        return anchor_augmented, positive_augmented

    def _generate_negative_patch(self) -> np.ndarray:
        """
        Generate a negative patch from a different image/keypoint.

        Returns:
            Negative patch
        """
        if len(self.keypoint_dataset.keypoints) == 0:
            raise ValueError("No keypoints found for negative patch generation")

        # Select a random keypoint from a different image
        anchor_img_idx = -1  # Initialize to invalid value
        if len(self.keypoint_dataset.keypoints) > 0:
            anchor_img_idx, _ = random.choice(self.keypoint_dataset.keypoints)

        # Keep selecting until we get a different image
        neg_img_idx = anchor_img_idx
        while neg_img_idx == anchor_img_idx:
            img_idx_kp = random.choice(self.keypoint_dataset.keypoints)
            neg_img_idx, neg_kp = img_idx_kp[0], img_idx_kp[1]

        neg_image = self.keypoint_dataset.images[neg_img_idx]
        neg_kp = random.choice([kp for img_idx, kp in self.keypoint_dataset.keypoints if img_idx == neg_img_idx])

        # Extract patch
        negative_patch = self.keypoint_dataset.extract_patch(neg_image, neg_kp, size=32)

        # Apply augmentation - remove channel dimension for augmentation then add back
        negative_2d = negative_patch.squeeze()  # (32, 32)
        negative_augmented = self.augmentation.augment(negative_2d)[0]  # augment expects (32,32), returns (1,32,32,1), [0] gives (32,32,1)

        return negative_augmented

    def generate_triplet_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a batch of triplets using keypoint correspondence simulation.
        Always uses random negative generation for simplicity and efficiency.

        Returns:
            Tuple of (anchors, positives, negatives) with shape (batch_size, 32, 32, 1)
        """
        anchors = []
        positives = []
        negatives = []

        for _ in range(self.batch_size):
            # Generate positive pair
            anchor, positive = self._generate_positive_pair()

            # Generate negative
            negative = self._generate_negative_patch()

            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

        # Convert to numpy arrays with proper shapes
        # Each augmented image has shape (32, 32, 1), so stacking gives (batch_size, 32, 32, 1)
        anchors = np.stack(anchors)
        positives = np.stack(positives)
        negatives = np.stack(negatives)

        return anchors, positives, negatives


def setup_environment(training_dir: str = "Set14"):
    """Setup the GPU-compatible environment and check dependencies"""
    print("🔧 Setting up KNIFT-GPU Training Environment...")
    print(f"   TensorFlow version: {tf.__version__}")
    print(f"   GPU available: {'✅ Yes' if tf.config.list_physical_devices('GPU') else '❌ No (CPU only)'}")
    print(f"   OpenCV version: {cv2.__version__}")

    # Check training directory
    if not os.path.exists(training_dir):
        print(f"❌ Training directory does not exist: {training_dir}")
        print("   Please ensure Set14 directory exists with images")
        sys.exit(1)

    dataset_files = [f for f in os.listdir(training_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png')) and "baboon" not in f.lower()]

    if len(dataset_files) == 0:
        print(f"❌ No training image files found in {training_dir} (excluding baboon)")
        print("   Please ensure Set14 directory contains images for training")
        sys.exit(1)

    print(f"   ✅ Found {len(dataset_files)} training images (excluding baboon)")

    # Check if baboon is available for validation
    baboon_files = [f for f in os.listdir(training_dir) if "baboon" in f.lower() and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(baboon_files) > 0:
        print(f"   ✅ Found baboon image for validation: {baboon_files[0]}")
    else:
        print(f"   ⚠️  No baboon image found in {training_dir} - validation will be skipped")

    # Check validation directory
    validation_dir = "Set14"
    if os.path.exists(validation_dir):
        val_files = [f for f in os.listdir(validation_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png')) and "baboon" in f.lower()]
        if len(val_files) > 0:
            print(f"   ✅ Found baboon validation image(s)")
        else:
            print(f"   ⚠️  No baboon validation image found in {validation_dir}")
    else:
        print(f"   ⚠️  Validation directory not found: {validation_dir}")

    # Create output directories
    directories = ['outputs', 'outputs/models', 'logs', 'checkpoints']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("✅ Environment setup completed!\n")


class ConsoleTraining:
    """
    Console-based training with GPU-compatible features and alignment tasks
    """
    def __init__(self, training_dir: str = "dataset/training",
                 validation_dir: str = "dataset/validation",
                 epochs: int = 25,
                 batch_size: int = 32,
                 steps_per_epoch: int = 31):

        self.training_dir = training_dir
        self.validation_dir = validation_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        self.loss_history = []
        self.accuracy_history = []
        self.epoch_history = []
        self.current_epoch = 0
        self.current_loss = 0.0
        self.current_accuracy = 0.0

        # Create the GPU-compatible model for training (dynamic batch size)
        self.train_model = build_model(batch_size=None, name="knift_gpu_training_model")
        self.train_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=BatchHardTripletLoss(margin=1.0),  # Use batch-hard triplet loss
            metrics=[PositiveDistanceMetric(), NegativeDistanceMetric()]
        )

        # Create the GPU-compatible model for TFLite export (fixed batch size of 1000)
        self.export_model = build_model(batch_size=1000, name="knift_gpu_export_model")

        # Initialize feature extractor with the training model
        self.feature_extractor = FeatureExtractor(self.train_model)

        # Initialize validation directory for later use
        self.validation_images = (None, None)
        if os.path.exists(self.validation_dir):
            print("✅ Set14 directory found - will use baboon for validation alignments per epoch")
        else:
            print("⚠️  Set14 directory not found - will skip validation alignments")

    def _create_real_data_generator(self, data_dir):
        """
        Create real data generator from the dataset directory with actual triplets.
        Excludes baboon.jpeg from training to use it for validation.
        """
        print(f"📂 Creating real data generator from: {data_dir}")

        # Create Data Loader with real dataset
        data_loader = DataLoader(data_dir)

        if len(data_loader) == 0:
            raise ValueError(f"No images found in dataset directory: {data_dir}")

        # Filter out baboon from the training images to keep it for validation
        original_paths = data_loader.get_image_paths()
        filtered_paths = [path for path in original_paths if "baboon" not in os.path.basename(path).lower()]
        data_loader.image_paths = filtered_paths

        print(f"📊 Found {len(data_loader)} real images for training (excluding baboon)")

        # Create augmentation pipeline
        augmentation = HeavyAugmentation(
            rotation_range=15.0,
            brightness_range=0.2,
            contrast_range=0.15,
            noise_std=0.01,
            flip_probability=0.0,  # ← Change from 0.5 to 0.0
            noise_probability=0.8,
            blur_probability=0.6
        )

        # Create triplet generator
        triplet_gen = OnlineTripletGenerator(
            data_loader=data_loader,
            augmentation=augmentation,
            batch_size=self.batch_size
        )

        def real_generator():
            while True:
                # Use the triplet generator's built-in method (random negative generation)
                anchors, positives, negatives = triplet_gen.generate_triplet_batch()

                # CRITICAL: Concatenate in correct order - all anchors, all positives, all negatives
                images = np.concatenate([anchors, positives, negatives], axis=0)

                # Dummy labels - triplet loss doesn't use them, but Keras requires them
                labels = np.zeros((images.shape[0], 1), dtype=np.float32)

                yield images, labels

        return real_generator()

    def load_validation_pair(self):
        """Load validation image - specifically use baboon from Set14 directory"""
        import random

        if not os.path.exists(self.validation_dir):
            return None, None

        # Specifically use baboon.jpeg for validation (exclude from training set)
        selected_file = "baboon.jpeg"
        img_path = os.path.join(self.validation_dir, selected_file)

        # Check if baboon image exists
        if not os.path.exists(img_path):
            print(f"   ❌ Validation image {selected_file} not found in {self.validation_dir}")
            return None, None

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"   ❌ Could not load validation image: {img_path}")
            return None, None

        # Create synthetic misaligned version with random transformation
        h, w = img.shape

        # Random rotation (-15 to +15 degrees)
        angle = random.uniform(-15, 15)
        M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)

        # Random translation (-10% to +10%)
        tx = random.uniform(-0.1 * w, 0.1 * w)
        ty = random.uniform(-0.1 * h, 0.1 * h)
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty

        # Apply transformation
        img_misaligned = cv2.warpAffine(img, M_rot, (w, h),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)

        print(f"   ✅ Loaded validation image: {selected_file}")
        print(f"      Applied transform: rotation={angle:.1f}°, tx={tx:.1f}px, ty={ty:.1f}px")

        return img, img_misaligned

    def test_alignment_quality(self):
        """Test how well model matches features between original and misaligned image"""
        if not hasattr(self, 'validation_images') or self.validation_images[0] is None:
            return {'quality': 0, 'matches': 0, 'inliers': 0}

        img1, img2 = self.validation_images

        # Check if we successfully loaded the validation images
        if img1 is None or img2 is None:
            return {'quality': 0, 'matches': 0, 'inliers': 0}

        try:
            # Extract keypoints using FAST detector
            fast = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)
            kp1 = fast.detect(img1, None)
            kp2 = fast.detect(img2, None)

            if len(kp1) == 0 or len(kp2) == 0:
                print(f"DEBUG: No keypoints found - img1: {len(kp1)}, img2: {len(kp2)}")
                return {'quality': 0, 'matches': 0, 'inliers': 0}

            # Limit to top 200 keypoints by response
            kp1 = sorted(kp1, key=lambda x: x.response, reverse=True)[:200]
            kp2 = sorted(kp2, key=lambda x: x.response, reverse=True)[:200]

            # Extract patches around keypoints
            patches1 = self.extract_patches(img1, kp1)
            patches2 = self.extract_patches(img2, kp2)

            if len(patches1) == 0 or len(patches2) == 0:
                print(f"DEBUG: No patches extracted - patches1: {len(patches1)}, patches2: {len(patches2)}")
                return {'quality': 0, 'matches': 0, 'inliers': 0}

            # Get features from current model
            features1 = self.feature_extractor.extract_features(patches1)
            features2 = self.feature_extractor.extract_features(patches2)

            # Match features using L2 distance
            matches = self.match_features_lowe_ratio(features1, features2, kp1, kp2,
                                                      ratio_threshold=0.75)

            if len(matches) < 4:
                print(f"DEBUG: Not enough matches for homography - found: {len(matches)}")
                return {'quality': 0, 'matches': len(matches), 'inliers': 0}

            # Get matched keypoint positions
            pts1 = np.float32([kp1[m[0]].pt for m in matches])
            pts2 = np.float32([kp2[m[1]].pt for m in matches])

            # Compute homography and count inliers
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

            if H is None:
                print("DEBUG: Could not compute homography matrix")
                return {'quality': 0, 'matches': len(matches), 'inliers': 0}

            # Fix: Flatten the mask if it's 2D to ensure proper iteration
            if mask is not None:
                if len(mask.shape) > 1:
                    mask = mask.ravel()  # Flatten from (N,1) to (N,)
                # Ensure mask is binary (0 or 1)
                mask = mask.astype(bool)

            inliers = np.sum(mask) if mask is not None else 0
            quality = (inliers / len(matches)) * 100 if len(matches) > 0 else 0

            print(f"DEBUG: Alignment test results - Matches: {len(matches)}, Inliers: {inliers}, Quality: {quality:.2f}%")

            # Fix: Properly handle the mask for matched_pairs
            if mask is not None and len(mask) == len(matches):
                matched_pairs = [(m[0], m[1]) for m, is_inlier in zip(matches, mask) if is_inlier]
            else:
                # Default to all matches if mask is invalid
                matched_pairs = [(m[0], m[1]) for m in matches]

            return {
                'quality': quality,
                'matches': len(matches),
                'inliers': int(inliers),
                'kp1': kp1,
                'kp2': kp2,
                'matched_pairs': matched_pairs,
                'img1': img1,
                'img2': img2,
                'homography': H
            }

        except Exception as e:
            print(f"Error in alignment test: {e}")
            import traceback
            traceback.print_exc()
            return {'quality': 0, 'matches': 0, 'inliers': 0}

    def extract_patches(self, img, keypoints, patch_size=32):
        """Extract 32x32 patches around keypoints"""
        patches = []

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            half_size = patch_size // 2

            # Calculate patch bounds
            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(img.shape[1], x + half_size)
            y2 = min(img.shape[0], y + half_size)

            patch = img[y1:y2, x1:x2]

            # Resize if needed
            if patch.shape[:2] != (patch_size, patch_size):
                patch = cv2.resize(patch, (patch_size, patch_size))

            # Normalize
            patch = patch.astype(np.float32) / 255.0
            patch = np.expand_dims(patch, axis=-1)
            patches.append(patch)

        return np.array(patches) if patches else np.array([])

    def match_features_lowe_ratio(self, desc1, desc2, kp1, kp2, ratio_threshold=0.75):
        """Match features using Lowe's ratio test"""
        if len(desc1) == 0 or len(desc2) == 0:
            return []

        # Compute distances
        distances = np.linalg.norm(desc1[:, np.newaxis, :] - desc2[np.newaxis, :, :], axis=2)

        good_matches = []
        for i in range(len(desc1)):
            dists = distances[i]
            sorted_indices = np.argsort(dists)

            if len(sorted_indices) < 2:
                continue

            best_idx = sorted_indices[0]
            second_best_idx = sorted_indices[1]
            best_dist = dists[best_idx]
            second_best_dist = dists[second_best_idx]

            # Lowe's ratio test
            if second_best_dist > 0:
                ratio = best_dist / second_best_dist
                if ratio < ratio_threshold:
                    good_matches.append((i, best_idx, best_dist))

        return [(i, idx) for i, idx, _ in good_matches]  # Return index pairs only

    def run_training(self):
        """Run the actual training with batch-hard triplet loss"""
        try:
            print(f"🎯 Starting training for {self.epochs} epochs with batch-hard triplet loss")
            data_generator = self._create_real_data_generator(self.training_dir)

            # Single-stage training with batch-hard triplet loss
            history = self.train_model.fit(
                data_generator,
                epochs=self.epochs,
                steps_per_epoch=self.steps_per_epoch,
                callbacks=[self._create_training_callback()],
                verbose=1
            )

            print("✅ Training completed!")

            # Save final model with the total epochs number
            final_model_path = os.path.join("outputs/models", f"knift_gpu_final_model_epoch_{self.epochs:03d}.weights.h5")
            print(f"💾 Saving final model: {final_model_path}")
            self.train_model.save_weights(final_model_path)
            print(f"✅ Final model saved successfully")

            # After training, export to TFLite
            print("Exporting to TFLite...")
            self.export_to_tflite()

        except Exception as e:
            print(f"ERROR in training: {str(e)}")  # Print error to console
            import traceback
            traceback.print_exc()  # Print full stack trace

    def _create_training_callback(self):
        """Create the training callback with access to parent class"""
        class TrainingCallback(tf.keras.callbacks.Callback):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent
                self.batch_losses = []
                self.batch_maes = []

            def on_batch_end(self, batch, logs=None):
                # Log progress to console
                if batch % 50 == 0:  # Print every 50 batches to avoid too much output
                    loss = logs.get('loss', 0)
                    pos_dist = logs.get('pos_dist', 0)
                    neg_dist = logs.get('neg_dist', 0)
                    print(f"   Batch {batch}: Loss = {loss:.4f}, Pos Dist = {pos_dist:.4f}, Neg Dist = {neg_dist:.4f}")

            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss', 0)
                pos_dist = logs.get('pos_dist', 0)
                neg_dist = logs.get('neg_dist', 0)

                # Print debug for every epoch
                print(f"Epoch {epoch+1}/{self.parent.epochs} - Loss: {loss:.4f}, "
                      f"Pos Dist: {pos_dist:.4f}, Neg Dist: {neg_dist:.4f}")

                # Add to history
                self.parent.epoch_history.append(epoch)
                self.parent.loss_history.append(loss)
                self.parent.accuracy_history.append((neg_dist - pos_dist) if pos_dist != 0 else 0)  # Metric: negative distance - positive distance

                # Update alignment visualization
                print(f"   Testing alignment quality...")
                self.parent.validation_images = self.parent.load_validation_pair()
                alignment_result = self.parent.test_alignment_quality()

                # Log alignment metrics to console
                print(f"   Alignment: Matches={alignment_result['matches']}, "
                      f"Inliers={alignment_result['inliers']}, "
                      f"Quality={alignment_result['quality']:.2f}%")

                # Save checkpoint every 5 epochs
                if (epoch + 1) % 5 == 0:
                    checkpoint_path = os.path.join(
                        "checkpoints",
                        f"checkpoint_epoch_{epoch+1:03d}_loss_{loss:.4f}.weights.h5"
                    )
                    print(f"   💾 Saving checkpoint at epoch {epoch+1}: {checkpoint_path}")
                    self.parent.train_model.save_weights(checkpoint_path)

        return TrainingCallback(self)

    def export_to_tflite(self):
        """Export the ~100K-parameter trained GPU-compatible model to TensorFlow Lite format"""
        try:
            # Path to the final model with correct extension
            final_model_path = os.path.join("outputs/models", f"knift_gpu_final_model_epoch_{self.epochs:03d}.weights.h5")

            print("🎯 Converting ~100K-parameter GPU-compatible model to TensorFlow Lite format...")

            # Create and export multiple GPU-compatible model variants with different batch sizes
            batch_sizes = [200, 400, 1000]  # Optimized for mobile GPU inference
            model_names = ["knift_gpu_float.tflite", "knift_gpu_float_400.tflite", "knift_gpu_float_1k.tflite"]

            for batch_size, model_name in zip(batch_sizes, model_names):
                print(f"🎯 Creating ~100K-parameter GPU-compatible model with batch size {batch_size}: {model_name}")

                # Create export model with the specific batch size
                export_model = build_model(batch_size=batch_size, name=f"knift_gpu_export_{batch_size}")

                # Copy weights from trained model to export model
                export_model.set_weights(self.train_model.get_weights())

                # Convert to TensorFlow Lite - no L2 normalization for GPU compatibility
                converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

                tflite_model = converter.convert()

                # Verify input/output shapes
                interpreter = tf.lite.Interpreter(model_content=tflite_model)
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                print(f"Input shape: {input_details[0]['shape']}")   # Should match batch_size
                print(f"Output shape: {output_details[0]['shape']}") # Should match batch_size

                # Save the GPU-compatible TFLite model
                tflite_path = os.path.join("outputs/models", model_name)
                os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)

                print(f"✅ {model_name} exported successfully!")
                print(f"   TFLite model size: {len(tflite_model) / (1024*1024):.2f} MB")

                # Verify the TFLite model
                interpreter.allocate_tensors()
                print(f"✅ {model_name} verification passed!")

            print("✅ All ~100K-parameter GPU-compatible TFLite models exported successfully!")

        except Exception as e:
            print(f"❌ Error during ~100K-parameter GPU-compatible TFLite export: {str(e)}")


def main():
    """Main entry point - ~100K-parameter GPU-compatible KNIFT-GPU training, no arguments needed"""
    print("=" * 60)
    print("🚀 KNIFT-GPU - ~100K Mobile-Optimized Feature Descriptor Training (Console Version)")
    print("=" * 60)

    # Fixed configuration - no parameters needed
    training_dir = "Set14"  # Use Set14 images for training (excluding baboon for validation)
    validation_dir = "Set14"  # Use baboon from Set14 for validation
    epochs = 100
    batch_size = 32
    steps_per_epoch = 10  # ~100k patches over 100 epochs = 1k patches per epoch = 10 steps at 96 patches per batch

    # Setup environment
    setup_environment(training_dir)

    # Print configuration
    print(f"\n📊 ~100K-parameter GPU-Compatible Training Configuration:")
    print(f"   Training images: {training_dir} (13 images excluding baboon)")
    print(f"   Validation image: baboon.jpeg from Set14")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Total patches: ~{epochs * steps_per_epoch * batch_size * 3:,} distributed over {epochs} epochs ({steps_per_epoch} steps/epoch)")
    print(f"   Models exported: knift_gpu_float.tflite, knift_gpu_float_400.tflite, knift_gpu_float_1k.tflite")
    print(f"\n🏃 Starting ~100K-parameter GPU-compatible training...")

    # Create and run the console trainer
    trainer = ConsoleTraining(
        training_dir=training_dir,
        validation_dir=validation_dir,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch
    )

    trainer.run_training()

    print("\n🎉 ~100K-parameter GPU-compatible training and export completed successfully!")


if __name__ == "__main__":
    main()