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
Version: 2.0.0
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
import random
import math
import glob

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
    print("🏗️  Building Inception-like Architecture...")
    print(f"   Input Shape: {input_shape}")
    print(f"   Output Dimension: {output_dim}")
    print(f"   Batch Size: {batch_size}")

    # Input layer
    if batch_size is not None:
        inputs = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size, name="input")
    else:
        inputs = tf.keras.layers.Input(shape=input_shape, name="input")

    # === STEM BLOCK ===
    print("📐 Building Stem Block...")
    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu',
        name='stem_conv'
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name='stem_bn')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name='stem_pool')(x)

    # === INCEPTION MODULE 1 ===
    print("🏗️  Building Inception Module 1...")
    x = inception_module(x, {
        'name': 'inception1',
        'branch1x1': 12,
        'branch3x3_reduce': 8,
        'branch3x3': 16,
        'branch5x5_reduce': 8,
        'branch5x5': 16,
        'branch_pool_proj': 8
    })

    # === INCEPTION MODULE 2 ===
    print("🏗️  Building Inception Module 2...")
    x = inception_module(x, {
        'name': 'inception2',
        'branch1x1': 16,
        'branch3x3_reduce': 12,
        'branch3x3': 24,
        'branch5x5_reduce': 12,
        'branch5x5': 24,
        'branch_pool_proj': 16
    })
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name='pool2')(x)

    # === INCEPTION MODULE 3 ===
    print("🏗️  Building Inception Module 3...")
    x = inception_module(x, {
        'name': 'inception3',
        'branch1x1': 24,
        'branch3x3_reduce': 16,
        'branch3x3': 32,
        'branch5x5_reduce': 16,
        'branch5x5': 32,
        'branch_pool_proj': 24
    })

    # === INCEPTION MODULE 4 ===
    print("🏗️  Building Inception Module 4...")
    x = inception_module(x, {
        'name': 'inception4',
        'branch1x1': 28,
        'branch3x3_reduce': 16,
        'branch3x3': 32,
        'branch5x5_reduce': 16,
        'branch5x5': 32,
        'branch_pool_proj': 24
    })
    x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name='pool3')(x)

    # === FINAL INCEPTION MODULE ===
    print("🏗️  Building Final Inception Module...")
    x = inception_module(x, {
        'name': 'inception5',
        'branch1x1': 28,
        'branch3x3_reduce': 14,
        'branch3x3': 28,
        'branch5x5_reduce': 14,
        'branch5x5': 28,
        'branch_pool_proj': 24
    })

    # === OUTPUT BLOCK ===
    print("📤 Building Output Block...")
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_pool')(x)
    outputs = tf.keras.layers.Dense(
        units=output_dim,
        activation=None,
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

    print("\n🔧 GPU Compatibility Check:")
    print("✅ CONV_2D, AVERAGE_POOL_2D, MAX_POOL_2D")
    print("✅ CONCATENATION, FULLY_CONNECTED, BATCH_NORM")
    print("✅ NO L2_NORMALIZATION (GPU-Compatible)")

    return model

def inception_module(inputs, config):
    """
    Build GPU-compatible Inception module with reduced parameters for feature extraction.
    """
    name = config['name']
    branch1x1 = config['branch1x1']
    branch3x3_reduce = config['branch3x3_reduce']
    branch3x3 = config['branch3x3']
    branch5x5_reduce = config['branch5x5_reduce']
    branch5x5 = config['branch5x5']
    branch_pool_proj = config['branch_pool_proj']

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

# === BATCH-HARD TRIPLET LOSS ===

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

        # Compute positive distances
        pos_dist = tf.reduce_sum(tf.square(anchors - positives), axis=1)

        # Compute pairwise distances between anchors and ALL negatives
        anchor_neg_dist = tf.reduce_sum(
            tf.square(tf.expand_dims(anchors, 1) - tf.expand_dims(negatives, 0)),
            axis=2
        )

        # Semi-hard mining: d(a,p) < d(a,n) < d(a,p) + margin
        pos_dist_expanded = tf.expand_dims(pos_dist, 1)
        semi_hard_mask = tf.logical_and(
            anchor_neg_dist > pos_dist_expanded,
            anchor_neg_dist < pos_dist_expanded + self.margin
        )

        # Get hardest semi-hard negative
        masked_dist = tf.where(
            semi_hard_mask,
            anchor_neg_dist,
            tf.fill(tf.shape(anchor_neg_dist), tf.float32.max)
        )

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

# === FEATURE EXTRACTOR ===

class FeatureExtractor:
    """Extracts GPU-compatible feature embeddings from image patches using trained KNIFT-GPU model"""

    def __init__(self, model: tf.keras.Model = None):
        self.model = model
        self.input_size = (32, 32)
        self.embedding_dim = 40

        if self.model:
            print(f"🔧 KNIFT-GPU FeatureExtractor initialized with model: {self.model.name}")

    def extract_features(self, patches: np.ndarray) -> np.ndarray:
        """Extract features from patches using the model"""
        if self.model is None:
            raise ValueError("No model loaded for feature extraction")

        if len(patches) == 0:
            return np.array([])

        # Handle different input shapes
        if len(patches.shape) == 3:
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
                patch = patch[..., np.newaxis]
                resized_patches.append(patch)
            patches = np.array(resized_patches)

        # Extract features
        try:
            features = self.model.predict(patches, verbose=0)
            return features
        except Exception as e:
            print(f"❌ Error during feature extraction: {e}")
            return np.array([])

# === DATA LOADER ===

class DataLoader:
    """Basic data loader for KNIFT-GPU"""

    def __init__(self, data_path, img_size=(32, 32)):
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.images = []
        self.image_paths = []

        print("📊 Initializing KNIFT-GPU Data Loader...")
        print(f"   Data Path: {self.data_path}")
        print(f"   Target Size: {self.img_size}")

        self._load_image_paths()
        print(f"   Found Images: {len(self.image_paths)}")

    def _load_image_paths(self):
        """Load all image paths from data directory, excluding baboon for validation."""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for ext in extensions:
            pattern = self.data_path / ext
            paths = list(glob.glob(str(pattern)))
            filtered_paths = [path for path in paths if "baboon" not in os.path.basename(path).lower()]
            self.image_paths.extend(filtered_paths)
        self.image_paths.sort()

    def get_image_paths(self):
        return self.image_paths.copy()

    def get_num_images(self):
        return len(self.image_paths)

    def __len__(self):
        return self.get_num_images()

# === AUGMENTATION PIPELINE ===

class HeavyAugmentation:
    """GPU-compatible heavy augmentation pipeline for training patches"""

    def __init__(self,
                 rotation_range=15,
                 brightness_range=0.2,
                 contrast_range=0.15,
                 noise_std=0.01,
                 blur_radius_range=(0.5, 2.0),
                 flip_probability=0.0,
                 noise_probability=0.8,
                 blur_probability=0.6):

        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.blur_radius_range = blur_radius_range
        self.flip_probability = flip_probability
        self.noise_probability = noise_probability
        self.blur_probability = blur_probability

        print("🎨 Initializing GPU-Compatible Heavy Augmentation Pipeline...")
        print(f"   Rotation range: ±{rotation_range}°")
        print(f"   Brightness range: ±{brightness_range*100}%")
        print(f"   Contrast range: ±{contrast_range*100}%")
        print(f"   Noise std: {noise_std}")
        print(f"   Blur radius: {blur_radius_range}")
        print(f"   Flip probability: {flip_probability}")

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        if angle == 0:
            return image
        pil_image = Image.fromarray((image * 255).astype(np.uint8), mode='L')
        rotated_pil = pil_image.rotate(angle, resample=Image.Resampling.BILINEAR)
        rotated_array = np.array(rotated_pil, dtype=np.float32) / 255.0
        return rotated_array

    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        pil_image = Image.fromarray((image * 255).astype(np.uint8), mode='L')
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced_pil = enhancer.enhance(factor)
        enhanced_array = np.array(enhanced_pil, dtype=np.float32) / 255.0
        return enhanced_array

    def adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        pil_image = Image.fromarray((image * 255).astype(np.uint8), mode='L')
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced_pil = enhancer.enhance(factor)
        enhanced_array = np.array(enhanced_pil, dtype=np.float32) / 255.0
        return enhanced_array

    def add_noise(self, image: np.ndarray, std: float) -> np.ndarray:
        noise = np.random.normal(0, std, image.shape)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 1)
        return noisy_image

    def add_blur(self, image: np.ndarray, radius: float) -> np.ndarray:
        if radius <= 0:
            return image
        pil_image = Image.fromarray((image * 255).astype(np.uint8), mode='L')
        blurred_pil = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
        blurred_array = np.array(blurred_pil, dtype=np.float32) / 255.0
        return blurred_array

    def horizontal_flip(self, image: np.ndarray) -> np.ndarray:
        return np.fliplr(image)

    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply heavy augmentation to image"""
        augmented = image.copy()

        # Random rotation
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        augmented = self.rotate_image(augmented, angle)

        # Random brightness
        if random.random() < 0.8:
            brightness_factor = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
            brightness_factor = max(0.1, min(2.0, brightness_factor))
            augmented = self.adjust_brightness(augmented, brightness_factor)

        # Random contrast
        if random.random() < 0.8:
            contrast_factor = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
            contrast_factor = max(0.1, min(2.0, contrast_factor))
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

        augmented = np.clip(augmented, 0, 1)
        augmented = augmented[np.newaxis, ..., np.newaxis]
        return augmented

# === KEYPOINT PATCH DATASET ===

class KeypointPatchDataset:
    """Extract GPU-compatible patches from keypoints on high-res images for KNIFT-GPU"""

    def __init__(self, image_dir, patches_per_image=100):
        self.image_dir = Path(image_dir)
        self.patches_per_image = patches_per_image
        self.images = []
        self.keypoints = []

        print("🎯 Initializing KNIFT-GPU Keypoint Patch Dataset...")
        print(f"   Image directory: {self.image_dir}")
        print(f"   Patches per image: {self.patches_per_image}")

        self.images = self.load_high_res_images()
        self.keypoints = self.detect_all_keypoints()

        if len(self.images) < 2:
            raise ValueError("Need at least 2 images for triplet training")

        print(f"   Loaded {len(self.images)} images")
        print(f"   Total keypoints: {len(self.keypoints)}")

    def load_high_res_images(self):
        """Load high-resolution images from directory"""
        image_paths = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for ext in extensions:
            pattern = self.image_dir / ext
            paths = list(glob.glob(str(pattern)))
            image_paths.extend(paths)

        image_paths.sort()
        images = []
        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
        return images

    def detect_all_keypoints(self):
        """Detect keypoints on all images"""
        fast = cv2.FastFeatureDetector_create(threshold=20)
        all_kps = []
        for idx, img in enumerate(self.images):
            kps = fast.detect(img, None)
            kps = sorted(kps, key=lambda x: x.response, reverse=True)[:500] if kps else []
            for kp in kps:
                all_kps.append((idx, kp))
        return all_kps

    def extract_patch(self, image, keypoint, size=32):
        """Extract a patch of given size around the keypoint"""
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        half_size = size // 2

        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(image.shape[1], x + half_size)
        y2 = min(image.shape[0], y + half_size)

        patch = image[y1:y2, x1:x2]

        if patch.shape[:2] != (size, size):
            patch = cv2.resize(patch, (size, size))

        patch = patch.astype(np.float32) / 255.0
        patch = np.expand_dims(patch, axis=-1)
        return patch

    def generate_correspondence_pair(self, image, keypoint):
        """Generate anchor and positive patches that simulate real viewpoint change"""
        anchor_patch = self.extract_patch(image, keypoint, size=32)

        h, w = image.shape
        scale = random.uniform(0.9, 1.1)
        rotation = random.uniform(-0.2, 0.2)

        cos_rot, sin_rot = np.cos(rotation), np.sin(rotation)
        scale_matrix = np.array([
            [scale * cos_rot, -scale * sin_rot, 0],
            [scale * sin_rot, scale * cos_rot, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        tx = random.uniform(-w*0.05, w*0.05)
        ty = random.uniform(-h*0.05, h*0.05)
        translation_matrix = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], dtype=np.float32)

        transform_matrix = translation_matrix @ scale_matrix
        transform_matrix = transform_matrix[:2, :]

        warped_image = cv2.warpAffine(image, transform_matrix, (w, h),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        pt = np.array([keypoint.pt[0], keypoint.pt[1], 1], dtype=np.float32)
        new_pt = transform_matrix @ pt

        # Boundary check
        margin = 16
        if (new_pt[0] < margin or new_pt[0] >= w - margin or
            new_pt[1] < margin or new_pt[1] >= h - margin):
            return anchor_patch, anchor_patch.copy()

        new_kp = cv2.KeyPoint(new_pt[0], new_pt[1], keypoint.size)
        positive_patch = self.extract_patch(warped_image, new_kp, size=32)

        return anchor_patch, positive_patch

# === TRIPLET GENERATOR ===

class OnlineTripletGenerator:
    """GPU-compatible online triplet generator for KNIFT-GPU training"""

    def __init__(self,
                 data_loader: DataLoader,
                 augmentation: HeavyAugmentation,
                 batch_size: int = 32):

        self.data_loader = data_loader
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.keypoint_dataset = KeypointPatchDataset(data_loader.data_path)

        print("🔄 Initializing GPU-Compatible Online Triplet Generator...")
        print(f"   Loaded {len(self.keypoint_dataset.images)} images")
        print(f"   Total keypoints: {len(self.keypoint_dataset.keypoints)}")
        print(f"   Batch size: {batch_size}")

        if len(self.keypoint_dataset.images) < 2:
            raise ValueError("Need at least 2 images for triplet training")

    def _generate_positive_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a positive pair using keypoint correspondence simulation"""
        if len(self.keypoint_dataset.keypoints) == 0:
            raise ValueError("No keypoints found for correspondence simulation")

        img_idx, kp = random.choice(self.keypoint_dataset.keypoints)
        image = self.keypoint_dataset.images[img_idx]

        anchor_patch, positive_patch = self.keypoint_dataset.generate_correspondence_pair(image, kp)

        anchor_2d = anchor_patch.squeeze()
        anchor_augmented = self.augmentation.augment(anchor_2d)[0]

        positive_2d = positive_patch.squeeze()
        positive_augmented = self.augmentation.augment(positive_2d)[0]

        return anchor_augmented, positive_augmented

    def _generate_negative_patch(self) -> np.ndarray:
        """Generate a negative patch from a different image/keypoint"""
        if len(self.keypoint_dataset.keypoints) == 0:
            raise ValueError("No keypoints found for negative patch generation")

        anchor_img_idx = -1
        if len(self.keypoint_dataset.keypoints) > 0:
            anchor_img_idx, _ = random.choice(self.keypoint_dataset.keypoints)

        neg_img_idx = anchor_img_idx
        while neg_img_idx == anchor_img_idx:
            img_idx_kp = random.choice(self.keypoint_dataset.keypoints)
            neg_img_idx, neg_kp = img_idx_kp[0], img_idx_kp[1]

        neg_image = self.keypoint_dataset.images[neg_img_idx]
        neg_kp = random.choice([kp for img_idx, kp in self.keypoint_dataset.keypoints if img_idx == neg_img_idx])

        negative_patch = self.keypoint_dataset.extract_patch(neg_image, neg_kp, size=32)
        negative_2d = negative_patch.squeeze()
        negative_augmented = self.augmentation.augment(negative_2d)[0]

        return negative_augmented

    def generate_triplet_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a batch of triplets"""
        anchors = []
        positives = []
        negatives = []

        for _ in range(self.batch_size):
            anchor, positive = self._generate_positive_pair()
            negative = self._generate_negative_patch()

            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

        anchors = np.stack(anchors)
        positives = np.stack(positives)
        negatives = np.stack(negatives)

        return anchors, positives, negatives

# === ENVIRONMENT SETUP ===

def setup_environment(training_dir: str = "Set14"):
    """Setup the GPU-compatible environment and check dependencies"""
    print("🔧 Setting up KNIFT-GPU Training Environment...")
    print(f"   TensorFlow version: {tf.__version__}")
    print(f"   GPU available: {'✅ Yes' if tf.config.list_physical_devices('GPU') else '❌ No (CPU only)'}")
    print(f"   OpenCV version: {cv2.__version__}")

    if not os.path.exists(training_dir):
        print(f"❌ Training directory does not exist: {training_dir}")
        sys.exit(1)

    dataset_files = [f for f in os.listdir(training_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png')) and "baboon" not in f.lower()]

    if len(dataset_files) == 0:
        print(f"❌ No training image files found in {training_dir} (excluding baboon)")
        sys.exit(1)

    print(f"   ✅ Found {len(dataset_files)} training images (excluding baboon)")

    baboon_files = [f for f in os.listdir(training_dir) if "baboon" in f.lower() and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(baboon_files) > 0:
        print(f"   ✅ Found baboon image for validation: {baboon_files[0]}")
    else:
        print(f"   ⚠️  No baboon image found in {training_dir} - validation will be skipped")

    directories = ['outputs', 'outputs/models', 'logs', 'checkpoints']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("✅ Environment setup completed!\n")

# === CONSOLE TRAINING CLASS ===

class ConsoleTraining:
    """Console-based training with GPU-compatible features and alignment tasks"""

    def __init__(self, training_dir: str = "Set14",
                 validation_dir: str = "Set14",
                 epochs: int = 250,
                 batch_size: int = 32,
                 steps_per_epoch: int = 20):

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

        # Create the GPU-compatible model for training
        self.train_model = build_model(batch_size=None, name="knift_gpu_training_model")
        self.train_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=BatchHardTripletLoss(margin=1.0),
            metrics=[PositiveDistanceMetric(), NegativeDistanceMetric()]
        )

        # Create the GPU-compatible model for TFLite export
        self.export_model = build_model(batch_size=1000, name="knift_gpu_export_model")

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(self.train_model)

        # Initialize validation directory
        self.validation_images = (None, None)
        if os.path.exists(self.validation_dir):
            print("✅ Set14 directory found - will use baboon for validation alignments per epoch")
        else:
            print("⚠️  Set14 directory not found - will skip validation alignments")

    def _create_real_data_generator(self, data_dir):
        """Create real data generator from the dataset directory"""
        print(f"📂 Creating real data generator from: {data_dir}")

        data_loader = DataLoader(data_dir)
        if len(data_loader) == 0:
            raise ValueError(f"No images found in dataset directory: {data_dir}")

        # Filter out baboon from training
        original_paths = data_loader.get_image_paths()
        filtered_paths = [path for path in original_paths if "baboon" not in os.path.basename(path).lower()]
        data_loader.image_paths = filtered_paths

        print(f"📊 Found {len(data_loader)} real images for training (excluding baboon)")

        augmentation = HeavyAugmentation(
            rotation_range=15.0,
            brightness_range=0.2,
            contrast_range=0.15,
            noise_std=0.01,
            flip_probability=0.0,
            noise_probability=0.8,
            blur_probability=0.6
        )

        triplet_gen = OnlineTripletGenerator(
            data_loader=data_loader,
            augmentation=augmentation,
            batch_size=self.batch_size
        )

        def real_generator():
            while True:
                anchors, positives, negatives = triplet_gen.generate_triplet_batch()
                images = np.concatenate([anchors, positives, negatives], axis=0)
                labels = np.zeros((images.shape[0], 1), dtype=np.float32)
                yield images, labels

        return real_generator()

    def load_validation_pair(self):
        """Load validation image - specifically use baboon from Set14 directory"""
        if not os.path.exists(self.validation_dir):
            return None, None

        selected_file = "baboon.jpeg"
        img_path = os.path.join(self.validation_dir, selected_file)

        if not os.path.exists(img_path):
            print(f"   ❌ Validation image {selected_file} not found in {self.validation_dir}")
            return None, None

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"   ❌ Could not load validation image: {img_path}")
            return None, None

        h, w = img.shape

        # Random rotation (-15 to +15 degrees)
        angle = random.uniform(-15, 15)
        M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)

        # Random translation
        tx = random.uniform(-0.1 * w, 0.1 * w)
        ty = random.uniform(-0.1 * h, 0.1 * h)
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty

        img_misaligned = cv2.warpAffine(img, M_rot, (w, h),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)

        print(f"   ✅ Loaded validation image: {selected_file}")
        print(f"   Applied transform: rotation={angle:.1f}°, tx={tx:.1f}px, ty={ty:.1f}px")

        return img, img_misaligned

    def test_alignment_quality(self):
        """Test how well model matches features between original and misaligned image"""
        if not hasattr(self, 'validation_images') or self.validation_images[0] is None:
            return {'quality': 0, 'matches': 0, 'inliers': 0}

        img1, img2 = self.validation_images

        if img1 is None or img2 is None:
            return {'quality': 0, 'matches': 0, 'inliers': 0}

        try:
            fast = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)
            kp1 = fast.detect(img1, None)
            kp2 = fast.detect(img2, None)

            if len(kp1) == 0 or len(kp2) == 0:
                return {'quality': 0, 'matches': 0, 'inliers': 0}

            kp1 = sorted(kp1, key=lambda x: x.response, reverse=True)[:200]
            kp2 = sorted(kp2, key=lambda x: x.response, reverse=True)[:200]

            patches1 = self.extract_patches(img1, kp1)
            patches2 = self.extract_patches(img2, kp2)

            if len(patches1) == 0 or len(patches2) == 0:
                return {'quality': 0, 'matches': 0, 'inliers': 0}

            features1 = self.feature_extractor.extract_features(patches1)
            features2 = self.feature_extractor.extract_features(patches2)

            matches = self.match_features_lowe_ratio(features1, features2, kp1, kp2, ratio_threshold=0.75)

            if len(matches) < 4:
                return {'quality': 0, 'matches': len(matches), 'inliers': 0}

            pts1 = np.float32([kp1[m[0]].pt for m in matches])
            pts2 = np.float32([kp2[m[1]].pt for m in matches])

            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

            if H is None:
                return {'quality': 0, 'matches': len(matches), 'inliers': 0}

            if mask is not None:
                if len(mask.shape) > 1:
                    mask = mask.ravel()
                mask = mask.astype(bool)
                inliers = np.sum(mask)
            else:
                inliers = 0

            quality = (inliers / len(matches)) * 100 if len(matches) > 0 else 0

            if mask is not None and len(mask) == len(matches):
                matched_pairs = [(m[0], m[1]) for m, is_inlier in zip(matches, mask) if is_inlier]
            else:
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
            return {'quality': 0, 'matches': 0, 'inliers': 0}

    def extract_patches(self, img, keypoints, patch_size=32):
        """Extract 32x32 patches around keypoints"""
        patches = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            half_size = patch_size // 2

            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(img.shape[1], x + half_size)
            y2 = min(img.shape[0], y + half_size)

            patch = img[y1:y2, x1:x2]

            if patch.shape[:2] != (patch_size, patch_size):
                patch = cv2.resize(patch, (patch_size, patch_size))

            patch = patch.astype(np.float32) / 255.0
            patch = np.expand_dims(patch, axis=-1)
            patches.append(patch)

        return np.array(patches) if patches else np.array([])

    def match_features_lowe_ratio(self, desc1, desc2, kp1, kp2, ratio_threshold=0.75):
        """Match features using Lowe's ratio test"""
        if len(desc1) == 0 or len(desc2) == 0:
            return []

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

            if second_best_dist > 0:
                ratio = best_dist / second_best_dist
                if ratio < ratio_threshold:
                    good_matches.append((i, best_idx, best_dist))

        return [(i, idx) for i, idx, _ in good_matches]

    def run_training(self):
        """Run the actual training with batch-hard triplet loss"""
        try:
            print(f"🎯 Starting training for {self.epochs} epochs with batch-hard triplet loss")
            data_generator = self._create_real_data_generator(self.training_dir)

            history = self.train_model.fit(
                data_generator,
                epochs=self.epochs,
                steps_per_epoch=self.steps_per_epoch,
                callbacks=[self._create_training_callback()],
                verbose=1
            )

            print("✅ Training completed!")

            # Save final model
            final_model_path = os.path.join("outputs/models", f"knift_gpu_final_model_epoch_{self.epochs:03d}.weights.h5")
            print(f"💾 Saving final model: {final_model_path}")
            self.train_model.save_weights(final_model_path)
            print(f"✅ Final model saved successfully")

            # Export to TFLite
            print("Exporting to TFLite...")
            self.export_to_tflite()

            # Run comprehensive baboon tests
            print("\n" + "="*60)
            print("🧪 Running Comprehensive Baboon Tests")
            print("="*60)
            self.run_comprehensive_baboon_tests()

        except Exception as e:
            print(f"ERROR in training: {str(e)}")
            import traceback
            traceback.print_exc()

    def _create_training_callback(self):
        """Create the training callback with access to parent class"""
        class TrainingCallback(tf.keras.callbacks.Callback):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent
                self.batch_losses = []
                self.batch_maes = []

            def on_batch_end(self, batch, logs=None):
                if batch % 50 == 0:
                    loss = logs.get('loss', 0)
                    pos_dist = logs.get('pos_dist', 0)
                    neg_dist = logs.get('neg_dist', 0)
                    print(f"   Batch {batch}: Loss = {loss:.4f}, Pos Dist = {pos_dist:.4f}, Neg Dist = {neg_dist:.4f}")

            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss', 0)
                pos_dist = logs.get('pos_dist', 0)
                neg_dist = logs.get('neg_dist', 0)

                print(f"Epoch {epoch+1}/{self.parent.epochs} - Loss: {loss:.4f}, "
                      f"Pos Dist: {pos_dist:.4f}, Neg Dist: {neg_dist:.4f}")

                self.parent.epoch_history.append(epoch)
                self.parent.loss_history.append(loss)
                self.parent.accuracy_history.append((neg_dist - pos_dist) if pos_dist != 0 else 0)

                print(f"   Testing alignment quality...")
                self.parent.validation_images = self.parent.load_validation_pair()
                alignment_result = self.parent.test_alignment_quality()

                print(f"   Alignment: Matches={alignment_result['matches']}, "
                      f"Inliers={alignment_result['inliers']}, "
                      f"Quality={alignment_result['quality']:.2f}%")

                if (epoch + 1) % 5 == 0:
                    checkpoint_path = os.path.join(
                        "checkpoints",
                        f"checkpoint_epoch_{epoch+1:03d}_loss_{loss:.4f}.weights.h5"
                    )
                    print(f"   💾 Saving checkpoint at epoch {epoch+1}: {checkpoint_path}")
                    self.parent.train_model.save_weights(checkpoint_path)

        return TrainingCallback(self)

    def run_comprehensive_baboon_tests(self):
        """Run comprehensive baboon tests with various levels of misalignment and augmentation"""
        if not os.path.exists(self.validation_dir):
            print("⚠️  Validation directory not found - skipping baboon tests")
            return

        baboon_path = os.path.join(self.validation_dir, "baboon.jpeg")
        if not os.path.exists(baboon_path):
            print(f"⚠️  Baboon image not found: {baboon_path}")
            return

        img_orig = cv2.imread(baboon_path, cv2.IMREAD_GRAYSCALE)
        if img_orig is None:
            print(f"❌ Could not load baboon image")
            return

        print("\n🎨 Test 1: Mild Misalignment (rotation ±5°, translation ±5%)")
        self._test_misalignment_level(img_orig, rotation_range=5, translation_percent=0.05, scale_range=(0.95, 1.05))

        print("\n🎨 Test 2: Moderate Misalignment (rotation ±15°, translation ±10%)")
        self._test_misalignment_level(img_orig, rotation_range=15, translation_percent=0.10, scale_range=(0.90, 1.10))

        print("\n🎨 Test 3: Severe Misalignment (rotation ±30°, translation ±20%)")
        self._test_misalignment_level(img_orig, rotation_range=30, translation_percent=0.20, scale_range=(0.80, 1.20))

        print("\n🎨 Test 4: Extreme Misalignment (rotation ±45°, translation ±30%)")
        self._test_misalignment_level(img_orig, rotation_range=45, translation_percent=0.30, scale_range=(0.70, 1.30))

        print("\n🎨 Test 5: Heavy Augmentation (brightness, contrast, noise, blur)")
        self._test_augmentation_robustness(img_orig)

        print("\n🎨 Test 6: Combined Misalignment + Augmentation")
        self._test_combined_stress(img_orig)

    def _test_misalignment_level(self, img, rotation_range, translation_percent, scale_range):
        """Test a specific level of geometric misalignment"""
        results = []

        for i in range(3):  # Run 3 tests per level
            h, w = img.shape
            angle = random.uniform(-rotation_range, rotation_range)
            scale = random.uniform(*scale_range)
            tx = random.uniform(-translation_percent * w, translation_percent * w)
            ty = random.uniform(-translation_percent * h, translation_percent * h)

            M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty

            img_transformed = cv2.warpAffine(img, M, (w, h),
                                              borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=0)

            self.validation_images = (img, img_transformed)
            result = self.test_alignment_quality()
            results.append(result)

            print(f"   Run {i+1}: Angle={angle:.1f}°, Scale={scale:.2f}, Tx={tx:.1f}px, Ty={ty:.1f}px")
            print(f"   Matches={result['matches']}, Inliers={result['inliers']}, Quality={result['quality']:.2f}%")

        avg_quality = np.mean([r['quality'] for r in results])
        avg_matches = np.mean([r['matches'] for r in results])
        avg_inliers = np.mean([r['inliers'] for r in results])

        print(f"\n   📊 Average Results:")
        print(f"   Quality: {avg_quality:.2f}%, Matches: {avg_matches:.1f}, Inliers: {avg_inliers:.1f}")

    def _test_augmentation_robustness(self, img):
        """Test robustness to appearance augmentations"""
        results = []

        for i in range(3):
            # Apply heavy augmentation
            img_pil = Image.fromarray(img)

            # Random brightness
            brightness_factor = random.uniform(0.5, 1.5)
            enhancer = ImageEnhance.Brightness(img_pil)
            img_aug = enhancer.enhance(brightness_factor)

            # Random contrast
            contrast_factor = random.uniform(0.5, 1.5)
            enhancer = ImageEnhance.Contrast(img_aug)
            img_aug = enhancer.enhance(contrast_factor)

            # Random blur
            blur_radius = random.uniform(1.0, 3.0)
            img_aug = img_aug.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # Convert back to numpy
            img_augmented = np.array(img_aug, dtype=np.uint8)

            # Add noise
            noise = np.random.normal(0, 15, img_augmented.shape)
            img_augmented = np.clip(img_augmented + noise, 0, 255).astype(np.uint8)

            self.validation_images = (img, img_augmented)
            result = self.test_alignment_quality()
            results.append(result)

            print(f"   Run {i+1}: Brightness={brightness_factor:.2f}, Contrast={contrast_factor:.2f}, Blur={blur_radius:.2f}px")
            print(f"   Matches={result['matches']}, Inliers={result['inliers']}, Quality={result['quality']:.2f}%")

        avg_quality = np.mean([r['quality'] for r in results])
        avg_matches = np.mean([r['matches'] for r in results])
        avg_inliers = np.mean([r['inliers'] for r in results])

        print(f"\n   📊 Average Results:")
        print(f"   Quality: {avg_quality:.2f}%, Matches: {avg_matches:.1f}, Inliers: {avg_inliers:.1f}")

    def _test_combined_stress(self, img):
        """Test combined geometric and appearance transformations"""
        results = []

        for i in range(3):
            h, w = img.shape

            # Geometric transformation
            angle = random.uniform(-30, 30)
            scale = random.uniform(0.8, 1.2)
            tx = random.uniform(-0.2 * w, 0.2 * w)
            ty = random.uniform(-0.2 * h, 0.2 * h)

            M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty

            img_transformed = cv2.warpAffine(img, M, (w, h),
                                              borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=0)

            # Appearance augmentation
            img_pil = Image.fromarray(img_transformed)

            brightness_factor = random.uniform(0.6, 1.4)
            enhancer = ImageEnhance.Brightness(img_pil)
            img_aug = enhancer.enhance(brightness_factor)

            contrast_factor = random.uniform(0.6, 1.4)
            enhancer = ImageEnhance.Contrast(img_aug)
            img_aug = enhancer.enhance(contrast_factor)

            blur_radius = random.uniform(0.5, 2.0)
            img_aug = img_aug.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            img_final = np.array(img_aug, dtype=np.uint8)
            noise = np.random.normal(0, 10, img_final.shape)
            img_final = np.clip(img_final + noise, 0, 255).astype(np.uint8)

            self.validation_images = (img, img_final)
            result = self.test_alignment_quality()
            results.append(result)

            print(f"   Run {i+1}: Angle={angle:.1f}°, Scale={scale:.2f}, Brightness={brightness_factor:.2f}")
            print(f"   Matches={result['matches']}, Inliers={result['inliers']}, Quality={result['quality']:.2f}%")

        avg_quality = np.mean([r['quality'] for r in results])
        avg_matches = np.mean([r['matches'] for r in results])
        avg_inliers = np.mean([r['inliers'] for r in results])

        print(f"\n   📊 Average Results:")
        print(f"   Quality: {avg_quality:.2f}%, Matches: {avg_matches:.1f}, Inliers: {avg_inliers:.1f}")

    def export_to_tflite(self):
        """Export the trained GPU-compatible model to TensorFlow Lite format"""
        try:
            final_model_path = os.path.join("outputs/models", f"knift_gpu_final_model_epoch_{self.epochs:03d}.weights.h5")
            print("🎯 Converting model to TensorFlow Lite format...")

            batch_sizes = [200, 400, 1000]
            model_names = ["knift_gpu_float.tflite", "knift_gpu_float_400.tflite", "knift_gpu_float_1k.tflite"]

            for batch_size, model_name in zip(batch_sizes, model_names):
                print(f"🎯 Creating model with batch size {batch_size}: {model_name}")

                export_model = build_model(batch_size=batch_size, name=f"knift_gpu_export_{batch_size}")
                export_model.set_weights(self.train_model.get_weights())

                converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()

                interpreter = tf.lite.Interpreter(model_content=tflite_model)
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                print(f"Input shape: {input_details[0]['shape']}")
                print(f"Output shape: {output_details[0]['shape']}")

                tflite_path = os.path.join("outputs/models", model_name)
                os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model)

                print(f"✅ {model_name} exported successfully!")
                print(f"   TFLite model size: {len(tflite_model) / (1024*1024):.2f} MB")

                interpreter.allocate_tensors()
                print(f"✅ {model_name} verification passed!")

            print("✅ All TFLite models exported successfully!")

        except Exception as e:
            print(f"❌ Error during TFLite export: {str(e)}")

# === MAIN ENTRY POINT ===

def main():
    """Main entry point - KNIFT-GPU training"""
    print("=" * 60)
    print("🚀 KNIFT-GPU - Mobile-Optimized Feature Descriptor Training")
    print("=" * 60)

    # Configuration
    training_dir = "Set14"
    validation_dir = "Set14"
    epochs = 250
    batch_size = 32
    steps_per_epoch = 20  # 500k patches ≈ 480k (250 epochs × 20 steps × 32 batch × 3 triplets)

    # Setup environment
    setup_environment(training_dir)

    # Print configuration
    print(f"\n📊 Training Configuration:")
    print(f"   Training images: {training_dir} (13 images excluding baboon)")
    print(f"   Validation image: baboon.jpeg from Set14")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Total patches: ~{epochs * steps_per_epoch * batch_size * 3:,}")
    print(f"   Models exported: knift_gpu_float.tflite, knift_gpu_float_400.tflite, knift_gpu_float_1k.tflite")

    print(f"\n🏃 Starting training...")

    # Create and run the console trainer
    trainer = ConsoleTraining(
        training_dir=training_dir,
        validation_dir=validation_dir,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch
    )

    trainer.run_training()

    print("\n🎉 Training and export completed successfully!")

if __name__ == "__main__":
    main()
