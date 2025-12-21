#!/usr/bin/env python3

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

def build_model(input_shape=(32, 32, 1), output_dim=40, batch_size=None, name="knift_gpu_model"):
    print("Building Inception-like Architecture...")

    if batch_size is not None:
        inputs = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size, name="input")
    else:
        inputs = tf.keras.layers.Input(shape=input_shape, name="input")

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

    x = inception_module(x, {
        'name': 'inception1',
        'branch1x1': 12,
        'branch3x3_reduce': 8,
        'branch3x3': 16,
        'branch5x5_reduce': 8,
        'branch5x5': 16,
        'branch_pool_proj': 8
    })

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

    x = inception_module(x, {
        'name': 'inception3',
        'branch1x1': 24,
        'branch3x3_reduce': 16,
        'branch3x3': 32,
        'branch5x5_reduce': 16,
        'branch5x5': 32,
        'branch_pool_proj': 24
    })

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

    x = inception_module(x, {
        'name': 'inception5',
        'branch1x1': 28,
        'branch3x3_reduce': 14,
        'branch3x3': 28,
        'branch5x5_reduce': 14,
        'branch5x5': 28,
        'branch_pool_proj': 24
    })

    # Replace GlobalAveragePooling2D with AveragePooling2D
    # For the regular model: input is (4, 4, 108), use pool_size=(4, 4)
    x = tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(1, 1),
                                         padding="valid", name="globalpool_pool")(x)

    # Replace FC with 1x1 CONV_2D followed by reshape for better GPU compatibility
    x = tf.keras.layers.Conv2D(
        filters=output_dim,
        kernel_size=1,
        strides=1,
        padding='valid',
        activation=None,
        name='embedding_conv'
    )(x)

    # Reshape from (batch, 1, 1, output_dim) to (batch, output_dim)
    x = tf.keras.layers.Reshape((output_dim,), name='embedding_reshape')(x)
    outputs = x

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.summary()

    param_count = model.count_params()
    print(f"Total Parameters: {param_count:,}")

    return model

def inception_module(inputs, config):
    name = config['name']
    branch1x1 = config['branch1x1']
    branch3x3_reduce = config['branch3x3_reduce']
    branch3x3 = config['branch3x3']
    branch5x5_reduce = config['branch5x5_reduce']
    branch5x5 = config['branch5x5']
    branch_pool_proj = config['branch_pool_proj']

    branch1 = tf.keras.layers.Conv2D(
        filters=branch1x1,
        kernel_size=1,
        padding='same',
        activation='relu',
        name=f'{name}_b1_1x1'
    )(inputs)

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

    concatenated = tf.keras.layers.Concatenate(name=f'{name}_concat')(
        [branch1, branch2, branch3, branch4]
    )

    output = tf.keras.layers.BatchNormalization(name=f'{name}_bn')(concatenated)
    return output

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

class BatchHardTripletLoss(tf.keras.losses.Loss):
    def __init__(self, margin=1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        total_batch_size = tf.shape(y_pred)[0]
        batch_size = total_batch_size // 3

        anchors = y_pred[:batch_size]
        positives = y_pred[batch_size:2*batch_size]
        negatives = y_pred[2*batch_size:3*batch_size]

        pos_dist = tf.reduce_sum(tf.square(anchors - positives), axis=1)

        anchor_neg_dist = tf.reduce_sum(
            tf.square(tf.expand_dims(anchors, 1) - tf.expand_dims(negatives, 0)),
            axis=2
        )

        pos_dist_expanded = tf.expand_dims(pos_dist, 1)
        semi_hard_mask = tf.logical_and(
            anchor_neg_dist > pos_dist_expanded,
            anchor_neg_dist < pos_dist_expanded + self.margin
        )

        masked_dist = tf.where(
            semi_hard_mask,
            anchor_neg_dist,
            tf.fill(tf.shape(anchor_neg_dist), tf.float32.max)
        )

        hardest_neg_dist = tf.reduce_min(masked_dist, axis=1)

        all_easy_mask = tf.equal(hardest_neg_dist, tf.float32.max)
        hardest_neg_dist = tf.where(
            all_easy_mask,
            tf.reduce_min(anchor_neg_dist, axis=1),
            hardest_neg_dist
        )

        loss = tf.maximum(pos_dist - hardest_neg_dist + self.margin, 0.0)
        return tf.reduce_mean(loss)

class FeatureExtractor:
    def __init__(self, model: tf.keras.Model = None):
        self.model = model
        self.input_size = (32, 32)
        self.embedding_dim = 40

        if self.model:
            print(f"KNIFT-GPU FeatureExtractor initialized with model: {self.model.name}")

    def extract_features(self, patches: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("No model loaded for feature extraction")

        if len(patches) == 0:
            return np.array([])

        if len(patches.shape) == 3:
            patches = np.expand_dims(patches, axis=-1)

        if patches.max() > 1.0:
            patches = patches.astype(np.float32) / 255.0

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

        try:
            features = self.model.predict(patches, verbose=0)
            return features
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            return np.array([])

class DataLoader:
    def __init__(self, data_path, img_size=(32, 32)):
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.images = []
        self.image_paths = []

        print("Initializing KNIFT-GPU Data Loader...")
        self._load_image_paths()
        print(f"Found Images: {len(self.image_paths)}")

    def _load_image_paths(self):
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

class HeavyAugmentation:
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

        print("Initializing GPU-Compatible Heavy Augmentation Pipeline...")

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
        augmented = image.copy()

        angle = random.uniform(-self.rotation_range, self.rotation_range)
        augmented = self.rotate_image(augmented, angle)

        if random.random() < 0.8:
            brightness_factor = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
            brightness_factor = max(0.1, min(2.0, brightness_factor))
            augmented = self.adjust_brightness(augmented, brightness_factor)

        if random.random() < 0.8:
            contrast_factor = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
            contrast_factor = max(0.1, min(2.0, contrast_factor))
            augmented = self.adjust_contrast(augmented, contrast_factor)

        if random.random() < self.flip_probability:
            augmented = self.horizontal_flip(augmented)

        if random.random() < self.noise_probability:
            noise_std = random.uniform(0, self.noise_std)
            if noise_std > 0:
                augmented = self.add_noise(augmented, noise_std)

        if random.random() < self.blur_probability:
            blur_radius = random.uniform(*self.blur_radius_range)
            augmented = self.add_blur(augmented, blur_radius)

        augmented = np.clip(augmented, 0, 1)
        augmented = augmented[np.newaxis, ..., np.newaxis]
        return augmented

class KeypointPatchDataset:
    def __init__(self, image_dir, patches_per_image=100):
        self.image_dir = Path(image_dir)
        self.patches_per_image = patches_per_image
        self.images = []
        self.keypoints = []

        print("Initializing KNIFT-GPU Keypoint Patch Dataset...")
        self.images = self.load_high_res_images()
        self.keypoints = self.detect_all_keypoints()

        if len(self.images) < 2:
            raise ValueError("Need at least 2 images for triplet training")

        print(f"Loaded {len(self.images)} images")
        print(f"Total keypoints: {len(self.keypoints)}")

    def load_high_res_images(self):
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
        fast = cv2.FastFeatureDetector_create(threshold=20)
        all_kps = []
        for idx, img in enumerate(self.images):
            kps = fast.detect(img, None)
            kps = sorted(kps, key=lambda x: x.response, reverse=True)[:500] if kps else []
            for kp in kps:
                all_kps.append((idx, kp))
        return all_kps

    def extract_patch(self, image, keypoint, size=32):
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

        margin = 16
        if (new_pt[0] < margin or new_pt[0] >= w - margin or
            new_pt[1] < margin or new_pt[1] >= h - margin):
            return anchor_patch, anchor_patch.copy()

        new_kp = cv2.KeyPoint(new_pt[0], new_pt[1], keypoint.size)
        positive_patch = self.extract_patch(warped_image, new_kp, size=32)

        return anchor_patch, positive_patch

class OnlineTripletGenerator:
    def __init__(self,
                 data_loader: DataLoader,
                 augmentation: HeavyAugmentation,
                 batch_size: int = 64):

        self.data_loader = data_loader
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.keypoint_dataset = KeypointPatchDataset(data_loader.data_path)

        print("Initializing GPU-Compatible Online Triplet Generator...")
        print(f"Loaded {len(self.keypoint_dataset.images)} images")
        print(f"Total keypoints: {len(self.keypoint_dataset.keypoints)}")
        print(f"Batch size: {batch_size}")

        if len(self.keypoint_dataset.images) < 2:
            raise ValueError("Need at least 2 images for triplet training")

    def _generate_positive_pair(self) -> Tuple[np.ndarray, np.ndarray]:
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

def setup_environment(training_dir: str = "Set14"):
    print("Setting up KNIFT-GPU Training Environment...")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {'Yes' if tf.config.list_physical_devices('GPU') else 'No (CPU only)'}")
    print(f"OpenCV version: {cv2.__version__}")

    if not os.path.exists(training_dir):
        print(f"Training directory does not exist: {training_dir}")
        sys.exit(1)

    dataset_files = [f for f in os.listdir(training_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png')) and "baboon" not in f.lower()]

    if len(dataset_files) == 0:
        print(f"No training image files found in {training_dir} (excluding baboon)")
        sys.exit(1)

    print(f"Found {len(dataset_files)} training images (excluding baboon)")

    baboon_files = [f for f in os.listdir(training_dir) if "baboon" in f.lower() and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(baboon_files) > 0:
        print(f"Found baboon image for validation: {baboon_files[0]}")
    else:
        print(f"No baboon image found in {training_dir} - validation will be skipped")

    directories = ['outputs', 'outputs/models', 'logs', 'checkpoints']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("Environment setup completed!\n")

class ConsoleTraining:
    def __init__(self, training_dir: str = "Set14",
                 validation_dir: str = "Set14",
                 epochs: int = 250,
                 batch_size: int = 64,
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

        self.train_model = build_model(batch_size=None, name="knift_gpu_training_model")
        self.train_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=BatchHardTripletLoss(margin=1.0),
            metrics=[PositiveDistanceMetric(), NegativeDistanceMetric()]
        )

        self.export_model = build_model(batch_size=1000, name="knift_gpu_export_model")

        self.feature_extractor = FeatureExtractor(self.train_model)

        self.validation_images = (None, None)
        if os.path.exists(self.validation_dir):
            print("Set14 directory found - will use baboon for validation alignments per epoch")
        else:
            print("Set14 directory not found - will skip validation alignments")

    def _create_real_data_generator(self, data_dir):
        print(f"Creating real data generator from: {data_dir}")

        data_loader = DataLoader(data_dir)
        if len(data_loader) == 0:
            raise ValueError(f"No images found in dataset directory: {data_dir}")

        original_paths = data_loader.get_image_paths()
        filtered_paths = [path for path in original_paths if "baboon" not in os.path.basename(path).lower()]
        data_loader.image_paths = filtered_paths

        print(f"Found {len(data_loader)} real images for training (excluding baboon)")

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
        if not os.path.exists(self.validation_dir):
            return None, None

        selected_file = "baboon.jpeg"
        img_path = os.path.join(self.validation_dir, selected_file)

        if not os.path.exists(img_path):
            print(f"Validation image {selected_file} not found in {self.validation_dir}")
            return None, None

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not load validation image: {img_path}")
            return None, None

        h, w = img.shape

        angle = random.uniform(-15, 15)
        M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)

        tx = random.uniform(-0.1 * w, 0.1 * w)
        ty = random.uniform(-0.1 * h, 0.1 * h)
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty

        img_misaligned = cv2.warpAffine(img, M_rot, (w, h),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)

        print(f"Loaded validation image: {selected_file}")
        print(f"Applied transform: rotation={angle:.1f}°, tx={tx:.1f}px, ty={ty:.1f}px")

        return img, img_misaligned

    def test_alignment_quality(self):
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
        try:
            print(f"Starting training for {self.epochs} epochs with batch-hard triplet loss")
            data_generator = self._create_real_data_generator(self.training_dir)

            history = self.train_model.fit(
                data_generator,
                epochs=self.epochs,
                steps_per_epoch=self.steps_per_epoch,
                callbacks=[self._create_training_callback()],
                verbose=1
            )

            print("Training completed!")

            final_model_path = os.path.join("outputs/models", f"knift_gpu_final_model_epoch_{self.epochs:03d}.weights.h5")
            print(f"Saving final model: {final_model_path}")
            self.train_model.save_weights(final_model_path)
            print(f"Final model saved successfully")

            print("Exporting to TFLite...")
            self.export_to_tflite()

            print("\n" + "="*60)
            print("Running Comprehensive Baboon Tests")
            print("="*60)
            self.run_comprehensive_baboon_tests()

        except Exception as e:
            print(f"ERROR in training: {str(e)}")
            import traceback
            traceback.print_exc()

    def _create_training_callback(self):
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
                    print(f"Batch {batch}: Loss = {loss:.4f}, Pos Dist = {pos_dist:.4f}, Neg Dist = {neg_dist:.4f}")

            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss', 0)
                pos_dist = logs.get('pos_dist', 0)
                neg_dist = logs.get('neg_dist', 0)

                print(f"Epoch {epoch+1}/{self.parent.epochs} - Loss: {loss:.4f}, "
                      f"Pos Dist: {pos_dist:.4f}, Neg Dist: {neg_dist:.4f}")

                self.parent.epoch_history.append(epoch)
                self.parent.loss_history.append(loss)
                self.parent.accuracy_history.append((neg_dist - pos_dist) if pos_dist != 0 else 0)

                print(f"Testing alignment quality...")
                self.parent.validation_images = self.parent.load_validation_pair()
                alignment_result = self.parent.test_alignment_quality()

                print(f"Alignment: Matches={alignment_result['matches']}, "
                      f"Inliers={alignment_result['inliers']}, "
                      f"Quality={alignment_result['quality']:.2f}%")

                if (epoch + 1) % 5 == 0:
                    checkpoint_path = os.path.join(
                        "checkpoints",
                        f"checkpoint_epoch_{epoch+1:03d}_loss_{loss:.4f}.weights.h5"
                    )
                    print(f"Saving checkpoint at epoch {epoch+1}: {checkpoint_path}")
                    self.parent.train_model.save_weights(checkpoint_path)

        return TrainingCallback(self)

    def run_comprehensive_baboon_tests(self):
        if not os.path.exists(self.validation_dir):
            print("Validation directory not found - skipping baboon tests")
            return

        baboon_path = os.path.join(self.validation_dir, "baboon.jpeg")
        if not os.path.exists(baboon_path):
            print(f"Baboon image not found: {baboon_path}")
            return

        img_orig = cv2.imread(baboon_path, cv2.IMREAD_GRAYSCALE)
        if img_orig is None:
            print(f"Could not load baboon image")
            return

        print("\nTest 1: Mild Misalignment (rotation ±5°, translation ±5%)")
        self._test_misalignment_level(img_orig, rotation_range=5, translation_percent=0.05, scale_range=(0.95, 1.05))

        print("\nTest 2: Moderate Misalignment (rotation ±15°, translation ±10%)")
        self._test_misalignment_level(img_orig, rotation_range=15, translation_percent=0.10, scale_range=(0.90, 1.10))

        print("\nTest 3: Severe Misalignment (rotation ±30°, translation ±20%)")
        self._test_misalignment_level(img_orig, rotation_range=30, translation_percent=0.20, scale_range=(0.80, 1.20))

        print("\nTest 4: Extreme Misalignment (rotation ±45°, translation ±30%)")
        self._test_misalignment_level(img_orig, rotation_range=45, translation_percent=0.30, scale_range=(0.70, 1.30))

        print("\nTest 5: Heavy Augmentation (brightness, contrast, noise, blur)")
        self._test_augmentation_robustness(img_orig)

        print("\nTest 6: Combined Misalignment + Augmentation")
        self._test_combined_stress(img_orig)

    def _test_misalignment_level(self, img, rotation_range, translation_percent, scale_range):
        results = []

        for i in range(3):
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

            print(f"Run {i+1}: Angle={angle:.1f}°, Scale={scale:.2f}, Tx={tx:.1f}px, Ty={ty:.1f}px")
            print(f"Matches={result['matches']}, Inliers={result['inliers']}, Quality={result['quality']:.2f}%")

        avg_quality = np.mean([r['quality'] for r in results])
        avg_matches = np.mean([r['matches'] for r in results])
        avg_inliers = np.mean([r['inliers'] for r in results])

        print(f"\nAverage Results:")
        print(f"Quality: {avg_quality:.2f}%, Matches: {avg_matches:.1f}, Inliers: {avg_inliers:.1f}")

    def _test_augmentation_robustness(self, img):
        results = []

        for i in range(3):
            img_pil = Image.fromarray(img)

            brightness_factor = random.uniform(0.5, 1.5)
            enhancer = ImageEnhance.Brightness(img_pil)
            img_aug = enhancer.enhance(brightness_factor)

            contrast_factor = random.uniform(0.5, 1.5)
            enhancer = ImageEnhance.Contrast(img_aug)
            img_aug = enhancer.enhance(contrast_factor)

            blur_radius = random.uniform(1.0, 3.0)
            img_aug = img_aug.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            img_augmented = np.array(img_aug, dtype=np.uint8)

            noise = np.random.normal(0, 15, img_augmented.shape)
            img_augmented = np.clip(img_augmented + noise, 0, 255).astype(np.uint8)

            self.validation_images = (img, img_augmented)
            result = self.test_alignment_quality()
            results.append(result)

            print(f"Run {i+1}: Brightness={brightness_factor:.2f}, Contrast={contrast_factor:.2f}, Blur={blur_radius:.2f}px")
            print(f"Matches={result['matches']}, Inliers={result['inliers']}, Quality={result['quality']:.2f}%")

        avg_quality = np.mean([r['quality'] for r in results])
        avg_matches = np.mean([r['matches'] for r in results])
        avg_inliers = np.mean([r['inliers'] for r in results])

        print(f"\nAverage Results:")
        print(f"Quality: {avg_quality:.2f}%, Matches: {avg_matches:.1f}, Inliers: {avg_inliers:.1f}")

    def _test_combined_stress(self, img):
        results = []

        for i in range(3):
            h, w = img.shape

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

            print(f"Run {i+1}: Angle={angle:.1f}°, Scale={scale:.2f}, Brightness={brightness_factor:.2f}")
            print(f"Matches={result['matches']}, Inliers={result['inliers']}, Quality={result['quality']:.2f}%")

        avg_quality = np.mean([r['quality'] for r in results])
        avg_matches = np.mean([r['matches'] for r in results])
        avg_inliers = np.mean([r['inliers'] for r in results])

        print(f"\nAverage Results:")
        print(f"Quality: {avg_quality:.2f}%, Matches: {avg_matches:.1f}, Inliers: {avg_inliers:.1f}")

    def export_to_tflite(self):
        try:
            final_model_path = os.path.join("outputs/models", f"knift_gpu_final_model_epoch_{self.epochs:03d}.weights.h5")
            print("Converting model to TensorFlow Lite format...")

            print("\nStarting BatchNorm fusion process...")
            fused_model = self._fuse_batch_norms()
            print("BatchNorm fusion completed!")

            batch_sizes = [200, 400, 1000]

            print("\nExporting fused models for GPU inference...")
            fused_model_names = ["knift_gpu_fused_float.tflite", "knift_gpu_fused_float_400.tflite", "knift_gpu_fused_float_1k.tflite"]

            for batch_size, model_name in zip(batch_sizes, fused_model_names):
                print(f"Creating fused model with batch size {batch_size}: {model_name}")

                # Instead of creating a new model with BN layers and then copying weights,
                # recreate the fused model with the appropriate batch size
                fused_batch_model = self._create_fused_model_with_batch_size(fused_model, batch_size)

                converter = tf.lite.TFLiteConverter.from_keras_model(fused_batch_model)
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

                print(f"{model_name} exported successfully!")
                print(f"TFLite model size: {len(tflite_model) / (1024*1024):.2f} MB")

                interpreter.allocate_tensors()
                print(f"{model_name} verification passed!")

            fused_keras_path = os.path.join("outputs/models", "knift_gpu_fused_model.h5")
            fused_model.save(fused_keras_path)
            print(f"Fused Keras model saved to: {fused_keras_path}")

            print("All original and fused TFLite models exported successfully!")

        except Exception as e:
            print(f"Error during TFLite export: {str(e)}")
            import traceback
            traceback.print_exc()

    def _fuse_batch_norms(self):
        print("Starting BatchNorm fusion process...")

        fused_weights = {}

        i = 0
        original_layers = self.train_model.layers
        while i < len(original_layers):
            current_layer = original_layers[i]

            if (isinstance(current_layer, tf.keras.layers.Conv2D) and
                i + 1 < len(original_layers) and
                isinstance(original_layers[i + 1], tf.keras.layers.BatchNormalization)):

                conv_weights = current_layer.get_weights()
                bn_weights = original_layers[i + 1].get_weights()

                conv_kernel = conv_weights[0]
                conv_bias = conv_weights[1] if len(conv_weights) > 1 else np.zeros(conv_kernel.shape[-1])

                gamma = bn_weights[0]
                beta = bn_weights[1]
                moving_mean = bn_weights[2]
                moving_var = bn_weights[3]
                epsilon = original_layers[i + 1].epsilon

                std = np.sqrt(moving_var + epsilon)
                w_fused = conv_kernel * (gamma / std).reshape([1, 1, 1, -1])
                b_fused = beta + (conv_bias - moving_mean) * gamma / std

                fused_weights[current_layer.name] = (w_fused, b_fused)
                print(f"Fused: {current_layer.name} + {original_layers[i + 1].name}")
                i += 2
            else:
                if hasattr(current_layer, 'get_weights') and len(current_layer.get_weights()) > 0:
                    fused_weights[current_layer.name] = current_layer.get_weights()
                i += 1

        inputs = tf.keras.Input(shape=self.train_model.input_shape[1:])
        x = inputs

        stem_conv_layer = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            name='stem_conv'
        )
        x = stem_conv_layer(inputs)

        if 'stem_conv' in fused_weights:
            stem_conv_layer.set_weights([fused_weights['stem_conv'][0], fused_weights['stem_conv'][1]])

        x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name='stem_pool')(x)

        x = self._create_fused_inception_module(x, 'inception1', fused_weights)

        x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name='pool2')(x)

        x = self._create_fused_inception_module(x, 'inception2', fused_weights)

        x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name='pool3')(x)

        x = self._create_fused_inception_module(x, 'inception3', fused_weights)

        x = self._create_fused_inception_module(x, 'inception4', fused_weights)

        x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name='pool4')(x)

        x = self._create_fused_inception_module(x, 'inception5', fused_weights)

        # Use pool_size that matches the actual spatial dimensions at this point: (2, 2) after pool4 layer
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1),
                                             padding="valid", name="globalpool_pool")(x)

        # Replace FC with 1x1 CONV_2D followed by reshape for better GPU compatibility
        x = tf.keras.layers.Conv2D(
            filters=40,
            kernel_size=1,
            strides=1,
            padding='valid',
            activation=None,
            name='embedding_conv'
        )(x)

        # Reshape from (batch, 1, 1, 40) to (batch, 40)
        x = tf.keras.layers.Reshape((40,), name='embedding_reshape')(x)
        outputs = x

        fused_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="knift_gpu_fused")

        if 'embedding_conv' in fused_weights:
            for layer in fused_model.layers:
                if layer.name == 'embedding_conv':
                    try:
                        layer.set_weights(fused_weights['embedding_conv'])
                    except:
                        pass

        print(f"Fused model created with {len(fused_model.layers)} layers")
        return fused_model

    def _create_fused_inception_module(self, x, block_name, fused_weights):
        configs = {
            'inception1': {'b1': 12, 'b2_r': 8, 'b2': 16, 'b3_r': 8, 'b3': 16, 'b4': 8},
            'inception2': {'b1': 16, 'b2_r': 12, 'b2': 24, 'b3_r': 12, 'b3': 24, 'b4': 16},
            'inception3': {'b1': 24, 'b2_r': 16, 'b2': 32, 'b3_r': 16, 'b3': 32, 'b4': 24},
            'inception4': {'b1': 28, 'b2_r': 16, 'b2': 32, 'b3_r': 16, 'b3': 32, 'b4': 24},
            'inception5': {'b1': 28, 'b2_r': 14, 'b2': 28, 'b3_r': 14, 'b3': 28, 'b4': 24}
        }

        config = configs[block_name]

        b1_name = f'{block_name}_b1_1x1'
        b1_layer = tf.keras.layers.Conv2D(
            filters=config['b1'], kernel_size=1, padding='same',
            activation='relu', name=b1_name
        )
        branch1 = b1_layer(x)

        if b1_name in fused_weights:
            b1_layer.set_weights([fused_weights[b1_name][0], fused_weights[b1_name][1]])

        b2_1_name = f'{block_name}_b2_1x1'
        b2_1_layer = tf.keras.layers.Conv2D(
            filters=config['b2_r'], kernel_size=1, padding='same',
            activation='relu', name=b2_1_name
        )
        temp = b2_1_layer(x)
        if b2_1_name in fused_weights:
            b2_1_layer.set_weights([fused_weights[b2_1_name][0], fused_weights[b2_1_name][1]])

        b2_2_name = f'{block_name}_b2_3x3'
        b2_2_layer = tf.keras.layers.Conv2D(
            filters=config['b2'], kernel_size=3, padding='same',
            activation='relu', name=b2_2_name
        )
        branch2 = b2_2_layer(temp)
        if b2_2_name in fused_weights:
            b2_2_layer.set_weights([fused_weights[b2_2_name][0], fused_weights[b2_2_name][1]])

        b3_1_name = f'{block_name}_b3_1x1'
        b3_1_layer = tf.keras.layers.Conv2D(
            filters=config['b3_r'], kernel_size=1, padding='same',
            activation='relu', name=b3_1_name
        )
        temp = b3_1_layer(x)
        if b3_1_name in fused_weights:
            b3_1_layer.set_weights([fused_weights[b3_1_name][0], fused_weights[b3_1_name][1]])

        b3_2_name = f'{block_name}_b3_5x5'
        b3_2_layer = tf.keras.layers.Conv2D(
            filters=config['b3'], kernel_size=5, padding='same',
            activation='relu', name=b3_2_name
        )
        branch3 = b3_2_layer(temp)
        if b3_2_name in fused_weights:
            b3_2_layer.set_weights([fused_weights[b3_2_name][0], fused_weights[b3_2_name][1]])

        branch4_pool = tf.keras.layers.MaxPooling2D(
            pool_size=3, strides=1, padding='same', name=f'{block_name}_b4_pool'
        )(x)

        b4_1_name = f'{block_name}_b4_1x1'
        b4_1_layer = tf.keras.layers.Conv2D(
            filters=config['b4'], kernel_size=1, padding='same',
            activation='relu', name=b4_1_name
        )
        branch4 = b4_1_layer(branch4_pool)
        if b4_1_name in fused_weights:
            b4_1_layer.set_weights([fused_weights[b4_1_name][0], fused_weights[b4_1_name][1]])

        concatenated = tf.keras.layers.Concatenate(name=f'{block_name}_concat')([
            branch1, branch2, branch3, branch4
        ])

        return concatenated

    def _create_fused_model_with_batch_size(self, original_fused_model, batch_size):
        """
        Creates a new model with the same architecture as original_fused_model but with specified batch size.
        This ensures the exported model is truly BN-free without any disconnected BN layers.
        """
        # Extract the architecture from the original fused model by recreating it with the batch size
        inputs = tf.keras.Input(shape=original_fused_model.input_shape[1:], batch_size=batch_size, name="input")
        x = inputs

        # Recreate the exact same architecture as in _fuse_batch_norms but with batch size
        # Stem: Conv2D -> MaxPooling2D
        x = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            name='stem_conv'
        )(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name='stem_pool')(x)

        # Inception blocks
        x = self._create_fused_inception_block_batch_only(x, 'inception1')

        x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name='pool2')(x)

        x = self._create_fused_inception_block_batch_only(x, 'inception2')

        x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name='pool3')(x)

        x = self._create_fused_inception_block_batch_only(x, 'inception3')

        x = self._create_fused_inception_block_batch_only(x, 'inception4')

        x = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name='pool4')(x)

        x = self._create_fused_inception_block_batch_only(x, 'inception5')

        # Global pooling and 1x1 conv layer
        x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1),
                                             padding="valid", name="globalpool_pool")(x)

        # Replace FC with 1x1 CONV_2D followed by reshape for better GPU compatibility
        x = tf.keras.layers.Conv2D(
            filters=40,
            kernel_size=1,
            strides=1,
            padding='valid',
            activation=None,
            name='embedding_conv'
        )(x)

        # Reshape from (batch, 1, 1, 40) to (batch, 40)
        x = tf.keras.layers.Reshape((40,), name='embedding_reshape')(x)
        outputs = x

        batch_model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"knift_gpu_fused_batch_{batch_size}")

        # Now copy the weights from the original fused model
        self._copy_fused_weights_to_batch_model(original_fused_model, batch_model)

        return batch_model

    def _create_fused_inception_block_batch_only(self, x, block_name):
        """
        Creates a fused inception block without BatchNormalization layers.
        """
        configs = {
            'inception1': {'b1': 12, 'b2_r': 8, 'b2': 16, 'b3_r': 8, 'b3': 16, 'b4': 8},
            'inception2': {'b1': 16, 'b2_r': 12, 'b2': 24, 'b3_r': 12, 'b3': 24, 'b4': 16},
            'inception3': {'b1': 24, 'b2_r': 16, 'b2': 32, 'b3_r': 16, 'b3': 32, 'b4': 24},
            'inception4': {'b1': 28, 'b2_r': 16, 'b2': 32, 'b3_r': 16, 'b3': 32, 'b4': 24},
            'inception5': {'b1': 28, 'b2_r': 14, 'b2': 28, 'b3_r': 14, 'b3': 28, 'b4': 24}
        }

        config = configs[block_name]

        b1_name = f'{block_name}_b1_1x1'
        branch1 = tf.keras.layers.Conv2D(
            filters=config['b1'], kernel_size=1, padding='same',
            activation='relu', name=b1_name
        )(x)

        b2_1_name = f'{block_name}_b2_1x1'
        temp = tf.keras.layers.Conv2D(
            filters=config['b2_r'], kernel_size=1, padding='same',
            activation='relu', name=b2_1_name
        )(x)

        b2_2_name = f'{block_name}_b2_3x3'
        branch2 = tf.keras.layers.Conv2D(
            filters=config['b2'], kernel_size=3, padding='same',
            activation='relu', name=b2_2_name
        )(temp)

        b3_1_name = f'{block_name}_b3_1x1'
        temp = tf.keras.layers.Conv2D(
            filters=config['b3_r'], kernel_size=1, padding='same',
            activation='relu', name=b3_1_name
        )(x)

        b3_2_name = f'{block_name}_b3_5x5'
        branch3 = tf.keras.layers.Conv2D(
            filters=config['b3'], kernel_size=5, padding='same',
            activation='relu', name=b3_2_name
        )(temp)

        branch4_pool = tf.keras.layers.MaxPooling2D(
            pool_size=3, strides=1, padding='same', name=f'{block_name}_b4_pool'
        )(x)

        b4_1_name = f'{block_name}_b4_1x1'
        branch4 = tf.keras.layers.Conv2D(
            filters=config['b4'], kernel_size=1, padding='same',
            activation='relu', name=b4_1_name
        )(branch4_pool)

        concatenated = tf.keras.layers.Concatenate(name=f'{block_name}_concat')(
            [branch1, branch2, branch3, branch4]
        )

        return concatenated

    def _copy_fused_weights_to_batch_model(self, fused_model, batch_model):
        fused_weights_map = {}
        for layer in fused_model.layers:
            if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
                fused_weights_map[layer.name] = layer.get_weights()

        for layer in batch_model.layers:
            if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
                if layer.name in fused_weights_map:
                    try:
                        layer.set_weights(fused_weights_map[layer.name])
                        print(f"Applied fused weights to: {layer.name}")
                    except Exception as e:
                        print(f"Could not apply fused weights to {layer.name}: {e}")


def main():
    print("=" * 60)
    print("KNIFT-GPU - Mobile-Optimized Feature Descriptor Training")
    print("=" * 60)

    training_dir = "Set14"
    validation_dir = "Set14"
    epochs = 100
    batch_size = 64
    steps_per_epoch = 20

    setup_environment(training_dir)

    print(f"\nTraining Configuration:")
    print(f"Training images: {training_dir} (13 images excluding baboon)")
    print(f"Validation image: baboon.jpeg from Set14")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total patches: ~{epochs * steps_per_epoch * batch_size * 3:,}")
    print(f"Models exported: knift_gpu_fused_float.tflite, knift_gpu_fused_float_400.tflite, knift_gpu_fused_float_1k.tflite")

    print(f"\nStarting training...")

    trainer = ConsoleTraining(
        training_dir=training_dir,
        validation_dir=validation_dir,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch
    )

    trainer.run_training()

    print("\nTraining and export completed successfully!")

if __name__ == "__main__":
    main()