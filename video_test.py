#!/usr/bin/env python3

"""
KNIFT Video Object Matcher (240p version)
This application matches sample images (sample1.jpg and sample2.jpg)
to objects in sample.mp4 using the KNIFT model with 240p downsampling for detection.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
import os
import time

class KNIFTVideoObjectMatcher:
    def __init__(self, root):
        self.root = root
        self.root.title("KNIFT Video Object Matcher (240p)")
        self.root.geometry("1200x800")

        # Initialize variables
        self.video_cap = None
        self.current_frame = None
        self.sample1_img = None
        self.sample2_img = None
        self.interpreter = None
        self.is_matching = False
        self.current_frame_number = 0
        self.total_frames = 0
        self.match_results = []

        # Frame skipping for performance
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_counter = 0

        # Store keypoints and descriptors
        self.sample1_keypoints = None
        self.sample1_descriptors = None
        self.sample2_keypoints = None
        self.sample2_descriptors = None
        # Store downscaled sample images for matching
        self.downscaled_sample1 = None
        self.downscaled_sample2 = None
        # Store match positions and keypoints for visualization
        self.match_positions = []
        self.current_frame_keypoints = None
        # Cache frame features for speed optimization
        self.last_frame_features = None  # Cache last frame
        self.frame_cache_id = -1

        # Keypoint visualization
        self.show_keypoints_var = tk.BooleanVar(value=True)  # Default to showing keypoints

        # Load images and video
        self.load_resources()

        # Create GUI
        self.create_widgets()

        # Load KNIFT model
        self.load_knift_model()

    def load_resources(self):
        """Load sample images and video"""
        try:
            # Load sample images
            sample1_path = os.path.join(os.path.dirname(__file__), "sample1.jpg")
            sample2_path = os.path.join(os.path.dirname(__file__), "sample2.jpg")
            video_path = os.path.join(os.path.dirname(__file__), "sample.mp4")

            if os.path.exists(sample1_path):
                self.sample1_img = cv2.imread(sample1_path)
                # Store actual image dimensions for UI layout
                if self.sample1_img is not None:
                    h, w = self.sample1_img.shape[:2]
                    # Calculate frame dimensions to match image aspect ratio, max 200px high
                    max_height = min(200, h)
                    aspect_ratio = w / h if h > 0 else 1
                    self.sample1_frame_height = max_height + 25  # Add space for title
                    self.sample1_frame_width = int(max_height * aspect_ratio)
                else:
                    # Default dimensions if image not loaded
                    self.sample1_frame_height = 200
                    self.sample1_frame_width = 200
            else:
                print(f"Warning: {sample1_path} not found")
                # Default dimensions if image not found
                self.sample1_frame_height = 200
                self.sample1_frame_width = 200

            if os.path.exists(sample2_path):
                self.sample2_img = cv2.imread(sample2_path)
                # Store actual image dimensions for UI layout
                if self.sample2_img is not None:
                    h, w = self.sample2_img.shape[:2]
                    # Calculate frame dimensions to match image aspect ratio, max 200px high
                    max_height = min(200, h)
                    aspect_ratio = w / h if h > 0 else 1
                    self.sample2_frame_height = max_height + 25  # Add space for title
                    self.sample2_frame_width = int(max_height * aspect_ratio)
                else:
                    # Default dimensions if image not loaded
                    self.sample2_frame_height = 200
                    self.sample2_frame_width = 200
            else:
                print(f"Warning: {sample2_path} not found")
                # Default dimensions if image not found
                self.sample2_frame_height = 200
                self.sample2_frame_width = 200

            if os.path.exists(video_path):
                self.video_cap = cv2.VideoCapture(video_path)
                self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                print(f"Warning: {video_path} not found")

        except Exception as e:
            print(f"Error loading resources: {e}")

    def create_widgets(self):
        """Create the main GUI widgets"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = ttk.Label(main_frame, text="KNIFT Video Object Matcher (240p)", font=("Arial", 16, "bold"))
        title_label.pack(pady=5)

        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        # Buttons
        self.start_btn = ttk.Button(control_frame, text="Start Matching", command=self.start_matching)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(control_frame, text="Stop Matching", command=self.stop_matching, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = ttk.Button(control_frame, text="Reset Video", command=self.reset_video)
        self.reset_btn.pack(side=tk.LEFT, padx=5)


        # Video and sample images display frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Calculate appropriate width for left frame based on sample images
        max_sample_width = max(self.sample1_frame_width, self.sample2_frame_width)

        # Left frame for sample images
        left_frame = ttk.LabelFrame(content_frame, text="Sample Images", width=max_sample_width + 20)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5)
        left_frame.pack_propagate(False)  # Prevent frame from shrinking to fit content

        # Sample 1 display
        sample1_frame = ttk.LabelFrame(left_frame, text="Sample 1", height=self.sample1_frame_height)
        sample1_frame.pack(fill=tk.X, padx=5, pady=5)
        sample1_frame.pack_propagate(False)  # Prevent frame from shrinking to fit content
        self.sample1_label = ttk.Label(sample1_frame, background="black")
        self.sample1_label.pack(fill=tk.BOTH, expand=True)

        # Sample 2 display
        sample2_frame = ttk.LabelFrame(left_frame, text="Sample 2", height=self.sample2_frame_height)
        sample2_frame.pack(fill=tk.X, padx=5, pady=5)
        sample2_frame.pack_propagate(False)  # Prevent frame from shrinking to fit content
        self.sample2_label = ttk.Label(sample2_frame, background="black")
        self.sample2_label.pack(fill=tk.BOTH, expand=True)

        # Right frame for video
        right_frame = ttk.LabelFrame(content_frame, text="Video Frame")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        right_frame.pack_propagate(False)

        self.video_label = ttk.Label(right_frame, text="Video will appear here", background="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Match visualization options
        vis_frame = ttk.Frame(right_frame)
        vis_frame.pack(fill=tk.X, padx=5, pady=5)

        self.show_matches_var = tk.BooleanVar(value=True)
        match_check = ttk.Checkbutton(vis_frame, text="Show Matches", variable=self.show_matches_var)
        match_check.pack(side=tk.LEFT, padx=5)

        keypoint_check = ttk.Checkbutton(vis_frame, text="Show Keypoints", variable=self.show_keypoints_var,
                                         command=self.toggle_keypoints_display)
        keypoint_check.pack(side=tk.LEFT, padx=5)

        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Matching Results")
        results_frame.pack(fill=tk.X, pady=5)

        # Results text widget
        self.results_text = tk.Text(results_frame, height=8)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to start matching")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Display sample images if available
        if self.sample1_img is not None:
            # Display sample image with keypoints if keypoints are available and visualization is enabled
            if self.show_keypoints_var.get() and self.sample1_keypoints is not None:
                img_with_kp = self.draw_keypoints(self.sample1_img, self.sample1_keypoints)
                self.display_image(img_with_kp, self.sample1_label)
            else:
                self.display_image(self.sample1_img, self.sample1_label)
        if self.sample2_img is not None:
            # Display sample image with keypoints if keypoints are available and visualization is enabled
            if self.show_keypoints_var.get() and self.sample2_keypoints is not None:
                img_with_kp = self.draw_keypoints(self.sample2_img, self.sample2_keypoints)
                self.display_image(img_with_kp, self.sample2_label)
            else:
                self.display_image(self.sample2_img, self.sample2_label)

    def load_knift_model(self):
        """Load the KNIFT model using the same approach as align_test.py"""
        model_paths = [
            "./outputs/models/knift_gpu_float.tflite"
        ]

        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    # Try XNNPACK delegate for CPU optimization
                    self.interpreter = tf.lite.Interpreter(
                        model_path=model_path,
                        num_threads=4
                    )
                    self.interpreter.allocate_tensors()
                    self.status_var.set(f"Loaded KNIFT model: {os.path.basename(model_path)}")
                    model_loaded = True

                    self.input_details = self.interpreter.get_input_details()
                    self.output_details = self.interpreter.get_output_details()
                    break
                except Exception as e:
                    print(f"Could not load model from {model_path}: {e}")

        if not model_loaded:
            self.status_var.set("Warning: Could not load KNIFT model - matching functionality will be limited")
            messagebox.showwarning("Model Warning", "Could not load KNIFT model. Matching functionality may be limited.")

    def extract_patches(self, img, grid_points, patch_size=32):
        """Extract patches around grid points - handles NORMALIZED coordinates (0..1)"""
        patches = []
        valid_indices = []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape

        for i, (x_norm, y_norm) in enumerate(grid_points):
            x = int(x_norm * w)
            y = int(y_norm * h)

            patch_center_x = x - patch_size//2
            patch_center_y = y - patch_size//2

            if (patch_center_x >= 0 and patch_center_x + patch_size <= w and
                patch_center_y >= 0 and patch_center_y + patch_size <= h):
                patch = gray[patch_center_y:patch_center_y + patch_size,
                            patch_center_x:patch_center_x + patch_size]
                patches.append(patch)
                valid_indices.append(i)

        return patches, valid_indices

    def detect_spatially_distributed_keypoints(self, img, target_count, grid_size=(8, 8), adaptive_threshold=True):
        """
        Detect keypoints with enforced spatial distribution using grid bucketing.
        Returns NORMALIZED coordinates (0..1) for resolution independence.

        Args:
            img: Input image (BGR or grayscale)
            target_count: Target number of keypoints to detect
            grid_size: Tuple (rows, cols) for grid division
            adaptive_threshold: Whether to use adaptive threshold for each cell

        Returns:
            numpy array of shape (N, 2) containing normalized keypoint coordinates (0..1)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape

        # Calculate keypoints per grid cell
        grid_rows, grid_cols = grid_size
        points_per_cell = max(1, target_count // (grid_rows * grid_cols))

        # Adjust grid size to better match target count if needed
        if (grid_rows * grid_cols * points_per_cell) < target_count:
            # Try to adjust grid size to get closer to target
            total_cells = target_count
            if total_cells <= 16:
                grid_rows, grid_cols = 4, 4
            elif total_cells <= 36:
                grid_rows, grid_cols = 6, 6
            elif total_cells <= 64:
                grid_rows, grid_cols = 8, 8
            else:
                grid_rows, grid_cols = 12, 12
            points_per_cell = max(1, target_count // (grid_rows * grid_cols))

        # Calculate cell dimensions
        cell_h = h // grid_rows
        cell_w = w // grid_cols

        all_keypoints = []
        # Keep track of which grid cells had keypoints found
        occupied_cells = set()

        # Detect keypoints in each grid cell independently
        for row in range(grid_rows):
            for col in range(grid_cols):
                # Define cell boundaries without overlap for initial detection
                y_start = row * cell_h
                y_end = min(h, (row + 1) * cell_h)
                x_start = col * cell_w
                x_end = min(w, (col + 1) * cell_w)

                # Extract cell
                cell = gray[y_start:y_end, x_start:x_end]

                # Skip very small cells
                if cell.shape[0] < 16 or cell.shape[1] < 16:
                    continue

                # Detect FAST keypoints in this cell
                cell_keypoints = []

                if adaptive_threshold:
                    # Use adaptive threshold that varies based on local content
                    threshold = 50
                    while threshold >= 5 and len(cell_keypoints) < points_per_cell:
                        fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=True)
                        cell_keypoints = fast.detect(cell, None)
                        threshold -= 5
                else:
                    # Use fixed threshold
                    fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
                    cell_keypoints = fast.detect(cell, None)

                # Sort by response and take top N for this cell
                if len(cell_keypoints) > points_per_cell:
                    cell_keypoints = sorted(cell_keypoints, key=lambda kp: kp.response, reverse=True)[:points_per_cell]

                # Convert keypoints to absolute image coordinates
                cell_keypoints_coords = []
                for kp in cell_keypoints:
                    abs_x = kp.pt[0] + x_start
                    abs_y = kp.pt[1] + y_start
                    cell_keypoints_coords.append([abs_x, abs_y])
                    # Mark this cell as occupied
                    occupied_cells.add((row, col))

                all_keypoints.extend(cell_keypoints_coords)

        # Convert to numpy array
        all_keypoints = np.array(all_keypoints) if all_keypoints else np.array([]).reshape(0, 2)

        # NORMALIZE to 0..1 range
        if len(all_keypoints) > 0:
            all_keypoints[:, 0] /= w  # normalize x
            all_keypoints[:, 1] /= h  # normalize y

        # Final check: return up to the target count
        if len(all_keypoints) > target_count:
            # Use spatial subdivision to maintain distribution when limiting
            # Create a grid to select evenly distributed points
            grid_cell_size_x = 1.0 / grid_cols  # normalized grid size
            grid_cell_size_y = 1.0 / grid_rows

            # Create a grid to distribute points
            cell_map = {}
            for pt in all_keypoints:
                x, y = pt
                cell_x = min(int(x / grid_cell_size_x), grid_cols - 1)
                cell_y = min(int(y / grid_cell_size_y), grid_rows - 1)
                cell_key = (cell_y, cell_x)

                if cell_key not in cell_map:
                    cell_map[cell_key] = []
                cell_map[cell_key].append(pt)

            selected_keypoints = []
            remaining_needed = target_count

            for cell_points in cell_map.values():
                if remaining_needed <= 0:
                    break
                n_to_take = min(len(cell_points), remaining_needed)
                selected_keypoints.extend(cell_points[:n_to_take])
                remaining_needed -= n_to_take

            all_keypoints = np.array(selected_keypoints)

        return all_keypoints if len(all_keypoints) > 0 else np.array([]).reshape(0, 2)

    def detect_fast_keypoints_with_threshold_dropping(self, img, target_count):
        """Detect FAST keypoints with threshold dropping - returns NORMALIZED coordinates"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape

        params = {
            'fast_initial_threshold': 50,
            'fast_min_threshold': 5,
            'fast_threshold_step': 5,
            'fast_nonmaxSuppression': True
        }

        threshold = params['fast_initial_threshold']
        while threshold >= params['fast_min_threshold']:
            fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=params['fast_nonmaxSuppression'])
            keypoints = fast.detect(gray, None)
            if len(keypoints) >= target_count:
                # Keep only the top target_count keypoints based on response
                keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:target_count]
                break
            threshold -= params['fast_threshold_step']

        if len(keypoints) == 0:
            # Fallback to cornerHarris if FAST fails
            gray_float = np.float32(gray)
            dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
            y_coords, x_coords = np.where(dst > 0.01 * dst.max())
            responses = dst[y_coords, x_coords]

            # Create KeyPoint objects
            keypoints = []
            for x, y, response in zip(x_coords, y_coords, responses):
                kp = cv2.KeyPoint(x, y, _size=7, _response=response)
                keypoints.append(kp)

        # Convert to numpy array and NORMALIZE
        if keypoints:
            points = np.array([[kp.pt[0] / w, kp.pt[1] / h] for kp in keypoints])
        else:
            points = np.array([]).reshape(0, 2)

        return points

    def detect_keypoints_with_spatial_distribution(self, img, target_count, grid_size=(8, 8)):
        """
        Alternative keypoint detection method that uses spatial distribution
        by default but falls back to the original method if needed.
        """
        # Use the enhanced spatially distributed detection
        keypoints = self.detect_spatially_distributed_keypoints(img, target_count, grid_size)

        # If we still don't have enough keypoints, try the original method as fallback
        if len(keypoints) < target_count:
            fallback_keypoints = self.detect_fast_keypoints_with_threshold_dropping(img, target_count)
            if len(fallback_keypoints) > len(keypoints):
                return fallback_keypoints

        return keypoints

    def extract_multiscale_features(self, img_original, target_keypoints=200, use_spatial_distribution=True):
        """Extract KNIFT features with keypoint detection - single scale only"""
        if self.interpreter is None or img_original is None:
            return None, None

        # Single scale detection - only use scale 1.0 (no multi-scale)
        # Use 100 keypoints for sample images to allow concatenation of 2 samples for 200 total
        img_scaled = img_original
        current_h, current_w = img_scaled.shape[:2]

        # Use spatial distribution if requested, otherwise use original method
        if use_spatial_distribution:
            points_scaled = self.detect_keypoints_with_spatial_distribution(
                img_scaled, target_keypoints
            )
        else:
            points_scaled = self.detect_fast_keypoints_with_threshold_dropping(
                img_scaled, target_keypoints
            )

        if len(points_scaled) == 0:
            return None, None

        patches, valid_indices = self.extract_patches(img_scaled, points_scaled)

        if not patches:
            return None, None

        patches_array = np.array(patches, dtype=np.float32) / 255.0
        patches_array = np.expand_dims(patches_array, axis=-1)

        # Get the expected batch size from the model - should match our target keypoints (200)
        expected_batch = self.input_details[0]['shape'][0]

        # Ensure our batch size matches the model's expected input
        # If expected batch is different from our target keypoints, we need to adjust
        actual_n = patches_array.shape[0]

        # Pad or truncate to exactly match expected batch size if needed
        if actual_n <= expected_batch:
            padded_input = np.zeros((expected_batch, 32, 32, 1), dtype=np.float32)
            padded_input[:actual_n] = patches_array
            limit = actual_n
        else:
            # If we have more patches than expected, only use the first 'expected_batch' patches
            padded_input = patches_array[:expected_batch]
            limit = expected_batch

        self.interpreter.set_tensor(self.input_details[0]['index'], padded_input)
        self.interpreter.invoke()

        full_output = self.interpreter.get_tensor(self.output_details[0]['index'])
        descriptors = full_output[:limit]

        # L2 normalize descriptors for better matching
        descriptors = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8)

        # No coordinate transformation needed since we're using the original scale
        valid_points_scaled = points_scaled[valid_indices[:limit]]

        return valid_points_scaled, descriptors

    def extract_multiscale_features_concatenated(self, img_original1, img_original2, target_keypoints_per_image=100, use_spatial_distribution=True):
        """Extract KNIFT features from two images together - 100 keypoints per image = 200 total for batch processing"""
        if self.interpreter is None or img_original1 is None or img_original2 is None:
            return None, None, None, None

        # Extract 100 keypoints from each image with the spatial distribution option
        if use_spatial_distribution:
            keypoints1 = self.detect_keypoints_with_spatial_distribution(
                img_original1, target_keypoints_per_image
            )

            keypoints2 = self.detect_keypoints_with_spatial_distribution(
                img_original2, target_keypoints_per_image
            )
        else:
            keypoints1 = self.detect_fast_keypoints_with_threshold_dropping(
                img_original1, target_keypoints_per_image
            )

            keypoints2 = self.detect_fast_keypoints_with_threshold_dropping(
                img_original2, target_keypoints_per_image
            )

        if len(keypoints1) == 0 or len(keypoints2) == 0:
            # Fallback to individual processing if we can't get keypoints from both
            kp1, desc1 = self.extract_multiscale_features(img_original1, target_keypoints_per_image, use_spatial_distribution) if len(keypoints1) > 0 else (None, None)
            kp2, desc2 = self.extract_multiscale_features(img_original2, target_keypoints_per_image, use_spatial_distribution) if len(keypoints2) > 0 else (None, None)
            return kp1, desc1, kp2, desc2

        # Extract patches from both images
        patches1, valid_indices1 = self.extract_patches(img_original1, keypoints1)
        patches2, valid_indices2 = self.extract_patches(img_original2, keypoints2)

        if not patches1 or not patches2:
            # Fallback if patches extraction failed
            kp1, desc1 = self.extract_multiscale_features(img_original1, target_keypoints_per_image, use_spatial_distribution) if patches1 else (None, None)
            kp2, desc2 = self.extract_multiscale_features(img_original2, target_keypoints_per_image, use_spatial_distribution) if patches2 else (None, None)
            return kp1, desc1, kp2, desc2

        # Combine patches from both images
        patches_array1 = np.array(patches1, dtype=np.float32) / 255.0
        patches_array1 = np.expand_dims(patches_array1, axis=-1)

        patches_array2 = np.array(patches2, dtype=np.float32) / 255.0
        patches_array2 = np.expand_dims(patches_array2, axis=-1)

        # Concatenate patches to form a batch of 200 (100 + 100)
        combined_patches = np.concatenate([patches_array1, patches_array2], axis=0)

        # Get the expected batch size from the model (should be 200)
        expected_batch = self.input_details[0]['shape'][0]

        # Prepare input tensor - pad or truncate to match expected batch size
        actual_n = combined_patches.shape[0]
        padded_input = np.zeros((expected_batch, 32, 32, 1), dtype=np.float32)

        if actual_n <= expected_batch:
            padded_input[:actual_n] = combined_patches
            limit = actual_n
        else:
            # If we have more patches than expected, only use the first 'expected_batch' patches
            padded_input = combined_patches[:expected_batch]
            limit = expected_batch

        self.interpreter.set_tensor(self.input_details[0]['index'], padded_input)
        self.interpreter.invoke()

        full_output = self.interpreter.get_tensor(self.output_details[0]['index'])

        # L2 normalize all descriptors
        full_output = full_output / (np.linalg.norm(full_output, axis=1, keepdims=True) + 1e-8)

        # Split the output back based on the original division (100 from each image)
        n1 = min(len(patches1), target_keypoints_per_image)
        n2 = min(len(patches2), target_keypoints_per_image)

        # Split descriptors back to individual images
        descriptors1 = full_output[:n1] if n1 > 0 else None
        descriptors2 = full_output[n1:n1+n2] if n1+n2 <= len(full_output) and n2 > 0 else None

        # Split keypoints back to individual images (using valid indices)
        keypoints1_valid = keypoints1[valid_indices1[:n1]] if len(valid_indices1) >= n1 else keypoints1
        keypoints2_valid = keypoints2[valid_indices2[:n2]] if len(valid_indices2) >= n2 else keypoints2

        return keypoints1_valid, descriptors1, keypoints2_valid, descriptors2

    def draw_keypoints(self, img, keypoints, color=(0, 255, 0), radius=3):
        """Draw keypoints on an image - handles NORMALIZED coordinates (0..1)"""
        if keypoints is None or len(keypoints) == 0:
            return img.copy()

        img_with_kp = img.copy()
        h, w = img.shape[:2]

        for pt in keypoints:
            x = int(pt[0] * w)
            y = int(pt[1] * h)
            cv2.circle(img_with_kp, (x, y), radius, color, -1)

        return img_with_kp


    def toggle_keypoints_display(self):
        """Callback to update displays when keypoint visualization toggle is changed"""
        # Update sample images if they exist and have keypoints computed
        if self.sample1_img is not None and self.sample1_keypoints is not None:
            if self.show_keypoints_var.get():
                # Show with keypoints
                img_with_kp = self.draw_keypoints(self.sample1_img, self.sample1_keypoints)
                self.display_image(img_with_kp, self.sample1_label)
            else:
                # Show without keypoints
                self.display_image(self.sample1_img, self.sample1_label)

        if self.sample2_img is not None and self.sample2_keypoints is not None:
            if self.show_keypoints_var.get():
                # Show with keypoints
                img_with_kp = self.draw_keypoints(self.sample2_img, self.sample2_keypoints)
                self.display_image(img_with_kp, self.sample2_label)
            else:
                # Show without keypoints
                self.display_image(self.sample2_img, self.sample2_label)

    def update_keypoint_visualization(self):
        """Update keypoints visualization in both sample images with blinking effect"""
        # Use frame counter to create blinking effect
        blink_state = (self.current_frame_number // 10) % 2  # Toggle every 10 frames

        if self.sample1_img is not None and self.sample1_keypoints is not None and self.show_keypoints_var.get():
            # Draw blinking keypoints on sample 1
            img_with_kp = self.draw_blinking_sample_keypoints(self.sample1_img, self.sample1_keypoints, blink_state)
            self.display_image(img_with_kp, self.sample1_label)

        if self.sample2_img is not None and self.sample2_keypoints is not None and self.show_keypoints_var.get():
            # Draw blinking keypoints on sample 2
            img_with_kp = self.draw_blinking_sample_keypoints(self.sample2_img, self.sample2_keypoints, blink_state)
            self.display_image(img_with_kp, self.sample2_label)

    def draw_blinking_sample_keypoints(self, img, keypoints, blink_state):
        """Draw blinking keypoints on sample images"""
        if keypoints is None or len(keypoints) == 0:
            return img.copy()

        img_with_kp = img.copy()
        h, w = img.shape[:2]

        for i, pt in enumerate(keypoints):
            # Convert normalized coordinates to pixel coordinates
            x = int(pt[0] * w)
            y = int(pt[1] * h)

            # Make some keypoints blink based on state
            if blink_state or i % 3 == 0:  # Every other frame or every 3rd keypoint
                cv2.circle(img_with_kp, (x, y), 3, (0, 255, 255), -1)  # Yellow blinking keypoints
            else:
                cv2.circle(img_with_kp, (x, y), 3, (0, 255, 0), -1)  # Green static keypoints

        return img_with_kp

    def downscale_frame_to_240p(self, frame):
        """Downscale frame to 240p (426x240 or similar resolution maintaining aspect ratio)"""
        if frame is None:
            return None

        h, w = frame.shape[:2]

        # Calculate target dimensions for 240p
        target_height = 240
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)

        # Resize the frame
        downscaled_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        return downscaled_frame

    def calculate_descriptor_distance(self, desc1, desc2):
        """Calculate distance between descriptors using L2 norm"""
        if desc1 is None or desc2 is None:
            return float('inf')

        # Calculate all pairwise distances
        if len(desc1.shape) == 1:
            desc1 = desc1.reshape(1, -1)
        if len(desc2.shape) == 1:
            desc2 = desc2.reshape(1, -1)

        distances = np.linalg.norm(desc1[:, np.newaxis, :] - desc2[np.newaxis, :, :], axis=2)
        return distances

    def match_descriptors(self, desc1, desc2, threshold=0.9):  # Changed from 0.7, relaxed to 0.9
        """Match descriptors using nearest neighbor with Lowe's ratio test"""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        # Calculate distances
        distances = self.calculate_descriptor_distance(desc1, desc2)

        matches = []
        for i in range(len(desc1)):
            if len(desc2) >= 2:
                # Get two closest matches
                sorted_indices = np.argsort(distances[i])
                first_idx = sorted_indices[0]
                second_idx = sorted_indices[1]

                # Apply Lowe's ratio test
                if distances[i, first_idx] < threshold * distances[i, second_idx]:
                    matches.append((i, first_idx))

        return matches

    def calculate_match_score(self, frame, sample_img, sample_kp, sample_desc):
        """Calculate match score and return matches for visualization"""
        if self.interpreter is None:
            return 0.0, [], None

        try:
            frame_keypoints, frame_descriptors = self.extract_multiscale_features(
                frame, target_keypoints=100, use_spatial_distribution=True
            )

            if frame_keypoints is None or sample_kp is None:
                return 0.0, [], None

            if frame_descriptors is None or sample_desc is None:
                return 0.0, [], None

            matches = self.match_descriptors(frame_descriptors, sample_desc)

            # Verify matches are valid (indices within bounds)
            # matches = (frame_desc_idx, sample_desc_idx),
            # but we need to ensure indices are valid for both descriptors and keypoints
            valid_matches = []
            for frame_idx, sample_idx in matches:
                if (frame_idx < len(frame_descriptors) and frame_idx < len(frame_keypoints) and
                    sample_idx < len(sample_desc) and sample_idx < len(sample_kp)):
                    valid_matches.append((frame_idx, sample_idx))

            if len(valid_matches) == 0:
                return 0.0, [], frame_keypoints

            match_score = len(valid_matches) / min(len(frame_descriptors), len(sample_desc))

            return min(match_score, 1.0), valid_matches, frame_keypoints

        except Exception as e:
            print(f"Error in KNIFT feature matching: {e}")
            return 0.0, [], None


    def calculate_match_bounding_box(self, frame_kp, matches, frame_shape):
        """Calculate very tight bounding box using median absolute deviation"""
        if len(matches) < 4:
            return None

        h, w = frame_shape[:2]

        # Get matched keypoint positions - corrected to use m[0] for frame_idx
        matched_pts = np.array([[frame_kp[m[0]][0] * w, frame_kp[m[0]][1] * h] for m in matches])

        x_coords = matched_pts[:, 0]
        y_coords = matched_pts[:, 1]

        # Use median and MAD for robust outlier rejection
        x_median = np.median(x_coords)
        y_median = np.median(y_coords)

        # Median Absolute Deviation
        x_mad = np.median(np.abs(x_coords - x_median))
        y_mad = np.median(np.abs(y_coords - y_median))

        # Filter points within 2 MADs (removes outliers)
        x_filtered = x_coords[np.abs(x_coords - x_median) < 2 * x_mad]
        y_filtered = y_coords[np.abs(y_coords - y_median) < 2 * y_mad]

        # If too many filtered out, fall back to all points
        if len(x_filtered) < 4 or len(y_filtered) < 4:
            x_filtered = x_coords
            y_filtered = y_coords

        # Calculate tight bounds
        x_min = x_filtered.min()
        x_max = x_filtered.max()
        y_min = y_filtered.min()
        y_max = y_filtered.max()

        # Minimal padding
        padding = 5
        x_min = max(0, int(x_min - padding))
        y_min = max(0, int(y_min - padding))
        x_max = min(w, int(x_max + padding))
        y_max = min(h, int(y_max + padding))

        # Ensure minimum box size
        min_box_size = 30
        if (x_max - x_min) < min_box_size:
            center_x = (x_min + x_max) / 2
            x_min = max(0, int(center_x - min_box_size / 2))
            x_max = min(w, int(center_x + min_box_size / 2))

        if (y_max - y_min) < min_box_size:
            center_y = (y_min + y_max) / 2
            y_min = max(0, int(center_y - min_box_size / 2))
            y_max = min(h, int(center_y + min_box_size / 2))

        return (x_min, y_min, x_max, y_max)

    def draw_match_visualization_colored(self, frame, sample_name, match_score, bbox, matches, color=(0, 255, 0)):
        """Draw bounding box with custom color"""
        if bbox is None:
            return frame

        h, w = frame.shape[:2]
        x_min, y_min, x_max, y_max = bbox

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)

        # Draw corner markers
        corner_size = 15
        cv2.line(frame, (x_min, y_min), (x_min + corner_size, y_min), color, 3)
        cv2.line(frame, (x_min, y_min), (x_min, y_min + corner_size), color, 3)
        cv2.line(frame, (x_max, y_min), (x_max - corner_size, y_min), color, 3)
        cv2.line(frame, (x_max, y_min), (x_max, y_min + corner_size), color, 3)
        cv2.line(frame, (x_min, y_max), (x_min + corner_size, y_max), color, 3)
        cv2.line(frame, (x_min, y_max), (x_min, y_max - corner_size), color, 3)
        cv2.line(frame, (x_max, y_max), (x_max - corner_size, y_max), color, 3)
        cv2.line(frame, (x_max, y_max), (x_max, y_max - corner_size), color, 3)

        # Draw center crosshair
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        cross_size = 10
        cv2.line(frame, (center_x - cross_size, center_y), (center_x + cross_size, center_y), color, 2)
        cv2.line(frame, (center_x, center_y - cross_size), (center_x, center_y + cross_size), color, 2)

        # Label
        label = f"{sample_name}: {match_score:.1%} ({len(matches)})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        label_y = max(y_min - 10, label_size[1] + 5)
        cv2.rectangle(frame,
                     (x_min, label_y - label_size[1] - 5),
                     (x_min + label_size[0] + 10, label_y + 5),
                     color, -1)

        cv2.putText(frame, label, (x_min + 5, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def scale_bounding_box(self, bbox, from_shape, to_shape):
        """Scale bounding box coordinates from one frame size to another"""
        if bbox is None:
            return None

        # Get dimensions
        from_h, from_w = from_shape[:2]
        to_h, to_w = to_shape[:2]

        # Calculate scaling factors
        scale_x = to_w / from_w
        scale_y = to_h / from_h

        x_min, y_min, x_max, y_max = bbox

        # Scale the coordinates
        scaled_x_min = int(x_min * scale_x)
        scaled_y_min = int(y_min * scale_y)
        scaled_x_max = int(x_max * scale_x)
        scaled_y_max = int(y_max * scale_y)

        # Ensure bounds are within the target frame
        scaled_x_min = max(0, min(to_w, scaled_x_min))
        scaled_y_min = max(0, min(to_h, scaled_y_min))
        scaled_x_max = max(0, min(to_w, scaled_x_max))
        scaled_y_max = max(0, min(to_h, scaled_y_max))

        return (scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max)

    def verify_matches_with_ransac(self, sample_kp, frame_kp, matches, frame_shape, threshold=10.0):
        """Use RANSAC to filter geometrically inconsistent matches"""
        if len(matches) < 8:  # Reduced minimum
            return [], None

        h, w = frame_shape[:2]

        # Validate matches before creating point arrays to avoid index errors
        # matches contain (frame_idx, sample_idx) tuples based on match_descriptors output
        valid_matches = []
        for frame_idx, sample_idx in matches:
            if frame_idx < len(frame_kp) and sample_idx < len(sample_kp):
                valid_matches.append((frame_idx, sample_idx))

        if len(valid_matches) < 8:  # Not enough valid matches for RANSAC
            return [], None

        # Denormalize keypoints to pixel coordinates
        # src_pts: sample keypoints (m[1] is sample_idx), dst_pts: frame keypoints (m[0] is frame_idx)
        src_pts = np.float32([[sample_kp[m[1]][0] * w, sample_kp[m[1]][1] * h] for m in valid_matches])
        dst_pts = np.float32([[frame_kp[m[0]][0] * w, frame_kp[m[0]][1] * h] for m in valid_matches])

        # Find homography with RANSAC (relaxed threshold for 240p)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)

        if H is None:
            return [], None

        # Filter inliers only
        inlier_matches = [valid_matches[i] for i in range(len(valid_matches)) if mask[i]]
        inlier_ratio = len(inlier_matches) / len(valid_matches) if len(valid_matches) > 0 else 0

        # Relaxed criteria: 40% inliers OR 12+ inliers
        is_valid = (inlier_ratio > 0.4 and len(inlier_matches) >= 6) or len(inlier_matches) >= 12

        # Don't print every frame - only when match found
        if is_valid:
            print(f"  ✓ RANSAC: {len(matches)} → {len(inlier_matches)} inliers ({inlier_ratio:.1%})")

        return inlier_matches if is_valid else [], H if is_valid else None

    def start_matching(self):
        """Start the video matching process"""
        if self.video_cap is None:
            messagebox.showerror("Error", "Video file not loaded")
            return

        # Pre-compute features for sample images together to form a batch of 200 keypoints
        if self.sample1_img is not None and self.sample2_img is not None:
            self.status_var.set("Pre-computing features for sample images together...")
            # Downscale sample images to match processing pipeline
            downscaled_sample1 = self.downscale_frame_to_240p(self.sample1_img)
            downscaled_sample2 = self.downscale_frame_to_240p(self.sample2_img)

            # Extract features for both samples together (100 keypoints each = 200 total) for efficient processing
            sample1_kp, sample1_desc, sample2_kp, sample2_desc = self.extract_multiscale_features_concatenated(
                downscaled_sample1, downscaled_sample2, target_keypoints_per_image=100, use_spatial_distribution=True
            )

            if sample1_kp is not None and sample1_desc is not None:
                self.sample1_keypoints = sample1_kp
                self.sample1_descriptors = sample1_desc
                self.downscaled_sample1 = downscaled_sample1  # Store for matching


                self.status_var.set("Features computed for sample 1")
                # Update sample 1 display with keypoints if visualization is enabled
                if self.show_keypoints_var.get():
                    img_with_kp = self.draw_keypoints(self.sample1_img, sample1_kp)
                    self.display_image(img_with_kp, self.sample1_label)

            if sample2_kp is not None and sample2_desc is not None:
                self.sample2_keypoints = sample2_kp
                self.sample2_descriptors = sample2_desc
                self.downscaled_sample2 = downscaled_sample2  # Store for matching


                self.status_var.set("Features computed for sample 2")
                # Update sample 2 display with keypoints if visualization is enabled
                if self.show_keypoints_var.get():
                    img_with_kp = self.draw_keypoints(self.sample2_img, sample2_kp)
                    self.display_image(img_with_kp, self.sample2_label)
        else:
            # Fallback to individual processing if only one sample is available
            if self.sample1_img is not None and self.sample1_keypoints is None:
                self.status_var.set("Pre-computing features for sample 1...")
                downscaled_sample1 = self.downscale_frame_to_240p(self.sample1_img)
                self.sample1_keypoints, self.sample1_descriptors = self.extract_multiscale_features(downscaled_sample1, target_keypoints=100, use_spatial_distribution=True)
                self.downscaled_sample1 = downscaled_sample1  # Store for matching


                self.status_var.set("Features computed for sample 1")
                # Update sample 1 display with keypoints if visualization is enabled
                if self.show_keypoints_var.get() and self.sample1_keypoints is not None:
                    img_with_kp = self.draw_keypoints(self.sample1_img, self.sample1_keypoints)
                    self.display_image(img_with_kp, self.sample1_label)

            if self.sample2_img is not None and self.sample2_keypoints is None:
                self.status_var.set("Pre-computing features for sample 2...")
                downscaled_sample2 = self.downscale_frame_to_240p(self.sample2_img)
                self.sample2_keypoints, self.sample2_descriptors = self.extract_multiscale_features(downscaled_sample2, target_keypoints=100, use_spatial_distribution=True)
                self.downscaled_sample2 = downscaled_sample2  # Store for matching


                self.status_var.set("Features computed for sample 2")
                # Update sample 2 display with keypoints if visualization is enabled
                if self.show_keypoints_var.get() and self.sample2_keypoints is not None:
                    img_with_kp = self.draw_keypoints(self.sample2_img, self.sample2_keypoints)
                    self.display_image(img_with_kp, self.sample2_label)

        self.is_matching = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Matching in progress...")

        # Start matching process
        self.match_video_frames()

    def stop_matching(self):
        """Stop the video matching process"""
        self.is_matching = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Matching stopped")

    def reset_video(self):
        """Reset video to the beginning"""
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_number = 0
            self.match_results = []
            self.results_text.delete(1.0, tk.END)
            self.status_var.set("Video reset to beginning")

    def match_video_frames(self):
        """Process video frames and match against sample images"""
        if not self.is_matching:
            return

        ret, frame = self.video_cap.read()
        if ret:
            self.current_frame = frame
            self.current_frame_number += 1

            # Display current frame
            self.display_image(frame, self.video_label)

            # Check if we should process this frame based on frame skip
            if self.frame_counter % self.frame_skip == 0:
                # Perform matching if we have sample images
                match_result = self.match_frame_to_samples(frame)

                # Log results
                if match_result:
                    self.match_results.append(match_result)
                    self.results_text.insert(tk.END, f"Frame {self.current_frame_number}: {match_result}\n")
                    self.results_text.see(tk.END)
            # Still continue processing the next frame regardless

            self.frame_counter += 1

            if self.frame_counter % 5 == 0:  # Update every 5 frames for performance
                self.update_keypoint_visualization()

            # Continue processing - adjust timing to match target FPS
            # With frame skipping, we can reduce the delay to process frames faster
            self.root.after(15, self.match_video_frames)  # Adjusted timing for better performance
        else:
            # Reached end of video
            self.stop_matching()
            self.status_var.set("Matching completed")
            messagebox.showinfo("Complete", f"Video matching completed. Found {len(self.match_results)} matches.")

    def match_frame_to_samples(self, frame):
        """Match the current frame against sample images using KNIFT features"""
        results = []

        # Create a copy of frame for visualization (keep original for display)
        vis_frame = frame.copy()

        # Downscale the frame for processing to 240p
        processed_frame = self.downscale_frame_to_240p(frame)

        # Cache frame features (extracted once, used for both samples)
        if self.frame_cache_id != self.current_frame_number:
            self.last_frame_features = self.extract_multiscale_features(
                processed_frame, target_keypoints=100, use_spatial_distribution=True
            )
            self.frame_cache_id = self.current_frame_number

        frame_kp, frame_desc = self.last_frame_features

        if frame_kp is None or frame_desc is None:
            self.display_image(vis_frame, self.video_label)
            return ""

        # Sample 1
        if self.sample1_img is not None and self.sample1_keypoints is not None:
            matches1 = self.match_descriptors(frame_desc, self.sample1_descriptors)

            if len(matches1) > 0:
                verified_matches1, H1 = self.verify_matches_with_ransac(
                    self.sample1_keypoints, frame_kp, matches1, processed_frame.shape, threshold=10.0
                )

                if len(verified_matches1) > 0:
                    verified_score1 = len(verified_matches1) / min(len(self.sample1_descriptors), 100)

                    if verified_score1 > 0.10:  # Lower threshold
                        results.append(f"Sample1: {verified_score1:.1%}")

                        if self.show_matches_var.get():
                            bbox1 = self.calculate_match_bounding_box(frame_kp, verified_matches1, processed_frame.shape)
                            # Scale bounding box from processed frame coordinates to original frame coordinates
                            scaled_bbox1 = self.scale_bounding_box(bbox1, processed_frame.shape, vis_frame.shape)
                            vis_frame = self.draw_match_visualization_colored(
                                vis_frame, "Sample 1", verified_score1, scaled_bbox1, verified_matches1,
                                color=(0, 255, 0)
                            )

        # Sample 2
        if self.sample2_img is not None and self.sample2_keypoints is not None:
            matches2 = self.match_descriptors(frame_desc, self.sample2_descriptors)

            if len(matches2) > 0:
                verified_matches2, H2 = self.verify_matches_with_ransac(
                    self.sample2_keypoints, frame_kp, matches2, processed_frame.shape, threshold=10.0
                )

                if len(verified_matches2) > 0:
                    verified_score2 = len(verified_matches2) / min(len(self.sample2_descriptors), 100)

                    if verified_score2 > 0.10:
                        results.append(f"Sample2: {verified_score2:.1%}")

                        if self.show_matches_var.get():
                            bbox2 = self.calculate_match_bounding_box(frame_kp, verified_matches2, processed_frame.shape)
                            # Scale bounding box from processed frame coordinates to original frame coordinates
                            scaled_bbox2 = self.scale_bounding_box(bbox2, processed_frame.shape, vis_frame.shape)
                            vis_frame = self.draw_match_visualization_colored(
                                vis_frame, "Sample 2", verified_score2, scaled_bbox2, verified_matches2,
                                color=(255, 0, 0)
                            )

        # Keypoints visualization (optional)
        if self.show_keypoints_var.get():
            vis_frame = self.draw_keypoints(vis_frame, frame_kp, color=(0, 255, 255), radius=2)

        self.display_image(vis_frame, self.video_label)

        return ", ".join(results) if results else ""


    def display_image(self, cv_img, label):
        """Display a CV2 image on a tkinter label with aspect ratio preserved"""
        # Convert CV2 image (BGR) to PIL image (RGB)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # Get the label's available dimensions
        label_width = label.winfo_width()
        label_height = label.winfo_height()

        # If no specific dimensions are set, use frame dimensions
        if label_width <= 1 or label_height <= 1:
            # For video display - fixed size
            if label == self.video_label:
                label_width, label_height = 800, 450
            # For sample images - use frame dimensions but respect image aspect ratio
            elif label in [self.sample1_label, self.sample2_label]:
                frame = label.master  # Get parent frame
                frame_width = frame.winfo_width() if frame.winfo_width() > 1 else 300
                frame_height = frame.winfo_height() if frame.winfo_height() > 1 else 200
                # Adjust for title bar space
                frame_height = max(1, frame_height - 25)

                # Determine which image this label displays
                if label == self.sample1_label and self.sample1_img is not None:
                    img_h, img_w = self.sample1_img.shape[:2]
                    # Calculate scale to fit image within frame while maintaining aspect ratio
                    scale_w = frame_width / img_w
                    scale_h = frame_height / img_h
                    scale = min(scale_w, scale_h)

                    label_width = int(img_w * scale)
                    label_height = int(img_h * scale)
                elif label == self.sample2_label and self.sample2_img is not None:
                    img_h, img_w = self.sample2_img.shape[:2]
                    # Calculate scale to fit image within frame while maintaining aspect ratio
                    scale_w = frame_width / img_w
                    scale_h = frame_height / img_h
                    scale = min(scale_w, scale_h)

                    label_width = int(img_w * scale)
                    label_height = int(img_h * scale)
                else:
                    # Fallback to defaults if images are not loaded
                    label_width, label_height = 300, 200
            else:
                label_width, label_height = 300, 200

        # Resize image to fit label while maintaining aspect ratio
        pil_img.thumbnail((label_width, label_height), Image.Resampling.LANCZOS)

        # Convert to PhotoImage and display
        photo = ImageTk.PhotoImage(pil_img)
        label.config(image=photo)
        label.image = photo  # Keep a reference to prevent garbage collection

    def cleanup(self):
        """Clean up resources"""
        if self.video_cap:
            self.video_cap.release()

def main():
    root = tk.Tk()
    app = KNIFTVideoObjectMatcher(root)

    # Handle window closing
    def on_closing():
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()