#!/usr/bin/env python3

"""
KNIFT Model Alignment Test Suite v10.0 - Automated Model Evaluation

This script automatically tests the alignment capabilities of the KNIFT model
by intentionally misaligning Set14 images and then evaluating the model's
ability to realign them. Provides comprehensive performance metrics and
success rate analysis for the trained model.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import tensorflow as tf
from typing import Optional, Tuple
import os
import time

class KNIFTAlignmentTestSuite:
    def __init__(self, root):
        self.root = root
        self.root.title("KNIFT Alignment Test Suite v10.0 - Model Performance Evaluation")
        self.root.geometry("1400x900")

        # Initialize variables
        self.img1_original = None
        self.img2_original = None
        self.img1_processed = None
        self.img2_original_copy = None
        self.homography = None
        self.interpreter = None

        # Store keypoints and descriptors
        self.keypoints_1 = None
        self.keypoints_2 = None
        self.desc1 = None
        self.desc2 = None

        # Animation state
        self.is_aligning = False
        self.current_iteration = 0
        self.iteration_results = []

        np.random.seed()

        # Fixed parameters (not in UI)
        self.params = {
            # Transformation parameters (for synthetic generation)
            'rotation_angle': 0.0,
            'translation_x': 0.0,
            'translation_y': 0.0,
            'scale_factor': 1.0,

            # Fixed algorithm parameters
            'pyramid_scales': [0.0625, 0.125, 0.25, 0.5, 1.0],  # Coarse to fine: 6.25%, 12.5%, 25%, 50%, 100% - 5 scales, dropping the 2x scale
            'target_keypoints': 1000,  # Use 1000 keypoints at each scale
            'fast_initial_threshold': 50,
            'fast_min_threshold': 5,
            'fast_threshold_step': 5,
            'fast_nonmaxSuppression': True,

            # Adaptive refinement parameters
            'max_iterations': 1,  # Only 1 iteration per scale now
        }

        # Create GUI
        # Multi-scale alignment state
        self.current_scale_index = 0
        self.scale_homographies = {}  # Store homography for each scale level
        self.scale_results = {}       # Store results for each scale level

        # Add variables for testing and reporting
        self.test_results = []        # Store results for each test image
        self.current_test_index = 0   # Index of current test image
        self.test_images = []         # List of test images to process

        self.create_widgets()

        # Try to auto-load the KNIFT model first
        self.auto_load_knift_model()

        # Load all Set14 images for testing
        self.load_test_images()

        # Load the first test image
        if self.test_images:
            self.load_current_test_image()

    def create_widgets(self):
        """Create the main GUI widgets"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Test Suite Status")
        status_frame.pack(fill=tk.X, padx=5, pady=5)

        # Status display
        self.status_var = tk.StringVar(value="Initializing KNIFT Model Test Suite...")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, wraplength=800)
        status_label.pack(padx=5, pady=5)

        # Progress display
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(progress_frame, text="Current Progress:").pack(side=tk.LEFT, padx=5)
        self.progress_var = tk.StringVar(value="Ready")
        progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        progress_label.pack(side=tk.LEFT, padx=5)

        # Images and result frame for visualization
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left frame for overlay view
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Overlay images frame
        overlay_frame = ttk.LabelFrame(left_frame, text="Current Alignment Progress")
        overlay_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Canvas for overlay view
        self.canvas_overlay = tk.Canvas(overlay_frame, bg='white')
        self.canvas_overlay.pack(fill=tk.BOTH, expand=True)

        # Right frame for visual feedback
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Aligned image frame
        aligned_frame = ttk.LabelFrame(right_frame, text="Misaligned vs Realigned Comparison")
        aligned_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Canvas for keypoint view
        self.canvas_aligned = tk.Canvas(aligned_frame, bg='white')
        self.canvas_aligned.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var.set("KNIFT Model Test Suite initialized. Starting automated tests...")

        # Start the automated test suite with a slight delay to allow GUI to initialize
        self.root.after(2000, self.start_full_test_suite)  # Start after 2 seconds to allow GUI to initialize

    def create_transformation_controls(self, parent):
        """Create transformation controls (for synthetic generation) - now only used internally"""
        # This method is retained for internal use but not called in the automated suite
        # Initialize variables that would normally be controlled by the GUI
        self.rotation_var = tk.DoubleVar(value=self.params['rotation_angle'])
        self.translation_x_var = tk.DoubleVar(value=self.params['translation_x'])
        self.translation_y_var = tk.DoubleVar(value=self.params['translation_y'])
        self.scale_var = tk.DoubleVar(value=self.params['scale_factor'])

    def load_test_images(self):
        """Load all unique images from the Set14 folder (excluding collage) for testing"""
        set14_paths = ["./Set14", "./Set14/Set14HQ"]  # Check both locations
        self.test_images = []
        seen_files = set()  # To avoid duplicate images

        for set14_path in set14_paths:
            if os.path.exists(set14_path):
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

                for file in os.listdir(set14_path):
                    if any(file.lower().endswith(ext) for ext in image_extensions) and file.lower() != 'collage.jpeg':
                        # Only add if not already seen (to avoid duplicates between paths)
                        if file not in seen_files:
                            image_path = os.path.join(set14_path, file)
                            self.test_images.append(image_path)
                            seen_files.add(file)

        if not self.test_images:
            self.status_var.set("No test images found in ./Set14 or ./Set14/Set14HQ folders (excluding collage).")
        else:
            self.status_var.set(f"Loaded {len(self.test_images)} unique test images for evaluation.")

    def load_current_test_image(self):
        """Load the current test image for alignment testing"""
        if self.current_test_index < len(self.test_images):
            image_path = self.test_images[self.current_test_index]
            self.img1_original = cv2.imread(image_path)
            if self.img1_original is not None:
                self.status_var.set(f"Loaded test image {self.current_test_index + 1}/{len(self.test_images)}: {os.path.basename(image_path)}")

                # Clear any previous state before generating the misaligned version
                self.img2_original = None
                self.img1_processed = None
                self.homography = None

                self.img2_original = self.generate_synthetic_transformations(self.img1_original)
                self.img2_original_copy = self.img2_original.copy()
                self.update_display()
                return True
        return False

    def generate_synthetic_transformations(self, img):
        """Generate synthetic transformations based on current transformation parameters"""
        rotation_angle = self.params['rotation_angle']
        translation_x = self.params['translation_x']
        translation_y = self.params['translation_y']
        scale_factor = self.params['scale_factor']

        h, w = img.shape[:2]

        cos_rot = np.cos(rotation_angle)
        sin_rot = np.sin(rotation_angle)

        scale_matrix = np.array([
            [scale_factor, 0, (1 - scale_factor) * w / 2],
            [0, scale_factor, (1 - scale_factor) * h / 2]
        ], dtype=np.float32)

        rot_matrix = np.array([
            [cos_rot, -sin_rot, (1-cos_rot)*(w/2) + sin_rot*(h/2)],
            [sin_rot, cos_rot, (1-cos_rot)*(h/2) - sin_rot*(w/2)]
        ], dtype=np.float32)

        trans_matrix = np.array([
            [1, 0, translation_x * w],
            [0, 1, translation_y * h]
        ], dtype=np.float32)

        scale_h = np.vstack([scale_matrix, [0, 0, 1]])
        rot_h = np.vstack([rot_matrix, [0, 0, 1]])
        trans_h = np.vstack([trans_matrix, [0, 0, 1]])

        combined_h = trans_h @ rot_h @ scale_h
        final_transform = combined_h[:2, :]

        transformed_img = cv2.warpAffine(img, final_transform, (w, h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))

        return transformed_img

    def auto_load_knift_model(self):
        """Automatically load the specific KNIFT model for GPU float 1k"""
        model_path = "./outputs/models/knift_gpu_float_1k.tflite"

        if os.path.exists(model_path):
            try:
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.status_var.set(f"KNIFT model auto-loaded: {os.path.basename(model_path)}")
                return
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                self.status_var.set(f"Failed to load KNIFT model: {os.path.basename(model_path)}")
                return
        else:
            self.status_var.set(f"KNIFT model not found: {model_path}")

    def update_transformation_params(self):
        """Update transformation parameters"""
        self.params['rotation_angle'] = self.rotation_var.get()
        self.params['translation_x'] = self.translation_x_var.get()
        self.params['translation_y'] = self.translation_y_var.get()
        self.params['scale_factor'] = self.scale_var.get()

    def update_synthetic_transform(self):
        """Update the synthetic transformation of image 2 based on parameters"""
        if self.img1_original is not None:
            self.update_transformation_params()
            self.img2_original = self.generate_synthetic_transformations(self.img1_original)
            self.img2_original_copy = self.img2_original.copy()

            # Reset alignment state
            self.keypoints_1 = None
            self.keypoints_2 = None
            self.desc1 = None
            self.desc2 = None
            self.homography = None
            self.img1_processed = None
            self.iteration_results = []

            self.update_display()

    def start_full_test_suite(self):
        """Start the full automated test suite on all Set14 images"""
        if not self.test_images:
            messagebox.showwarning("Warning", "No test images loaded")
            return

        if self.is_aligning:
            messagebox.showinfo("Info", "Test suite already in progress")
            return

        # Reset test results
        self.test_results = []
        self.current_test_index = 0
        self.is_aligning = True

        # Start processing the first image
        self.process_next_test_image()

    def stop_alignment(self):
        """Stop the alignment process"""
        self.is_aligning = False
        self.status_var.set("Alignment stopped by user")

    def start_alignment_process(self):
        """Start the hierarchical multi-scale alignment process"""
        if self.img1_original is None or self.img2_original is None:
            messagebox.showwarning("Warning", "Please load images first")
            return

        if self.is_aligning:
            messagebox.showinfo("Info", "Alignment already in progress")
            return

        self.is_aligning = True
        self.current_scale_index = 0
        self.scale_homographies = {}
        self.scale_results = {}
        self.homography = None  # Reset the final homography
        self.img1_processed = None  # Reset the processed image

        # Use the max iterations value as the maximum number of scales to process
        # Though we'll process all scales in the pyramid regardless
        self.params['max_iterations'] = self.max_iter_var.get()

        # Start the multi-scale alignment
        self.root.after(100, self.run_next_scale)

    def process_next_test_image(self):
        """Process the next image in the test suite"""
        if not self.is_aligning:
            return

        if self.current_test_index >= len(self.test_images):
            # All images have been processed, show final report
            self.finalize_test_suite()
            return

        # Reset alignment state for this image - ensure clean slate
        self.current_scale_index = 0
        self.scale_homographies = {}
        self.scale_results = {}
        self.homography = None
        self.img1_processed = None
        self.iteration_results = []
        self.keypoints_1 = None
        self.keypoints_2 = None
        self.desc1 = None
        self.desc2 = None

        # Clear the display canvases before loading the new image
        self.canvas_overlay.delete("all")
        self.canvas_aligned.delete("all")

        # Use root.after to ensure proper separation between resets and loading
        self.root.after(50, self._load_and_process_current_image)

    def _load_and_process_current_image(self):
        """Helper method to load and process the current image after reset"""
        # Load the next test image
        success = self.load_current_test_image()

        if not success:
            # If loading failed, move to next or finish
            if self.current_test_index < len(self.test_images):
                self.root.after(1000, self.process_next_test_image)  # Skip this image and try the next
            else:
                self.finalize_test_suite()
            return

        # Set random transformation parameters for this test
        self.set_random_transformation_params()

        # Update status
        self.status_var.set(f"Testing image {self.current_test_index + 1}/{len(self.test_images)}: {os.path.basename(self.test_images[self.current_test_index])}")

        # Start alignment for this image
        self.root.after(100, self.run_next_scale)

    def set_random_transformation_params(self):
        """Set random transformation parameters to misalign the test image"""
        import random
        # Generate random transformation parameters to create a challenging misalignment
        self.params['rotation_angle'] = random.uniform(-0.2, 0.2)  # -11.4 to 11.4 degrees
        self.params['translation_x'] = random.uniform(-0.1, 0.1)   # -10% to 10% of width
        self.params['translation_y'] = random.uniform(-0.1, 0.1)   # -10% to 10% of height
        self.params['scale_factor'] = random.uniform(0.9, 1.1)     # 90% to 110% scaling

        # Update GUI elements if they exist
        try:
            self.rotation_var.set(self.params['rotation_angle'])
            self.translation_x_var.set(self.params['translation_x'])
            self.translation_y_var.set(self.params['translation_y'])
            self.scale_var.set(self.params['scale_factor'])
        except:
            pass  # Ignore if GUI elements not initialized yet

        # Generate misaligned version
        self.img2_original = self.generate_synthetic_transformations(self.img1_original)
        self.img2_original_copy = self.img2_original.copy()

    def run_next_scale(self):
        """Process the next scale in the hierarchical multi-scale alignment"""
        if not self.is_aligning:
            return

        if self.current_scale_index >= len(self.params['pyramid_scales']):
            self.finalize_current_image()
            return

        # Get current scale
        current_scale = self.params['pyramid_scales'][self.current_scale_index]
        self.progress_var.set(f"Scale: {current_scale} ({self.current_scale_index+1}/{len(self.params['pyramid_scales'])})")
        self.status_var.set(f"Processing scale {current_scale:.4f} ({self.current_scale_index+1}/{len(self.params['pyramid_scales'])})...")

        # Calculate adaptive parameters for this scale
        # Use tighter constraints for finer scales
        progress = self.current_scale_index / max(1, len(self.params['pyramid_scales']) - 1)

        # Adaptive ratio threshold: 0.8 -> 0.6 (stricter for finer scales)
        ratio_threshold = 0.8 - (progress * 0.2)

        # Adaptive distance threshold: 1.5 -> 0.5 (stricter for finer scales)
        distance_threshold = 1.5 - (progress * 1.0)

        # Adaptive RANSAC threshold: 10.0 -> 1.5 (stricter for finer scales)
        ransac_threshold = 10.0 - (progress * 8.5)

        try:
            # Get image dimensions
            h_orig, w_orig = self.img2_original.shape[:2]

            # Create scaled versions of both images
            if current_scale == 1.0:
                img1_scaled = self.img1_original
                img2_scaled = self.img2_original
            else:
                new_w1, new_h1 = int(w_orig * current_scale), int(h_orig * current_scale)
                new_w2, new_h2 = int(w_orig * current_scale), int(h_orig * current_scale)

                if current_scale < 1.0 and (new_w1 < 36 or new_h1 < 36 or new_w2 < 36 or new_h2 < 36):
                    # Skip this scale if too small
                    print(f"Scale {current_scale} too small, skipping...")
                    self.current_scale_index += 1
                    self.root.after(100, self.run_next_scale)
                    return

                # Use linear interpolation for scaling
                interpolation = cv2.INTER_LINEAR

                img1_scaled = cv2.resize(self.img1_original, (new_w1, new_h1), interpolation=interpolation)
                img2_scaled = cv2.resize(self.img2_original, (new_w2, new_h2), interpolation=interpolation)

            # Prepare transformation from previous scale if exists
            prev_homography_scaled = None
            if self.current_scale_index > 0:
                # Get the homography from the previous (coarser) scale and adjust for scale difference
                prev_scale = self.params['pyramid_scales'][self.current_scale_index - 1]

                if prev_scale in self.scale_homographies:
                    # All scales are downscaling now, no upscaling at 2x
                    # Standard pyramid processing case
                    scale_factor = current_scale / prev_scale
                    prev_H = self.scale_homographies[prev_scale].copy()

                    # Adjust translation components for the new scale
                    prev_H[0, 2] *= (current_scale / prev_scale)  # tx
                    prev_H[1, 2] *= (current_scale / prev_scale)  # ty
                    # For rotation/scale components, we keep them as they should be scale-invariant
                    # but we might need to adjust them too depending on the specific transformation
                    prev_homography_scaled = prev_H
            else:
                # For the first scale, we start with identity
                prev_homography_scaled = np.eye(3, dtype=np.float32)

            # Extract features from scaled images
            self.status_var.set(f"Scale {current_scale:.4f}: Extracting features...")
            self.root.update_idletasks()

            # Temporarily adjust target keypoints and pyramid scales for this specific scale processing
            original_target_kp = self.params['target_keypoints']
            self.params['target_keypoints'] = 1000  # Use 1000 keypoints at each scale as requested

            # Extract features from both scaled images
            keypoints_1_scaled, desc1_scaled = self.extract_multiscale_features_single_scale(img1_scaled, 1.0)
            keypoints_2_scaled, desc2_scaled = self.extract_multiscale_features_single_scale(img2_scaled, 1.0)

            # Restore original target keypoints value
            self.params['target_keypoints'] = original_target_kp

            if keypoints_1_scaled is None or keypoints_2_scaled is None:
                self.status_var.set(f"Could not extract features at scale {current_scale}.")
                self.is_aligning = False
                return

            # Estimate homography with adaptive parameters
            self.status_var.set(f"Scale {current_scale:.4f}: Computing homography...")
            self.root.update_idletasks()

            scale_homography = self.estimate_homography_adaptive(
                keypoints_1_scaled, keypoints_2_scaled,
                desc1_scaled, desc2_scaled,
                ratio_threshold, distance_threshold, ransac_threshold
            )

            if scale_homography is not None:
                # If we have a previous homography for this scale, evaluate if the new one is better
                current_homography_before = self.homography.copy() if self.homography is not None else None

                # Apply the scale-level homography to the original-sized images
                # Since we're only using downscaling scales now (no upscaling), we simplify the logic

                if prev_homography_scaled is not None and self.current_scale_index > 0:
                    # Standard pyramid composition for downscaling levels
                    # Compose: new transformation applied to previous transformation
                    # The scale_homography transforms the scaled image, so we need to appropriately apply it
                    # First, we need to scale up the transformation
                    scale_homography_upscaled = scale_homography.copy()
                    scale_homography_upscaled[0, 2] *= (1.0 / current_scale)  # tx
                    scale_homography_upscaled[1, 2] *= (1.0 / current_scale)  # ty
                    # The rotation and scale components stay the same

                    # Then compose with the previous transformation
                    self.homography = scale_homography_upscaled
                    if self.scale_homographies:  # If we have previous scales' results
                        # We need to combine the transformations properly
                        # This is a simplified approach - in full implementation we'd need to be more careful
                        # about how transformations from different scales are combined
                        pass
                else:
                    # This is the first scale, no previous transformation to compose with
                    # Scale the homography to work with original-sized images
                    scale_factor = 1.0 / current_scale
                    self.homography = scale_homography.copy()
                    if current_scale != 1.0:  # Only scale if not at 1.0x already
                        self.homography[0, 2] *= (1.0 / current_scale)  # tx
                        self.homography[1, 2] *= (1.0 / current_scale)  # ty

                # Store this scale's homography
                self.scale_homographies[current_scale] = scale_homography.copy()

                # Apply the current homography to get the aligned image
                h, w = self.img2_original.shape[:2]
                self.img1_processed = cv2.warpPerspective(
                    self.img1_original, self.homography, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0)
                )

                # Calculate error
                gray1 = cv2.cvtColor(self.img1_processed, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(self.img2_original, cv2.COLOR_BGR2GRAY)
                avg_error = np.mean(cv2.absdiff(gray1, gray2))

                # Count matches at this scale
                # We'll use the original descriptors for consistency in evaluation
                # But for now, let's just store the keypoints for the scaled images
                good_matches = self.apply_lowe_ratio_test(desc1_scaled, desc2_scaled, ratio_threshold)
                filtered_matches = [(i, j, dist) for i, j, dist in good_matches if dist < distance_threshold]
                matches_count = len(filtered_matches)

                # Store scale result
                scale_result = {
                    'scale': current_scale,
                    'scale_index': self.current_scale_index,
                    'avg_error': avg_error,
                    'matches': matches_count,
                    'ratio_threshold': ratio_threshold,
                    'ransac_threshold': ransac_threshold,
                    'homography': self.homography.copy(),
                    'scale_homography': scale_homography.copy()  # The homography at this specific scale
                }
                self.scale_results[current_scale] = scale_result

                # Print scale info
                print(f"\n=== Scale Level {self.current_scale_index+1}: {current_scale:.4f} ===")
                print(f"Parameters:")
                print(f"  Ratio Threshold: {ratio_threshold:.3f}")
                print(f"  Distance Threshold: {distance_threshold:.3f}")
                print(f"  RANSAC Threshold: {ransac_threshold:.3f}")
                print(f"Results:")
                print(f"  Good matches: {matches_count}")
                print(f"  Average pixel error: {avg_error:.3f}")
                print(f"Homography (for original size):")
                print(self.homography)
                print(f"==================\n")

                # Update display to show current scale result
                self.update_display()
                self.root.update_idletasks()

                # Check if this scale made the alignment worse
                if current_homography_before is not None:
                    # Calculate error with previous homography
                    old_aligned = cv2.warpPerspective(
                        self.img1_original, current_homography_before, (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0)
                    )
                    old_gray1 = cv2.cvtColor(old_aligned, cv2.COLOR_BGR2GRAY)
                    old_gray2 = cv2.cvtColor(self.img2_original, cv2.COLOR_BGR2GRAY)
                    old_error = np.mean(cv2.absdiff(old_gray1, old_gray2))

                    if avg_error > old_error:
                        print(f"Alignment got worse at scale {current_scale} (error: {old_error:.3f} -> {avg_error:.3f}), rejecting homography")
                        # Reject the new homography and keep the previous one
                        self.homography = current_homography_before
                        # Also restore the corresponding image
                        self.img1_processed = old_aligned
                        # Update display with the better result
                        self.update_display()

                # Move to next scale
                self.current_scale_index += 1

                # Schedule next scale with a small delay for animation
                self.root.after(500, self.run_next_scale)
            else:
                print(f"Scale {current_scale}: Homography estimation failed")
                # Store the failure but continue to next scale
                self.current_scale_index += 1
                self.root.after(100, self.run_next_scale)

        except Exception as e:
            self.status_var.set(f"Error at scale {current_scale}: {str(e)}")
            print(f"Error at scale {current_scale}: {e}")
            import traceback
            traceback.print_exc()
            self.is_aligning = False

    def finalize_current_image(self):
        """Finalize the current image processing and record results"""
        # Record the results for this image
        result = {
            'image_name': os.path.basename(self.test_images[self.current_test_index]),
            'scale_used': self.params['pyramid_scales'],
            'final_homography': self.homography.copy() if self.homography is not None else None,
            'original_transformation': {
                'rotation_angle': self.params['rotation_angle'],
                'translation_x': self.params['translation_x'],
                'translation_y': self.params['translation_y'],
                'scale_factor': self.params['scale_factor']
            }
        }

        # Calculate alignment error if possible
        if self.img1_processed is not None and self.img2_original is not None:
            gray1 = cv2.cvtColor(self.img1_processed, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(self.img2_original, cv2.COLOR_BGR2GRAY)
            result['final_error'] = float(np.mean(cv2.absdiff(gray1, gray2)))
        else:
            result['final_error'] = float('inf')

        self.test_results.append(result)

        # Move to the next image
        self.current_test_index += 1

        if self.current_test_index < len(self.test_images):
            # Process the next image after a 3-second pause
            self.status_var.set(f"Completed image {self.current_test_index-1}/{len(self.test_images)}. Pausing 3 seconds before next image...")
            self.root.after(3000, self.process_next_test_image)
        else:
            # All images processed
            self.finalize_test_suite()

    def finalize_test_suite(self):
        """Finalize the test suite and display the results"""
        self.is_aligning = False

        # Generate and display the final report
        report = self.generate_test_report()

        self.status_var.set(f"Test suite completed. {len(self.test_results)} images processed.")
        print("\n" + "="*60)
        print("KNIFT MODEL TEST SUITE - FINAL REPORT")
        print("="*60)
        print(report)
        print("="*60)

        # Show summary in a messagebox
        messagebox.showinfo("Test Suite Complete", f"KNIFT Model Test Suite Complete!\n\n{len(self.test_results)} images processed.\nCheck console for detailed report.")

    def generate_test_report(self):
        """Generate a detailed test report"""
        if not self.test_results:
            return "No results to report."

        total_images = len(self.test_results)
        successful_alignments = 0
        total_error = 0
        avg_error = 0

        report_lines = []
        report_lines.append(f"Total Images Tested: {total_images}")
        report_lines.append(f"Pyramid Scales Used: {self.params['pyramid_scales']}")
        report_lines.append("")
        report_lines.append("Individual Results:")
        report_lines.append("-" * 50)

        for i, result in enumerate(self.test_results):
            error = result.get('final_error', float('inf'))
            total_error += error

            # Consider alignment successful if error is below a threshold
            error_threshold = 10.0  # This threshold can be adjusted based on requirements
            is_success = error < error_threshold
            if is_success:
                successful_alignments += 1

            report_lines.append(f"Image {i+1}: {result['image_name']}")
            report_lines.append(f"  - Original rotation: {result['original_transformation']['rotation_angle']:.3f} rad")
            report_lines.append(f"  - Original translation: ({result['original_transformation']['translation_x']:.3f}, {result['original_transformation']['translation_y']:.3f})")
            report_lines.append(f"  - Original scale: {result['original_transformation']['scale_factor']:.3f}")
            report_lines.append(f"  - Final alignment error: {error:.3f}")
            report_lines.append(f"  - Status: {'SUCCESS' if is_success else 'FAILED'}")
            report_lines.append("")

        avg_error = total_error / total_images if total_images > 0 else 0
        success_rate = (successful_alignments / total_images) * 100 if total_images > 0 else 0

        report_lines.append("SUMMARY:")
        report_lines.append("-" * 20)
        report_lines.append(f"Successful Alignments: {successful_alignments}/{total_images}")
        report_lines.append(f"Success Rate: {success_rate:.2f}%")
        report_lines.append(f"Average Final Error: {avg_error:.3f}")

        return "\n".join(report_lines)

    def extract_multiscale_features_single_scale(self, img, scale_factor):
        """Extract features at a single scale level - helper for multi-scale processing"""
        if self.interpreter is None or img is None:
            return None, None

        # Use the configured target keypoints for this scale
        target_keypoints = min(self.params['target_keypoints'], 1000)  # Limit to 1000 as suggested

        all_keypoints = []
        all_descriptors = []

        points_scaled = self.detect_fast_keypoints_with_threshold_dropping(
            img, target_keypoints
        )

        if len(points_scaled) == 0:
            return None, None

        patches, valid_indices = self.extract_patches(img, points_scaled)

        if not patches:
            return None, None

        patches_array = np.array(patches, dtype=np.float32) / 255.0
        patches_array = np.expand_dims(patches_array, axis=-1)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        expected_batch = input_details[0]['shape'][0]

        actual_n = patches_array.shape[0]
        padded_input = np.zeros((expected_batch, 32, 32, 1), dtype=np.float32)

        limit = min(actual_n, expected_batch)
        padded_input[:limit] = patches_array[:limit]

        self.interpreter.set_tensor(input_details[0]['index'], padded_input)
        self.interpreter.invoke()

        full_output = self.interpreter.get_tensor(output_details[0]['index'])
        descriptors = full_output[:limit]

        valid_points_scaled = points_scaled[valid_indices[:limit]]
        points_original_coords = valid_points_scaled  # Already in the scale's coordinate system

        all_keypoints.append(points_original_coords)
        all_descriptors.append(descriptors)

        if len(all_keypoints) > 0:
            final_keypoints = np.vstack(all_keypoints)
            final_descriptors = np.vstack(all_descriptors)
            return final_keypoints, final_descriptors
        else:
            return None, None

    def finalize_alignment(self):
        """Finalize the alignment process"""
        self.is_aligning = False

        if self.scale_results:
            # Find best result across all scales
            best_scale_result = min(self.scale_results.values(), key=lambda x: x['avg_error'])
            best_scale = best_scale_result['scale']

            print(f"\n=== Alignment Complete ===")
            print(f"Transformation parameters used:")
            print(f"  Rotation: {self.params['rotation_angle']:.3f} rad")
            print(f"  Translation X: {self.params['translation_x']:.3f}")
            print(f"  Translation Y: {self.params['translation_y']:.3f}")
            print(f"  Scale Factor: {self.params['scale_factor']:.3f}")
            print(f"\nMulti-scale pyramid processed: {self.params['pyramid_scales']}")
            print(f"\nBest result from scale level {best_scale_result['scale_index']+1} ({best_scale}):")
            print(f"  Average pixel error: {best_scale_result['avg_error']:.3f}")
            print(f"  Good matches: {best_scale_result['matches']}")
            print(f"\nFinal Homography Matrix:")
            print(best_scale_result['homography'])
            print(f"========================\n")

            # Update to use the best homography
            self.homography = best_scale_result['homography']
            h, w = self.img2_original.shape[:2]
            self.img1_processed = cv2.warpPerspective(
                self.img1_original, self.homography, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            self.status_var.set(f"Alignment complete! Best scale: {best_scale}, "
                              f"Error: {best_scale_result['avg_error']:.3f}px, Matches: {best_scale_result['matches']}")
            self.update_display()
        else:
            self.status_var.set("Alignment failed - no successful scales")

    def detect_fast_keypoints_with_threshold_dropping(self, img, target_count):
        """Detect FAST keypoints with threshold dropping and spatial sampling"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        current_threshold = self.params['fast_initial_threshold']
        min_threshold = self.params['fast_min_threshold']
        threshold_step = self.params['fast_threshold_step']

        while current_threshold >= min_threshold:
            fast = cv2.FastFeatureDetector_create(
                threshold=current_threshold,
                nonmaxSuppression=self.params['fast_nonmaxSuppression']
            )

            keypoints = fast.detect(gray, None)

            if len(keypoints) >= target_count:
                # Sort by response (corner quality)
                keypoints_sorted = sorted(keypoints, key=lambda x: x.response, reverse=True)

                # Take top keypoints and apply spatial sampling
                kp_coords = np.array([kp.pt for kp in keypoints_sorted[:target_count * 3]], dtype=np.float32)
                sampled_coords = self.spatial_sample_keypoints_with_response(
                    keypoints_sorted[:target_count * 3], target_count, img.shape
                )
                return sampled_coords

            current_threshold -= threshold_step

        # Return what we have
        kp_coords = np.array([kp.pt for kp in keypoints], dtype=np.float32) if keypoints else np.empty((0, 2), dtype=np.float32)
        return kp_coords

    def spatial_sample_keypoints_with_response(self, keypoints_objs, target_count, img_shape):
        """Spatial sampling that prefers stronger corner responses"""
        if len(keypoints_objs) <= target_count:
            return np.array([kp.pt for kp in keypoints_objs], dtype=np.float32)

        h, w = img_shape[:2]
        grid_size = int(np.sqrt(target_count))
        grid_rows = grid_size
        grid_cols = grid_size

        if grid_rows * grid_cols < target_count:
            grid_cols += 1
        if grid_rows * grid_cols < target_count:
            grid_rows += 1

        cell_h = h / grid_rows
        cell_w = w / grid_cols

        # Group keypoints by grid cell
        grid_cells = {}
        for kp in keypoints_objs:
            row = int(kp.pt[1] / cell_h)
            col = int(kp.pt[0] / cell_w)
            cell_id = (row, col)

            if cell_id not in grid_cells:
                grid_cells[cell_id] = []
            grid_cells[cell_id].append(kp)

        # Select best keypoint from each cell
        selected_kps = []
        for cell_id, cell_kps in grid_cells.items():
            best_kp = max(cell_kps, key=lambda x: x.response)
            selected_kps.append(best_kp.pt)

            if len(selected_kps) >= target_count:
                break

        return np.array(selected_kps, dtype=np.float32)

    def extract_patches(self, img, grid_points, patch_size=32):
        """Extract patches around grid points"""
        patches = []
        valid_indices = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape

        for i, (x, y) in enumerate(grid_points):
            x, y = int(x), int(y)
            patch_center_x = x - patch_size//2
            patch_center_y = y - patch_size//2

            if (patch_center_x >= 0 and patch_center_x + patch_size <= w and
                patch_center_y >= 0 and patch_center_y + patch_size <= h):
                patch = gray[patch_center_y:patch_center_y + patch_size,
                           patch_center_x:patch_center_x + patch_size]
                patches.append(patch)
                valid_indices.append(i)

        return patches, valid_indices

    def extract_multiscale_features(self, img_original):
        """Extract KNIFT features on multiple scales with FAST keypoint detection"""
        if self.interpreter is None or img_original is None:
            return None, None

        all_keypoints = []
        all_descriptors = []

        scales = self.params['pyramid_scales']
        target_keypoints = self.params['target_keypoints']

        h_orig, w_orig = img_original.shape[:2]

        for scale in scales:
            if scale == 1.0:
                img_scaled = img_original
            else:
                new_w, new_h = int(w_orig * scale), int(h_orig * scale)
                if new_w < 36 or new_h < 36:
                    continue
                img_scaled = cv2.resize(img_original, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            current_h, current_w = img_scaled.shape[:2]

            points_scaled = self.detect_fast_keypoints_with_threshold_dropping(
                img_scaled, target_keypoints
            )

            if len(points_scaled) == 0:
                continue

            patches, valid_indices = self.extract_patches(img_scaled, points_scaled)

            if not patches:
                continue

            patches_array = np.array(patches, dtype=np.float32) / 255.0
            patches_array = np.expand_dims(patches_array, axis=-1)

            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            expected_batch = input_details[0]['shape'][0]

            actual_n = patches_array.shape[0]
            padded_input = np.zeros((expected_batch, 32, 32, 1), dtype=np.float32)

            limit = min(actual_n, expected_batch)
            padded_input[:limit] = patches_array[:limit]

            self.interpreter.set_tensor(input_details[0]['index'], padded_input)
            self.interpreter.invoke()

            full_output = self.interpreter.get_tensor(output_details[0]['index'])
            descriptors = full_output[:limit]

            valid_points_scaled = points_scaled[valid_indices[:limit]]
            points_original_coords = valid_points_scaled * (1.0 / scale)

            all_keypoints.append(points_original_coords)
            all_descriptors.append(descriptors)

        if len(all_keypoints) > 0:
            final_keypoints = np.vstack(all_keypoints)
            final_descriptors = np.vstack(all_descriptors)
            return final_keypoints, final_descriptors
        else:
            return None, None

    def apply_lowe_ratio_test(self, desc1, desc2, ratio_threshold=0.7):
        """Apply Lowe's Ratio Test to find good matches"""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        try:
            distances = np.linalg.norm(desc1[:, np.newaxis, :] - desc2[np.newaxis, :, :], axis=2)
            good_matches = []

            for i in range(len(desc1)):
                dists = distances[i]
                sorted_indices = np.argsort(dists)

                if len(sorted_indices) < 2:
                    continue

                best_idx, second_best_idx = sorted_indices[0], sorted_indices[1]
                best_dist, second_best_dist = dists[best_idx], dists[second_best_idx]

                if second_best_dist > 0:
                    ratio = best_dist / second_best_dist
                    if ratio < ratio_threshold:
                        good_matches.append((i, best_idx, best_dist))

            return good_matches
        except Exception as e:
            print(f"Error in Lowe's Ratio Test: {e}")
            return []

    def estimate_homography_adaptive(self, keypoints_1, keypoints_2, desc1, desc2,
                                    ratio_threshold, distance_threshold, ransac_threshold):
        """Adaptive homography estimation with iterative refinement"""
        if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
            return None

        try:
            # Apply Lowe's Ratio Test
            good_matches = self.apply_lowe_ratio_test(desc1, desc2, ratio_threshold)

            # Filter matches by distance threshold
            filtered_matches = [(i, j, dist) for i, j, dist in good_matches if dist < distance_threshold]

            if len(filtered_matches) < 8:
                return None

            src_pts = np.float32([keypoints_1[i] for i, j, _ in filtered_matches])
            dst_pts = np.float32([keypoints_2[j] for i, j, _ in filtered_matches])

            # Initial RANSAC
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)

            if homography is None:
                return None

            # Iterative refinement
            inliers_src = src_pts[mask.ravel() == 1]
            inliers_dst = dst_pts[mask.ravel() == 1]

            # Compute reprojection errors
            inliers_src_h = np.column_stack([inliers_src, np.ones(len(inliers_src))])
            projected = (homography @ inliers_src_h.T).T
            projected = projected[:, :2] / projected[:, 2:3]
            errors = np.linalg.norm(inliers_dst - projected, axis=1)

            # Keep only points with low reprojection error
            strict_threshold = ransac_threshold * 0.6
            refined_mask = errors < strict_threshold

            if np.sum(refined_mask) >= 8:
                refined_src = inliers_src[refined_mask]
                refined_dst = inliers_dst[refined_mask]
                homography_refined, _ = cv2.findHomography(refined_src, refined_dst, 0)

                if homography_refined is not None:
                    return homography_refined

            return homography

        except Exception as e:
            print(f"Error in homography estimation: {e}")
            return None

    def create_50_percent_overlay(self):
        """Create a 50% overlay of the two images"""
        if self.img2_original is None:
            return None

        h, w = self.img2_original.shape[:2]

        if self.img1_processed is not None:
            img1_to_use = self.img1_processed
        elif self.img1_original is not None:
            img1_to_use = self.img1_original
        else:
            return None

        try:
            if img1_to_use.shape[:2] != (h, w):
                img1_to_use = cv2.resize(img1_to_use, (w, h))

            if img1_to_use.dtype != np.uint8:
                img1_to_use = img1_to_use.astype(np.uint8)

            if self.img2_original.dtype != np.uint8:
                img2_safe = self.img2_original.astype(np.uint8)
            else:
                img2_safe = self.img2_original

            overlay = cv2.addWeighted(img1_to_use, 0.5, img2_safe, 0.5, 0)
            return overlay
        except Exception as e:
            print(f"Error creating overlay: {e}")
            return None

    def update_display(self):
        """Update the display"""
        # Update overlay view
        overlay_img = self.create_50_percent_overlay()
        if overlay_img is not None:
            overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
            overlay_pil = Image.fromarray(overlay_rgb)

            self.root.update_idletasks()
            canvas_width = self.canvas_overlay.winfo_width()
            if canvas_width <= 1: canvas_width = 600
            canvas_height = self.canvas_overlay.winfo_height()
            if canvas_height <= 1: canvas_height = 400

            img_ratio = overlay_pil.width / overlay_pil.height
            display_width = min(canvas_width - 20, overlay_pil.width)
            display_height = min(canvas_height - 20, int(display_width / img_ratio))

            if display_height > canvas_height - 20:
                display_height = canvas_height - 20
                display_width = int(display_height * img_ratio)

            overlay_display = overlay_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)

            self.overlay_tk = ImageTk.PhotoImage(overlay_display)
            self.canvas_overlay.delete("all")

            x_pos = (canvas_width - display_width) // 2
            y_pos = (canvas_height - display_height) // 2
            self.canvas_overlay.create_image(x_pos, y_pos, anchor=tk.NW, image=self.overlay_tk)

            label_text = f"Iteration {self.current_iteration}" if self.is_aligning else "Overlay (50%)"
            if self.img1_processed is not None:
                label_text += " - Aligned"

            self.canvas_overlay.create_text(display_width // 2 + x_pos, 5, 
                                           anchor=tk.N, text=label_text, 
                                           fill="lime", font=("Arial", 12, "bold"))

        # Update keypoint visualization
        if self.img1_original is not None and self.img2_original is not None:
            h1, w1 = self.img1_original.shape[:2]
            h2, w2 = self.img2_original.shape[:2]
            max_h = max(h1, h2)
            total_w = w1 + w2

            combined_img = np.zeros((max_h, total_w, 3), dtype=np.uint8)
            combined_img[:h1, :w1] = self.img1_original
            combined_img[:h2, w1:w1+w2] = self.img2_original

            combined_rgb = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
            combined_pil = Image.fromarray(combined_rgb)

            canvas_width = self.canvas_aligned.winfo_width() or 600
            canvas_height = self.canvas_aligned.winfo_height() or 400

            img_ratio = combined_pil.width / combined_pil.height
            display_width = min(canvas_width - 20, combined_pil.width)
            display_height = min(canvas_height - 40, int(display_width / img_ratio))

            if display_height > canvas_height - 40:
                display_height = canvas_height - 40
                display_width = int(display_height * img_ratio)

            combined_display = combined_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)

            self.combined_tk = ImageTk.PhotoImage(combined_display)
            self.canvas_aligned.delete("all")

            x_pos = (canvas_width - display_width) // 2
            y_pos = (canvas_height - display_height) // 2
            self.canvas_aligned.create_image(x_pos, y_pos, anchor=tk.NW, image=self.combined_tk)

            # Draw keypoints
            if self.keypoints_1 is not None and len(self.keypoints_1) > 0:
                scale_x = display_width / combined_pil.width
                scale_y = display_height / combined_pil.height

                for pt in self.keypoints_1:
                    x = int(pt[0] * scale_x) + x_pos
                    y = int(pt[1] * scale_y) + y_pos
                    self.canvas_aligned.create_oval(x-2, y-2, x+2, y+2, fill='green', outline='green')

                for pt in self.keypoints_2:
                    x = int((pt[0] + w1) * scale_x) + x_pos
                    y = int(pt[1] * scale_y) + y_pos
                    self.canvas_aligned.create_oval(x-2, y-2, x+2, y+2, fill='blue', outline='blue')

                if self.homography is not None:
                    transformed_pts = cv2.perspectiveTransform(
                        self.keypoints_1.reshape(-1, 1, 2), self.homography
                    ).reshape(-1, 2)

                    for pt in transformed_pts:
                        x = int((pt[0] + w1) * scale_x) + x_pos
                        y = int(pt[1] * scale_y) + y_pos
                        self.canvas_aligned.create_oval(x-1, y-1, x+1, y+1, 
                                                       fill='', outline='red', width=1)

def main():
    root = tk.Tk()
    app = KNIFTAlignmentTestSuite(root)
    root.mainloop()

if __name__ == "__main__":
    main()
