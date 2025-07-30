import os
import logging
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.feature import hog
import matplotlib.pyplot as plt
import json

class ImageProcessor:
    """
    Image processing utilities for Pokémon card analysis.
    Handles image preprocessing, normalization, and enhancement for AI model input.
    """
    
    def __init__(self):
        """
        Initialize image processor with standard card dimensions and processing parameters.
        """
        self.target_size = (224, 224)  # Standard input size for MobileNetV2
        self.card_aspect_ratio = 2.5 / 3.5  # Standard Pokémon card aspect ratio
        
        logging.info("Image Processor initialized")
    
    def preprocess_for_ai(self, image_path):
        """
        Preprocess card image for AI model input.
        
        Args:
            image_path: Path to the card image file
            
        Returns:
            numpy.ndarray: Preprocessed image array ready for AI model
        """
        try:
            # Load and validate image
            image = self._load_and_validate_image(image_path)
            if image is None:
                return None
            
            # Enhance image quality
            enhanced_image = self._enhance_image_quality(image)
            
            # Normalize card orientation and crop
            normalized_image = self._normalize_card_orientation(enhanced_image)
            
            # Resize to model input size
            resized_image = self._resize_for_model(normalized_image)
            
            # Convert to array and normalize pixel values
            image_array = np.array(resized_image).astype(np.float32) / 255.0
            
            logging.info(f"Image preprocessed successfully: {image_array.shape}")
            return image_array
            
        except Exception as e:
            logging.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def _load_and_validate_image(self, image_path):
        """
        Load image and perform basic validation.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL.Image: Loaded and validated image
        """
        try:
            if not os.path.exists(image_path):
                logging.error(f"Image file not found: {image_path}")
                return None
            
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Validate image dimensions
            width, height = image.size
            if width < 100 or height < 100:
                logging.warning("Image too small for reliable analysis")
                return None
            
            if width > 4000 or height > 4000:
                logging.info("Large image detected, will be resized")
            
            return image
            
        except Exception as e:
            logging.error(f"Error loading image: {str(e)}")
            return None
    
    def _enhance_image_quality(self, image):
        """
        Enhance image quality for better analysis.
        
        Args:
            image: PIL Image object
            
        Returns:
            PIL.Image: Enhanced image
        """
        try:
            # Auto-orient image based on EXIF data
            image = ImageOps.exif_transpose(image)
            
            # Enhance contrast slightly
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
            
            # Enhance color saturation slightly
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.05)
            
            # Enhance sharpness slightly
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            return image
            
        except Exception as e:
            logging.warning(f"Image enhancement failed: {str(e)}")
            return image
    
    def _normalize_card_orientation(self, image):
        """
        Normalize card orientation and attempt to crop to card boundaries.
        
        Args:
            image: PIL Image object
            
        Returns:
            PIL.Image: Normalized and cropped image
        """
        try:
            # Convert to OpenCV format for advanced processing
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect card boundaries using edge detection
            cropped_image = self._detect_and_crop_card(cv_image)
            
            if cropped_image is not None:
                # Convert back to PIL Image
                cropped_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                return cropped_pil
            else:
                # Fallback: just ensure proper aspect ratio
                return self._crop_to_aspect_ratio(image)
                
        except Exception as e:
            logging.warning(f"Card normalization failed: {str(e)}")
            return self._crop_to_aspect_ratio(image)
    
    def _detect_and_crop_card(self, cv_image):
        """
        Detect card boundaries using computer vision techniques.
        
        Args:
            cv_image: OpenCV image array
            
        Returns:
            numpy.ndarray: Cropped card image or None if detection fails
        """
        try:
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (likely the card)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Validate aspect ratio (cards should be roughly rectangular)
                aspect_ratio = w / h
                if 0.6 < aspect_ratio < 0.8:  # Typical card aspect ratio range
                    # Add some padding
                    padding = 10
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(cv_image.shape[1] - x, w + 2 * padding)
                    h = min(cv_image.shape[0] - y, h + 2 * padding)
                    
                    return cv_image[y:y+h, x:x+w]
            
            return None
            
        except Exception as e:
            logging.warning(f"Card detection failed: {str(e)}")
            return None
    
    def _crop_to_aspect_ratio(self, image):
        """
        Crop image to standard card aspect ratio.
        
        Args:
            image: PIL Image object
            
        Returns:
            PIL.Image: Cropped image with correct aspect ratio
        """
        try:
            width, height = image.size
            current_ratio = width / height
            
            if abs(current_ratio - self.card_aspect_ratio) < 0.1:
                # Already close to correct aspect ratio
                return image
            
            # Calculate crop dimensions
            if current_ratio > self.card_aspect_ratio:
                # Image is too wide
                new_width = int(height * self.card_aspect_ratio)
                left = (width - new_width) // 2
                crop_box = (left, 0, left + new_width, height)
            else:
                # Image is too tall
                new_height = int(width / self.card_aspect_ratio)
                top = (height - new_height) // 2
                crop_box = (0, top, width, top + new_height)
            
            return image.crop(crop_box)
            
        except Exception as e:
            logging.warning(f"Aspect ratio cropping failed: {str(e)}")
            return image
    
    def _resize_for_model(self, image):
        """
        Resize image to model input dimensions.
        
        Args:
            image: PIL Image object
            
        Returns:
            PIL.Image: Resized image
        """
        try:
            # Use high-quality resampling
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
            
        except Exception as e:
            logging.error(f"Image resizing failed: {str(e)}")
            # Fallback to basic resize
            return image.resize(self.target_size)
    
    def create_comparison_view(self, original_path, official_card_path=None):
        """
        Create a side-by-side comparison view for visual analysis.
        
        Args:
            original_path: Path to the uploaded card image
            official_card_path: Path to official card image (optional)
            
        Returns:
            PIL.Image: Combined comparison image
        """
        try:
            # Load the uploaded card
            uploaded_card = Image.open(original_path)
            uploaded_card = uploaded_card.convert('RGB')
            
            # Load official card or create placeholder
            if official_card_path and os.path.exists(official_card_path):
                official_card = Image.open(official_card_path)
                official_card = official_card.convert('RGB')
            else:
                # Create placeholder for official card
                official_card = self._create_placeholder_card()
            
            # Resize both images to same height
            target_height = 400
            uploaded_width = int(uploaded_card.width * target_height / uploaded_card.height)
            official_width = int(official_card.width * target_height / official_card.height)
            
            uploaded_card = uploaded_card.resize((uploaded_width, target_height), Image.Resampling.LANCZOS)
            official_card = official_card.resize((official_width, target_height), Image.Resampling.LANCZOS)
            
            # Create combined image
            total_width = uploaded_width + official_width + 20  # 20px gap
            comparison_image = Image.new('RGB', (total_width, target_height), color='white')
            
            # Paste images
            comparison_image.paste(uploaded_card, (0, 0))
            comparison_image.paste(official_card, (uploaded_width + 20, 0))
            
            return comparison_image
            
        except Exception as e:
            logging.error(f"Error creating comparison view: {str(e)}")
            return None
    
    def _create_placeholder_card(self):
        """
        Create a placeholder image for official card comparison.
        
        Returns:
            PIL.Image: Placeholder card image
        """
        from PIL import ImageDraw, ImageFont
        
        # Create a card-sized placeholder
        width, height = 250, 350
        placeholder = Image.new('RGB', (width, height), color='#f0f0f0')
        draw = ImageDraw.Draw(placeholder)
        
        # Draw border
        draw.rectangle([5, 5, width-5, height-5], outline='#cccccc', width=2)
        
        # Add text
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
        
        text = "Official Card\nImage\nNot Available"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (width - text_width) // 2
        text_y = (height - text_height) // 2
        
        draw.text((text_x, text_y), text, fill='#666666', font=font)
        
        return placeholder
    
    def extract_card_regions(self, image_path):
        """
        Extract specific regions of the card for detailed analysis.
        
        Args:
            image_path: Path to the card image
            
        Returns:
            dict: Dictionary containing different card regions
        """
        try:
            image = Image.open(image_path)
            width, height = image.size
            
            # Define approximate regions based on standard card layout
            regions = {
                'name_region': image.crop((0, 0, width, height // 8)),
                'artwork_region': image.crop((width // 10, height // 8, 
                                            9 * width // 10, 5 * height // 8)),
                'text_region': image.crop((width // 10, 5 * height // 8, 
                                         9 * width // 10, 7 * height // 8)),
                'bottom_region': image.crop((0, 7 * height // 8, width, height))
            }
            
            return regions
            
        except Exception as e:
            logging.error(f"Error extracting card regions: {str(e)}")
            return {}
    
    def compare_visual_similarity(self, uploaded_path, reference_path):
        """
        Compare visual similarity between uploaded card and reference card.
        
        Args:
            uploaded_path: Path to uploaded card image
            reference_path: Path to reference card image
            
        Returns:
            dict: Similarity analysis results
        """
        try:
            # Load and preprocess images
            uploaded_img = cv2.imread(uploaded_path)
            reference_img = cv2.imread(reference_path) if os.path.exists(reference_path) else None
            
            if uploaded_img is None:
                return {"error": "Could not load uploaded image"}
            
            if reference_img is None:
                return {"error": "Reference image not available"}
            
            # Resize images to same size for comparison
            target_size = (400, 560)  # Card aspect ratio
            uploaded_resized = cv2.resize(uploaded_img, target_size)
            reference_resized = cv2.resize(reference_img, target_size)
            
            # Convert to grayscale for SSIM
            uploaded_gray = cv2.cvtColor(uploaded_resized, cv2.COLOR_BGR2GRAY)
            reference_gray = cv2.cvtColor(reference_resized, cv2.COLOR_BGR2GRAY)
            
            # Calculate SSIM
            ssim_score = ssim(uploaded_gray, reference_gray)
            
            # Feature matching using ORB
            orb_similarity = self._calculate_orb_similarity(uploaded_gray, reference_gray)
            
            # Color histogram comparison
            color_similarity = self._compare_color_histograms(uploaded_resized, reference_resized)
            
            # Edge analysis
            edge_analysis = self._analyze_edge_similarity(uploaded_gray, reference_gray)
            
            return {
                "ssim_score": ssim_score,
                "orb_similarity": orb_similarity,
                "color_similarity": color_similarity,
                "edge_analysis": edge_analysis,
                "overall_similarity": (ssim_score + orb_similarity + color_similarity) / 3
            }
            
        except Exception as e:
            logging.error(f"Error in visual similarity comparison: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_orb_similarity(self, img1, img2):
        """
        Calculate ORB feature similarity between two images.
        
        Args:
            img1, img2: Grayscale images
            
        Returns:
            float: Similarity score (0-1)
        """
        try:
            # Initialize ORB detector
            orb = cv2.ORB_create(nfeatures=1000)
            
            # Detect keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            
            if des1 is None or des2 is None:
                return 0.0
            
            # Match descriptors
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Calculate similarity based on good matches
            good_matches = [m for m in matches if m.distance < 50]
            similarity = len(good_matches) / max(len(kp1), len(kp2), 1)
            
            return min(similarity, 1.0)
            
        except Exception as e:
            logging.warning(f"ORB similarity calculation failed: {str(e)}")
            return 0.0
    
    def _compare_color_histograms(self, img1, img2):
        """
        Compare color histograms between two images.
        
        Args:
            img1, img2: BGR images
            
        Returns:
            float: Similarity score (0-1)
        """
        try:
            # Convert to HSV for better color analysis
            hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms
            hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
            hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
            
            # Compare histograms using correlation
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            return max(0.0, correlation)
            
        except Exception as e:
            logging.warning(f"Color histogram comparison failed: {str(e)}")
            return 0.0
    
    def _analyze_edge_similarity(self, img1, img2):
        """
        Analyze edge patterns similarity between two images.
        
        Args:
            img1, img2: Grayscale images
            
        Returns:
            dict: Edge analysis results
        """
        try:
            # Apply Canny edge detection
            edges1 = cv2.Canny(img1, 50, 150)
            edges2 = cv2.Canny(img2, 50, 150)
            
            # Calculate edge density
            edge_density1 = np.sum(edges1 > 0) / (edges1.shape[0] * edges1.shape[1])
            edge_density2 = np.sum(edges2 > 0) / (edges2.shape[0] * edges2.shape[1])
            
            # Compare edge patterns using XOR
            edge_diff = cv2.bitwise_xor(edges1, edges2)
            edge_similarity = 1.0 - (np.sum(edge_diff > 0) / (edge_diff.shape[0] * edge_diff.shape[1]))
            
            return {
                "edge_similarity": edge_similarity,
                "edge_density_uploaded": edge_density1,
                "edge_density_reference": edge_density2,
                "edge_density_difference": abs(edge_density1 - edge_density2)
            }
            
        except Exception as e:
            logging.warning(f"Edge analysis failed: {str(e)}")
            return {"edge_similarity": 0.0}
    
    def analyze_print_quality(self, image_path):
        """
        Analyze print quality indicators in the card image.
        
        Args:
            image_path: Path to the card image
            
        Returns:
            dict: Print quality analysis results
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Sharpness analysis using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Edge strength analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_strength = np.mean(edges)
            
            # Text clarity analysis (focusing on text regions)
            height, width = gray.shape
            text_region = gray[int(height*0.6):int(height*0.9), int(width*0.1):int(width*0.9)]
            text_clarity = cv2.Laplacian(text_region, cv2.CV_64F).var()
            
            # Color consistency analysis
            color_std = np.std(image, axis=(0, 1))
            color_consistency = 1.0 / (1.0 + np.mean(color_std) / 255.0)
            
            return {
                "sharpness_score": min(laplacian_var / 100.0, 1.0),
                "edge_strength": edge_strength / 255.0,
                "text_clarity": min(text_clarity / 50.0, 1.0),
                "color_consistency": color_consistency,
                "overall_quality": (min(laplacian_var / 100.0, 1.0) + 
                                  edge_strength / 255.0 + 
                                  min(text_clarity / 50.0, 1.0) + 
                                  color_consistency) / 4
            }
            
        except Exception as e:
            logging.error(f"Error analyzing print quality: {str(e)}")
            return {"error": str(e)}
    
    def detect_geometric_anomalies(self, image_path):
        """
        Detect geometric anomalies that might indicate a fake card.
        
        Args:
            image_path: Path to the card image
            
        Returns:
            dict: Geometric analysis results
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Border analysis
            border_analysis = self._analyze_card_borders(gray)
            
            # Corner roundness analysis
            corner_analysis = self._analyze_corner_roundness(gray)
            
            # Text box alignment analysis
            alignment_analysis = self._analyze_text_alignment(gray)
            
            return {
                "border_analysis": border_analysis,
                "corner_analysis": corner_analysis,
                "alignment_analysis": alignment_analysis
            }
            
        except Exception as e:
            logging.error(f"Error detecting geometric anomalies: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_card_borders(self, gray_image):
        """
        Analyze card border characteristics.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            dict: Border analysis results
        """
        try:
            height, width = gray_image.shape
            
            # Extract border regions
            border_width = min(width // 20, height // 20)
            top_border = gray_image[0:border_width, :]
            bottom_border = gray_image[height-border_width:height, :]
            left_border = gray_image[:, 0:border_width]
            right_border = gray_image[:, width-border_width:width]
            
            # Analyze border consistency
            border_means = [
                np.mean(top_border),
                np.mean(bottom_border),
                np.mean(left_border),
                np.mean(right_border)
            ]
            
            border_std = np.std(border_means)
            border_consistency = 1.0 / (1.0 + border_std / 255.0)
            
            return {
                "border_consistency": border_consistency,
                "border_means": border_means,
                "border_std": border_std
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_corner_roundness(self, gray_image):
        """
        Analyze corner roundness characteristics.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            dict: Corner analysis results
        """
        try:
            height, width = gray_image.shape
            corner_size = min(width // 10, height // 10)
            
            # Extract corners
            corners = {
                "top_left": gray_image[0:corner_size, 0:corner_size],
                "top_right": gray_image[0:corner_size, width-corner_size:width],
                "bottom_left": gray_image[height-corner_size:height, 0:corner_size],
                "bottom_right": gray_image[height-corner_size:height, width-corner_size:width]
            }
            
            corner_scores = {}
            for corner_name, corner_img in corners.items():
                # Detect circular features in corners
                circles = cv2.HoughCircles(corner_img, cv2.HOUGH_GRADIENT, 1, 20,
                                         param1=50, param2=30, minRadius=5, maxRadius=corner_size//2)
                
                corner_scores[corner_name] = len(circles[0]) if circles is not None else 0
            
            return {
                "corner_roundness_scores": corner_scores,
                "corner_consistency": np.std(list(corner_scores.values()))
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_text_alignment(self, gray_image):
        """
        Analyze text alignment and layout consistency.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            dict: Alignment analysis results
        """
        try:
            # Apply edge detection to find text boxes
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Find horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            # Analyze line straightness
            horizontal_score = np.sum(horizontal_lines) / (horizontal_lines.shape[0] * horizontal_lines.shape[1])
            vertical_score = np.sum(vertical_lines) / (vertical_lines.shape[0] * vertical_lines.shape[1])
            
            return {
                "horizontal_alignment": horizontal_score,
                "vertical_alignment": vertical_score,
                "overall_alignment": (horizontal_score + vertical_score) / 2
            }
            
        except Exception as e:
            return {"error": str(e)}
