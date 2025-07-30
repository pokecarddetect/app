import os
import logging
import numpy as np
from PIL import Image

# Delay TensorFlow imports to avoid startup errors
tf = None
MobileNetV2 = None
preprocess_input = None
Dense = None
GlobalAveragePooling2D = None
Model = None

class PokemonCardClassifier:
    """
    AI model for classifying Pokémon cards as Original or Fake.
    Uses MobileNetV2 as base model with transfer learning for efficient classification.
    """
    
    def __init__(self, model_type='auto'):
        """
        Initialize the classifier with selectable model architecture.
        
        Args:
            model_type: Model architecture ('auto', 'efficientnet', 'resnet50', 'mobilenet')
        """
        self.model = None
        self.model_type = model_type
        self.input_shape = (224, 224, 3)
        self.class_names = ['Fake', 'Original', 'Proxy', 'Custom Art', 'Altered', 'Test Print']  # Multiple categories
        self.tensorflow_loaded = False
        self._build_model(model_type)
    
    def _import_tensorflow(self):
        """Import TensorFlow components safely"""
        global tf, MobileNetV2, preprocess_input, Dense, GlobalAveragePooling2D, Model
        
        try:
            if not self.tensorflow_loaded:
                import tensorflow as tf_module
                from tensorflow.keras.applications import MobileNetV2 as MobileNetV2_module
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_module
                from tensorflow.keras.layers import Dense as Dense_module, GlobalAveragePooling2D as GlobalAveragePooling2D_module
                from tensorflow.keras.models import Model as Model_module
                
                tf = tf_module
                MobileNetV2 = MobileNetV2_module
                preprocess_input = preprocess_input_module
                Dense = Dense_module
                GlobalAveragePooling2D = GlobalAveragePooling2D_module
                Model = Model_module
                
                self.tensorflow_loaded = True
                logging.info("TensorFlow loaded successfully")
        except Exception as e:
            logging.error(f"Failed to import TensorFlow: {str(e)}")
            self.tensorflow_loaded = False
    
    def _tensorflow_available(self):
        """Check if TensorFlow is available"""
        return self.tensorflow_loaded and tf is not None
        
    def _build_model(self, model_type='auto'):
        """
        Build the classification model with fallback to lightweight alternatives.
        
        For deployment optimization, this uses a simplified model without TensorFlow dependency.
        """
        import os
        try:
            # Skip TensorFlow for cloud deployment (Railway, Heroku, etc.)
            if (os.environ.get('DEPLOYMENT_MODE') == 'true' or 
                os.environ.get('PORT') or 
                os.environ.get('RAILWAY_ENVIRONMENT') or
                os.environ.get('HEROKU_APP_NAME')):
                logging.info("Cloud deployment detected: Using enhanced fallback model")
                self._create_fallback_model()
                return
                
            # Skip TensorFlow for deployment optimization
            if os.environ.get('DEPLOYMENT_MODE') == 'true':
                logging.info("Deployment mode: Using lightweight fallback model")
                self._create_lightweight_fallback_model()
                return
                
            # Import TensorFlow components only if not in deployment mode
            self._import_tensorflow()
            
            if not self._tensorflow_available():
                logging.warning("TensorFlow not available, using fallback model")
                self._create_fallback_model()
                return
            
            # Try models in order of preference: EfficientNet -> ResNet50 -> MobileNetV2
            model_attempts = []
            if model_type == 'auto':
                model_attempts = ['efficientnet', 'resnet50', 'mobilenet']
            elif model_type in ['efficientnet', 'resnet50', 'mobilenet']:
                model_attempts = [model_type]
            else:
                model_attempts = ['mobilenet']  # Default fallback
            
            for attempt_type in model_attempts:
                try:
                    logging.info(f"Building {attempt_type.upper()} classification model...")
                    
                    # Select and load base model
                    if attempt_type == 'efficientnet':
                        try:
                            base_model = tf.keras.applications.EfficientNetB0(
                                weights='imagenet',
                                include_top=False,
                                input_shape=self.input_shape
                            )
                            model_name = "EfficientNetB0"
                            dense_units = 256
                            dropout_rate = 0.3
                            learning_rate = 0.00005
                        except Exception as e:
                            logging.warning(f"EfficientNet not available: {e}")
                            continue
                            
                    elif attempt_type == 'resnet50':
                        base_model = tf.keras.applications.ResNet50(
                            weights='imagenet',
                            include_top=False,
                            input_shape=self.input_shape
                        )
                        model_name = "ResNet50"
                        dense_units = 512
                        dropout_rate = 0.4
                        learning_rate = 0.0001
                        
                    else:  # mobilenet
                        base_model = MobileNetV2(
                            weights='imagenet',
                            include_top=False,
                            input_shape=self.input_shape
                        )
                        model_name = "MobileNetV2"
                        dense_units = 128
                        dropout_rate = 0.2
                        learning_rate = 0.0001
                    
                    # Freeze base model layers
                    base_model.trainable = False
                    
                    # Build optimized classification head
                    x = base_model.output
                    x = tf.keras.layers.GlobalAveragePooling2D()(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.Dropout(dropout_rate)(x)
                    x = tf.keras.layers.Dense(dense_units, activation='relu', name='feature_dense')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.Dropout(dropout_rate/2)(x)
                    
                    # Add secondary feature layer for complex models
                    if attempt_type in ['efficientnet', 'resnet50']:
                        x = tf.keras.layers.Dense(dense_units//4, activation='relu', name='feature_dense2')(x)
                        x = tf.keras.layers.Dropout(dropout_rate/4)(x)
                    
                    predictions = tf.keras.layers.Dense(1, activation='sigmoid', name='classification')(x)
                    
                    # Create the complete model
                    self.model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
                    
                    # Compile with optimized settings
                    self.model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss='binary_crossentropy',
                        metrics=['accuracy', 'precision', 'recall']
                    )
                    
                    self.model_type = attempt_type
                    logging.info(f"{model_name} model built successfully")
                    return  # Success, exit loop
                    
                except Exception as e:
                    logging.warning(f"Failed to build {attempt_type} model: {e}")
                    continue
            
            # If all models failed, use fallback
            logging.error("All model types failed, using fallback")
            self._create_fallback_model()
            
        except Exception as e:
            logging.error(f"Error building AI model: {str(e)}")
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """
        Create an enhanced fallback model using image processing techniques.
        This ensures the application continues to work even without TensorFlow.
        """
        logging.warning("Creating enhanced fallback model due to TensorFlow loading error")
        
        class EnhancedFallbackModel:
            def __init__(self):
                self.features_analyzed = 0
                
            def predict(self, x):
                """Enhanced heuristic-based prediction using image analysis"""
                try:
                    import cv2
                    
                    if hasattr(x, 'shape') and len(x.shape) == 4:
                        batch_size = x.shape[0]
                        predictions = []
                        
                        for i in range(batch_size):
                            img = x[i]
                            
                            # Convert to appropriate format for OpenCV
                            if img.max() <= 1.0:  # Normalized image
                                img = (img * 255).astype(np.uint8)
                            
                            # Convert to grayscale for analysis
                            if len(img.shape) == 3:
                                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                            else:
                                gray = img
                            
                            score = 0.0
                            
                            # 1. Sharpness analysis (authentic cards are usually sharp)
                            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                            if laplacian_var > 100:  # Good sharpness
                                score += 0.25
                            elif laplacian_var > 50:
                                score += 0.15
                            
                            # 2. Edge detection (authentic cards have clean edges)
                            edges = cv2.Canny(gray, 50, 150)
                            edge_density = np.mean(edges)
                            if 10 < edge_density < 50:  # Good edge density
                                score += 0.2
                            
                            # 3. Color distribution (authentic cards have balanced colors)
                            if len(img.shape) == 3:
                                color_std = np.std([np.std(img[:,:,c]) for c in range(3)])
                                if 15 < color_std < 40:  # Balanced color variation
                                    score += 0.2
                            
                            # 4. Brightness and contrast analysis
                            mean_brightness = np.mean(gray)
                            contrast = np.std(gray)
                            if 80 < mean_brightness < 180 and contrast > 30:
                                score += 0.2
                            
                            # 5. Texture analysis using Local Binary Pattern concept
                            try:
                                # Simple texture measure
                                texture_score = np.std(gray)
                                if texture_score > 25:  # Good texture variation
                                    score += 0.15
                            except:
                                pass
                            
                            # Convert score to prediction (lower = more likely original)
                            prediction = max(0.1, min(0.9, 1.0 - score))
                            predictions.append([prediction])
                        
                        self.features_analyzed += batch_size
                        return np.array(predictions)
                    else:
                        return np.array([[0.5]])
                        
                except Exception as e:
                    logging.error(f"Fallback prediction error: {e}")
                    # Simple fallback within fallback
                    if hasattr(x, 'shape') and len(x.shape) >= 3:
                        mean_val = float(np.mean(x))
                        prediction = 0.3 if 0.3 <= mean_val <= 0.7 else 0.7
                        return np.array([[prediction]])
                    return np.array([[0.5]])
        
        self.model = EnhancedFallbackModel()
        self.model_type = 'enhanced_fallback'
        logging.info("Enhanced fallback model created successfully")
    
    def _get_model_info(self):
        """Get information about the current model architecture"""
        model_info = {
            'efficientnet': 'EfficientNetB0 - Modern compound scaling architecture',
            'resnet50': 'ResNet50 - Deep residual network with skip connections',
            'mobilenet': 'MobileNetV2 - Lightweight mobile-optimized architecture',
            'enhanced_fallback': 'Enhanced Computer Vision Fallback - OpenCV-based analysis'
        }
        return model_info.get(getattr(self, 'model_type', 'unknown'), 'Unknown model architecture')
    
    def classify_card(self, image_array):
        """
        Classify a preprocessed card image as Original or Fake.
        
        Args:
            image_array: Preprocessed image array from image_utils
            
        Returns:
            tuple: (prediction_label, confidence_percentage, feature_analysis)
        """
        try:
            if self.model is None:
                return "Error", 0.0, {"error": "Model not loaded"}
            
            # Handle fallback models
            if isinstance(self.model, str) or hasattr(self.model, 'features_analyzed'):
                return self._fallback_classify(image_array)
            
            # Ensure image is in correct format for TensorFlow model
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, axis=0)
            
            # Make prediction using TensorFlow model
            prediction = self.model.predict(image_array, verbose=0)
            confidence = float(prediction[0][0]) * 100
            
            # Interpret results
            if confidence > 50:
                prediction_label = "Original"
                confidence_score = confidence
            else:
                prediction_label = "Fake"
                confidence_score = 100 - confidence
            
            # Extract feature analysis for visualization
            feature_analysis = self._analyze_features(image_array)
            
            # Add model information to features
            feature_analysis.update({
                'model_type': getattr(self, 'model_type', 'unknown'),
                'architecture': self._get_model_info(),
                'model_prediction': float(prediction[0][0]),
                'classification_threshold': 0.5
            })
            
            logging.info(f"AI Classification ({self._get_model_info()}): {prediction_label} ({confidence_score:.2f}% confidence)")
            
            return prediction_label, confidence_score, feature_analysis
            
        except Exception as e:
            logging.error(f"Error in AI classification: {str(e)}")
            return self._fallback_classify(image_array)
    
    def _fallback_classify(self, image_array):
        """
        Enhanced fallback classification using comprehensive feature analysis.
        Better handles Japanese cards and various card types.
        
        Args:
            image_array: Preprocessed image array
            
        Returns:
            tuple: (prediction_label, confidence_percentage, feature_analysis)
        """
        try:
            # Comprehensive feature analysis
            feature_analysis = self._analyze_features(image_array)
            
            # Initialize authenticity score
            authenticity_score = 50.0  # Neutral starting point
            
            # Analyze multiple quality indicators
            quality_indicators = self._extract_quality_indicators(image_array)
            
            # Print quality assessment (30% weight)
            print_quality = quality_indicators.get('print_quality', 0.5)
            authenticity_score += (print_quality - 0.5) * 30
            
            # Edge sharpness (25% weight) - real cards have crisp edges
            edge_sharpness = quality_indicators.get('edge_sharpness', 0.5)
            authenticity_score += (edge_sharpness - 0.5) * 25
            
            # Color consistency (20% weight) - real cards have consistent colors
            color_consistency = quality_indicators.get('color_consistency', 0.5)
            authenticity_score += (color_consistency - 0.5) * 20
            
            # Text clarity (15% weight) - important for OCR correlation
            text_clarity = quality_indicators.get('text_clarity', 0.5)
            authenticity_score += (text_clarity - 0.5) * 15
            
            # Overall composition (10% weight) - proper centering, proportions
            composition_score = quality_indicators.get('composition', 0.5)
            authenticity_score += (composition_score - 0.5) * 10
            
            # Apply regional adjustments for Japanese cards
            if self._detect_japanese_text_regions(image_array):
                # Japanese cards often have different characteristics
                authenticity_score += 5  # Slight boost for detected Japanese text
                feature_analysis['region_detected'] = 'Japanese'
            
            # Normalize score to 0-100 range
            authenticity_score = max(5.0, min(95.0, authenticity_score))
            
            # Multi-category classification based on features
            prediction_label, confidence_score = self._classify_card_category(image_array, authenticity_score, quality_indicators)
            
            # Cap confidence for fallback model (more conservative)
            confidence_score = min(confidence_score, 85.0)
            
            logging.info(f"Fallback Classification: {prediction_label} ({confidence_score:.2f}% confidence)")
            
            # Enhanced feature analysis with quality indicators
            feature_analysis.update({
                'quality_indicators': quality_indicators,
                'authenticity_factors': {
                    'print_quality': print_quality,
                    'edge_sharpness': edge_sharpness,
                    'color_consistency': color_consistency,
                    'text_clarity': text_clarity,
                    'composition': composition_score
                },
                'model_type': 'enhanced_fallback'
            })
            if feature_analysis:
                feature_analysis['model_type'] = 'fallback'
            else:
                feature_analysis = {'model_type': 'fallback'}
            
            return prediction_label, confidence_score, feature_analysis
            
        except Exception as e:
            logging.error(f"Error in fallback classification: {str(e)}")
            return "Original", 50.0, {"error": "Classification failed", "model_type": "fallback"}
    
    def _classify_card_category(self, image_array, authenticity_score, quality_indicators):
        """
        Classify card into specific category based on features and quality analysis.
        
        Categories:
        - Original: Authentic official card
        - Fake: Poor quality counterfeit
        - Proxy: High quality reproduction for gameplay
        - Custom Art: Fan-made with custom artwork
        - Altered: Modified official card (altered art, foiling, etc.)
        - Test Print: Prototype or test print
        """
        import cv2
        
        # Convert to uint8 for OpenCV operations
        img_uint8 = (image_array * 255).astype(np.uint8)
        
        # Feature analysis for categorization
        color_variance = self._analyze_color_variance(img_uint8)
        edge_quality = quality_indicators.get('edge_sharpness', 0.5)
        print_quality = quality_indicators.get('print_quality', 0.5)
        text_clarity = quality_indicators.get('text_clarity', 0.5)
        
        # Art style analysis (detect custom/altered artwork)
        art_style_score = self._analyze_art_style(img_uint8)
        
        # Initialize category scores
        category_scores = {
            'Original': authenticity_score,
            'Fake': 100 - authenticity_score,
            'Proxy': 0,
            'Custom Art': 0,
            'Altered': 0,
            'Test Print': 0
        }
        
        # Proxy detection (high quality but non-official)
        if print_quality > 0.7 and edge_quality > 0.6 and text_clarity > 0.6:
            if authenticity_score < 70:  # Good quality but not quite original
                category_scores['Proxy'] = 60 + (print_quality * 20)
        
        # Custom Art detection (unusual colors/art style)
        if art_style_score > 0.7 and color_variance > 0.6:
            category_scores['Custom Art'] = 50 + (art_style_score * 30)
        
        # Altered card detection (mixed quality indicators)
        if edge_quality > 0.5 and print_quality < 0.6:
            category_scores['Altered'] = 40 + ((edge_quality - print_quality) * 30)
        
        # Test Print detection (unusual printing characteristics)
        if print_quality < 0.4 and edge_quality > 0.7:
            category_scores['Test Print'] = 45 + (edge_quality * 25)
        
        # Find highest scoring category
        best_category = max(category_scores, key=category_scores.get)
        confidence = category_scores[best_category]
        
        # Apply minimum confidence thresholds
        if confidence < 30:
            # Default to simple binary classification for low confidence
            if authenticity_score > 50:
                return "Original", min(authenticity_score, 75.0)
            else:
                return "Fake", min(100 - authenticity_score, 75.0)
        
        # Cap confidence for multi-category classification
        confidence = min(confidence, 85.0)
        
        logging.info(f"Multi-category Classification: {best_category} ({confidence:.2f}% confidence)")
        
        return best_category, confidence
    
    def _analyze_color_variance(self, img_uint8):
        """Analyze color distribution variance to detect custom artwork"""
        import cv2
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        
        # Calculate color histogram
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
        
        # Calculate variance in color distribution
        h_variance = np.var(hist_h)
        s_variance = np.var(hist_s)
        
        # Combine variances (higher = more varied colors)
        color_variance = (h_variance + s_variance) / 2
        
        return min(color_variance * 10, 1.0)  # Normalize to 0-1
    
    def _analyze_art_style(self, img_uint8):
        """Analyze artwork style to detect custom art"""
        import cv2
        
        # Edge detection for art style analysis
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density in artwork region (middle of card)
        h, w = edges.shape
        art_region = edges[int(h*0.2):int(h*0.7), int(w*0.1):int(w*0.9)]
        edge_density = np.sum(art_region > 0) / art_region.size
        
        # Texture analysis using standard deviation
        texture_std = np.std(gray[int(h*0.2):int(h*0.7), int(w*0.1):int(w*0.9)])
        texture_score = min(texture_std / 50.0, 1.0)
        
        # Combine metrics (higher = more artistic/custom)
        art_style_score = (edge_density * 0.6 + texture_score * 0.4)
        
        return min(art_style_score, 1.0)
    
    def _extract_quality_indicators(self, image_array):
        """Extract comprehensive quality indicators from image"""
        import cv2
        
        # Convert to uint8 for OpenCV operations
        img_uint8 = (image_array * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        indicators = {}
        
        # Print quality - measure edge strength and sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        indicators['print_quality'] = min(1.0, laplacian_var / 1000.0)
        
        # Edge sharpness - use Sobel gradients
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        indicators['edge_sharpness'] = min(1.0, np.mean(edge_magnitude) / 100.0)
        
        # Color consistency - measure color distribution uniformity
        color_std = np.std(image_array, axis=(0, 1))
        indicators['color_consistency'] = 1.0 - min(1.0, np.mean(color_std) * 3)
        
        # Text clarity - measure high-frequency content in text regions
        edges = cv2.Canny(gray, 50, 150)
        text_score = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        indicators['text_clarity'] = min(1.0, text_score * 10)
        
        # Composition - measure center bias and symmetry
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        composition_score = np.mean(center_region) / (np.mean(gray) + 1e-6)
        indicators['composition'] = min(1.0, max(0.0, (composition_score - 0.5) * 2))
        
        return indicators
    
    def _detect_japanese_text_regions(self, image_array):
        """Detect if image contains Japanese text characteristics"""
        import cv2
        
        try:
            # Convert to grayscale for text analysis
            img_uint8 = (image_array * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            
            # Look for complex character patterns typical in Japanese text
            # Japanese characters tend to have more intricate shapes than Latin text
            kernel = np.ones((3,3), np.uint8)
            morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Find contours that might be text
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            complex_shapes = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 5000:  # Size range for text elements
                    # Check complexity by approximating contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Japanese characters tend to have more complex shapes
                    if len(approx) > 8:  # Complex shape indicator
                        complex_shapes += 1
            
            # If we find multiple complex shapes, likely Japanese text
            return complex_shapes >= 3
            
        except Exception as e:
            logging.warning(f"Japanese text detection failed: {e}")
            return False
    
    def _analyze_features(self, image_array):
        """
        Analyze key features that influence the classification decision.
        This provides insight into what the AI is detecting.
        
        Args:
            image_array: Input image array
            
        Returns:
            dict: Feature analysis results
        """
        try:
            # Basic image analysis
            features = {
                "image_quality": self._assess_image_quality(image_array),
                "color_distribution": self._analyze_colors(image_array),
                "edge_sharpness": self._measure_edge_sharpness(image_array),
                "text_regions": self._detect_text_regions(image_array)
            }
            
            return features
            
        except Exception as e:
            logging.error(f"Error in feature analysis: {str(e)}")
            return {"error": "Feature analysis failed"}
    
    def _assess_image_quality(self, image_array):
        """Assess overall image quality metrics"""
        try:
            # Calculate basic quality metrics
            mean_brightness = np.mean(image_array)
            std_brightness = np.std(image_array)
            
            # Quality score based on brightness distribution
            if 50 <= mean_brightness <= 200 and std_brightness > 30:
                quality = "Good"
            elif std_brightness < 20:
                quality = "Low contrast"
            else:
                quality = "Poor lighting"
                
            return {
                "score": quality,
                "brightness": float(mean_brightness),
                "contrast": float(std_brightness)
            }
        except:
            return {"score": "Unknown", "brightness": 0, "contrast": 0}
    
    def _analyze_colors(self, image_array):
        """Analyze color distribution in the image"""
        try:
            # Convert to 0-255 range if needed
            if image_array.max() <= 1.0:
                image_array = image_array * 255
            
            # Calculate color statistics
            rgb_means = np.mean(image_array[0], axis=(0, 1))
            
            return {
                "red_intensity": float(rgb_means[0]),
                "green_intensity": float(rgb_means[1]),
                "blue_intensity": float(rgb_means[2]),
                "color_balance": "Balanced" if np.std(rgb_means) < 30 else "Imbalanced"
            }
        except:
            return {"red_intensity": 0, "green_intensity": 0, "blue_intensity": 0, "color_balance": "Unknown"}
    
    def _measure_edge_sharpness(self, image_array):
        """Measure edge sharpness to detect print quality"""
        try:
            # Convert to grayscale for edge detection
            gray = np.mean(image_array[0], axis=2)
            
            # Simple edge detection using gradients
            grad_x = np.abs(np.gradient(gray, axis=1))
            grad_y = np.abs(np.gradient(gray, axis=0))
            edge_strength = np.mean(grad_x + grad_y)
            
            if edge_strength > 10:
                sharpness = "Sharp"
            elif edge_strength > 5:
                sharpness = "Moderate"
            else:
                sharpness = "Blurry"
                
            return {
                "score": sharpness,
                "edge_strength": float(edge_strength)
            }
        except:
            return {"score": "Unknown", "edge_strength": 0}
    
    def _detect_text_regions(self, image_array):
        """Detect potential text regions for OCR analysis"""
        try:
            # Simple text region detection based on horizontal patterns
            gray = np.mean(image_array[0], axis=2)
            
            # Look for horizontal text-like patterns
            horizontal_variance = np.var(gray, axis=1)
            text_regions = np.sum(horizontal_variance > np.mean(horizontal_variance))
            
            return {
                "detected_regions": int(text_regions),
                "text_likelihood": "High" if text_regions > 50 else "Low"
            }
        except:
            return {"detected_regions": 0, "text_likelihood": "Unknown"}
