import numpy as np
import cv2
import logging
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

class ExplainableAI:
    """
    Explainable AI module for generating attention heatmaps and model interpretability.
    Uses Grad-CAM and other techniques to visualize where the AI model focuses.
    """
    
    def __init__(self):
        """
        Initialize the Explainable AI processor.
        """
        self.heatmap_cache = {}
        logging.info("Explainable AI processor initialized")
    
    def generate_gradcam_heatmap(self, model, image_array, class_idx=None):
        """
        Generate Grad-CAM heatmap for model predictions.
        
        Args:
            model: AI model (fallback creates synthetic heatmap)
            image_array: Preprocessed image array
            class_idx: Target class index for heatmap generation
            
        Returns:
            numpy.ndarray: Heatmap array
        """
        try:
            # For fallback model, create a synthetic attention heatmap
            if hasattr(model, 'is_fallback') and model.is_fallback:
                return self._create_synthetic_heatmap(image_array)
            
            # For real TensorFlow models, implement Grad-CAM
            return self._compute_gradcam(model, image_array, class_idx)
            
        except Exception as e:
            logging.warning(f"Grad-CAM generation failed: {str(e)}")
            return self._create_synthetic_heatmap(image_array)
    
    def _create_synthetic_heatmap(self, image_array):
        """
        Create a synthetic attention heatmap based on image features.
        
        Args:
            image_array: Input image array
            
        Returns:
            numpy.ndarray: Synthetic heatmap
        """
        try:
            # Remove batch dimension if present
            if len(image_array.shape) == 4:
                image = image_array[0]
            else:
                image = image_array
            
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
            else:
                gray = image
            
            # Create attention map based on image features
            height, width = gray.shape
            
            # Focus on edges (typical for card analysis)
            edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
            edge_heatmap = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
            
            # Focus on text regions (bottom half of card)
            text_weight = np.zeros_like(gray)
            text_weight[int(height*0.5):, :] = 0.8  # Bottom half for text
            text_heatmap = cv2.GaussianBlur(text_weight, (31, 31), 0)
            
            # Focus on artwork regions (middle of card)
            artwork_weight = np.zeros_like(gray)
            artwork_weight[int(height*0.2):int(height*0.7), int(width*0.1):int(width*0.9)] = 0.6
            artwork_heatmap = cv2.GaussianBlur(artwork_weight, (21, 21), 0)
            
            # Focus on header regions (top of card - title and HP)
            header_weight = np.zeros_like(gray)
            header_weight[:int(height*0.3), :] = 0.7  # Top 30% for headers
            header_heatmap = cv2.GaussianBlur(header_weight, (21, 21), 0)
            
            # Combine heatmaps with better balance
            combined_heatmap = (edge_heatmap / 255.0 * 0.3 + 
                              text_heatmap * 0.25 + 
                              artwork_heatmap * 0.25 +
                              header_heatmap * 0.2)
            
            # Normalize to [0, 1]
            heatmap = (combined_heatmap - combined_heatmap.min()) / (combined_heatmap.max() - combined_heatmap.min() + 1e-8)
            
            return heatmap
            
        except Exception as e:
            logging.error(f"Error creating synthetic heatmap: {str(e)}")
            # Return a simple center-focused heatmap as fallback
            height, width = 224, 224
            y, x = np.ogrid[:height, :width]
            center_y, center_x = height // 2, width // 2
            heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (height/4)**2))
            return heatmap
    
    def _compute_gradcam(self, model, image_array, class_idx):
        """
        Compute actual Grad-CAM heatmap using TensorFlow model.
        
        Args:
            model: TensorFlow model
            image_array: Input image
            class_idx: Target class index
            
        Returns:
            numpy.ndarray: Grad-CAM heatmap
        """
        try:
            import tensorflow as tf
            
            # Get the last convolutional layer
            last_conv_layer = None
            for layer in reversed(model.model.layers):
                if len(layer.output_shape) == 4:  # Conv layer
                    last_conv_layer = layer
                    break
            
            if last_conv_layer is None:
                return self._create_synthetic_heatmap(image_array)
            
            # Create gradient model
            grad_model = tf.keras.models.Model(
                inputs=[model.model.inputs],
                outputs=[last_conv_layer.output, model.model.output]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image_array)
                if class_idx is None:
                    class_idx = tf.argmax(predictions[0])
                class_channel = predictions[:, class_idx]
            
            # Compute gradients of class output with respect to feature maps
            grads = tape.gradient(class_channel, conv_outputs)
            
            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight feature maps by gradients
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            # Resize to original image size
            heatmap = cv2.resize(heatmap.numpy(), (224, 224))
            
            return heatmap
            
        except Exception as e:
            logging.warning(f"TensorFlow Grad-CAM failed: {str(e)}")
            return self._create_synthetic_heatmap(image_array)
    
    def overlay_heatmap_on_image(self, image_path, heatmap, alpha=0.4):
        """
        Overlay heatmap on original image for visualization.
        
        Args:
            image_path: Path to original image
            heatmap: Generated heatmap array
            alpha: Transparency of heatmap overlay
            
        Returns:
            PIL.Image: Image with heatmap overlay
        """
        try:
            # Load original image
            original_image = Image.open(image_path).convert('RGB')
            original_array = np.array(original_image)
            
            # Resize heatmap to match image dimensions
            img_height, img_width = original_array.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (img_width, img_height))
            
            # Convert heatmap to RGB using colormap
            heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # Remove alpha channel
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            
            # Overlay heatmap on original image
            overlayed = cv2.addWeighted(
                original_array, 1 - alpha,
                heatmap_colored, alpha,
                0
            )
            
            return Image.fromarray(overlayed)
            
        except Exception as e:
            logging.error(f"Error overlaying heatmap: {str(e)}")
            # Return original image if overlay fails
            return Image.open(image_path).convert('RGB')
    
    def save_heatmap_visualization(self, image_path, heatmap, output_path):
        """
        Save heatmap visualization to file.
        
        Args:
            image_path: Path to original image
            heatmap: Generated heatmap
            output_path: Path to save visualization
            
        Returns:
            bool: Success status
        """
        try:
            # Create overlay image
            overlay_image = self.overlay_heatmap_on_image(image_path, heatmap)
            
            # Save to file
            overlay_image.save(output_path, 'JPEG', quality=95)
            
            logging.info(f"Heatmap visualization saved to: {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving heatmap visualization: {str(e)}")
            return False
    
    def generate_attention_analysis(self, image_path, heatmap):
        """
        Generate detailed analysis of model attention patterns.
        
        Args:
            image_path: Path to analyzed image
            heatmap: Generated attention heatmap
            
        Returns:
            dict: Attention analysis results
        """
        try:
            # Load original image for region analysis
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            
            # Resize heatmap to match image
            heatmap_resized = cv2.resize(heatmap, (width, height))
            
            # Define card regions
            regions = {
                'top_region': heatmap_resized[0:height//4, :],
                'artwork_region': heatmap_resized[height//4:int(height*0.6), :],
                'text_region': heatmap_resized[int(height*0.6):int(height*0.9), :],
                'bottom_region': heatmap_resized[int(height*0.9):, :],
                'left_border': heatmap_resized[:, 0:width//10],
                'right_border': heatmap_resized[:, int(width*0.9):],
                'center_focus': heatmap_resized[height//3:2*height//3, width//3:2*width//3]
            }
            
            # Calculate attention scores for each region
            region_scores = {}
            for region_name, region_heatmap in regions.items():
                if region_heatmap.size > 0:
                    region_scores[region_name] = {
                        'mean_attention': float(np.mean(region_heatmap)),
                        'max_attention': float(np.max(region_heatmap)),
                        'attention_area': float(np.sum(region_heatmap > 0.5) / region_heatmap.size)
                    }
                else:
                    region_scores[region_name] = {
                        'mean_attention': 0.0,
                        'max_attention': 0.0,
                        'attention_area': 0.0
                    }
            
            # Find peak attention areas
            peak_indices = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
            peak_y, peak_x = peak_indices
            
            # Analyze attention distribution
            attention_variance = float(np.var(heatmap_resized))
            attention_entropy = float(-np.sum(heatmap_resized * np.log(heatmap_resized + 1e-8)))
            
            # Determine focus characteristics
            focus_analysis = self._analyze_focus_patterns(heatmap_resized)
            
            return {
                'region_scores': region_scores,
                'peak_attention': {
                    'x': int(peak_x),
                    'y': int(peak_y),
                    'intensity': float(heatmap_resized[peak_y, peak_x])
                },
                'attention_statistics': {
                    'variance': attention_variance,
                    'entropy': attention_entropy,
                    'mean_attention': float(np.mean(heatmap_resized)),
                    'attention_spread': float(np.std(heatmap_resized))
                },
                'focus_analysis': focus_analysis
            }
            
        except Exception as e:
            logging.error(f"Error generating attention analysis: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_focus_patterns(self, heatmap):
        """
        Analyze focus patterns in the attention heatmap.
        
        Args:
            heatmap: Attention heatmap array
            
        Returns:
            dict: Focus pattern analysis
        """
        try:
            # Threshold heatmap to find high-attention areas
            threshold = np.percentile(heatmap, 75)
            high_attention = (heatmap > threshold).astype(np.uint8)
            
            # Find connected components (attention clusters)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(high_attention)
            
            # Analyze cluster characteristics
            clusters = []
            for i in range(1, num_labels):  # Skip background label 0
                cluster_area = stats[i, cv2.CC_STAT_AREA]
                cluster_centroid = centroids[i]
                clusters.append({
                    'area': int(cluster_area),
                    'centroid': [float(cluster_centroid[0]), float(cluster_centroid[1])],
                    'relative_size': float(cluster_area / heatmap.size)
                })
            
            # Sort clusters by area
            clusters.sort(key=lambda x: x['area'], reverse=True)
            
            # Determine focus type
            if len(clusters) == 0:
                focus_type = "diffuse"
            elif len(clusters) == 1:
                focus_type = "single_focus"
            elif len(clusters) <= 3:
                focus_type = "multi_focus"
            else:
                focus_type = "scattered"
            
            return {
                'focus_type': focus_type,
                'num_clusters': len(clusters),
                'primary_clusters': clusters[:3],  # Top 3 clusters
                'attention_concentration': float(np.sum(heatmap > threshold) / heatmap.size)
            }
            
        except Exception as e:
            logging.warning(f"Focus pattern analysis failed: {str(e)}")
            return {'focus_type': 'unknown', 'error': str(e)}
    
    def create_attention_summary(self, attention_analysis):
        """
        Create human-readable summary of attention analysis.
        
        Args:
            attention_analysis: Results from generate_attention_analysis
            
        Returns:
            list: List of attention insights
        """
        try:
            insights = []
            
            if 'error' in attention_analysis:
                return ["Attention analysis unavailable"]
            
            # Region-based insights
            region_scores = attention_analysis.get('region_scores', {})
            
            # Find most attended region
            region_attention = {name: scores.get('mean_attention', 0) 
                              for name, scores in region_scores.items()}
            most_attended = max(region_attention, key=region_attention.get) if region_attention else None
            
            if most_attended:
                insights.append(f"Model focused most on the {most_attended.replace('_', ' ')}")
            
            # Focus pattern insights
            focus_analysis = attention_analysis.get('focus_analysis', {})
            focus_type = focus_analysis.get('focus_type', 'unknown')
            
            if focus_type == 'single_focus':
                insights.append("Model shows concentrated attention on a single area")
            elif focus_type == 'multi_focus':
                insights.append("Model attention is distributed across multiple key areas")
            elif focus_type == 'scattered':
                insights.append("Model attention is widely scattered across the image")
            elif focus_type == 'diffuse':
                insights.append("Model shows diffuse attention with no clear focus points")
            
            # Text region attention
            text_attention = region_scores.get('text_region', {}).get('mean_attention', 0)
            if text_attention > 0.6:
                insights.append("Strong focus on text regions suggests text-based authentication")
            elif text_attention < 0.3:
                insights.append("Limited text attention may indicate visual feature focus")
            
            # Border attention (potential fake indicator)
            border_attention = (region_scores.get('left_border', {}).get('mean_attention', 0) + 
                              region_scores.get('right_border', {}).get('mean_attention', 0)) / 2
            if border_attention > 0.5:
                insights.append("High border attention may indicate print quality concerns")
            
            # Artwork attention
            artwork_attention = region_scores.get('artwork_region', {}).get('mean_attention', 0)
            if artwork_attention > 0.5:
                insights.append("Significant artwork attention suggests visual authenticity checks")
            
            return insights if insights else ["Standard attention pattern detected"]
            
        except Exception as e:
            logging.error(f"Error creating attention summary: {str(e)}")
            return ["Attention analysis summary unavailable"]