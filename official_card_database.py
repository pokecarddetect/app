"""
Official Pokémon Card Database for Visual Comparison
Manages and compares cards against official reference database using PostgreSQL
"""

import os
import json
import logging
import cv2
import numpy as np
from PIL import Image
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from app import db
from models import OfficialCard, PokemonSet

class OfficialCardDatabase:
    def __init__(self):
        # Initialize with PostgreSQL database instead of local files
        self.cards_index = self._load_cards_from_database()
        self.feature_cache = {}  # In-memory cache for features
        
        logging.info("Official Card Database initialized")
    
    def _load_cards_from_database(self) -> Dict:
        """Load cards index from PostgreSQL database"""
        cards_index = {}
        
        try:
            # Query cards from database
            cards = OfficialCard.query.all()
            
            for card in cards:
                # Use remote image URLs from API instead of local files
                image_url = card.image_url_large or card.image_url_small
                
                if image_url:
                    cards_index[card.id] = {
                        "name": card.name,
                        "set": card.set_name or 'Unknown Set',
                        "set_id": card.set_id or '',
                        "number": card.number or '',
                        "rarity": card.rarity or 'Unknown',
                        "types": json.loads(card.types) if card.types else [],
                        "image_url": image_url,
                        "features": None,  # Will be computed when needed
                        "added_date": datetime.now().strftime("%Y-%m-%d"),
                        "api_data": True
                    }
            
            logging.info(f"Loaded {len(cards_index)} cards from database")
                    
        except Exception as e:
            logging.warning(f"Failed to load cards from database: {e}")
            logging.info("No cards found in database, using fallback sample data")
        
        return cards_index
    
    def extract_card_features(self, image_input) -> Dict:
        """Extract comprehensive features from a card image (path or URL)"""
        try:
            # Load image from path or URL
            if isinstance(image_input, str) and image_input.startswith('http'):
                # Load from URL
                response = requests.get(image_input)
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            else:
                # Load from local path
                image = cv2.imread(image_input)
            
            if image is None:
                return {}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            features = {}
            
            # 1. ORB Features for keypoint matching
            orb = cv2.ORB_create(nfeatures=500)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            if descriptors is not None:
                features['orb_descriptors'] = descriptors.tolist()
                features['orb_keypoints'] = [(kp.pt[0], kp.pt[1], kp.angle, kp.size) for kp in keypoints]
            
            # 2. Color histogram
            hist_b = cv2.calcHist([image], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [32], [0, 256])
            
            features['color_histogram'] = {
                'blue': hist_b.flatten().tolist(),
                'green': hist_g.flatten().tolist(),
                'red': hist_r.flatten().tolist()
            }
            
            # 3. Edge features
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))
            
            # 4. Texture features using LBP
            features['texture_variance'] = float(np.var(gray))
            features['texture_mean'] = float(np.mean(gray))
            
            # 5. Card dimensions and aspect ratio
            h, w = image.shape[:2]
            features['dimensions'] = {
                'width': w,
                'height': h,
                'aspect_ratio': float(w / h)
            }
            
            # 6. Border detection
            border_features = self._extract_border_features(gray)
            features['border'] = border_features
            
            # 7. Template regions (corners, center, etc.)
            region_features = self._extract_region_features(image)
            features['regions'] = region_features
            
            return features
            
        except Exception as e:
            logging.error(f"Feature extraction failed: {e}")
            return {}
    
    def _extract_border_features(self, gray_image: np.ndarray) -> Dict:
        """Extract features specific to card borders"""
        h, w = gray_image.shape
        border_width = min(h, w) // 20  # 5% border
        
        # Extract border regions
        top_border = gray_image[:border_width, :]
        bottom_border = gray_image[-border_width:, :]
        left_border = gray_image[:, :border_width]
        right_border = gray_image[:, -border_width:]
        
        return {
            'top_uniformity': float(np.std(top_border)),
            'bottom_uniformity': float(np.std(bottom_border)),
            'left_uniformity': float(np.std(left_border)),
            'right_uniformity': float(np.std(right_border)),
            'corner_sharpness': self._measure_corner_sharpness(gray_image)
        }
    
    def _measure_corner_sharpness(self, gray_image: np.ndarray) -> float:
        """Measure the sharpness of card corners"""
        h, w = gray_image.shape
        corner_size = min(h, w) // 10
        
        # Extract corners
        corners = [
            gray_image[:corner_size, :corner_size],  # Top-left
            gray_image[:corner_size, -corner_size:],  # Top-right
            gray_image[-corner_size:, :corner_size],  # Bottom-left
            gray_image[-corner_size:, -corner_size:]  # Bottom-right
        ]
        
        sharpness_scores = []
        for corner in corners:
            laplacian_var = cv2.Laplacian(corner, cv2.CV_64F).var()
            sharpness_scores.append(laplacian_var)
        
        return float(np.mean(sharpness_scores))
    
    def _extract_region_features(self, image: np.ndarray) -> Dict:
        """Extract features from specific card regions"""
        h, w = image.shape[:2]
        
        # Define regions
        center_region = image[h//4:3*h//4, w//4:3*w//4]
        top_region = image[:h//3, :]
        bottom_region = image[2*h//3:, :]
        
        regions = {
            'center_brightness': float(np.mean(center_region)),
            'center_contrast': float(np.std(center_region)),
            'top_region_color': [float(np.mean(top_region[:,:,i])) for i in range(3)],
            'bottom_region_color': [float(np.mean(bottom_region[:,:,i])) for i in range(3)]
        }
        
        return regions
    
    def compare_with_database(self, uploaded_image_path: str, top_k: int = 5) -> List[Dict]:
        """
        Compare uploaded image with official database
        Returns list of matches with similarity scores
        """
        try:
            # Extract features from uploaded image
            uploaded_features = self.extract_card_features(uploaded_image_path)
            if not uploaded_features:
                return []
            
            matches = []
            
            for card_id, card_info in self.cards_index.items():
                # Compute features if not cached
                if not card_info.get('features') and card_info.get('image_path'):
                    card_features = self.extract_card_features(card_info['image_path'])
                    if card_features:
                        card_info['features'] = card_features
                        self.feature_cache[card_id] = card_features
                        self._save_feature_cache()
                
                if card_info.get('features'):
                    similarity = self._calculate_similarity(uploaded_features, card_info['features'])
                    
                    match = {
                        'card_id': card_id,
                        'card_info': card_info,
                        'similarity_score': similarity,
                        'confidence': min(100.0, similarity * 100),
                        'match_type': self._determine_match_type(similarity)
                    }
                    matches.append(match)
            
            # Sort by similarity score
            matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return matches[:top_k]
            
        except Exception as e:
            logging.error(f"Database comparison failed: {e}")
            return []
    
    def _calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate comprehensive similarity between two feature sets"""
        similarities = []
        weights = []
        
        # 1. Color histogram comparison
        if 'color_histogram' in features1 and 'color_histogram' in features2:
            color_sim = self._compare_color_histograms(
                features1['color_histogram'], 
                features2['color_histogram']
            )
            similarities.append(color_sim)
            weights.append(0.3)  # 30% weight
        
        # 2. ORB feature matching
        if 'orb_descriptors' in features1 and 'orb_descriptors' in features2:
            orb_sim = self._compare_orb_features(
                features1['orb_descriptors'],
                features2['orb_descriptors']
            )
            similarities.append(orb_sim)
            weights.append(0.25)  # 25% weight
        
        # 3. Dimension and aspect ratio
        if 'dimensions' in features1 and 'dimensions' in features2:
            dim_sim = self._compare_dimensions(
                features1['dimensions'],
                features2['dimensions']
            )
            similarities.append(dim_sim)
            weights.append(0.15)  # 15% weight
        
        # 4. Border features
        if 'border' in features1 and 'border' in features2:
            border_sim = self._compare_border_features(
                features1['border'],
                features2['border']
            )
            similarities.append(border_sim)
            weights.append(0.2)  # 20% weight
        
        # 5. Region features
        if 'regions' in features1 and 'regions' in features2:
            region_sim = self._compare_region_features(
                features1['regions'],
                features2['regions']
            )
            similarities.append(region_sim)
            weights.append(0.1)  # 10% weight
        
        if not similarities:
            return 0.0
        
        # Weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            weighted_sim = sum(s * w for s, w in zip(similarities, weights)) / total_weight
        else:
            weighted_sim = sum(similarities) / len(similarities)
        
        return min(1.0, max(0.0, weighted_sim))
    
    def _compare_color_histograms(self, hist1: Dict, hist2: Dict) -> float:
        """Compare color histograms using correlation"""
        try:
            correlations = []
            for channel in ['red', 'green', 'blue']:
                if channel in hist1 and channel in hist2:
                    h1 = np.array(hist1[channel])
                    h2 = np.array(hist2[channel])
                    corr = cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CORREL)
                    correlations.append(max(0, corr))
            
            return np.mean(correlations) if correlations else 0.0
        except:
            return 0.0
    
    def _compare_orb_features(self, desc1: List, desc2: List) -> float:
        """Compare ORB descriptors using matcher"""
        try:
            if not desc1 or not desc2:
                return 0.0
            
            desc1_np = np.array(desc1, dtype=np.uint8)
            desc2_np = np.array(desc2, dtype=np.uint8)
            
            # Use BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(desc1_np, desc2_np)
            
            if len(matches) == 0:
                return 0.0
            
            # Calculate good matches ratio
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]  # Threshold for good matches
            
            match_ratio = len(good_matches) / max(len(desc1), len(desc2))
            return min(1.0, match_ratio * 2)  # Scale appropriately
            
        except:
            return 0.0
    
    def _compare_dimensions(self, dim1: Dict, dim2: Dict) -> float:
        """Compare card dimensions and aspect ratios"""
        try:
            # Compare aspect ratios (most important for cards)
            ar1 = dim1.get('aspect_ratio', 1.0)
            ar2 = dim2.get('aspect_ratio', 1.0)
            
            ar_diff = abs(ar1 - ar2)
            ar_similarity = max(0, 1.0 - ar_diff / 0.5)  # Allow 0.5 difference
            
            return ar_similarity
        except:
            return 0.0
    
    def _compare_border_features(self, border1: Dict, border2: Dict) -> float:
        """Compare border uniformity and corner sharpness"""
        try:
            similarities = []
            
            # Compare corner sharpness
            if 'corner_sharpness' in border1 and 'corner_sharpness' in border2:
                sharp1 = border1['corner_sharpness']
                sharp2 = border2['corner_sharpness']
                sharp_diff = abs(sharp1 - sharp2) / max(sharp1, sharp2, 1.0)
                similarities.append(max(0, 1.0 - sharp_diff))
            
            # Compare border uniformity
            for side in ['top_uniformity', 'bottom_uniformity', 'left_uniformity', 'right_uniformity']:
                if side in border1 and side in border2:
                    u1 = border1[side]
                    u2 = border2[side]
                    u_diff = abs(u1 - u2) / max(u1, u2, 1.0)
                    similarities.append(max(0, 1.0 - u_diff))
            
            return np.mean(similarities) if similarities else 0.0
        except:
            return 0.0
    
    def _compare_region_features(self, region1: Dict, region2: Dict) -> float:
        """Compare region-specific features"""
        try:
            similarities = []
            
            # Compare brightness and contrast
            for feature in ['center_brightness', 'center_contrast']:
                if feature in region1 and feature in region2:
                    v1 = region1[feature]
                    v2 = region2[feature]
                    diff = abs(v1 - v2) / max(v1, v2, 1.0)
                    similarities.append(max(0, 1.0 - diff))
            
            # Compare region colors
            for region in ['top_region_color', 'bottom_region_color']:
                if region in region1 and region in region2:
                    color1 = np.array(region1[region])
                    color2 = np.array(region2[region])
                    color_diff = np.linalg.norm(color1 - color2) / 255.0
                    similarities.append(max(0, 1.0 - color_diff))
            
            return np.mean(similarities) if similarities else 0.0
        except:
            return 0.0
    
    def _determine_match_type(self, similarity: float) -> str:
        """Determine the type of match based on similarity score"""
        if similarity >= 0.85:
            return "Exact Match"
        elif similarity >= 0.70:
            return "Strong Match"
        elif similarity >= 0.50:
            return "Moderate Match"
        elif similarity >= 0.30:
            return "Weak Match"
        else:
            return "No Match"
    
    def add_official_card(self, card_id: str, card_info: Dict, image_path: str = None) -> bool:
        """Add a new official card to the database"""
        try:
            # Extract features if image is provided
            features = None
            if image_path and os.path.exists(image_path):
                features = self.extract_card_features(image_path)
            
            # Add to index
            self.cards_index[card_id] = {
                **card_info,
                'features': features,
                'image_path': image_path,
                'added_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            # Save updated index
            self._save_cards_index()
            
            logging.info(f"Added official card: {card_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to add official card: {e}")
            return False
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the database"""
        total_cards = len(self.cards_index)
        cards_with_images = sum(1 for card in self.cards_index.values() if card.get('image_path'))
        cards_with_features = sum(1 for card in self.cards_index.values() if card.get('features'))
        
        return {
            'total_cards': total_cards,
            'cards_with_images': cards_with_images,
            'cards_with_features': cards_with_features,
            'coverage_percentage': (cards_with_features / total_cards * 100) if total_cards > 0 else 0
        }