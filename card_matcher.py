"""
Pokemon Card Matching and Comparison System
Matches uploaded cards against the official database for authenticity verification.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
from models import OfficialCard
from app import app, db

logger = logging.getLogger(__name__)

class CardMatcher:
    """
    Matches uploaded cards against official database and identifies discrepancies.
    """
    
    def __init__(self):
        """Initialize the card matcher."""
        self.similarity_threshold = 0.7  # Minimum similarity for a match
        
    def find_matching_cards(self, card_name: str, set_name: Optional[str] = None, 
                          card_number: Optional[str] = None) -> List[OfficialCard]:
        """
        Find official cards that match the given criteria.
        
        Args:
            card_name: Name of the Pokemon card
            set_name: Optional set name for more precise matching
            card_number: Optional card number within the set
            
        Returns:
            List of matching official cards
        """
        with app.app_context():
            query = OfficialCard.query
            
            # Filter by card name with fuzzy matching
            if card_name:
                # Clean card name for better matching
                clean_name = self._clean_card_name(card_name)
                query = query.filter(
                    OfficialCard.name.ilike(f'%{clean_name}%')
                )
            
            # Filter by set name if provided
            if set_name:
                clean_set = self._clean_set_name(set_name)
                query = query.filter(
                    OfficialCard.set_name.ilike(f'%{clean_set}%')
                )
            
            # Filter by card number if provided
            if card_number:
                query = query.filter(OfficialCard.number == card_number)
            
            matches = query.all()
            
            # If no exact matches, try broader search
            if not matches and card_name:
                # Split card name and search for individual words
                name_words = clean_name.split()
                for word in name_words:
                    if len(word) > 3:  # Only search for meaningful words
                        matches = OfficialCard.query.filter(
                            OfficialCard.name.ilike(f'%{word}%')
                        ).limit(5).all()
                        if matches:
                            break
            
            return matches
    
    def calculate_card_similarity(self, ocr_text: str, official_card: OfficialCard) -> float:
        """
        Calculate similarity between OCR extracted text and official card data.
        
        Args:
            ocr_text: Text extracted from the uploaded card
            official_card: Official card data from database
            
        Returns:
            Similarity score between 0 and 1
        """
        if not ocr_text or not official_card:
            return 0.0
        
        # Extract key information from official card
        official_text = self._extract_official_card_text(official_card)
        
        # Clean both texts for comparison
        ocr_clean = self._clean_text_for_comparison(ocr_text)
        official_clean = self._clean_text_for_comparison(official_text)
        
        # Calculate similarity using different methods
        name_similarity = self._calculate_name_similarity(ocr_clean, official_card.name)
        text_similarity = SequenceMatcher(None, ocr_clean, official_clean).ratio()
        
        # Weight different aspects
        overall_similarity = (name_similarity * 0.4) + (text_similarity * 0.6)
        
        return overall_similarity
    
    def analyze_card_authenticity(self, ocr_text: str, card_name: str = None, 
                                set_name: str = None) -> Dict:
        """
        Analyze card authenticity by comparing against official database.
        
        Args:
            ocr_text: Text extracted from uploaded card
            card_name: Detected card name (if any)
            set_name: Detected set name (if any)
            
        Returns:
            Dictionary with analysis results
        """
        analysis_result = {
            "official_match_found": False,
            "best_match": None,
            "similarity_score": 0.0,
            "discrepancies": [],
            "authenticity_indicators": [],
            "confidence_adjustment": 0.0
        }
        
        try:
            # Extract card name from OCR if not provided
            if not card_name:
                card_name = self._extract_card_name_from_ocr(ocr_text)
            
            if not card_name:
                analysis_result["discrepancies"].append("Could not identify card name from image")
                analysis_result["confidence_adjustment"] = -15.0
                return analysis_result
            
            # Find matching official cards
            matching_cards = self.find_matching_cards(card_name, set_name)
            
            if not matching_cards:
                analysis_result["discrepancies"].append(f"No official card found matching '{card_name}'")
                analysis_result["confidence_adjustment"] = -25.0
                return analysis_result
            
            # Find best match
            best_match = None
            best_similarity = 0.0
            
            for card in matching_cards:
                similarity = self.calculate_card_similarity(ocr_text, card)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = card
            
            if best_match:
                analysis_result["official_match_found"] = True
                analysis_result["best_match"] = {
                    "id": best_match.id,
                    "name": best_match.name,
                    "set_name": best_match.set_name,
                    "rarity": best_match.rarity,
                    "artist": best_match.artist
                }
                analysis_result["similarity_score"] = best_similarity
                
                # Analyze discrepancies
                discrepancies = self._find_discrepancies(ocr_text, best_match)
                analysis_result["discrepancies"] = discrepancies
                
                # Calculate confidence adjustment based on similarity
                if best_similarity > 0.9:
                    analysis_result["confidence_adjustment"] = 20.0
                    analysis_result["authenticity_indicators"].append("High text similarity to official card")
                elif best_similarity > 0.7:
                    analysis_result["confidence_adjustment"] = 10.0
                    analysis_result["authenticity_indicators"].append("Good text similarity to official card")
                elif best_similarity > 0.5:
                    analysis_result["confidence_adjustment"] = -5.0
                    analysis_result["authenticity_indicators"].append("Moderate text similarity to official card")
                else:
                    analysis_result["confidence_adjustment"] = -15.0
                    analysis_result["discrepancies"].append("Low text similarity to official card")
                
                # Adjust confidence based on number of discrepancies
                if len(discrepancies) > 5:
                    analysis_result["confidence_adjustment"] -= 20.0
                elif len(discrepancies) > 3:
                    analysis_result["confidence_adjustment"] -= 10.0
                elif len(discrepancies) == 0:
                    analysis_result["confidence_adjustment"] += 10.0
            
        except Exception as e:
            logger.error(f"Error in card authenticity analysis: {str(e)}")
            analysis_result["discrepancies"].append("Error occurred during official card comparison")
            analysis_result["confidence_adjustment"] = -10.0
        
        return analysis_result
    
    def _clean_card_name(self, name: str) -> str:
        """Clean card name for better matching."""
        if not name:
            return ""
        
        # Remove special characters and normalize
        cleaned = re.sub(r'[^\w\s-]', '', name)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Handle common variations
        cleaned = cleaned.replace('-EX', ' EX')
        cleaned = cleaned.replace('-GX', ' GX')
        cleaned = cleaned.replace('Pokemon', 'Pokémon')
        
        return cleaned
    
    def _clean_set_name(self, set_name: str) -> str:
        """Clean set name for better matching."""
        if not set_name:
            return ""
        
        # Remove common prefixes/suffixes
        cleaned = set_name.replace('Pokemon', '').replace('TCG', '').strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned
    
    def _clean_text_for_comparison(self, text: str) -> str:
        """Clean text for similarity comparison."""
        if not text:
            return ""
        
        # Convert to lowercase and remove extra whitespace
        cleaned = text.lower()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove common OCR artifacts
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _extract_official_card_text(self, official_card: OfficialCard) -> str:
        """Extract comparable text from official card data."""
        text_parts = []
        
        # Add card name
        if official_card.name:
            text_parts.append(official_card.name)
        
        # Add set name
        if official_card.set_name:
            text_parts.append(official_card.set_name)
        
        # Add HP if available
        if official_card.hp:
            text_parts.append(f"HP {official_card.hp}")
        
        # Add types
        if official_card.types:
            try:
                types = json.loads(official_card.types)
                text_parts.extend(types)
            except:
                pass
        
        # Add attacks
        if official_card.attacks:
            try:
                attacks = json.loads(official_card.attacks)
                for attack in attacks:
                    if isinstance(attack, dict) and 'name' in attack:
                        text_parts.append(attack['name'])
                        if 'text' in attack and attack['text']:
                            text_parts.append(attack['text'])
            except:
                pass
        
        # Add artist
        if official_card.artist:
            text_parts.append(official_card.artist)
        
        # Add rarity
        if official_card.rarity:
            text_parts.append(official_card.rarity)
        
        return ' '.join(text_parts)
    
    def _calculate_name_similarity(self, ocr_text: str, official_name: str) -> float:
        """Calculate similarity specifically for card names."""
        if not ocr_text or not official_name:
            return 0.0
        
        ocr_words = set(self._clean_text_for_comparison(ocr_text).split())
        official_words = set(self._clean_text_for_comparison(official_name).split())
        
        if not official_words:
            return 0.0
        
        # Calculate word overlap
        overlap = len(ocr_words.intersection(official_words))
        total_words = len(official_words)
        
        return overlap / total_words
    
    def _extract_card_name_from_ocr(self, ocr_text: str) -> Optional[str]:
        """Extract potential card name from OCR text."""
        if not ocr_text:
            return None
        
        lines = ocr_text.split('\n')
        
        # Common Pokemon names that might appear on cards
        pokemon_indicators = [
            'pikachu', 'charizard', 'blastoise', 'venusaur', 'mewtwo', 'mew',
            'lugia', 'ho-oh', 'rayquaza', 'kyogre', 'groudon', 'dialga', 'palkia',
            'giratina', 'arceus', 'reshiram', 'zekrom', 'kyurem'
        ]
        
        # Look for Pokemon names in the text
        for line in lines[:3]:  # Check first few lines
            line_clean = self._clean_text_for_comparison(line)
            for pokemon in pokemon_indicators:
                if pokemon in line_clean:
                    return line.strip()
        
        # If no known Pokemon found, return the first substantial line
        for line in lines:
            if len(line.strip()) > 3 and not line.strip().isdigit():
                return line.strip()
        
        return None
    
    def _find_discrepancies(self, ocr_text: str, official_card: OfficialCard) -> List[str]:
        """Find specific discrepancies between OCR text and official card."""
        discrepancies = []
        
        if not ocr_text or not official_card:
            return discrepancies
        
        ocr_clean = self._clean_text_for_comparison(ocr_text)
        
        # Check HP mismatch
        if official_card.hp:
            hp_pattern = r'hp\s*(\d+)'
            ocr_hp_match = re.search(hp_pattern, ocr_clean)
            if ocr_hp_match:
                ocr_hp = int(ocr_hp_match.group(1))
                if ocr_hp != official_card.hp:
                    discrepancies.append(f"HP mismatch: found {ocr_hp}, expected {official_card.hp}")
            else:
                discrepancies.append(f"HP not found in card text (expected {official_card.hp})")
        
        # Check artist name
        if official_card.artist:
            artist_clean = self._clean_text_for_comparison(official_card.artist)
            if artist_clean not in ocr_clean:
                discrepancies.append(f"Artist name '{official_card.artist}' not found")
        
        # Check set information
        if official_card.set_name:
            set_clean = self._clean_text_for_comparison(official_card.set_name)
            if set_clean not in ocr_clean:
                discrepancies.append(f"Set name '{official_card.set_name}' not found")
        
        # Check rarity information
        if official_card.rarity:
            rarity_clean = self._clean_text_for_comparison(official_card.rarity)
            if 'rare' in rarity_clean and 'rare' not in ocr_clean:
                discrepancies.append("Rarity information not found or incorrect")
        
        return discrepancies