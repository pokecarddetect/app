import os
import re
import logging
from PIL import Image
import pytesseract

class OCRProcessor:
    """
    OCR (Optical Character Recognition) processor for analyzing text on Pokémon cards.
    Detects spelling errors, font inconsistencies, and other textual anomalies common in fake cards.
    """
    
    def __init__(self):
        """
        Initialize OCR processor with Tesseract configuration.
        Sets up language models and confidence thresholds for text extraction.
        """
        # Configure Tesseract for optimal card text recognition
        self.tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,/:()-™® '
        
        # Common Pokémon card text patterns and legitimate terms
        self.pokemon_terms = {
            'pokemon', 'pikachu', 'charizard', 'blastoise', 'venusaur', 'mewtwo', 
            'mew', 'lucario', 'garchomp', 'rayquaza', 'kyogre', 'groudon',
            'energy', 'trainer', 'supporter', 'item', 'stadium', 'basic', 'stage',
            'evolution', 'hp', 'attack', 'damage', 'weakness', 'resistance',
            'retreat', 'cost', 'nintendo', 'gamefreak', 'creatures'
        }
        
        # Common fake card text errors and misspellings
        self.common_fakes = {
            'nintnedo': 'nintendo',
            'pokmon': 'pokemon',
            'trainor': 'trainer',
            'atack': 'attack',
            'retreet': 'retreat',
            'weaknes': 'weakness'
        }
        
        logging.info("OCR Processor initialized")
    
    def analyze_card_text(self, image_path):
        """
        Perform comprehensive OCR analysis on a Pokémon card image.
        
        Args:
            image_path: Path to the card image file
            
        Returns:
            tuple: (extracted_text, issues_found)
        """
        try:
            # Extract text from the image
            extracted_text = self._extract_text(image_path)
            
            if not extracted_text.strip():
                return "", ["No text detected - possible scanning issue"]
            
            # Analyze the extracted text for issues
            issues = self._analyze_text_issues(extracted_text)
            
            logging.info(f"OCR analysis completed. Found {len(issues)} potential issues.")
            
            return extracted_text, issues
            
        except Exception as e:
            logging.error(f"Error in OCR analysis: {str(e)}")
            return "", [f"OCR processing error: {str(e)}"]
    
    def _extract_text(self, image_path):
        """
        Extract text from the card image using Tesseract OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Extracted text from the image
        """
        try:
            # Open and preprocess image for better OCR
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance image for better text recognition
            enhanced_image = self._enhance_for_ocr(image)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(enhanced_image, config=self.tesseract_config)
            
            return text.strip()
            
        except Exception as e:
            logging.error(f"Error extracting text: {str(e)}")
            # Fallback: try with original image
            try:
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image)
                return text.strip()
            except:
                return ""
    
    def _enhance_for_ocr(self, image):
        """
        Enhance image quality for better OCR recognition.
        
        Args:
            image: PIL Image object
            
        Returns:
            PIL Image: Enhanced image for OCR
        """
        try:
            from PIL import ImageEnhance, ImageFilter
            
            # Resize image for better text recognition
            width, height = image.size
            if width < 1000:
                scale_factor = 1000 / width
                new_size = (int(width * scale_factor), int(height * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Enhance contrast for better text visibility
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Apply slight blur to reduce noise
            image = image.filter(ImageFilter.SMOOTH_MORE)
            
            return image
            
        except Exception as e:
            logging.warning(f"Image enhancement failed: {str(e)}")
            return image
    
    def _analyze_text_issues(self, text):
        """
        Analyze extracted text for common fake card indicators.
        
        Args:
            text: Extracted text from OCR
            
        Returns:
            list: List of issues found in the text
        """
        issues = []
        text_lower = text.lower()
        
        # Check for common misspellings in fake cards
        for fake_word, correct_word in self.common_fakes.items():
            if fake_word in text_lower:
                issues.append(f"Misspelling detected: '{fake_word}' should be '{correct_word}'")
        
        # Check for missing copyright information
        if not any(term in text_lower for term in ['©', 'copyright', 'nintendo', 'gamefreak']):
            issues.append("Missing or incomplete copyright information")
        
        # Check for unusual character patterns
        if re.search(r'[^\w\s.,/:()™®-]', text):
            issues.append("Unusual characters detected - possible printing errors")
        
        # Check for repeated characters (common in poor printing)
        if re.search(r'(.)\1{3,}', text):
            issues.append("Repeated character sequences detected")
        
        # Check text quality and legibility
        text_issues = self._assess_text_quality(text)
        issues.extend(text_issues)
        
        # Check for Pokémon-specific terms
        pokemon_issues = self._check_pokemon_terminology(text_lower)
        issues.extend(pokemon_issues)
        
        return issues
    
    def _assess_text_quality(self, text):
        """
        Assess overall text quality and detect OCR confidence issues.
        
        Args:
            text: Extracted text
            
        Returns:
            list: List of text quality issues
        """
        issues = []
        
        # Check for very short text (might indicate poor OCR)
        if len(text.strip()) < 10:
            issues.append("Very little text detected - image quality may be poor")
        
        # Check for excessive special characters (OCR noise)
        special_char_ratio = len(re.findall(r'[^\w\s]', text)) / max(len(text), 1)
        if special_char_ratio > 0.3:
            issues.append("High ratio of special characters - possible OCR errors")
        
        # Check for broken words (spaces within words)
        words = text.split()
        single_chars = [w for w in words if len(w) == 1 and w.isalpha()]
        if len(single_chars) > len(words) * 0.2:
            issues.append("Many single characters detected - possible text fragmentation")
        
        # Check for number consistency
        numbers = re.findall(r'\d+', text)
        if numbers:
            try:
                # Check if numbers are reasonable for Pokémon cards
                for num in numbers:
                    if len(num) > 4:  # HP values, damage, etc. shouldn't be too large
                        issues.append(f"Unusually large number detected: {num}")
            except:
                pass
        
        return issues
    
    def _check_pokemon_terminology(self, text_lower):
        """
        Check for proper Pokémon terminology and detect suspicious text.
        
        Args:
            text_lower: Lowercase version of extracted text
            
        Returns:
            list: List of terminology-related issues
        """
        issues = []
        
        # Check if text contains any Pokémon-related terms
        has_pokemon_terms = any(term in text_lower for term in self.pokemon_terms)
        
        if not has_pokemon_terms and len(text_lower) > 20:
            issues.append("No recognizable Pokémon terminology found")
        
        # Check for common fake card phrases
        fake_indicators = [
            'not official',
            'fan made',
            'proxy',
            'custom',
            'reproduction'
        ]
        
        for indicator in fake_indicators:
            if indicator in text_lower:
                issues.append(f"Suspicious text found: '{indicator}'")
        
        # Check for missing set information
        set_indicators = ['©', 'nintendo', 'gamefreak', 'creatures']
        if not any(indicator in text_lower for indicator in set_indicators):
            issues.append("Missing official set/copyright information")
        
        return issues
    
    def get_text_confidence(self, image_path):
        """
        Get OCR confidence scores for the extracted text.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Confidence metrics for the OCR result
        """
        try:
            image = Image.open(image_path)
            
            # Get detailed OCR data with confidence scores
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'average_confidence': avg_confidence,
                'word_count': len([word for word in data['text'] if word.strip()]),
                'low_confidence_words': len([conf for conf in confidences if conf < 50])
            }
            
        except Exception as e:
            logging.error(f"Error getting OCR confidence: {str(e)}")
            return {'average_confidence': 0, 'word_count': 0, 'low_confidence_words': 0}
