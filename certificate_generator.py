"""
Certificate Generator for Pokémon Card Authentication
Creates shareable certificates with QR codes for card analysis results
"""

import os
import json
import uuid
import qrcode
from qrcode import constants
import logging
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Optional, List
import hashlib
import numpy as np

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class CertificateGenerator:
    def __init__(self):
        self.certificates_dir = "certificates"
        self.qr_codes_dir = os.path.join(self.certificates_dir, "qr_codes")
        self.cert_images_dir = os.path.join(self.certificates_dir, "images")
        self.cert_data_dir = os.path.join(self.certificates_dir, "data")
        
        # Create directories
        for directory in [self.certificates_dir, self.qr_codes_dir, 
                         self.cert_images_dir, self.cert_data_dir]:
            os.makedirs(directory, exist_ok=True)
        
        logging.info("Certificate Generator initialized")
    
    def generate_certificate_id(self, analysis_data: Dict) -> str:
        """Generate unique certificate ID based on analysis data"""
        # Create hash from key analysis components
        hash_input = f"{analysis_data.get('filename', '')}{analysis_data.get('prediction', '')}{analysis_data.get('confidence', '')}{datetime.now().isoformat()}"
        cert_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        return f"PKM-CERT-{cert_hash.upper()}"
    
    def create_certificate_data(self, analysis_data: Dict, certificate_id: str) -> Dict:
        """Create certificate data structure"""
        # Clean the analysis data to remove numpy types
        clean_analysis_data = self._clean_for_json(analysis_data)
        
        certificate_data = {
            "certificate_id": certificate_id,
            "created_at": datetime.now().isoformat(),
            "card_analysis": {
                "filename": clean_analysis_data.get('filename', 'Unknown'),
                "prediction": clean_analysis_data.get('prediction', 'Unknown'),
                "confidence": round(float(clean_analysis_data.get('confidence', 0)), 2),
                "ai_features": self._clean_for_json(clean_analysis_data.get('ai_features', {})),
                "ocr_text": clean_analysis_data.get('ocr_text', ''),
                "ocr_issues": clean_analysis_data.get('ocr_issues', []),
                "card_match": self._clean_for_json(clean_analysis_data.get('card_match_analysis', {})),
                "visual_analysis": self._clean_for_json(clean_analysis_data.get('visual_analysis', {})),
                "attention_analysis": self._clean_for_json(clean_analysis_data.get('attention_analysis', {}))
            },
            "verification": {
                "authenticated": True,
                "method": "AI + OCR + Visual Analysis",
                "database_version": "2025-07-29",
                "analysis_timestamp": clean_analysis_data.get('analysis_date', datetime.now().isoformat())
            },
            "certificate_url": f"/certificate/{certificate_id}",
            "qr_code_url": f"/certificates/qr_codes/{certificate_id}.png"
        }
        
        return certificate_data
    
    def generate_qr_code(self, certificate_id: str, base_url: str = "https://pokemon-card-detector.repl.co") -> str:
        """Generate QR code for certificate"""
        certificate_url = f"{base_url}/certificate/{certificate_id}"
        
        # Create QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(certificate_url)
        qr.make(fit=True)
        
        # Create QR code image
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Save QR code
        qr_path = os.path.join(self.qr_codes_dir, f"{certificate_id}.png")
        qr_img.save(qr_path)
        
        return qr_path
    
    def create_certificate_image(self, certificate_data: Dict, card_image_path: Optional[str] = None) -> str:
        """Create visual certificate image"""
        # Certificate dimensions
        cert_width = 800
        cert_height = 600
        
        # Create certificate image
        cert_img = Image.new('RGB', (cert_width, cert_height), color='white')
        draw = ImageDraw.Draw(cert_img)
        
        # Try to load fonts (fallback to default if not available)
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Colors
        primary_color = "#2c3e50"
        accent_color = "#3498db"
        success_color = "#27ae60"
        warning_color = "#e74c3c"
        
        # Draw border
        border_width = 3
        draw.rectangle([border_width, border_width, cert_width-border_width, cert_height-border_width], 
                      outline=primary_color, width=border_width)
        
        # Draw header
        draw.rectangle([10, 10, cert_width-10, 80], fill=primary_color)
        draw.text((cert_width//2, 45), "POKÉMON CARD AUTHENTICITY CERTIFICATE", 
                 fill="white", font=title_font, anchor="mm")
        
        # Certificate ID
        draw.text((cert_width//2, 100), f"Certificate ID: {certificate_data['certificate_id']}", 
                 fill=primary_color, font=header_font, anchor="mm")
        
        # Analysis results
        analysis = certificate_data['card_analysis']
        prediction = analysis['prediction']
        confidence = analysis['confidence']
        
        # Result color
        result_color = success_color if prediction == "Original" else warning_color
        
        # Draw result box
        result_y = 130
        draw.rectangle([50, result_y, cert_width-50, result_y+80], outline=result_color, width=2)
        
        # Result text
        draw.text((cert_width//2, result_y+25), f"RESULT: {prediction.upper()}", 
                 fill=result_color, font=header_font, anchor="mm")
        draw.text((cert_width//2, result_y+50), f"Confidence: {confidence}%", 
                 fill=result_color, font=body_font, anchor="mm")
        
        # Analysis details
        details_y = 230
        draw.text((70, details_y), "Analysis Details:", fill=primary_color, font=header_font)
        
        details = [
            f"• Card: {analysis['filename']}",
            f"• Analysis Method: AI Classification + OCR + Visual Comparison",
            f"• OCR Issues Found: {len(analysis.get('ocr_issues', []))}",
            f"• Database Matches: {analysis.get('card_match', {}).get('matches_found', 'N/A')}",
        ]
        
        for i, detail in enumerate(details):
            draw.text((70, details_y + 30 + i*20), detail, fill=primary_color, font=body_font)
        
        # Verification info
        verification_y = 370
        draw.text((70, verification_y), "Verification:", fill=primary_color, font=header_font)
        
        verification = certificate_data['verification']
        verification_details = [
            f"• Authenticated: {verification['authenticated']}",
            f"• Database Version: {verification['database_version']}",
            f"• Analysis Date: {verification['analysis_timestamp'][:10]}",
        ]
        
        for i, detail in enumerate(verification_details):
            draw.text((70, verification_y + 30 + i*20), detail, fill=primary_color, font=body_font)
        
        # Add QR code
        qr_code_path = os.path.join(self.qr_codes_dir, f"{certificate_data['certificate_id']}.png")
        if os.path.exists(qr_code_path):
            try:
                qr_img = Image.open(qr_code_path)
                qr_img = qr_img.resize((120, 120))
                cert_img.paste(qr_img, (cert_width-150, cert_height-150))
                
                # QR code label
                draw.text((cert_width-90, cert_height-25), "Scan to verify", 
                         fill=primary_color, font=small_font, anchor="mm")
            except Exception as e:
                logging.warning(f"Failed to add QR code to certificate: {e}")
        
        # Add card image if available
        if card_image_path and os.path.exists(card_image_path):
            try:
                card_img = Image.open(card_image_path)
                # Resize card image to fit
                card_img.thumbnail((150, 200), Image.Resampling.LANCZOS)
                # Position card image
                card_x = cert_width - 200
                card_y = 130
                cert_img.paste(card_img, (card_x, card_y))
            except Exception as e:
                logging.warning(f"Failed to add card image to certificate: {e}")
        
        # Footer
        draw.text((cert_width//2, cert_height-20), 
                 "This certificate verifies the authenticity analysis of the specified Pokémon card", 
                 fill=primary_color, font=small_font, anchor="mm")
        
        # Save certificate image
        cert_image_path = os.path.join(self.cert_images_dir, f"{certificate_data['certificate_id']}.png")
        cert_img.save(cert_image_path)
        
        return cert_image_path
    
    def generate_certificate(self, analysis_data: Dict, card_image_path: Optional[str] = None, 
                           base_url: str = "https://pokemon-card-detector.repl.co") -> Dict:
        """Generate complete certificate with QR code and image"""
        try:
            # Generate certificate ID
            certificate_id = self.generate_certificate_id(analysis_data)
            
            # Create certificate data
            certificate_data = self.create_certificate_data(analysis_data, certificate_id)
            
            # Generate QR code
            qr_code_path = self.generate_qr_code(certificate_id, base_url)
            
            # Create certificate image
            cert_image_path = self.create_certificate_image(certificate_data, card_image_path)
            
            # Save certificate data
            cert_data_path = os.path.join(self.cert_data_dir, f"{certificate_id}.json")
            with open(cert_data_path, 'w', encoding='utf-8') as f:
                json.dump(certificate_data, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
            
            # Update certificate data with file paths
            certificate_data.update({
                "qr_code_path": qr_code_path,
                "certificate_image_path": cert_image_path,
                "certificate_data_path": cert_data_path
            })
            
            logging.info(f"Certificate generated successfully: {certificate_id}")
            return certificate_data
            
        except Exception as e:
            logging.error(f"Failed to generate certificate: {e}")
            return {"error": f"Certificate generation failed: {e}"}
    
    def get_certificate(self, certificate_id: str) -> Optional[Dict]:
        """Retrieve certificate data by ID"""
        cert_data_path = os.path.join(self.cert_data_dir, f"{certificate_id}.json")
        
        if not os.path.exists(cert_data_path):
            return None
        
        try:
            with open(cert_data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load certificate {certificate_id}: {e}")
            return None
    
    def list_certificates(self) -> List[Dict]:
        """List all certificates"""
        certificates = []
        
        if not os.path.exists(self.cert_data_dir):
            return certificates
        
        for filename in os.listdir(self.cert_data_dir):
            if filename.endswith('.json'):
                cert_id = filename[:-5]  # Remove .json extension
                cert_data = self.get_certificate(cert_id)
                if cert_data:
                    certificates.append({
                        "certificate_id": cert_id,
                        "created_at": cert_data.get('created_at'),
                        "prediction": cert_data.get('card_analysis', {}).get('prediction'),
                        "confidence": cert_data.get('card_analysis', {}).get('confidence'),
                        "filename": cert_data.get('card_analysis', {}).get('filename')
                    })
        
        # Sort by creation date (newest first)
        certificates.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return certificates

    def _clean_for_json(self, obj):
        """Convert numpy types and other non-serializable objects to JSON-serializable types"""
        import numpy as np
        
        if isinstance(obj, (np.integer, np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._clean_for_json(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)  # Convert objects to string representation
        else:
            return obj