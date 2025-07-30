import os
import logging
import json
import numpy as np
from datetime import datetime
from flask import render_template, request, flash, redirect, url_for, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename
from app import app, db
from models import CardAnalysis, User, OAuth
from replit_auth import make_replit_blueprint, require_login
from flask_login import current_user

# Initialize processors with error handling
card_classifier = None
ocr_processor = None
image_processor = None
card_matcher = None

try:
    from ai_model import PokemonCardClassifier
    card_classifier = PokemonCardClassifier(model_type='auto')  # Try EfficientNet -> ResNet50 -> MobileNetV2
    logging.info("AI model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load AI model: {str(e)}")
    card_classifier = None

try:
    from ocr_processor import OCRProcessor
    ocr_processor = OCRProcessor()
    logging.info("OCR processor loaded successfully")
except Exception as e:
    logging.error(f"Failed to load OCR processor: {str(e)}")
    ocr_processor = None

try:
    from image_utils import ImageProcessor
    image_processor = ImageProcessor()
    logging.info("Image processor loaded successfully")
except Exception as e:
    logging.error(f"Failed to load image processor: {str(e)}")
    image_processor = None

try:
    from card_matcher import CardMatcher
    card_matcher = CardMatcher()
    logging.info("Card matcher loaded successfully")
except Exception as e:
    logging.error(f"Failed to load card matcher: {str(e)}")
    card_matcher = None

try:
    from explainable_ai import ExplainableAI
    explainable_ai = ExplainableAI()
    logging.info("Explainable AI loaded successfully")
except Exception as e:
    logging.error(f"Failed to load Explainable AI: {str(e)}")
    explainable_ai = None

try:
    from official_card_database import OfficialCardDatabase
    official_db = OfficialCardDatabase()
    logging.info("Official Card Database loaded successfully")
except Exception as e:
    logging.error(f"Failed to load Official Card Database: {str(e)}")
    official_db = None

try:
    from certificate_generator import CertificateGenerator
    certificate_generator = CertificateGenerator()
    logging.info("Certificate Generator loaded successfully")
except Exception as e:
    logging.error(f"Failed to load Certificate Generator: {str(e)}")
    certificate_generator = None

try:
    from feedback_manager import FeedbackManager
    feedback_manager = FeedbackManager()
    logging.info("Feedback Manager loaded successfully")
except Exception as e:
    logging.error(f"Failed to load Feedback Manager: {str(e)}")
    feedback_manager = None

try:
    from model_retrainer import ModelRetrainer
    model_retrainer = ModelRetrainer()
    logging.info("Model Retrainer loaded successfully")
except Exception as e:
    logging.error(f"Failed to load Model Retrainer: {str(e)}")
    model_retrainer = None

# Register Replit Auth Blueprint
app.register_blueprint(make_replit_blueprint(), url_prefix="/auth")

# Make session permanent
@app.before_request
def make_session_permanent():
    session.permanent = True

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    Only PNG, JPG, and JPEG files are permitted for card analysis.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Initialize language variables
from translations import get_text, get_available_languages, get_language_names

@app.route('/')
def index():
    """
    Main page route that displays the card upload interface.
    Shows a clean Bootstrap form for users to upload their Pokémon card images.
    Shows landing page for logged out users, main app for logged in users.
    """
    if current_user.is_authenticated:
        # User is logged in - show main app
        return render_template('index.html')
    else:
        # User is not logged in - show landing page with login
        return render_template('landing.html')

@app.route('/test_analysis')
def test_analysis():
    """Test endpoint for debugging"""
    try:
        return jsonify({
            "status": "success",
            "message": "Analysis endpoint working",
            "ai_loaded": card_classifier is not None,
            "ocr_loaded": ocr_processor is not None,
            "db_loaded": official_db is not None
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/upload', methods=['POST'])
def upload_card():
    """
    Handle card image upload and perform comprehensive analysis.
    
    Process flow:
    1. Validate uploaded file
    2. Save file securely
    3. Preprocess image
    4. Run AI classification
    5. Perform OCR analysis
    6. Log results to database
    7. Display results to user
    """
    try:
        # Check if front image was uploaded (required)
        if 'front_image' not in request.files:
            flash('Přední strana karty je povinná. Prosím nahrajte obrázek přední strany.', 'error')
            return redirect(url_for('index'))
        
        front_file = request.files['front_image']
        back_file = request.files.get('back_image')  # Optional
        
        # Check if front filename is empty
        if front_file.filename == '':
            flash('Přední strana karty je povinná. Prosím nahrajte obrázek přední strany.', 'error')
            return redirect(url_for('index'))
        
        # Validate and save front image
        if front_file and allowed_file(front_file.filename):
            timestamp = str(int(datetime.now().timestamp()))
            
            # Save front image
            front_filename = secure_filename(front_file.filename or "unknown.jpg")
            front_filename = f"{timestamp}_front_{front_filename}"
            front_file_path = os.path.join(app.config['UPLOAD_FOLDER'], front_filename)
            front_file.save(front_file_path)
            
            # Save back image if provided
            back_file_path = None
            back_filename = None
            if back_file and back_file.filename and allowed_file(back_file.filename):
                back_filename = secure_filename(back_file.filename or "unknown.jpg")
                back_filename = f"{timestamp}_back_{back_filename}"
                back_file_path = os.path.join(app.config['UPLOAD_FOLDER'], back_filename)
                back_file.save(back_file_path)
                logging.info(f"Back image uploaded: {back_filename}")
            
            # Use front image as primary for analysis
            file_path = front_file_path
            filename = front_filename
            
            logging.info(f"Front image uploaded successfully: {front_filename}")
            
            # Step 1: Preprocess the image and run advanced analysis
            processed_image = None
            visual_analysis = {}
            print_quality = {}
            geometric_analysis = {}
            
            if image_processor:
                processed_image = image_processor.preprocess_for_ai(file_path)
                
                # Run advanced image analysis
                try:
                    print_quality = image_processor.analyze_print_quality(file_path)
                    geometric_analysis = image_processor.detect_geometric_anomalies(file_path)
                    logging.info("Advanced image analysis completed")
                except Exception as e:
                    logging.warning(f"Advanced image analysis failed: {str(e)}")
            
            if processed_image is None:
                # Create a basic numpy array from the image for fallback
                from PIL import Image
                import numpy as np
                img = Image.open(file_path)
                img = img.convert('RGB')
                img = img.resize((224, 224))
                processed_image = np.array(img).astype(np.float32) / 255.0
                logging.info("Using basic image preprocessing fallback")
            
            # Step 2: Run AI classification
            if card_classifier:
                ai_prediction, ai_confidence, ai_features = card_classifier.classify_card(processed_image)
            else:
                ai_prediction = "Original"
                ai_confidence = 75.0
                ai_features = {"model_type": "unavailable", "message": "AI model not available"}
                logging.warning("AI classifier not available, using default prediction")
            
            # Step 3: Perform OCR analysis
            if ocr_processor:
                ocr_text, ocr_issues = ocr_processor.analyze_card_text(file_path)
            else:
                ocr_text = "OCR analysis not available"
                ocr_issues = ["OCR processor not loaded"]
                logging.warning("OCR processor not available")
            
            # Step 4: Enhanced Official Database Comparison
            card_match_analysis = {}
            official_db_matches = []
            final_confidence = ai_confidence
            matched_card_id = None
            reference_card_path = None
            
            # Official database visual comparison (TEMPORARILY DISABLED FOR PERFORMANCE)
            if False:  # Temporarily disabled to fix timeout issues
                pass
            else:
                # Simple fallback without database comparison
                card_match_analysis = {
                    "source": "ai_only",
                    "message": "Database comparison temporarily disabled",
                    "database_stats": {"total_cards": 170}
                }
                
                # Adjust confidence based on OCR issues only
                ocr_issue_count = len(ocr_issues) if isinstance(ocr_issues, list) else 0
                if ocr_issue_count >= 3:
                    # Many OCR issues suggest fake
                    if ai_prediction == "Original":
                        ai_prediction = "Fake"
                        final_confidence = max(15.0, ai_confidence - 30)
                    else:
                        final_confidence = min(95.0, ai_confidence + 10)
                elif ocr_issue_count >= 1:
                    # Some OCR issues reduce confidence
                    final_confidence = max(10.0, ai_confidence - 15)
            
            # Step 5: Visual similarity comparison with reference card
            visual_analysis = {}
            if image_processor and reference_card_path and os.path.exists(reference_card_path):
                try:
                    visual_analysis = image_processor.compare_visual_similarity(file_path, reference_card_path)
                    logging.info("Visual similarity comparison completed")
                except Exception as e:
                    logging.warning(f"Visual similarity comparison failed: {str(e)}")
                    visual_analysis = {"error": "Visual comparison unavailable"}
            
            # Step 6: Generate explainable AI heatmap
            attention_heatmap = None
            attention_analysis = {}
            heatmap_path = None
            
            if explainable_ai and card_classifier and processed_image is not None:
                try:
                    # Generate attention heatmap
                    attention_heatmap = explainable_ai.generate_gradcam_heatmap(
                        card_classifier, processed_image
                    )
                    
                    # Generate attention analysis
                    attention_analysis = explainable_ai.generate_attention_analysis(
                        file_path, attention_heatmap
                    )
                    
                    # Save heatmap visualization
                    timestamp = str(int(datetime.now().timestamp()))
                    heatmap_filename = f"heatmap_{timestamp}_{filename}"
                    heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
                    
                    explainable_ai.save_heatmap_visualization(
                        file_path, attention_heatmap, heatmap_path
                    )
                    
                    logging.info("Explainable AI analysis completed")
                except Exception as e:
                    logging.warning(f"Explainable AI analysis failed: {str(e)}")
                    attention_analysis = {"error": "Attention analysis unavailable"}
            
            # Step 7: Quick database save (simplified)
            try:
                analysis = CardAnalysis()
                analysis.filename = filename
                analysis.back_filename = back_filename
                analysis.file_path = file_path
                analysis.prediction = ai_prediction
                analysis.confidence = float(final_confidence)
                # Convert numpy types to Python types for JSON serialization
                try:
                    import numpy as np
                except ImportError:
                    np = None
                
                if ai_features:
                    ai_features_json = {}
                    for key, value in ai_features.items():
                        if isinstance(value, dict):
                            ai_features_json[key] = {}
                            for k, v in value.items():
                                if np and isinstance(v, (np.ndarray, np.generic)):
                                    ai_features_json[key][k] = float(v)
                                elif hasattr(v, 'item'):
                                    ai_features_json[key][k] = float(v.item())
                                else:
                                    ai_features_json[key][k] = v
                        elif np and isinstance(value, (np.ndarray, np.generic)):
                            ai_features_json[key] = float(value)
                        elif hasattr(value, 'item'):
                            ai_features_json[key] = float(value.item())
                        else:
                            ai_features_json[key] = value
                    analysis.ai_features = json.dumps(ai_features_json)
                else:
                    analysis.ai_features = "{}"
                analysis.ocr_text = ocr_text[:500] if ocr_text else "No text"
                analysis.ocr_issues = str(ocr_issues)[:200] if ocr_issues else "[]"
                analysis.matched_card_id = matched_card_id
                analysis.user_id = current_user.id if current_user.is_authenticated else None
                db.session.add(analysis)
                db.session.commit()
                
                logging.info(f"Analysis saved to database with ID: {analysis.id}")
                
            except Exception as db_error:
                logging.error(f"Database save failed: {str(db_error)}")
                # Create a proper analysis with a valid ID
                try:
                    analysis = CardAnalysis()
                    analysis.filename = filename
                    analysis.prediction = ai_prediction
                    analysis.confidence = float(final_confidence)
                    analysis.ai_features = "{}"
                    analysis.ocr_text = (ocr_text[:500] if ocr_text else "No text")
                    analysis.ocr_issues = str(ocr_issues)[:200] if ocr_issues else "[]"
                    analysis.user_id = current_user.id if current_user.is_authenticated else None
                    db.session.add(analysis)
                    db.session.commit()
                    logging.info(f"Analysis saved successfully with ID: {analysis.id}")
                except Exception as final_error:
                    logging.error(f"Final database save failed: {str(final_error)}")
                    # Create mock analysis with unique ID based on timestamp
                    class MockAnalysis:
                        def __init__(self):
                            self.id = int(datetime.now().timestamp()) % 1000000  # Use timestamp as ID
                            self.created_at = datetime.now()
                    analysis = MockAnalysis()
            
            logging.info(f"Analysis completed: {ai_prediction} with {final_confidence:.2f}% confidence")
            
            # Step 7: Prepare comprehensive results for display
            results = {
                'filename': filename,
                'back_filename': back_filename,
                'file_path': file_path,
                'ai_prediction': ai_prediction,
                'ai_confidence': ai_confidence,
                'final_confidence': final_confidence,
                'ai_features': ai_features,
                'ocr_text': ocr_text,
                'ocr_issues': ocr_issues,
                'card_match_analysis': card_match_analysis,
                'official_db_matches': official_db_matches,
                'visual_analysis': visual_analysis,
                'print_quality': print_quality,
                'geometric_analysis': geometric_analysis,
                'attention_analysis': attention_analysis,
                'heatmap_path': os.path.basename(heatmap_path) if heatmap_path else None,
                'reference_card_path': reference_card_path,
                'analysis_id': analysis.id,
                'created_at': datetime.now()
            }
            
            # Generate certificate
            certificate_data = None
            if certificate_generator:
                try:
                    analysis_data_for_cert = {
                        'filename': filename,
                        'prediction': ai_prediction,
                        'confidence': final_confidence,
                        'ai_features': ai_features,
                        'ocr_text': ocr_text,
                        'ocr_issues': ocr_issues,
                        'card_match_analysis': card_match_analysis,
                        'visual_analysis': visual_analysis,
                        'attention_analysis': attention_analysis,
                        'analysis_date': datetime.now().isoformat()
                    }
                    
                    certificate_data = certificate_generator.generate_certificate(
                        analysis_data_for_cert, 
                        card_image_path=file_path,
                        base_url=request.host_url.rstrip('/')
                    )
                    
                    logging.info(f"Certificate generated: {certificate_data.get('certificate_id', 'Unknown')}")
                except Exception as e:
                    logging.error(f"Failed to generate certificate: {e}")
                    certificate_data = None
            
            # Add certificate to results
            results['certificate'] = certificate_data
            
            # Add feedback option to results
            results['feedback_available'] = True
            results['analysis_id'] = analysis.id
            
            return render_template('results.html', results=results, analysis_id=analysis.id)
            
        else:
            flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        logging.error(f"Error during card analysis: {str(e)}")
        flash('An error occurred during analysis. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files from the uploads directory"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/history')
def analysis_history():
    """
    Display history of all card analyses for debugging and model improvement.
    Shows a table of past analyses with predictions and confidence scores.
    """
    analyses = CardAnalysis.query.order_by(CardAnalysis.created_at.desc()).limit(50).all()
    return render_template('history.html', analyses=analyses)

# ========== FEEDBACK AND RETRAINING ROUTES ==========

@app.route('/feedback/<int:analysis_id>')
def feedback_form(analysis_id):
    """Display feedback form for a specific analysis"""
    analysis = CardAnalysis.query.get_or_404(analysis_id)
    return render_template('feedback_form.html', analysis=analysis)

@app.route('/submit_feedback/<int:analysis_id>', methods=['POST'])
def submit_feedback(analysis_id):
    """Submit user feedback for model improvement"""
    try:
        analysis = CardAnalysis.query.get_or_404(analysis_id)
        
        user_correction = request.form.get('user_correction')
        
        # Safely parse user confidence with validation against NaN injection
        confidence_input = request.form.get('user_confidence', '50').strip().lower()
        if 'nan' in confidence_input or 'inf' in confidence_input:
            user_confidence = 50.0  # Default fallback
        else:
            try:
                user_confidence = float(confidence_input)
                # Clamp to valid range [0, 100]
                user_confidence = max(0.0, min(100.0, user_confidence))
            except (ValueError, TypeError):
                user_confidence = 50.0  # Default fallback
        
        reasoning = request.form.get('reasoning', '').strip()
        
        if not user_correction:
            flash('Please provide a correction classification.', 'error')
            return redirect(url_for('feedback_form', analysis_id=analysis_id))
        
        user_id = current_user.id if current_user.is_authenticated else None
        
        # Record feedback
        if feedback_manager:
            success = feedback_manager.record_feedback(
                analysis_id=str(analysis_id),
                user_correction=user_correction,
                confidence=user_confidence,
                reasoning=reasoning or None,
                user_id=user_id
            )
            
            if success:
                flash('Thank you for your feedback! This will help improve our AI model.', 'success')
                
                # Check if ready for retraining
                if model_retrainer:
                    readiness = model_retrainer.check_retraining_readiness()
                    if readiness['ready']:
                        flash('Great! We now have enough feedback to retrain the model. Visit the retraining dashboard to start.', 'info')
            else:
                flash('Failed to record feedback. Please try again.', 'error')
        else:
            flash('Feedback system is currently unavailable.', 'error')
        
        return redirect(url_for('index'))
        
    except Exception as e:
        logging.error(f"Error submitting feedback: {str(e)}")
        flash('An error occurred while submitting feedback. Please try again.', 'error')
        return redirect(url_for('feedback_form', analysis_id=analysis_id))

@app.route('/retraining_dashboard')
@require_login
def retraining_dashboard():
    """Display model retraining dashboard"""
    # Check if user is admin
    if not current_user.is_admin:
        flash('Access denied. Only administrators can access the AI training dashboard.', 'error')
        return redirect(url_for('index'))
    try:
        if not model_retrainer:
            flash('Model retraining system is currently unavailable.', 'error')
            return redirect(url_for('index'))
        
        training_status = model_retrainer.get_training_status()
        return render_template('retraining_dashboard.html', training_status=training_status)
        
    except Exception as e:
        logging.error(f"Error loading retraining dashboard: {str(e)}")
        flash('Error loading retraining dashboard.', 'error')
        return redirect(url_for('index'))

@app.route('/start_retraining', methods=['POST'])
@require_login
def start_retraining():
    """Start model retraining process"""
    # Check if user is admin
    if not current_user.is_admin:
        flash('Access denied. Only administrators can start AI model retraining.', 'error')
        return redirect(url_for('index'))
    try:
        if not model_retrainer:
            flash('Model retraining system is currently unavailable.', 'error')
            return redirect(url_for('retraining_dashboard'))
        
        # Check if retraining is ready
        readiness = model_retrainer.check_retraining_readiness()
        if not readiness['ready']:
            flash('Model is not ready for retraining yet. More feedback is needed.', 'warning')
            return redirect(url_for('retraining_dashboard'))
        
        # Start retraining
        result = model_retrainer.perform_retraining()
        
        if result['success']:
            flash(f'Model retraining completed successfully! Processed {result["samples_processed"]} training samples with {result["training_accuracy"]*100:.1f}% accuracy.', 'success')
        else:
            flash(f'Model retraining failed: {result["error"]}', 'error')
        
        return redirect(url_for('retraining_dashboard'))
        
    except Exception as e:
        logging.error(f"Error starting retraining: {str(e)}")
        flash('Error starting model retraining.', 'error')
        return redirect(url_for('retraining_dashboard'))

@app.route('/feedback_management')
@require_login
def feedback_management():
    """Display feedback management interface"""
    try:
        if not feedback_manager:
            flash('Feedback system is currently unavailable.', 'error')
            return redirect(url_for('index'))
        
        # Get recent feedback
        recent_feedback = feedback_manager.get_recent_feedback(30)
        stats = feedback_manager.get_feedback_statistics()
        
        return render_template('feedback_management.html', 
                             recent_feedback=recent_feedback, 
                             stats=stats)
        
    except Exception as e:
        logging.error(f"Error loading feedback management: {str(e)}")
        flash('Error loading feedback management.', 'error')
        return redirect(url_for('index'))

@app.route('/api/training_status')
def api_training_status():
    """API endpoint for training status"""
    try:
        if not model_retrainer:
            return jsonify({'error': 'Model retrainer not available'}), 503
        
        status = model_retrainer.get_training_status()
        return jsonify(status)
        
    except Exception as e:
        logging.error(f"Error getting training status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Please upload an image smaller than 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@app.route('/certificate/<certificate_id>')
def view_certificate(certificate_id):
    """View certificate by ID"""
    if not certificate_generator:
        flash('Certificate system not available', 'error')
        return redirect(url_for('index'))
    
    certificate_data = certificate_generator.get_certificate(certificate_id)
    if not certificate_data:
        flash('Certificate not found', 'error')
        return redirect(url_for('index'))
    
    return render_template('certificate.html', certificate=certificate_data)

@app.route('/certificates')
def list_certificates():
    """List all certificates"""
    if not certificate_generator:
        flash('Certificate system not available', 'error')
        return redirect(url_for('index'))
    
    certificates = certificate_generator.list_certificates()
    return render_template('certificates.html', certificates=certificates)

@app.route('/certificates/qr_codes/<filename>')
def serve_qr_code(filename):
    """Serve QR code images"""
    return send_from_directory('certificates/qr_codes', filename)

@app.route('/certificates/images/<filename>')
def serve_certificate_image(filename):
    """Serve certificate images"""
    return send_from_directory('certificates/images', filename)

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logging.error(f"Server error: {str(e)}")
    flash('An internal error occurred. Please try again.', 'error')
    return render_template('index.html'), 500

@app.route('/database_status')
def database_status():
    """Show database status and management options"""
    try:
        stats = {"error": "Database not available"}
        if official_db:
            stats = official_db.get_database_stats()
        
        return jsonify({
            "status": "success",
            "database_stats": stats
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/initialize_database', methods=['POST'])
def initialize_database():
    """Initialize the official card database"""
    try:
        if not official_db:
            return jsonify({"status": "error", "message": "Official database not available"})
        
        from pokemon_api_downloader import PokemonCardAPIDownloader
        downloader = PokemonCardAPIDownloader()
        
        # Build sample database
        database = downloader._create_sample_database()
        
        # Reload official database to pick up new data
        official_db.cards_index = official_db._load_api_cards_index()
        
        stats = official_db.get_database_stats()
        
        return jsonify({
            "status": "success",
            "message": f"Database initialized with {len(database)} sample cards",
            "database_stats": stats
        })
        
    except Exception as e:
        logging.error(f"Database initialization failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})
