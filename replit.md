# Pokémon Card Authenticity Detector

## Overview

This is a comprehensive Flask web application that uses advanced artificial intelligence to classify Pokémon cards into multiple categories including Original, Fake, Proxy, Custom Art, Altered, and Test Print cards. The system combines multiple analysis techniques including CNN deep learning, OCR text analysis, visual similarity comparison, explainable AI with attention heatmaps, print quality assessment, and geometric analysis to provide highly accurate multi-category predictions with detailed explanations.

**NEW: Complete Model Retraining Pipeline** - The system now includes a comprehensive user feedback system and automated model retraining capabilities that allow continuous improvement based on real-world usage and user corrections.

## User Preferences

Preferred communication style: Simple, everyday language.
Language: Czech (česky)

## System Architecture

### Web Framework
- **Flask**: Lightweight Python web framework serving as the main application backbone
- **SQLAlchemy**: ORM for database operations with SQLite as the default database
- **Bootstrap**: Frontend CSS framework for responsive UI design

### AI/ML Components
- **TensorFlow/Keras**: Deep learning framework for the core AI model
- **Multiple AI Architectures**: 
  - **EfficientNetB0**: Modern compound scaling architecture with optimal efficiency (preferred)
  - **ResNet50**: Deep residual network with skip connections for robust feature extraction
  - **MobileNetV2**: Lightweight mobile-optimized architecture for fast inference
  - **Enhanced Fallback**: OpenCV-based computer vision analysis when TensorFlow unavailable
- **Auto Model Selection**: Automatically tries models in order: EfficientNet → ResNet50 → MobileNetV2 → Fallback
- **Tesseract OCR**: Text extraction and analysis from card images
- **PIL/OpenCV**: Advanced image processing and computer vision operations
- **Scikit-Image**: Advanced image analysis including SSIM, HOG features, and edge detection
- **Explainable AI**: Grad-CAM implementation for attention heatmap generation
- **Visual Similarity**: ORB feature matching, color histogram analysis, and structural comparison

### Database
- **SQLite**: Default lightweight database for storing analysis results
- **SQLAlchemy Models**: CardAnalysis model tracks predictions, confidence scores, and metadata

## Key Components

### Feedback and Retraining System (`feedback_manager.py`, `model_retrainer.py`)
- **FeedbackManager**: Collects and manages user feedback for model improvement
- **ModelRetrainer**: Handles automated model retraining with user feedback data
- **Training Pipeline**: Complete retraining workflow with statistics and progress tracking
- **Data Storage**: JSON-based feedback storage with database integration
- **Readiness Checking**: Automatic assessment of when model is ready for retraining
- **Performance Tracking**: Training history and accuracy improvements monitoring

### AI Classification (`ai_model.py`)
- **PokemonCardClassifier**: Multi-architecture multi-category classifier with automatic model selection
- **Supported Models**:
  - **EfficientNetB0**: 256-unit dense layers with BatchNormalization and optimized dropout (0.3)
  - **ResNet50**: 512-unit dense layers with deeper architecture for complex patterns
  - **MobileNetV2**: 128-unit lightweight architecture for fast inference
  - **Enhanced Fallback**: OpenCV-based analysis using sharpness, edge detection, color consistency
- Uses transfer learning with frozen ImageNet-pretrained base layers
- Custom optimized classification heads tailored for each architecture
- Automatic fallback system ensures functionality even without TensorFlow
- Input size: 224x224x3 (standard for all supported models)
- **Multi-category Classification**: Supports 6 categories:
  - **Original**: Authentic official cards
  - **Fake**: Poor quality counterfeits
  - **Proxy**: High quality reproductions for gameplay
  - **Custom Art**: Fan-made cards with custom artwork
  - **Altered**: Modified official cards (altered art, foiling, etc.)
  - **Test Print**: Prototype or test prints

### OCR Processing (`ocr_processor.py`)
- **OCRProcessor**: Extracts and analyzes text from card images
- Detects common misspellings and font inconsistencies in fake cards
- Maintains dictionaries of legitimate Pokémon terms and common fake errors
- Configured for optimal card text recognition

### Image Processing (`image_utils.py`)
- **ImageProcessor**: Handles image preprocessing for AI model input
- Normalizes card orientation and aspect ratio (2.5:3.5)
- Enhances image quality and resizes to model requirements
- Converts images to normalized float32 arrays

### Web Routes (`routes.py`)
- **Upload endpoint**: Handles file uploads with security validation
- **Analysis workflow**: Coordinates AI classification and OCR processing
- **Results display**: Renders analysis results with confidence scores
- **History tracking**: Stores results in database for future reference
- **Feedback system**: User feedback collection routes for model improvement
- **Retraining dashboard**: Model retraining management interface

### Database Models (`models.py`)
- **CardAnalysis**: Stores analysis results including:
  - File information and paths
  - AI predictions and confidence scores
  - OCR extracted text and detected issues
  - Timestamps for analysis tracking
  - **User feedback fields**: Stores user corrections, confidence, and reasoning for model retraining

## Data Flow

1. **Image Upload**: User uploads card image through web interface
2. **File Validation**: System validates file type and size constraints
3. **Image Preprocessing**: Image is normalized, enhanced, and resized
4. **AI Analysis**: MobileNetV2 model classifies card as Original/Fake
5. **OCR Processing**: Text is extracted and analyzed for anomalies
6. **Results Storage**: Analysis results are saved to database
7. **Response**: User receives prediction with confidence score and details

## External Dependencies

### Python Packages
- `tensorflow`: AI model framework
- `flask`: Web application framework
- `sqlalchemy`: Database ORM
- `pillow`: Image processing
- `opencv-python`: Advanced image operations
- `pytesseract`: OCR functionality
- `numpy`: Numerical operations

### System Dependencies
- **Tesseract OCR**: System-level OCR engine installation required
- **ImageNet Weights**: Pre-trained MobileNetV2 weights downloaded automatically

### File Storage
- **uploads/**: Directory for user-uploaded card images
- **official_cards/**: Directory for reference card images (future expansion)

## Deployment Strategy

### Development Setup
- **Flask Debug Mode**: Enabled for development with auto-reload
- **SQLite Database**: File-based database for simple deployment
- **Local File Storage**: Images stored in local directories

### Configuration
- **Environment Variables**: 
  - `DATABASE_URL`: Database connection string
  - `SESSION_SECRET`: Flask session encryption key
- **File Upload Limits**: 16MB maximum file size
- **Allowed Extensions**: PNG, JPG, JPEG only

### Scalability Considerations
- **Database**: Can be switched to PostgreSQL by changing DATABASE_URL
- **File Storage**: Upload directory structure ready for cloud storage migration
- **Model Loading**: AI model initialization optimized for production deployment
- **Logging**: Comprehensive logging system for debugging and monitoring

### Security Features
- **File Validation**: Strict file type and size checking
- **Secure Filenames**: Uses werkzeug secure_filename for safe file handling
- **CSRF Protection**: Flask secret key configuration for session security
- **Proxy Headers**: ProxyFix middleware for proper HTTPS handling in production
- **Admin Access Control**: AI training and retraining restricted to admin accounts only

## Recent Updates (July 29, 2025)

### Deployment Optimization (Latest)
- **Size Reduction**: Resolved deployment size issues by removing local Pokemon card database (167MB) and uploads directory (97MB)
- **PostgreSQL Migration**: Migrated from local JSON files to PostgreSQL database for card data storage
- **External Storage**: Implemented remote image URL support instead of local file storage
- **TensorFlow Removal**: Removed TensorFlow dependency (28 packages) to reduce deployment size from 2.4GB to under 1GB
- **Package Optimization**: Removed matplotlib and scikit-image dependencies (12 packages) for further size reduction
- **Enhanced Fallback Model**: Improved OpenCV-based model for deployment without TensorFlow dependencies
- **Database Schema**: Updated models to support external card data storage with PostgreSQL
- **Remote Images**: Modified image processing to handle both local files and remote URLs

### Previous Updates
- **Security Fix**: Patched critical security vulnerability in user input validation (routes.py line 476) that could allow NaN injection attacks through float() typecast
- **Input Validation**: Added comprehensive validation for user_confidence parameter to prevent malicious input including 'nan', 'inf' values with safe fallback to default values
- **Dark Theme Restoration**: Restored original dark Bootstrap theme throughout the application per user preference
- **Admin Access Control**: Added is_admin field to User model, restricted AI training dashboard and retraining to admin accounts only
- **Feedback System**: Fixed feedback form accessibility after card analysis completion - green feedback button appears after each analysis
- **Language Switching**: Fixed language selection dropdown with Czech as default language
- **Czech Localization**: Enhanced Czech language support throughout the application
- **Navigation Fixes**: Resolved duplicate route conflicts between app.py and routes.py
- **Community Learning**: System learns from all user feedback while restricting model retraining to admin accounts