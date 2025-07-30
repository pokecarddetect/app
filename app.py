import os
import logging
from flask import Flask, session, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

# Create database instance
db = SQLAlchemy(model_class=Base)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "pokemon-card-detector-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # needed for url_for to generate with https

# Configure the database, relative to the app instance folder
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///pokemon_cards.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Configure file upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('official_cards', exist_ok=True)

# Initialize the app with the extension
db.init_app(app)

# Add language support context processor
@app.context_processor
def inject_language():
    lang = session.get('language', 'cs')  # Default to Czech
    from translations import get_text, get_available_languages, get_language_names
    return dict(
        get_text=lambda key: get_text(key, lang),
        current_language=lang,
        available_languages=get_available_languages(),
        language_names=get_language_names()
    )

# Language switching route
@app.route('/set_language/<language>')
def set_language(language):
    from translations import get_available_languages
    from flask import request, session, redirect
    from urllib.parse import urlparse
    
    if language in get_available_languages():
        session['language'] = language
    
    # Validate referrer URL to prevent open redirect attacks
    referrer = request.referrer
    if referrer:
        parsed_referrer = urlparse(referrer)
        parsed_request = urlparse(request.url)
        # Only redirect to same host
        if parsed_referrer.netloc == parsed_request.netloc:
            return redirect(referrer)
    
    return redirect('/')

with app.app_context():
    # Import models and routes
    import models  # noqa: F401
    import routes  # noqa: F401
    
    # Create all database tables
    db.create_all()
