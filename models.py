from app import db
from datetime import datetime
from flask_dance.consumer.storage.sqla import OAuthConsumerMixin
from flask_login import UserMixin
from sqlalchemy import UniqueConstraint

# User table for Replit Auth
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.String, primary_key=True)
    email = db.Column(db.String, unique=True, nullable=True)
    first_name = db.Column(db.String, nullable=True)
    last_name = db.Column(db.String, nullable=True)
    profile_image_url = db.Column(db.String, nullable=True)
    is_admin = db.Column(db.Boolean, default=False)  # Admin flag for AI training access

    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationship with card analyses
    analyses = db.relationship('CardAnalysis', backref='user', lazy=True)

# OAuth table for Replit Auth
class OAuth(OAuthConsumerMixin, db.Model):
    user_id = db.Column(db.String, db.ForeignKey(User.id))
    browser_session_key = db.Column(db.String, nullable=False)
    user = db.relationship(User)

    __table_args__ = (UniqueConstraint(
        'user_id',
        'browser_session_key',
        'provider',
        name='uq_user_browser_session_key_provider',
    ),)

class CardAnalysis(db.Model):
    """
    Database model to store card analysis results for logging and potential model retraining.
    Tracks each uploaded card analysis with timestamp, prediction, and confidence.
    """
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    back_filename = db.Column(db.String(255), nullable=True)  # Optional back image
    file_path = db.Column(db.String(500), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)  # 'Original', 'Fake', 'Proxy', 'Custom Art', etc.
    confidence = db.Column(db.Float, nullable=False)
    ai_features = db.Column(db.Text)  # JSON string of AI analysis features
    ocr_text = db.Column(db.Text)  # Extracted OCR text
    ocr_issues = db.Column(db.Text)  # JSON string of OCR-detected issues
    matched_card_id = db.Column(db.String(255))  # Link to official card if matched
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Link to user (nullable for backward compatibility)
    user_id = db.Column(db.String, db.ForeignKey('users.id'), nullable=True)
    
    # User feedback fields for retraining
    user_feedback = db.Column(db.String(50), nullable=True)  # User's correction
    feedback_confidence = db.Column(db.Float, nullable=True)  # User's confidence
    feedback_reasoning = db.Column(db.Text, nullable=True)  # User's explanation
    feedback_date = db.Column(db.DateTime, nullable=True)
    
    def __repr__(self):
        return f'<CardAnalysis {self.filename}: {self.prediction} ({self.confidence:.2f}%)>'

class OfficialCard(db.Model):
    """
    Database model to store official Pokemon card data from the Pokemon TCG API.
    Used for reference comparison and authenticity validation.
    """
    id = db.Column(db.String(255), primary_key=True)  # Pokemon TCG API card ID
    name = db.Column(db.String(255), nullable=False)
    set_id = db.Column(db.String(255))
    set_name = db.Column(db.String(255))
    number = db.Column(db.String(50))
    rarity = db.Column(db.String(100))
    artist = db.Column(db.String(255))
    hp = db.Column(db.Integer)
    types = db.Column(db.Text)  # JSON array of types
    attacks = db.Column(db.Text)  # JSON array of attacks
    weaknesses = db.Column(db.Text)  # JSON array of weaknesses
    resistances = db.Column(db.Text)  # JSON array of resistances
    retreat_cost = db.Column(db.Integer)
    converted_energy_cost = db.Column(db.Integer)
    card_market_url = db.Column(db.Text)
    tcgplayer_url = db.Column(db.Text)
    image_url_small = db.Column(db.Text)
    image_url_large = db.Column(db.Text)
    local_image_path = db.Column(db.Text)
    data_json = db.Column(db.Text)  # Complete JSON data from API
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<OfficialCard {self.name} ({self.set_name})>'

class PokemonSet(db.Model):
    """
    Database model to store Pokemon TCG set information.
    """
    id = db.Column(db.String(255), primary_key=True)  # Pokemon TCG API set ID
    name = db.Column(db.String(255), nullable=False)
    series = db.Column(db.String(255))
    printed_total = db.Column(db.Integer)
    total = db.Column(db.Integer)
    legalities = db.Column(db.Text)  # JSON object of format legalities
    ptcgo_code = db.Column(db.String(50))
    release_date = db.Column(db.String(50))
    updated_at = db.Column(db.String(50))
    symbol_url = db.Column(db.Text)
    logo_url = db.Column(db.Text)
    data_json = db.Column(db.Text)  # Complete JSON data from API
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<PokemonSet {self.name} ({self.id})>'
