"""
Database Models
Defines User and TranslationHistory models for authentication and history tracking
"""

from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from datetime import datetime
import json

db = SQLAlchemy()
bcrypt = Bcrypt()


class User(db.Model):
    """User model for authentication"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=True)  # Nullable for Google OAuth users
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Google OAuth fields
    google_id = db.Column(db.String(255), unique=True, nullable=True)
    avatar_url = db.Column(db.String(500), nullable=True)
    auth_provider = db.Column(db.String(20), default='local')  # 'local' or 'google'

    # Relationship with translation history
    translations = db.relationship('TranslationHistory', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        """Verify password"""
        if not self.password_hash:
            return False  # Google OAuth users don't have passwords
        return bcrypt.check_password_hash(self.password_hash, password)

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'total_translations': len(self.translations),
            'avatar_url': self.avatar_url,
            'auth_provider': self.auth_provider
        }


class TranslationHistory(db.Model):
    """Translation history model"""
    __tablename__ = 'translation_history'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Original content info
    content_type = db.Column(db.String(50), nullable=False)  # 'text', 'audio', 'video'
    original_filename = db.Column(db.String(255), nullable=True)
    file_size = db.Column(db.Integer, nullable=True)  # in bytes

    # Content
    original_text = db.Column(db.Text, nullable=False)
    translated_text = db.Column(db.Text, nullable=False)

    # Processing info
    processing_time = db.Column(db.Float, nullable=True)  # in seconds
    audio_duration = db.Column(db.Float, nullable=True)  # in seconds
    mode = db.Column(db.String(50), nullable=True)  # 'text', 'audio-only', 'audio-visual'

    # Metrics stored as JSON string
    metrics_json = db.Column(db.Text, nullable=True)

    # Timestamp
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary"""
        # Parse metrics_json if available
        metrics = None
        if self.metrics_json:
            try:
                metrics = json.loads(self.metrics_json)
            except (json.JSONDecodeError, TypeError):
                metrics = None

        return {
            'id': self.id,
            'user_id': self.user_id,
            'content_type': self.content_type,
            'original_filename': self.original_filename,
            'file_size': self.file_size,
            'original_text': self.original_text,
            'translated_text': self.translated_text,
            'processing_time': self.processing_time,
            'audio_duration': self.audio_duration,
            'mode': self.mode,
            'metrics': metrics,
            'created_at': self.created_at.isoformat()
        }

    @staticmethod
    def get_user_stats(user_id):
        """Get user translation statistics"""
        histories = TranslationHistory.query.filter_by(user_id=user_id).all()

        if not histories:
            return {
                'total_translations': 0,
                'avg_processing_time': 0,
                'total_characters_translated': 0,
                'recent_translations_7days': 0
            }

        total = len(histories)
        avg_time = sum(h.processing_time or 0 for h in histories) / total
        total_chars = sum(len(h.translated_text) for h in histories)

        # Count translations in last 7 days
        from datetime import datetime, timedelta
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_count = TranslationHistory.query.filter_by(user_id=user_id).filter(
            TranslationHistory.created_at >= week_ago
        ).count()

        return {
            'total_translations': total,
            'avg_processing_time': round(avg_time, 3),
            'total_characters_translated': total_chars,
            'recent_translations_7days': recent_count
        }