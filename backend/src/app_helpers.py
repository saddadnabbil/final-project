"""
Helper functions for Flask app (DRY principle)
Extract common patterns to reduce code duplication
"""

from flask import request, jsonify
import jwt
import json
import time
from metrics import calculate_metrics
from models_db import db, TranslationHistory


def handle_options():
    """Handle OPTIONS request for CORS"""
    return jsonify({}), 200


def validate_file_upload(file_key, max_size_mb, file_type="file"):
    """
    Validate file upload request.

    Args:
        file_key: Key name in request.files
        max_size_mb: Maximum file size in MB
        file_type: "audio" or "video" for error messages

    Returns:
        tuple: (file_bytes, filename, error_response)
               error_response is None if validation passes
    """
    # Check if file exists
    if file_key not in request.files:
        return None, None, (jsonify({'error': f'No {file_type} file provided'}), 400)

    file = request.files[file_key]

    # Check if filename is empty
    if file.filename == '':
        return None, None, (jsonify({'error': 'No file selected'}), 400)

    # Read file
    file_bytes = file.read()
    print(f'[HTTP] File {file_type} diterima: {file.filename}, size: {len(file_bytes)} bytes')

    # Validate size
    if len(file_bytes) == 0:
        return None, None, (jsonify({'error': f'{file_type.capitalize()} file is empty'}), 400)

    max_bytes = max_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        return None, None, (jsonify({'error': f'{file_type.capitalize()} file too large (max {max_size_mb}MB)'}), 400)

    return file_bytes, file.filename, None


def get_user_from_token(app):
    """
    Extract user_id from JWT token in Authorization header.

    Args:
        app: Flask app instance (for JWT_SECRET_KEY)

    Returns:
        int or None: user_id if valid token, None otherwise
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None

    try:
        token = auth_header.split(' ')[1]
        data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
        return data.get('user_id')
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, KeyError):
        return None


def save_translation_history(user_id, mode, transcription, translation, metrics, content_type='audio'):
    """
    Save translation to database history.

    Args:
        user_id: User ID
        mode: "audio-only" or "audio-visual"
        transcription: Transcription text
        translation: Translation text
        metrics: Metrics dict (can be None)
        content_type: "audio" or "video"

    Returns:
        bool: True if saved successfully
    """
    try:
        # Serialize metrics to JSON if available
        metrics_json = None
        if metrics:
            metrics_json = json.dumps(metrics)

        history = TranslationHistory(
            user_id=user_id,
            content_type=content_type,
            original_text=transcription,
            translated_text=translation,
            processing_time=metrics.get('processing_time') if metrics else None,
            audio_duration=metrics.get('audio_duration') if metrics else None,
            mode=mode,
            metrics_json=metrics_json
        )
        db.session.add(history)
        db.session.commit()
        print(f'[DB] Translation history saved for user {user_id}')
        return True
    except Exception as e:
        print(f'[DB] Failed to save history: {e}')
        db.session.rollback()
        return False


def create_success_response(transcription, translation, mode, metrics=None, visual_augmentation=None):
    """
    Create standardized success response.

    Args:
        transcription: Transcription text
        translation: Translation text
        mode: "audio-only" or "audio-visual"
        metrics: Metrics dict (optional)
        visual_augmentation: Visual augmentation data (optional, for audio-visual mode)

    Returns:
        tuple: (response_dict, status_code)
    """
    response = {
        'success': True,
        'transcription': transcription,
        'translation': translation,
        'mode': mode
    }

    if metrics is not None:
        response['metrics'] = metrics

    if visual_augmentation is not None:
        response['visual_augmentation'] = visual_augmentation

    return jsonify(response), 200


def create_error_response(message, status_code=500):
    """
    Create standardized error response.

    Args:
        message: Error message
        status_code: HTTP status code

    Returns:
        tuple: (response_dict, status_code)
    """
    return jsonify({'error': message}), status_code
