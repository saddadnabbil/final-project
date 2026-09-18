# backend/src/app.py

import sys
import os

# Add src directory to Python path so imports work when running from backend/
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta
from functools import wraps
import jwt

# Import configuration
from config import config

# Import database
from models_db import db, bcrypt, User, TranslationHistory

# Import metrics calculation
from metrics import calculate_metrics

# Import helper functions for DRY principle
from app_helpers import (
    handle_options,
    validate_file_upload,
    get_user_from_token,
    save_translation_history,
    create_success_response,
    create_error_response
)

# Conditional import for ML models (Python 3.13 compatibility)
skip_models = os.getenv('SKIP_MODEL_LOADING', 'false').lower() == 'true'

if skip_models:
    print("Skipping model loading for testing - using mock models...")
    # Create mock classes/functions for testing
    class MockAudioVisualModel:
        def process_translation(self, *args, **kwargs):
            return {"transcription": "Mock transcription - models not available", "translation": "Mock translation - models not available"}
        
        def translate(self, text, **kwargs):
            return f"Mock translation: {text}"
    
    AudioVisualModel = MockAudioVisualModel
    extract_audio_from_video = lambda *args, **kwargs: b"mock_audio"
    extract_video_frame = lambda *args, **kwargs: b"mock_frame"
    cleanup_temp_video = lambda: None
    models_available = False
else:
    try:
        from models import AudioVisualModel
        from utils import extract_audio_from_video, extract_video_frame, cleanup_temp_video
        models_available = True
        print("ML models loaded successfully")
    except Exception as e:
        print(f"Warning: Could not import ML models: {e}")
        print("Running in authentication-only mode.")
        
        # Create mock classes/functions for testing
        class MockAudioVisualModel:
            def process_translation(self, *args, **kwargs):
                return {"transcription": "Mock transcription - models not available", "translation": "Mock translation - models not available"}
            
            def translate(self, text, **kwargs):
                return f"Mock translation: {text}"
        
        AudioVisualModel = MockAudioVisualModel
        extract_audio_from_video = lambda *args, **kwargs: b"mock_audio"
        extract_video_frame = lambda *args, **kwargs: b"mock_frame"
        cleanup_temp_video = lambda: None
        models_available = False

# Load .env from backend directory (one level up from src/)
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)
load_dotenv()  # Also try loading from current directory


app = Flask(__name__)

# CORS Configuration from config
print(f"[CONFIG] CORS enabled for origins: {config.ALLOWED_ORIGINS}")
CORS(app,
     origins=config.ALLOWED_ORIGINS,
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "Accept"],
     supports_credentials=True)

# Database configuration from config
if config.DATABASE_URI.startswith('sqlite:///'):
    # Ensure instance directory exists for SQLite
    db_path = config.DATABASE_URI.replace('sqlite:///', '')
    if not os.path.isabs(db_path):
        # Relative path - make it relative to backend directory
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'instance', db_path)
        app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    else:
        app.config['SQLALCHEMY_DATABASE_URI'] = config.DATABASE_URI
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = config.DATABASE_URI

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['JWT_SECRET_KEY'] = config.JWT_SECRET_KEY

# File upload limits from config
app.config['MAX_CONTENT_LENGTH'] = config.MAX_VIDEO_SIZE  # Use max video size as overall limit

# JWT Configuration from config
app.config['JWT_ACCESS_TOKEN_EXPIRE'] = timedelta(days=config.JWT_EXPIRATION_DAYS)

print(f"[CONFIG] Environment: {config.FLASK_ENV}")
print(f"[CONFIG] Debug mode: {config.DEBUG_MODE}")
print(f"[CONFIG] Max audio size: {config.MAX_AUDIO_SIZE_MB}MB")
print(f"[CONFIG] Max video size: {config.MAX_VIDEO_SIZE_MB}MB")
print("Initializing application...")

# Initialize Bcrypt
bcrypt.init_app(app)

if models_available:
    print("Initializing AI Model...")
    av_model = AudioVisualModel()
    print("AI Model ready.")
else:
    print("Running without AI models - authentication features available.")
    av_model = AudioVisualModel()  # This will use our mock class

# Initialize database
db.init_app(app)

# Create database tables
with app.app_context():
    db.create_all()
    print("Database tables created/verified.")

# Request logging middleware
@app.before_request
def log_request_info():
    """Log all incoming requests"""
    # Skip health check endpoints to reduce spam
    if request.path not in ['/', '/api/health']:
        print(f"\n[REQUEST] {request.method} {request.path}", flush=True)
        if request.args:
            print(f"[REQUEST] Query params: {dict(request.args)}", flush=True)
        if request.content_type and 'application/json' in request.content_type:
            try:
                print(f"[REQUEST] JSON body: {request.get_json()}", flush=True)
            except:
                pass

@app.after_request
def log_response_info(response):
    """Log all responses"""
    # Skip health check endpoints to reduce spam
    if request.path not in ['/', '/api/health']:
        print(f"[RESPONSE] {response.status_code} {request.method} {request.path}\n", flush=True)
    return response

# JWT protected decorator
def jwt_required_custom(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Handle OPTIONS request for CORS preflight
        if request.method == 'OPTIONS':
            return jsonify({}), 200

        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid token'}), 401

        token = auth_header.split(' ')[1]
        try:
            data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            current_user_id = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401

        # Add current_user to request context
        request.current_user_id = current_user_id
        return f(*args, **kwargs)
    return decorated_function

# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring and tunnel verification"""
    return jsonify({
        'status': 'healthy',
        'service': 'Skripsi Sunda Translation API',
        'version': '1.0.0',
        'models_loaded': models_available,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/health', methods=['GET'])
def api_health():
    """Detailed health check with system info"""
    return jsonify({
        'status': 'healthy',
        'service': 'Skripsi Sunda Translation API',
        'version': '1.0.0',
        'environment': config.FLASK_ENV,
        'models': {
            'whisper': {
                'available': models_available,
                'type': config.WHISPER_MODEL_TYPE,
                'device': config.get_device('whisper')
            },
            'nllb': {
                'available': models_available,
                'name': config.NLLB_MODEL_NAME.split('/')[-1],
                'device': config.get_device('nllb')
            },
            'visual': {
                'enabled': config.USE_VISUAL_AUGMENTATION,
                'available': models_available
            }
        },
        'endpoints': [
            '/api/register',
            '/api/login',
            '/api/logout',
            '/api/profile',
            '/api/upload-audio',
            '/api/upload-video',
            '/api/translation-history',
            '/api/config'
        ],
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/config', methods=['GET'])
def api_config():
    """Get current configuration (for debugging)"""
    return jsonify({
        'status': 'success',
        'config': config.to_dict()
    }), 200


# ============================================
# MODEL SELECTION ENDPOINTS
# ============================================

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available ASR model variants"""
    try:
        from whisper_finetuned import get_model_manager
        manager = get_model_manager()

        models = manager.get_available_models()
        current = manager.get_current_variant() or config.WHISPER_MODEL_VARIANT

        return jsonify({
            'status': 'success',
            'models': models,
            'current': current
        }), 200
    except Exception as e:
        print(f'[API] Error getting models: {e}')
        return jsonify({
            'status': 'success',
            'models': [
                {'id': 'cp1500', 'name': 'Fine-tuned Whisper (1500 steps)', 'status': 'unknown'},
                {'id': 'cp4500', 'name': 'Fine-tuned Whisper (4500 steps)', 'status': 'unknown'},
                {'id': 'google', 'name': 'Google Speech API', 'status': 'available'}
            ],
            'current': config.WHISPER_MODEL_VARIANT
        }), 200


@app.route('/api/models/select', methods=['POST', 'OPTIONS'])
def select_model():
    """Select which ASR model variant to use"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = request.get_json()
        if not data or not data.get('model'):
            return jsonify({'error': 'Model ID required'}), 400

        model_id = data['model']
        valid_models = ['cp1500', 'cp4500', 'google']

        if model_id not in valid_models:
            return jsonify({'error': f'Invalid model. Choose from: {valid_models}'}), 400

        from whisper_finetuned import get_model_manager
        manager = get_model_manager()

        # Pre-load the model (except for google which is API-based)
        if model_id != 'google':
            try:
                manager.load_model(model_id, device=config.get_device('whisper'))
            except FileNotFoundError as e:
                return jsonify({'error': f'Model not found: {e}'}), 404
        else:
            manager._current_variant = 'google'

        return jsonify({
            'status': 'success',
            'message': f'Model changed to {model_id}',
            'current': model_id
        }), 200

    except Exception as e:
        print(f'[API] Error selecting model: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# Authentication endpoints
@app.route('/api/register', methods=['POST', 'OPTIONS'])
def register():
    """Register a new user"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = request.get_json()

        if not data or not data.get('username') or not data.get('password'):
            return jsonify({'error': 'Username and password required'}), 400

        username = data['username'].strip()
        password = data['password']
        email = data.get('email', '').strip()

        # Validate email
        if not email:
            return jsonify({'error': 'Email is required'}), 400

        if len(username) < 3 or len(password) < 6:
            return jsonify({'error': 'Username must be at least 3 characters, password at least 6 characters'}), 400

        # Check if user already exists
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 409

        # Check if email already exists
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already exists'}), 409

        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        return jsonify({
            'message': 'User registered successfully',
            'user_id': user.id
        }), 201

    except Exception as e:
        db.session.rollback()
        print(f'Registration error: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/login', methods=['POST', 'OPTIONS'])
def login():
    """Authenticate user and return JWT token"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200
        
    try:
        data = request.get_json()
        
        if not data or not data.get('username') or not data.get('password'):
            return jsonify({'error': 'Username and password required'}), 400
        
        username = data['username'].strip()
        password = data['password']
        
        # Find user
        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            return jsonify({'error': 'Invalid username or password'}), 401
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': user.id,
            'username': user.username,
            'exp': datetime.utcnow() + timedelta(days=7)
        }, app.config['JWT_SECRET_KEY'], algorithm='HS256')

        # Get user stats
        stats = TranslationHistory.get_user_stats(user.id)

        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'created_at': user.created_at.isoformat(),
                'stats': stats
            }
        }), 200
        
    except Exception as e:
        print(f'Login error: {e}')
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/logout', methods=['POST', 'OPTIONS'])
@jwt_required_custom
def logout():
    """Logout user (client should discard token)"""
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/api/auth/google', methods=['POST', 'OPTIONS'])
def google_auth():
    """Authenticate or register user with Google OAuth"""
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        data = request.get_json()

        if not data or not data.get('credential'):
            return jsonify({'error': 'Google credential required'}), 400

        # Verify Google ID token
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests

        # Your Google Client ID (should match frontend)
        GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')

        if not GOOGLE_CLIENT_ID:
            return jsonify({'error': 'Google OAuth not configured on server'}), 500

        try:
            # Verify the token
            idinfo = id_token.verify_oauth2_token(
                data['credential'],
                google_requests.Request(),
                GOOGLE_CLIENT_ID
            )

            # Get user info from token
            google_id = idinfo['sub']
            email = idinfo.get('email', '')
            name = idinfo.get('name', '')
            picture = idinfo.get('picture', '')

            # Check if user exists by google_id or email
            user = User.query.filter_by(google_id=google_id).first()

            if not user:
                # Check if email already exists (user registered with email/password)
                user = User.query.filter_by(email=email).first()
                if user:
                    # Link Google account to existing user
                    user.google_id = google_id
                    user.avatar_url = picture
                    if user.auth_provider == 'local':
                        user.auth_provider = 'both'  # User has both local and Google
                    db.session.commit()
                else:
                    # Create new user
                    # Generate unique username from name or email
                    base_username = name.lower().replace(' ', '_')[:20] if name else email.split('@')[0]
                    username = base_username
                    counter = 1
                    while User.query.filter_by(username=username).first():
                        username = f"{base_username}_{counter}"
                        counter += 1

                    user = User(
                        username=username,
                        email=email,
                        google_id=google_id,
                        avatar_url=picture,
                        auth_provider='google'
                    )
                    db.session.add(user)
                    db.session.commit()

            # Generate JWT token
            token = jwt.encode({
                'user_id': user.id,
                'username': user.username,
                'exp': datetime.utcnow() + timedelta(days=7)
            }, app.config['JWT_SECRET_KEY'], algorithm='HS256')

            # Get user stats
            stats = TranslationHistory.get_user_stats(user.id)

            return jsonify({
                'message': 'Google authentication successful',
                'token': token,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'created_at': user.created_at.isoformat(),
                    'avatar_url': user.avatar_url,
                    'auth_provider': user.auth_provider,
                    'stats': stats
                }
            }), 200

        except ValueError as e:
            print(f'Google token verification failed: {e}')
            return jsonify({'error': 'Invalid Google token'}), 401

    except Exception as e:
        db.session.rollback()
        print(f'Google auth error: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Google authentication failed'}), 500

@app.route('/api/profile', methods=['GET', 'OPTIONS'])
@jwt_required_custom
def get_profile():
    """Get current user profile"""
    try:
        user = db.session.get(User, request.current_user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get user stats from TranslationHistory
        stats = TranslationHistory.get_user_stats(user.id)

        return jsonify({
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'created_at': user.created_at.isoformat(),
                'stats': stats
            }
        }), 200

    except Exception as e:
        print(f'Profile error: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to get profile'}), 500

@app.route('/api/upload-audio', methods=['POST', 'OPTIONS'])
def upload_audio():
    """
    Endpoint untuk upload file audio (audio-only mode).
    Hanya untuk file upload, TIDAK untuk recording.
    """
    if request.method == 'OPTIONS':
        return handle_options()

    print('[HTTP] Menerima request upload-audio')
    processing_start_time = time.time()

    try:
        # Validate file upload
        audio_bytes, filename, error_response = validate_file_upload('audio', 16, 'audio')
        if error_response:
            return error_response

        print(f'[HTTP] File audio diterima: {filename}, size: {len(audio_bytes)} bytes')

        # Process audio-only (no visual), with auto-translate
        result = av_model.process_translation(audio_bytes, video_frame_bytes=None, use_visual=False, auto_translate=True)

        if not result:
            return create_error_response('Failed to process audio', 500)

        # Check if result is dict (with translation) or string (transcription only)
        if isinstance(result, dict):
            transcription = result.get('transcription', '')
            translation = result.get('translation', '')
            print(f'[HTTP] Hasil transkripsi: {transcription}')
            print(f'[HTTP] Hasil terjemahan: {translation}')

            # Calculate metrics
            metrics = calculate_metrics(
                transcription=transcription,
                translation=translation,
                audio_bytes=audio_bytes,
                processing_start_time=processing_start_time
            )
            print(f'[HTTP] Metrics calculated: {metrics}')

            # Save to database if user is authenticated
            user_id = get_user_from_token(app)
            if user_id:
                save_translation_history(user_id, 'audio-only', transcription, translation, metrics, content_type='audio')

            return create_success_response(transcription, translation, 'audio-only', metrics)
        else:
            # Fallback: jika hanya string (backward compatibility)
            print(f'[HTTP] Hasil transkripsi: {result}')

            # Save to database if user is authenticated
            user_id = get_user_from_token(app)
            if user_id:
                save_translation_history(user_id, 'audio-only', result, '', None, content_type='audio')

            return create_success_response(result, '', 'audio-only')

    except Exception as e:
        print(f'[HTTP] Error pada upload-audio: {e}')
        return create_error_response(str(e), 500)


@app.route('/api/upload-video', methods=['POST', 'OPTIONS'])
def upload_video():
    """
    Endpoint untuk upload file video (audio-visual mode).
    Ekstrak audio dari video dan proses dengan visual features.
    """
    if request.method == 'OPTIONS':
        return handle_options()

    print('[HTTP] Menerima request upload-video')
    processing_start_time = time.time()

    try:
        # Validate file upload
        video_bytes, filename, error_response = validate_file_upload('video', 100, 'video')
        if error_response:
            return error_response

        print(f'[HTTP] File video diterima: {filename}, size: {len(video_bytes)} bytes')

        # Extract audio from video
        print('[HTTP] Extracting audio from video...')
        audio_bytes = extract_audio_from_video(video_bytes)

        if not audio_bytes or len(audio_bytes) == 0:
            return create_error_response('Failed to extract audio from video', 500)

        print(f'[HTTP] Audio extracted: {len(audio_bytes)} bytes')

        # Extract video frame for visual processing
        print('[HTTP] Extracting video frame...')
        video_frame_bytes = extract_video_frame(video_bytes)

        # Validate video frame before attempting visual processing
        use_visual_mode = video_frame_bytes is not None and len(video_frame_bytes) > 100
        if not use_visual_mode:
            print('[HTTP] Video frame extraction failed or too small, using audio-only mode')

        # Process with audio-visual model
        result = av_model.process_translation(
            audio_bytes,
            video_frame_bytes=video_frame_bytes if use_visual_mode else None,
            use_visual=use_visual_mode,
            auto_translate=True
        )

        if not result:
            return create_error_response('Failed to process video', 500)

        # Determine actual mode used
        actual_mode = 'audio-visual' if use_visual_mode else 'audio-only'

        # Check if result is dict (with translation) or string (transcription only)
        if isinstance(result, dict):
            transcription = result.get('transcription', '')
            translation = result.get('translation', '')
            visual_augmentation = result.get('visual_augmentation', None)
            print(f'[HTTP] Hasil transkripsi: {transcription}')
            print(f'[HTTP] Hasil terjemahan: {translation}')
            if visual_augmentation:
                print(f'[HTTP] Visual augmentation data available: attention_weights={visual_augmentation.get("attention_weights", {})}')

            # Calculate metrics
            metrics = calculate_metrics(
                transcription=transcription,
                translation=translation,
                audio_bytes=audio_bytes,
                processing_start_time=processing_start_time
            )
            print(f'[HTTP] Metrics calculated: {metrics}')

            # Save to database if user is authenticated
            user_id = get_user_from_token(app)
            if user_id:
                save_translation_history(user_id, actual_mode, transcription, translation, metrics, content_type='video')

            return create_success_response(transcription, translation, actual_mode, metrics, visual_augmentation)
        else:
            # Fallback: jika hanya string (backward compatibility)
            print(f'[HTTP] Hasil transkripsi: {result}')

            # Save to database if user is authenticated
            user_id = get_user_from_token(app)
            if user_id:
                save_translation_history(user_id, actual_mode, result, '', None, content_type='video')

            return create_success_response(result, '', actual_mode)

    except Exception as e:
        print(f'[HTTP] Error pada upload-video: {e}')
        return create_error_response(str(e), 500)
    
    finally:
        # Clean up cached temp video file
        if models_available:
            try:
                cleanup_temp_video()
            except:
                pass


# Text translation endpoint disabled - app focuses on audio/video only
# @app.route('/translate-text', methods=['POST', 'OPTIONS'])
# def translate_text():
#     """
#     Endpoint untuk terjemahan teks Sunda ke Indonesia atau sebaliknya.
#     """
#     if request.method == 'OPTIONS':
#         return jsonify({}), 200
#
#     print('[HTTP] Menerima request translate-text')
#
#     try:
#         data = request.get_json()
#         if not data or 'text' not in data:
#             return jsonify({'error': 'No text provided'}), 400
#
#         text = data['text'].strip()
#         source_lang = data.get('source_lang', 'sun')  # Default Sunda
#         target_lang = data.get('target_lang', 'ind')  # Default Indonesia
#
#         if not text:
#             return jsonify({'error': 'Text is empty'}), 400
#
#         # Translate using the model
#         translated_text = av_model.translate(text, source_lang=source_lang, target_lang=target_lang)
#
#         print(f'[HTTP] Teks asli: {text}')
#         print(f'[HTTP] Terjemahan: {translated_text}')
#
#         # Save to database if user is authenticated
#         auth_header = request.headers.get('Authorization')
#         if auth_header and auth_header.startswith('Bearer '):
#             try:
#                 token = auth_header.split(' ')[1]
#                 data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
#                 user_id = data['user_id']
#
#                 # Save translation history
#                 history = TranslationHistory(
#                     user_id=user_id,
#                     content_type='text',
#                     original_text=text,
#                     translated_text=translated_text,
#                     mode='text'
#                 )
#                 db.session.add(history)
#                 db.session.commit()
#                 print(f'[DB] Translation history saved for user {user_id}')
#             except Exception as db_error:
#                 print(f'[DB] Failed to save history: {db_error}')
#                 db.session.rollback()
#
#         return jsonify({
#             'success': True,
#             'original_text': text,
#             'translated_text': translated_text,
#             'source_lang': source_lang,
#             'target_lang': target_lang
#         }), 200
#
#     except Exception as e:
#         print(f'[HTTP] Error pada translate-text: {e}')
#         return jsonify({'error': str(e)}), 500


@app.route('/api/translation-history', methods=['GET', 'OPTIONS'])
@jwt_required_custom
def get_translation_history():
    """Get user's translation history"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        
        # Validate pagination parameters
        if page < 1 or per_page < 1 or per_page > 50:
            return jsonify({'error': 'Invalid pagination parameters'}), 400
        
        # Get user's translation history
        histories = TranslationHistory.query.filter_by(user_id=request.current_user_id)\
            .order_by(TranslationHistory.created_at.desc())\
            .paginate(page=page, per_page=per_page, error_out=False)
        
        return jsonify({
            'success': True,
            'histories': [h.to_dict() for h in histories.items],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': histories.total,
                'pages': histories.pages,
                'has_next': histories.has_next,
                'has_prev': histories.has_prev
            }
        }), 200
        
    except Exception as e:
        print(f'[HTTP] Error getting translation history: {e}')
        return jsonify({'error': 'Failed to get translation history'}), 500


@app.route('/api/translation-history/<int:history_id>', methods=['DELETE', 'OPTIONS'])
@jwt_required_custom
def delete_translation_history(history_id):
    """Delete a specific translation history entry"""
    try:
        history = TranslationHistory.query.filter_by(
            id=history_id, 
            user_id=request.current_user_id
        ).first()
        
        if not history:
            return jsonify({'error': 'Translation history not found'}), 404
        
        db.session.delete(history)
        db.session.commit()
        
        return jsonify({'message': 'Translation history deleted successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        print(f'[HTTP] Error deleting translation history: {e}')
        return jsonify({'error': 'Failed to delete translation history'}), 500


if __name__ == '__main__':
    # WARNING: Development server only - DO NOT use in production!
    # For production, use: python run_production.py
    print("=" * 80)
    print("WARNING: Development Server")
    print("=" * 80)
    print("This is a development server. DO NOT use in production!")
    print("")
    print("For production deployment, use:")
    print("  python run_production.py")
    print("")
    print("Or with Gunicorn (Linux/Mac):")
    print("  gunicorn -c gunicorn.conf.py src.app:app")
    print("=" * 80)
    print("")
    print("Starting development server at http://0.0.0.0:5001")
    # threaded=False to avoid CUDA DLL loading issues in worker threads
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=False)
