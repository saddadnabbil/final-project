"""
Production server runner for Sundanese Translator Backend.
Uses Waitress (Windows-compatible) or Gunicorn (Linux/Mac).

Usage:
    python run_production.py
"""
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_production_server():
    """Run production WSGI server."""
    import platform
    from src.app import app

    # Server configuration
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5001))
    workers = int(os.getenv('WORKERS', 2))  # Number of worker processes

    print(f"[PRODUCTION] Starting production server...")
    print(f"[PRODUCTION] Platform: {platform.system()}")
    print(f"[PRODUCTION] Host: {host}:{port}")
    print(f"[PRODUCTION] Workers: {workers}")

    # Use Waitress for Windows, Gunicorn for Linux/Mac
    if platform.system() == 'Windows':
        try:
            from waitress import serve
            print("[PRODUCTION] Using Waitress WSGI server (Windows)")
            print(f"[PRODUCTION] Server running at http://{host}:{port}")
            print("[PRODUCTION] Press Ctrl+C to stop")

            serve(
                app,
                host=host,
                port=port,
                threads=workers * 2,  # Waitress uses threads instead of processes
                channel_timeout=120,  # Increased timeout for ML processing
                cleanup_interval=30,
                url_scheme='http'
            )
        except ImportError:
            print("[ERROR] Waitress not installed. Install with: pip install waitress")
            sys.exit(1)
    else:
        # For Linux/Mac, recommend using Gunicorn via command line
        print("[INFO] On Linux/Mac, we recommend running with Gunicorn:")
        print(f"[INFO] gunicorn --bind {host}:{port} --workers {workers} --timeout 120 --chdir src app:app")
        print("\n[INFO] Falling back to Waitress...")

        try:
            from waitress import serve
            print(f"[PRODUCTION] Server running at http://{host}:{port}")
            serve(
                app,
                host=host,
                port=port,
                threads=workers * 2,
                channel_timeout=120,
                cleanup_interval=30
            )
        except ImportError:
            print("[ERROR] Please install waitress or gunicorn:")
            print("  pip install waitress  # or")
            print("  pip install gunicorn")
            sys.exit(1)

if __name__ == '__main__':
    run_production_server()
