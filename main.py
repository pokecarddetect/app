import os
from app import app

if __name__ == '__main__':
    # Get port from environment variable (for Railway, Heroku, etc.) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Run the Flask application on dynamic port for external access
    app.run(host='0.0.0.0', port=port, debug=False)
