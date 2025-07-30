"""
WSGI entry point for deployment platforms like Heroku, Railway, etc.
"""
from main import app

if __name__ == "__main__":
    app.run()