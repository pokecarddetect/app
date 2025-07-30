#!/bin/bash
# Quick start script for Pokemon Card Authentication App

echo "🃏 Pokémon Card Authentication App - Quick Start"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -e .

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "⚠️  WARNING: DATABASE_URL environment variable not set"
    echo "   Using SQLite as fallback (not recommended for production)"
    export DATABASE_URL="sqlite:///pokemon_cards.db"
fi

# Check if SESSION_SECRET is set
if [ -z "$SESSION_SECRET" ]; then
    echo "⚠️  WARNING: SESSION_SECRET not set, using default (change in production!)"
    export SESSION_SECRET="dev-secret-key-change-in-production"
fi

# Start the application
echo "🚀 Starting Pokemon Card Authentication App..."
echo "   Access the app at: http://localhost:5000"
echo "   Press Ctrl+C to stop"
echo ""

python main.py