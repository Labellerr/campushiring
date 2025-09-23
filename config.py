import os

class Config:
    # Generate a good secret key
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change-in-production'
    
    # SQLAlchemy settings
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///bunny_journal.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
