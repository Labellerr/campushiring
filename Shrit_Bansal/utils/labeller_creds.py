
import os

def get_labeller_credentials():
    return {
        "api_key": os.getenv("Labeller_API_KEY"),
        "api_secret": os.getenv("Labeller_API_SECRET"),
        "client_id": os.getenv("Labeller_CLIENT_ID"),
        "project_id": os.getenv("Labeller_PROJECT_ID"),
        "email": os.getenv("Labeller_EMAIL"),
    }
