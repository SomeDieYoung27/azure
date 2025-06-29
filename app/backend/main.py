import os

from app import create_app
from load_azd_env import load_azd_env

RUNNING_ON_AZURE = os.getenv("WEBSITE_HOSTNAME") is not None or os.getenv("RUNNING_IN_PRODUCTION") is not None

if not RUNNING_ON_AZURE:
    load_azd_env()

app = create_app()