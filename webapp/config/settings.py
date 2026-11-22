import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = Path(os.environ.get("ARROUTE_ROOT", BASE_DIR.parent))
sys.path.insert(0, str(REPO_ROOT))          # makes `import arroute` work

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-change-me")
DEBUG = os.environ.get("DJANGO_DEBUG", "0") == "1"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",")

INSTALLED_APPS = ["django.contrib.staticfiles", "rag"]
MIDDLEWARE = ["django.middleware.security.SecurityMiddleware", "django.middleware.common.CommonMiddleware",
              "django.middleware.csrf.CsrfViewMiddleware"]
ROOT_URLCONF = "config.urls"
TEMPLATES = [{"BACKEND": "django.template.backends.django.DjangoTemplates", "APP_DIRS": True,
              "OPTIONS": {"context_processors": ["django.template.context_processors.request"]}}]
WSGI_APPLICATION = "config.wsgi.application"
DATABASES = {}
USE_TZ = True
STATIC_URL = "/static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# RAG settings
MODELS_DIR = os.environ.get("ARROUTE_MODELS", str(REPO_ROOT / "models"))
CORPUS_DIR = os.environ.get("ARROUTE_CORPUS", str(REPO_ROOT / "data" / "corpus"))
BUDGET_USD = float(os.environ.get("ARROUTE_BUDGET_USD", "5"))
LLM_MODEL = os.environ.get("ARROUTE_LLM_MODEL", "gpt-4.1-mini")
