import os
import threading

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
application = get_wsgi_application()

if os.environ.get("RAG_WARMUP", "1") == "1":     # load the models in the background; /healthz turns 200 when done
    from rag import services
    threading.Thread(target=services.load, daemon=True).start()
