import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from . import services


@require_GET
def index(request):
    return render(request, "rag/index.html")


@require_GET
def healthz(request):
    ok = services.ready()
    return JsonResponse({"ready": ok}, status=200 if ok else 503)


@csrf_exempt          # JSON API, no cookies/session involved
@require_POST
def ask(request):
    try:
        question = (json.loads(request.body or b"{}").get("question") or "").strip()
    except ValueError:
        return JsonResponse({"error": "invalid JSON"}, status=400)
    if not question:
        return JsonResponse({"error": "question is required"}, status=400)
    if len(question) > 1000:
        return JsonResponse({"error": "question too long"}, status=400)
    try:
        return JsonResponse(services.answer_question(question), json_dumps_params={"ensure_ascii": False})
    except Exception as e:                      # budget exceeded, API failure, ...
        return JsonResponse({"error": f"{type(e).__name__}: {e}"}, status=500)
