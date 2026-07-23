import os
import json
import shutil
import platform
import subprocess
from datetime import datetime

import redis
import requests
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from apps.accounts.decorators import admin_required
from apps.accounts.forms import UserCreationForm, UserEditForm, PasswordResetForm

User = get_user_model()

FASTAPI_BASE = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


# ---------------------------------------------------------------------------
# Dashboard home
# ---------------------------------------------------------------------------

@admin_required
def dashboard_home(request):
    context = {
        "total_users": User.objects.count(),
        "hec_users": User.objects.filter(role="HEAD_OF_EXAM_CELL").count(),
        "admin_users": User.objects.filter(role="ADMIN").count(),
        "active_page": "home",
    }
    return render(request, "admin_panel/home.html", context)


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

@admin_required
def user_list(request):
    users = User.objects.order_by("-created_at")
    return render(request, "admin_panel/user_list.html", {"users": users, "active_page": "users"})


@admin_required
def user_create(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "User created successfully.")
            return redirect("admin_panel:user_list")
    else:
        form = UserCreationForm()
    return render(request, "admin_panel/user_form.html", {"form": form, "action": "Create", "active_page": "users"})


@admin_required
def user_edit(request, pk):
    user = get_object_or_404(User, pk=pk)
    if request.method == "POST":
        form = UserEditForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, "User updated.")
            return redirect("admin_panel:user_list")
    else:
        form = UserEditForm(instance=user)
    return render(request, "admin_panel/user_form.html", {"form": form, "action": "Edit", "user": user, "active_page": "users"})


@admin_required
@require_POST
def user_delete(request, pk):
    user = get_object_or_404(User, pk=pk)
    if user == request.user:
        messages.error(request, "You cannot delete your own account.")
    else:
        user.delete()
        messages.success(request, f"User {user.email} deleted.")
    return redirect("admin_panel:user_list")


@admin_required
def user_reset_password(request, pk):
    user = get_object_or_404(User, pk=pk)
    if request.method == "POST":
        form = PasswordResetForm(request.POST)
        if form.is_valid():
            user.set_password(form.cleaned_data["new_password"])
            user.save()
            messages.success(request, f"Password reset for {user.email}.")
            return redirect("admin_panel:user_list")
    else:
        form = PasswordResetForm()
    return render(request, "admin_panel/reset_password.html", {"form": form, "target_user": user, "active_page": "users"})


# ---------------------------------------------------------------------------
# Model weight management
# ---------------------------------------------------------------------------

ML_WEIGHTS_DIR = os.getenv("ML_WEIGHTS_DIR", "/app/ml/weights")


@admin_required
def model_list(request):
    weights = []
    if os.path.isdir(ML_WEIGHTS_DIR):
        for fname in sorted(os.listdir(ML_WEIGHTS_DIR)):
            fpath = os.path.join(ML_WEIGHTS_DIR, fname)
            if os.path.isfile(fpath):
                stat = os.stat(fpath)
                weights.append({
                    "name": fname,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                })
    return render(request, "admin_panel/model_list.html", {"weights": weights, "active_page": "models"})


@admin_required
def model_upload(request):
    if request.method == "POST" and request.FILES.get("weight_file"):
        f = request.FILES["weight_file"]
        allowed_exts = {".pt", ".pth", ".engine", ".onnx"}
        ext = os.path.splitext(f.name)[1].lower()
        if ext not in allowed_exts:
            messages.error(request, f"Invalid file type '{ext}'. Allowed: {', '.join(allowed_exts)}")
        else:
            os.makedirs(ML_WEIGHTS_DIR, exist_ok=True)
            dest = os.path.join(ML_WEIGHTS_DIR, f.name)
            with open(dest, "wb+") as dest_file:
                for chunk in f.chunks():
                    dest_file.write(chunk)
            messages.success(request, f"Model '{f.name}' uploaded successfully.")
        return redirect("admin_panel:model_list")
    return render(request, "admin_panel/model_upload.html", {"active_page": "models"})


# ---------------------------------------------------------------------------
# System health
# ---------------------------------------------------------------------------

def _get_redis_info():
    try:
        r = redis.from_url(REDIS_URL, socket_connect_timeout=2)
        info = r.info()
        queue_depth = r.llen("frame_queue") if r.exists("frame_queue") else 0
        return {
            "status": "ok",
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "N/A"),
            "queue_depth": queue_depth,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _get_fastapi_info():
    try:
        resp = requests.get(f"{FASTAPI_BASE}/api/health", timeout=3)
        return resp.json() if resp.ok else {"status": "error", "code": resp.status_code}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _get_gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        name, temp, util, mem_used, mem_total = [x.strip() for x in out.split(",")]
        return {
            "status": "ok",
            "name": name,
            "temperature": f"{temp}°C",
            "utilization": f"{util}%",
            "memory": f"{mem_used} / {mem_total} MiB",
        }
    except FileNotFoundError:
        return {"status": "unavailable", "error": "nvidia-smi not found"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@admin_required
def system_health(request):
    context = {
        "active_page": "health",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "timestamp": timezone.now(),
    }
    return render(request, "admin_panel/health.html", context)


@admin_required
def health_api(request):
    """HTMX polling endpoint — returns JSON of live metrics."""
    data = {
        "redis": _get_redis_info(),
        "fastapi": _get_fastapi_info(),
        "gpu": _get_gpu_info(),
        "timestamp": timezone.now().isoformat(),
    }
    return JsonResponse(data)
