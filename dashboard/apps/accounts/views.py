"""
Accounts views: login, logout, profile.
"""
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages
from django.utils import timezone

from apps.accounts.forms import LoginForm


def login_view(request):
    if request.user.is_authenticated:
        return redirect("exam_control:dashboard")

    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["email"]
            password = form.cleaned_data["password"]
            user = authenticate(request, username=email, password=password)
            if user is not None:
                login(request, user)
                user.last_login_at = timezone.now()
                user.save(update_fields=["last_login_at"])
                messages.success(request, f"Welcome back, {user.full_name}!")
                next_url = request.GET.get("next", "exam_control:dashboard")
                return redirect(next_url)
            else:
                messages.error(request, "Invalid email or password.")
    else:
        form = LoginForm()

    return render(request, "accounts/login.html", {"form": form})


def logout_view(request):
    logout(request)
    messages.info(request, "You have been signed out.")
    return redirect("accounts:login")


@login_required
def profile_view(request):
    return render(request, "accounts/profile.html", {"user": request.user})
