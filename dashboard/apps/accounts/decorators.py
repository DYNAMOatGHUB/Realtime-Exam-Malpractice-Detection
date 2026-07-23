"""
Role-based access decorators for views.
"""
from functools import wraps
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied


def admin_required(view_func):
    """Restrict view to ADMIN role only."""
    @wraps(view_func)
    @login_required
    def wrapper(request, *args, **kwargs):
        if not request.user.is_admin:
            raise PermissionDenied
        return view_func(request, *args, **kwargs)
    return wrapper


def hec_or_admin_required(view_func):
    """Allow HEC and ADMIN roles (most operational views)."""
    @wraps(view_func)
    @login_required
    def wrapper(request, *args, **kwargs):
        if not (request.user.is_hec or request.user.is_admin):
            raise PermissionDenied
        return view_func(request, *args, **kwargs)
    return wrapper
