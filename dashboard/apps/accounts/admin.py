"""Accounts admin registration."""
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from apps.accounts.models import User


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ("email", "full_name", "role", "is_active", "created_at")
    list_filter = ("role", "is_active")
    search_fields = ("email", "full_name")
    ordering = ("-created_at",)
    fieldsets = (
        (None, {"fields": ("email", "password")}),
        ("Personal info", {"fields": ("full_name", "role")}),
        ("Permissions", {"fields": ("is_active", "is_staff", "is_superuser", "groups", "user_permissions")}),
        ("Dates", {"fields": ("last_login", "last_login_at")}),
    )
    add_fieldsets = (
        (None, {
            "classes": ("wide",),
            "fields": ("email", "full_name", "role", "password1", "password2"),
        }),
    )
