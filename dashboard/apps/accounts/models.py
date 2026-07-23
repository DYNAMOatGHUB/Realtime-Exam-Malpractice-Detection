"""
Custom User model with RBAC roles: ADMIN and HEAD_OF_EXAM_CELL.
"""
from __future__ import annotations

from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models


class Role(models.TextChoices):
    ADMIN = "ADMIN", "System Administrator"
    HEAD_OF_EXAM_CELL = "HEAD_OF_EXAM_CELL", "Head of Exam Cell"


class UserManager(BaseUserManager):
    def create_user(self, email: str, password: str, **extra_fields):
        if not email:
            raise ValueError("Email is required")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email: str, password: str, **extra_fields):
        extra_fields.setdefault("role", Role.ADMIN)
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        return self.create_user(email, password, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(unique=True, db_index=True)
    full_name = models.CharField(max_length=255)
    role = models.CharField(max_length=30, choices=Role.choices, default=Role.HEAD_OF_EXAM_CELL)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    last_login_at = models.DateTimeField(null=True, blank=True)

    objects = UserManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["full_name"]

    class Meta:
        verbose_name = "User"
        verbose_name_plural = "Users"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.full_name} ({self.role})"

    @property
    def is_admin(self) -> bool:
        return self.role == Role.ADMIN

    @property
    def is_hec(self) -> bool:
        return self.role == Role.HEAD_OF_EXAM_CELL
