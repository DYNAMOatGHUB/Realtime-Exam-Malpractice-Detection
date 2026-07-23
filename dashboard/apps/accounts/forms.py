"""Forms for accounts app."""
from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError

User = get_user_model()


class LoginForm(forms.Form):
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={"placeholder": "admin@university.edu", "autocomplete": "email"})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={"placeholder": "Password", "autocomplete": "current-password"})
    )


class UserCreationForm(forms.ModelForm):
    """Admin-side user creation form."""
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={"placeholder": "Password"}),
        validators=[validate_password],
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={"placeholder": "Confirm password"}),
    )

    class Meta:
        model = User
        fields = ["email", "full_name", "role"]
        widgets = {
            "email": forms.EmailInput(attrs={"placeholder": "email@university.edu"}),
            "full_name": forms.TextInput(attrs={"placeholder": "Full name"}),
        }

    def clean(self):
        cleaned = super().clean()
        p1 = cleaned.get("password")
        p2 = cleaned.get("confirm_password")
        if p1 and p2 and p1 != p2:
            raise ValidationError("Passwords do not match.")
        return cleaned

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password"])
        if commit:
            user.save()
        return user


class UserEditForm(forms.ModelForm):
    """Admin-side user edit form (no password field)."""

    class Meta:
        model = User
        fields = ["email", "full_name", "role", "is_active"]
        widgets = {
            "email": forms.EmailInput(attrs={"placeholder": "email@university.edu"}),
            "full_name": forms.TextInput(attrs={"placeholder": "Full name"}),
        }


class PasswordResetForm(forms.Form):
    """Admin-side password reset for any user."""
    new_password = forms.CharField(
        widget=forms.PasswordInput(attrs={"placeholder": "New password"}),
        validators=[validate_password],
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={"placeholder": "Confirm new password"}),
    )

    def clean(self):
        cleaned = super().clean()
        p1 = cleaned.get("new_password")
        p2 = cleaned.get("confirm_password")
        if p1 and p2 and p1 != p2:
            raise ValidationError("Passwords do not match.")
        return cleaned
