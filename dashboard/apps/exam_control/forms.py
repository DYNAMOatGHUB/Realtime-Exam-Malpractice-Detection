"""Forms for exam_control app."""
from django import forms
from apps.exam_control.models import LHMapping, LectureHall, Invigilator, VideoAnalysisJob



class LHMappingForm(forms.ModelForm):
    lh_code = forms.CharField(label="Lecture Hall Code", max_length=50, widget=forms.TextInput(attrs={"placeholder": "e.g. LH-101", "class": "form-input"}))
    lh_name = forms.CharField(label="Lecture Hall Name", max_length=255, widget=forms.TextInput(attrs={"placeholder": "e.g. Main Auditorium", "class": "form-input"}))
    invigilator_name = forms.CharField(label="Invigilator Name", max_length=255, widget=forms.TextInput(attrs={"placeholder": "e.g. John Doe", "class": "form-input"}))
    invigilator_email = forms.EmailField(label="Invigilator Email", widget=forms.EmailInput(attrs={"placeholder": "e.g. john@exam.edu", "class": "form-input"}))

    class Meta:
        model = LHMapping
        fields = ["lh_code", "lh_name", "invigilator_name", "invigilator_email", "exam_session_label"]
        widgets = {
            "exam_session_label": forms.TextInput(attrs={"placeholder": "Mid-Sem Jan 2026", "class": "form-input"}),
        }


class InvigilatorForm(forms.ModelForm):
    class Meta:
        model = Invigilator
        fields = ["full_name", "email", "department", "phone"]
        widgets = {
            "full_name": forms.TextInput(attrs={"class": "form-input"}),
            "email": forms.EmailInput(attrs={"class": "form-input"}),
            "department": forms.TextInput(attrs={"class": "form-input"}),
            "phone": forms.TextInput(attrs={"class": "form-input"}),
        }


class StartStreamForm(forms.Form):
    camera_id = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={"placeholder": "cam_lh01", "class": "form-input"}),
    )
    rtsp_url = forms.CharField(
        max_length=512,
        widget=forms.TextInput(attrs={"placeholder": "rtsp://... or /path/to/video.mp4", "class": "form-input"}),
    )
    target_fps = forms.IntegerField(
        initial=8, min_value=1, max_value=30,
        widget=forms.NumberInput(attrs={"class": "form-input"}),
    )


class VideoAnalysisJobForm(forms.ModelForm):
    class Meta:
        model = VideoAnalysisJob
        fields = ["mapping", "video_file"]
        widgets = {
            "mapping": forms.Select(attrs={"class": "form-select"}),
            "video_file": forms.FileInput(attrs={"class": "form-input"}),
        }

