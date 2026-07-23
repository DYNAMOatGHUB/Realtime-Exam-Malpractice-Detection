"""Models package — import all models here so SQLAlchemy discovers them."""
from app.models.user import User, UserRole
from app.models.lh_mapping import Invigilator, LectureHall, LHMapping
from app.models.detection import DetectionEvent, MalpracticeClass, TriageStatus
from app.models.alert import AlertLog, AlertStatus

__all__ = [
    "User", "UserRole",
    "Invigilator", "LectureHall", "LHMapping",
    "DetectionEvent", "MalpracticeClass", "TriageStatus",
    "AlertLog", "AlertStatus",
]
