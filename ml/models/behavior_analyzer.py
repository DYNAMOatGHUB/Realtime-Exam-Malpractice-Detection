import logging
import math
from collections import defaultdict
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Person States
STATE_NORMAL = "NORMAL"
STATE_WARNING = "WARNING"
STATE_SUSPICIOUS = "SUSPICIOUS"

# COCO Keypoint mapping
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

class BehaviorAnalyzer:
    """
    Analyzes pose keypoints over time to flag anomalous behavior.
    Implements a strict persistence rule:
      - Suspicious if anomaly is sustained for >= `required_frames`
      - Suspicious if anomaly occurs >= `required_episodes` times
    """
    def __init__(self, fps: float = 10.0, sustained_seconds: float = 5.0, required_episodes: int = 5):
        self.fps = fps
        self.required_frames = int(fps * sustained_seconds)
        self.required_episodes = required_episodes
        
        # State per person: track_id -> dict
        self.person_states: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
            "consecutive_anomaly_frames": 0,
            "anomaly_episodes": 0,
            "in_episode": False,
            "suspicious": False,
            "current_state": STATE_NORMAL,
        })

    def _calculate_distance(self, kp1, kp2):
        """Calculate Euclidean distance between two keypoints [x, y, conf]"""
        if kp1[2] < 0.3 or kp2[2] < 0.3:
            return -1 # Not visible
        return math.hypot(kp1[0] - kp2[0], kp1[1] - kp2[1])

    def _is_anomalous_frame(self, keypoints: List[List[float]]) -> bool:
        """
        Rule-based short-window anomaly detection.
        keypoints is a list of [x, y, conf] for 17 COCO keypoints.
        """
        if len(keypoints) < 17:
            return False

        nose = keypoints[0]
        l_ear = keypoints[3]
        r_ear = keypoints[4]
        l_shoulder = keypoints[5]
        r_shoulder = keypoints[6]
        l_wrist = keypoints[9]
        r_wrist = keypoints[10]

        is_anomalous = False

        # Rule 1: Extreme Head Turn (nose is closer to one shoulder/ear extremely than the other)
        if nose[2] > 0.4 and l_ear[2] > 0.4 and r_ear[2] > 0.4:
            dist_l = self._calculate_distance(nose, l_ear)
            dist_r = self._calculate_distance(nose, r_ear)
            if dist_l > 0 and dist_r > 0:
                ratio = dist_l / dist_r
                if ratio > 3.0 or ratio < 0.33:
                    is_anomalous = True

        # Rule 2: Hands away from desk (hands raised high above shoulders)
        if l_shoulder[2] > 0.5 and l_wrist[2] > 0.4:
            if l_wrist[1] < (l_shoulder[1] - 0.05):
                is_anomalous = True

        if r_shoulder[2] > 0.5 and r_wrist[2] > 0.4:
            if r_wrist[1] < (r_shoulder[1] - 0.05):
                is_anomalous = True

        return is_anomalous

    def update_person(self, track_id: int, keypoints: List[List[float]]) -> str:
        """
        Update the state for a tracked person and return their current status and confidence score.
        Returns: Tuple[str, float] - (STATE_NORMAL/WARNING/SUSPICIOUS, confidence 0.0-1.0)
        """
        state = self.person_states[track_id]
        
        if state["suspicious"]:
            return STATE_SUSPICIOUS, 1.0

        is_anomalous = self._is_anomalous_frame(keypoints)

        if is_anomalous:
            state["consecutive_anomaly_frames"] += 1
            if not state["in_episode"]:
                state["in_episode"] = True
                
            state["current_state"] = STATE_WARNING
            
            # Check duration rule
            if state["consecutive_anomaly_frames"] >= self.required_frames:
                state["suspicious"] = True
                state["current_state"] = STATE_SUSPICIOUS
                
        else:
            if state["in_episode"]:
                state["anomaly_episodes"] += 1
                state["in_episode"] = False
                
            state["consecutive_anomaly_frames"] = 0
            state["current_state"] = STATE_NORMAL
            
            # Check frequency rule
            if state["anomaly_episodes"] >= self.required_episodes:
                state["suspicious"] = True
                state["current_state"] = STATE_SUSPICIOUS

        # Calculate dynamic confidence
        prog_dur = min(1.0, state["consecutive_anomaly_frames"] / max(1, self.required_frames))
        prog_freq = min(1.0, state["anomaly_episodes"] / max(1, self.required_episodes))
        confidence = max(prog_dur, prog_freq)
        
        if state["suspicious"]:
            confidence = 1.0

        return state["current_state"], round(confidence, 2)
