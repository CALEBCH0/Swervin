
# import os
# import cv2
# import json
# import time
# from typing import Dict, Any, Optional
# import numpy as np

# """
# Recorder class that saves frames and stats(model outputs and driver state fields) as png and json.
# Uses batch saving to avoid real-time frame drop while recording.
# Set recording interval and batch buffer size for more flexibility.
# (5 interval = recording every 5 frames)
# (1000 buffer size = saves up to 1000 frames per session)
# """
# class FrameRecorder:
#     def __init__(self, session_name=None, record_interval=10, buffer_size=500, base_dir="tests/data", test_mode=False):
#         self.record_interval = record_interval
#         self.buffer_size = buffer_size
#         self.base_dir = base_dir
#         self.test_mode = test_mode
#         self.is_recording = False
#         self.frame_counter = 0
#         self.frame_idx = 0
#         self.frame_buffer = []
#         self.state_buffer = []
#         self.driver_status = None
        
#         if session_name is not None:
#             self.current_session = session_name
#         else:
#             self.current_session = f"session_{int(time.time())}"
#         self.session_dir = None
        
#     def get_stat(self, attr: str, default):
#         return getattr(self.driver_status, attr, default=default)

#     def extract_stats(self, driver_status) -> Dict[str, Any]:
#         """Extract stats organized by model outputs and derived states"""
#         frame_data = {
#             "frame_idx": self.frame_idx,
#             "obj_detector": {
#                 "outputs": {
#                     "face_box": self.get_stat('face_box', []),
#                     "body_box": self.get_stat('body_box', []),
#                     "hand_box": self.get_stat('hand_box', []),
#                 },
#                 "states": {
#                     "is_face": self.get_stat('is_face', False),
#                     "is_hand": self.get_stat('is_hand', False),
#                 }
#             }
#         }
#         return frame_data

#     def _check_eyes_open(self, driver_status):
#         """Check if eyes are open using EAR or eye state"""
#         try:
#             if hasattr(driver_status, 'is_eye_open') and callable(getattr(driver_status, 'is_eye_open')):
#                 return driver_status.is_eye_open()
#             else:
#                 ear = getattr(driver_status, 'EAR', 0.0)
#                 ear_threshold = getattr(driver_status, 'ear_open_threshold', 0.2)
#                 return ear > ear_threshold
#         except:
#             return None

#     def _convert_hand_gesture(self, hand_gesture_str: str) -> int:
#         """Convert hand gesture string to integer"""
#         hand_gesture_map = {
#             'Zero': 0, 'One': 1, 'Two': 2, 'Three': 3, 
#             'Four': 4, 'Five': 5, 'None': 6
#         }
#         return hand_gesture_map.get(str(hand_gesture_str), 6)

#     def record(self, frame, driver_status):
#         """Record frame and driver status"""
#         # Update driver status
#         self.driver_status = driver_status
        
#         if not self.is_recording or self.frame_counter % self.record_interval != 0:
#             self.frame_counter += 1
#             return False
        
#         if len(self.frame_buffer) >= self.buffer_size:
#             print("Buffer full, dropping frame")
#             self.frame_counter += 1
#             return False
        
#         # Extract structured stats
#         frame_stats = self.extract_stats(driver_status)
        
#         if not self.test_mode:
#             self.frame_buffer.append((frame.copy(), self.frame_idx))
#         self.state_buffer.append(frame_stats)
        
#         self.frame_counter += 1
#         self.frame_idx += 1
#         return True

#     def start_recording(self):
#         if self.is_recording:
#             print("Already recording")
#             return
        
#         self.create_data_dir()
#         self.frame_counter = 0
#         self.frame_idx = 0
#         self.frame_buffer.clear()
#         self.state_buffer.clear()
#         self.is_recording = True
#         print(f"Recording started for session: {self.current_session}")

#     def stop_recording(self):
#         if not self.is_recording:
#             print("Not currently recording")
#             return
        
#         self.is_recording = False
#         self._save_all()
#         self.frame_buffer.clear()
#         self.state_buffer.clear()
        
#         if self.test_mode:
#             print(f"Test session({self.current_session}) stats saved in {self.session_dir}")
#         else:
#             print(f"Session({self.current_session}) saved {self.frame_idx} frames to {self.session_dir}")

#     def create_data_dir(self):
#         """Create directory structure for recording session"""
#         import os
        
#         if not os.path.exists(self.base_dir):
#             os.makedirs(self.base_dir, exist_ok=True)
        
#         self.session_dir = os.path.join(self.base_dir, self.current_session)
#         if not self.test_mode:
#             frames_dir = os.path.join(self.session_dir, "frames")
#             os.makedirs(frames_dir, exist_ok=True)
#         stats_dir = os.path.join(self.session_dir, "stats")
#         os.makedirs(stats_dir, exist_ok=True)
        
#         print(f"Created recording session: {self.current_session}")
#         print(f"Data will be saved in: {self.session_dir}")
        
#         return self.session_dir

#     def _save_all(self):
#         """Save all buffered data"""
#         try:
#             import cv2
            
#             if not self.test_mode:
#                 frames_dir = os.path.join(self.session_dir, "frames")
#                 print(f"Saving {len(self.frame_buffer)} frames...")
                
#                 for i, (frame, frame_idx) in enumerate(self.frame_buffer):
#                     if self.frame_idx > 200 and frame_idx % 50 == 0:
#                         print(f"{frame_idx}/{self.frame_idx} saved")
#                     frame_filename = os.path.join(frames_dir, f"{self.current_session}_{frame_idx:05d}_frame.png")
#                     success = cv2.imwrite(frame_filename, frame)
#                     if not success:
#                         print(f"Warning: Failed to save frame {frame_idx}")
            
#             # Save stats as indexed dictionary
#             self._save_all_stats()
#             print("All data saved successfully!")
            
#         except Exception as e:
#             print(f"Error saving recording data: {e}")





# claude version working but not customized
import os
import cv2
import json
import time
from typing import Dict, Any, Optional
import numpy as np

class FrameRecorder:
    def __init__(self, session_name='default', record_interval=10, buffer_size=500, base_dir="tests/data", test_mode=False):
        self.record_interval = record_interval
        self.buffer_size = buffer_size
        self.base_dir = base_dir
        self.test_mode = test_mode
        self.is_recording = False
        self.frame_counter = 0
        self.frame_idx = 0
        self.frame_buffer = []
        self.state_buffer = []
        
        if session_name != 'default':
            self.current_session = session_name
        else:
            self.current_session = f"session_{int(time.time())}"
        self.session_dir = None

    def extract_stats(self, driver_status) -> Dict[str, Any]:
        """Extract stats organized by model with outputs and derived states"""
        
        def ensure_json_serializable(value):
            """Ensure all values are JSON serializable"""
            if value is None:
                return None
            elif isinstance(value, (np.integer, np.int64, np.int32)):
                return int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                if np.isnan(value) or np.isinf(value):
                    return None
                return float(value)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (list, tuple)):
                return [ensure_json_serializable(item) for item in value]
            elif isinstance(value, dict):
                return {str(k): ensure_json_serializable(v) for k, v in value.items()}
            elif isinstance(value, (np.bool_, bool)):
                return bool(value)
            elif hasattr(value, 'item'):  # numpy scalar
                return ensure_json_serializable(value.item())
            elif isinstance(value, (str, int, float)):
                return value
            else:
                try:
                    if hasattr(value, 'tolist'):
                        return value.tolist()
                    else:
                        return str(value)
                except:
                    return str(value)
                
        def safe_get_array(obj, attr, default):
            try:
                value = getattr(obj, attr, default)
                return ensure_json_serializable(value)
            except:
                return ensure_json_serializable(default)
        
        frame_data = {
            "frame_idx": int(self.frame_idx),
            
            # ================================================================
            # OBJECT DETECTOR
            # ================================================================
            "object_detector": {
                "outputs": {
                    "face_boxes": ensure_json_serializable(getattr(driver_status, 'face_boxes', [])),
                    "hand_boxes": ensure_json_serializable(getattr(driver_status, 'hand_boxes', [])),
                    "body_boxes": ensure_json_serializable(getattr(driver_status, 'body_boxes', [])),
                },
                "states": {
                    "is_face": bool(getattr(driver_status, 'is_face', False)),
                    "recent_face_location": ensure_json_serializable(getattr(driver_status, 'recent_face_location', None)),
                    "face_area": float(getattr(driver_status, 'face_area', 0)),
                    "same_face": bool(getattr(driver_status, 'same_face', False)),
                    "is_hand": bool(getattr(driver_status, 'is_hand', False)),
                    "hand_box": ensure_json_serializable(getattr(driver_status, 'hand_box', None))
                }
            },
            
            # ================================================================
            # FACE LANDMARK - CRITICAL FIX
            # ================================================================
            "face_landmark": {
                "outputs": {
                    "landmarks_68": ensure_json_serializable(getattr(driver_status, 'facelandmark', [[0.0, 0.0]] * 68))
                },
                "states": {
                    "landmark_confidence": float(getattr(driver_status, 'landmark_confidence', 0.0))
                }
            },
            
            # ================================================================
            # HEAD POSE
            # ================================================================
            "head_pose": {
                "outputs": {
                    "rotation_matrix": ensure_json_serializable(safe_get_array(driver_status, 'rotation_matrix', [[1,0,0],[0,1,0],[0,0,1]])),
                    "pose_angles": ensure_json_serializable(safe_get_array(driver_status, 'head_pose', [0, 0, 0]))
                },
                "states": {
                    "head_pose": ensure_json_serializable(safe_get_array(driver_status, 'head_pose', [0, 0, 0])),
                    "pitch": float(safe_get_array(driver_status, 'head_pose', [0, 0, 0])[0]),
                    "yaw": float(safe_get_array(driver_status, 'head_pose', [0, 0, 0])[1]),
                    "roll": float(safe_get_array(driver_status, 'head_pose', [0, 0, 0])[2]),
                    "HP_READY": bool(getattr(driver_status, 'HP_READY', False)),
                    "PYRavg": ensure_json_serializable(getattr(driver_status, 'PYRavg', [0, 0, 0]))
                }
            },
            
            # ================================================================
            # GAZE ESTIMATOR
            # ================================================================
            "gaze_estimator": {
                "outputs": {
                    "pitch_bins": ensure_json_serializable(getattr(driver_status, 'gaze_pitch_bins', [0.0] * 90)),
                    "yaw_bins": ensure_json_serializable(getattr(driver_status, 'gaze_yaw_bins', [0.0] * 90)),
                    "gaze_raw": ensure_json_serializable(safe_get_array(driver_status, 'gaze_raw', [0, 0]))
                },
                "states": {
                    "gaze": ensure_json_serializable(safe_get_array(driver_status, 'gaze', [0, 0])),
                    "gaze_pitch": float(safe_get_array(driver_status, 'gaze', [0, 0])[1]) if len(safe_get_array(driver_status, 'gaze', [0, 0])) > 1 else 0.0,
                    "gaze_yaw": float(safe_get_array(driver_status, 'gaze', [0, 0])[0]) if len(safe_get_array(driver_status, 'gaze', [0, 0])) > 0 else 0.0,
                    "eyes_open": self._check_eyes_open(driver_status)
                }
            },
            
            # ================================================================
            # HAND DETECTOR & CLASSIFIER
            # ================================================================
            "hand_detector": {
                "outputs": {
                    "hand_detections": ensure_json_serializable(getattr(driver_status, 'hand_detections', [])),
                    "selected_hand_box": ensure_json_serializable(getattr(driver_status, 'hand_box', None))
                },
                "states": {
                    "hand_box": ensure_json_serializable(getattr(driver_status, 'hand_box', None)),
                    "hand_not_detected": int(getattr(driver_status, 'hand_not_detected', 0))
                }
            },
            
            "hand_classifier": {
                "outputs": {
                    "gesture_logits": ensure_json_serializable(getattr(driver_status, 'gesture_logits', [0.0] * 10)),
                    "gesture_class": int(getattr(driver_status, 'gesture_class', 6))
                },
                "states": {
                    "hand_gesture": str(getattr(driver_status, 'hand_gesture', 'None')),
                    "hand_gesture_id": int(self._convert_hand_gesture(getattr(driver_status, 'hand_gesture', 'None'))),
                    "activate_hand_gesture": bool(getattr(driver_status, 'activate_hand_gesture', False)),
                    "_hand_gesture_count": int(getattr(driver_status, '_hand_gesture_count', 0))
                }
            },
            
            # ================================================================
            # PHONE DETECTOR
            # ================================================================
            "phone_detector": {
                "outputs": {
                    "phone_detections": ensure_json_serializable(getattr(driver_status, 'phone_detections', [])),
                    "phone_boxes": ensure_json_serializable(getattr(driver_status, 'phone_boxes', []))
                },
                "states": {
                    "phone_counter": int(getattr(driver_status, 'phone_counter', 0))
                }
            },
            
            # ================================================================
            # ACTION DETECTOR
            # ================================================================
            "action_detector": {
                "outputs": {
                    "smoking_confidence": float(safe_get_array(driver_status, 'action_val', [0.0, 0.0])[0]),
                    "phone_confidence": float(safe_get_array(driver_status, 'action_val', [0.0, 0.0])[1]),
                    "action_raw": ensure_json_serializable(safe_get_array(driver_status, 'action_val', [0.0, 0.0]))
                },
                "states": {
                    "action_val": ensure_json_serializable(safe_get_array(driver_status, 'action_val', [0.0, 0.0])),
                    "smoking": int(getattr(driver_status, 'smoking', 0)),
                    "usingphone": int(getattr(driver_status, 'usingphone', 0)),
                    "action_thr": ensure_json_serializable(getattr(driver_status, 'action_thr', [0.9, 0.7]))
                }
            },
            
            # ================================================================
            # FACE ATTRIBUTES
            # ================================================================
            "face_info_detector": {
                "outputs": {
                    "gender_age_logits": ensure_json_serializable(safe_get_array(driver_status, 'face_info_raw', [0.0] * 5)),
                    "face_info_scores": ensure_json_serializable(safe_get_array(driver_status, 'face_info_val', [0.0] * 5))
                },
                "states": {
                    "face_info_val": ensure_json_serializable(safe_get_array(driver_status, 'face_info_val', [0.0] * 5)),
                    "gender": int(getattr(driver_status, 'gender', 3)),
                    "age": int(getattr(driver_status, 'age', 3)),
                    "fix_face_info": bool(getattr(driver_status, 'fix_face_info', False)),
                    "face_info_thr": float(getattr(driver_status, 'face_info_thr', 0.5))
                }
            },
            
            "face_acce_detector": {
                "outputs": {
                    "accessory_logits": ensure_json_serializable(safe_get_array(driver_status, 'face_acce_raw', [0.0] * 4)),
                    "accessory_scores": ensure_json_serializable(safe_get_array(driver_status, 'face_acce_val', [0.0] * 2))
                },
                "states": {
                    "face_acce_val": ensure_json_serializable(safe_get_array(driver_status, 'face_acce_val', [0.0] * 2)),
                    "glasses": int(getattr(driver_status, 'glasses', 0)),
                    "mask": int(getattr(driver_status, 'mask', 0)),
                    "face_acce_thr": float(getattr(driver_status, 'face_acce_thr', 0.9))
                }
            },
            
            # ================================================================
            # BODY KEYPOINT (if enabled)
            # ================================================================
            "body_keypoint": {
                "outputs": {
                    "keypoint_heatmaps": ensure_json_serializable(getattr(driver_status, 'keypoint_heatmaps', [])),
                    "raw_keypoints": ensure_json_serializable(safe_get_array(driver_status, 'raw_bodykeypoints', np.zeros((10, 3))))
                },
                "states": {
                    "bodykeypoints": ensure_json_serializable(safe_get_array(driver_status, 'bodykeypoints', np.zeros((10, 3)))),
                    "vis_bodykeypoints": ensure_json_serializable(safe_get_array(driver_status, 'vis_bodykeypoints', np.zeros((10, 2)))),
                    "activate_body": bool(getattr(driver_status, 'activate_body', False)),
                    "thr_bodykeypoint": float(getattr(driver_status, 'thr_bodykeypoint', 0.2))
                }
            },
            
            # ================================================================
            # COMPUTED METRICS (derived from model outputs)
            # ================================================================
            "eye_metrics": {
                "outputs": {
                    "raw_ear": float(getattr(driver_status, 'EAR_raw', 0.0)),
                    "ear_history": ensure_json_serializable(getattr(driver_status, 'EARs', []))
                },
                "states": {
                    "EAR": float(getattr(driver_status, 'EAR', 0.0)),
                    "EAR_thres": float(getattr(driver_status, 'EAR_thres', 0.4)),
                    "microsleep": bool(getattr(driver_status, 'microsleep', False)),
                    "sleep": bool(getattr(driver_status, 'sleep', False)),
                    "drowsy_val": float(getattr(driver_status, 'drowsy_val', 0.0)),
                    "ear_open_threshold": float(getattr(driver_status, 'ear_open_threshold', 0.2))
                }
            },
            
            "mouth_metrics": {
                "outputs": {
                    "raw_lar": float(getattr(driver_status, 'LAR_raw', 0.0))
                },
                "states": {
                    "LAR": float(getattr(driver_status, 'LAR', 0.0)),
                    "LAR_thresh": float(getattr(driver_status, 'LAR_thresh', 0.4)),
                    "yawning": bool(getattr(driver_status, 'yawning', False)),
                    "_yawn_count": int(getattr(driver_status, '_yawn_count', 0)),
                    "yawn_consec_frames": int(getattr(driver_status, 'yawn_consec_frames', 10))
                }
            },
            
            "head_gestures": {
                "outputs": {
                    "talking_detection": bool(getattr(driver_status, 'talking_raw', False)),
                    "nodding_detection": bool(getattr(driver_status, 'nodding_raw', False)),
                    "shaking_detection": bool(getattr(driver_status, 'shaking_raw', False))
                },
                "states": {
                    "talking": bool(getattr(driver_status, 'talking', False)),
                    "nodding": bool(getattr(driver_status, 'nodding', False)),
                    "shaking": bool(getattr(driver_status, 'shaking', False))
                }
            },
            
            "distraction_metrics": {
                "outputs": {
                    "distraction_angle": float(getattr(driver_status, 'distraction_angle', 0.0)),
                    "distraction_history": ensure_json_serializable(getattr(driver_status, 'distracted_status', []))
                },
                "states": {
                    "distracted": bool(getattr(driver_status, 'distracted', False)),
                    "long_distracted": bool(getattr(driver_status, 'long_distracted', False)),
                    "distracted_val": float(getattr(driver_status, 'distracted_val', 0.0)),
                    "distracted_angle_thres": float(getattr(driver_status, 'distracted_angle_thres', 40)),
                    "distraction_thres": float(getattr(driver_status, 'distraction_thres', 0.6)),
                    "long_distracted_thre": float(getattr(driver_status, 'long_distracted_thre', 0.6))
                }
            },
            
            # ================================================================
            # OVERALL SYSTEM STATE
            # ================================================================
            "system": {
                "outputs": {
                    "processing_fps": float(getattr(driver_status, 'total_frame_value', 0)),
                    "frame_timestamp": int(time.time())
                },
                "states": {
                    "ID": str(getattr(driver_status, 'ID', 'Unknown')),
                    "driver_id": str(getattr(driver_status, 'ID', 'Unknown')),
                    "same_face": bool(getattr(driver_status, 'same_face', False)),
                    "session_name": str(getattr(self, 'current_session', 'unknown'))
                }
            }
        }
        
        return frame_data

    def _check_eyes_open(self, driver_status):
        """Check if eyes are open using EAR or eye state"""
        try:
            if hasattr(driver_status, 'is_eye_open') and callable(getattr(driver_status, 'is_eye_open')):
                return driver_status.is_eye_open()
            else:
                ear = getattr(driver_status, 'EAR', 0.0)
                ear_threshold = getattr(driver_status, 'ear_open_threshold', 0.2)
                return ear > ear_threshold
        except:
            return None

    def _convert_hand_gesture(self, hand_gesture_str: str) -> int:
        """Convert hand gesture string to integer"""
        hand_gesture_map = {
            'Zero': 0, 'One': 1, 'Two': 2, 'Three': 3, 
            'Four': 4, 'Five': 5, 'None': 6
        }
        return hand_gesture_map.get(str(hand_gesture_str), 6)

    def record(self, frame, driver_status):
        """Record frame and driver status"""
        if not self.is_recording or self.frame_counter % self.record_interval != 0:
            self.frame_counter += 1
            return False
        
        if len(self.frame_buffer) >= self.buffer_size:
            print("Buffer full, dropping frame")
            self.frame_counter += 1
            return False
        
        # Extract structured stats
        frame_stats = self.extract_stats(driver_status)
        
        if not self.test_mode:
            self.frame_buffer.append((frame.copy(), self.frame_idx))
        self.state_buffer.append(frame_stats)
        
        self.frame_counter += 1
        self.frame_idx += 1
        return True

    def start_recording(self):
        if self.is_recording:
            print("Already recording")
            return
        
        self.create_data_dir()
        self.frame_counter = 0
        self.frame_idx = 0
        self.frame_buffer.clear()
        self.state_buffer.clear()
        self.is_recording = True
        print(f"Recording started for session: {self.current_session}")

    def stop_recording(self):
        if not self.is_recording:
            print("Not currently recording")
            return
        
        self.is_recording = False
        self._save_all()
        self.frame_buffer.clear()
        self.state_buffer.clear()
        
        if self.test_mode:
            print(f"Test session({self.current_session}) stats saved in {self.session_dir}")
        else:
            print(f"Session({self.current_session}) saved {self.frame_idx} frames to {self.session_dir}")

    def create_data_dir(self):
        """Create directory structure for recording session"""
        import os
        
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)
        
        self.session_dir = os.path.join(self.base_dir, self.current_session)
        if not self.test_mode:
            frames_dir = os.path.join(self.session_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
        stats_dir = os.path.join(self.session_dir, "stats")
        os.makedirs(stats_dir, exist_ok=True)
        
        print(f"Created recording session: {self.current_session}")
        print(f"Data will be saved in: {self.session_dir}")
        
        return self.session_dir

    def _save_all(self):
        """Save all buffered data"""
        try:
            import cv2
            
            if not self.test_mode:
                frames_dir = os.path.join(self.session_dir, "frames")
                print(f"Saving {len(self.frame_buffer)} frames...")
                
                for i, (frame, frame_idx) in enumerate(self.frame_buffer):
                    if self.frame_idx > 200 and frame_idx % 50 == 0:
                        print(f"{frame_idx}/{self.frame_idx} saved")
                    frame_filename = os.path.join(frames_dir, f"{self.current_session}_{frame_idx:05d}_frame.png")
                    success = cv2.imwrite(frame_filename, frame)
                    if not success:
                        print(f"Warning: Failed to save frame {frame_idx}")
            
            # Save stats as indexed dictionary
            self._save_all_stats()
            print("All data saved successfully!")
            
        except Exception as e:
            print(f"Error saving recording data: {e}")

    def _save_all_stats(self):
        """Save stats indexed by frame_idx for easy access"""
        stats_dir = os.path.join(self.session_dir, "stats")
        
        # Create indexed dictionary: {frame_idx: {model: {outputs/states: ...}}}
        indexed_stats = {}
        for frame_data in self.state_buffer:
            frame_idx = frame_data['frame_idx']
            # Remove frame_idx from the data since it's now the key
            frame_stats = {k: v for k, v in frame_data.items() if k != 'frame_idx'}
            indexed_stats[frame_idx] = frame_stats
        
        # Validate JSON serializability before saving
        try:
            test_json = json.dumps(indexed_stats, separators=(',', ':'))
            print("✅ JSON validation passed")
        except (TypeError, ValueError) as e:
            print(f"❌ JSON validation failed: {e}")
            print("Attempting to clean data...")
            # Clean the data more aggressively
            indexed_stats = self._clean_for_json(indexed_stats)
        
        # Save indexed stats - NO default=str parameter!
        indexed_filename = os.path.join(stats_dir, f"{self.current_session}_all_stats.json")
        try:
            with open(indexed_filename, 'w') as f:
                json.dump(indexed_stats, f, separators=(',', ':'), indent=2)
            print(f"Saved indexed stats to {indexed_filename}")
            
            # Verify the saved file can be loaded
            with open(indexed_filename, 'r') as f:
                test_load = json.load(f)
            print("✅ Saved JSON file validation passed")
            
        except Exception as e:
            print(f"❌ Error saving JSON: {e}")
            # Save as pickle as fallback
            import pickle
            fallback_filename = os.path.join(stats_dir, f"{self.current_session}_all_stats.pkl")
            with open(fallback_filename, 'wb') as f:
                pickle.dump(indexed_stats, f)
            print(f"Saved as pickle fallback: {fallback_filename}")
    
    def _clean_for_json(self, data):
        """Aggressively clean data for JSON compatibility"""
        if isinstance(data, dict):
            return {str(k): self._clean_for_json(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._clean_for_json(item) for item in data]
        elif data is None:
            return None
        elif isinstance(data, (bool, int, float, str)):
            return data
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif hasattr(data, 'item'):  # numpy scalar
            val = data.item()
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                return None
            return val
        else:
            # Last resort - convert to string
            return str(data)

    
    # def _save_all_stats(self):
    #     """Save stats indexed by frame_idx for easy access"""
    #     stats_dir = os.path.join(self.session_dir, "stats")
        
    #     # Create indexed dictionary: {frame_idx: {model: {outputs/states: ...}}}
    #     indexed_stats = {}
    #     for frame_data in self.state_buffer:
    #         frame_idx = frame_data['frame_idx']
    #         # Remove frame_idx from the data since it's now the key
    #         frame_stats = {k: v for k, v in frame_data.items() if k != 'frame_idx'}
    #         indexed_stats[frame_idx] = frame_stats
        
    #     # Validate JSON serializability before saving
    #     try:
    #         test_json = json.dumps(indexed_stats, separators=(',', ':'))
    #         print("✅ JSON validation passed")
    #     except (TypeError, ValueError) as e:
    #         print(f"❌ JSON validation failed: {e}")
    #         print("Attempting to clean data...")
    #         indexed_stats = self._clean_for_json(indexed_stats)
        
    #     # Save indexed stats - NO default=str parameter!
    #     indexed_filename = os.path.join(stats_dir, f"{self.current_session}_all_stats.json")
    #     try:
    #         with open(indexed_filename, 'w') as f:
    #             json.dump(indexed_stats, f, separators=(',', ':'), indent=2)
    #         print(f"Saved indexed stats to {indexed_filename}")
            
    #         # Verify the saved file can be loaded
    #         with open(indexed_filename, 'r') as f:
    #             test_load = json.load(f)
    #         print("✅ Saved JSON file validation passed")
            
    #     except Exception as e:
    #         print(f"❌ Error saving JSON: {e}")
    #         # Save as pickle as fallback
    #         import pickle
    #         fallback_filename = os.path.join(stats_dir, f"{self.current_session}_all_stats.pkl")
    #         with open(fallback_filename, 'wb') as f:
    #             pickle.dump(indexed_stats, f)
    #         print(f"Saved as pickle fallback: {fallback_filename}")

    # def _clean_for_json(self, data):
    #     """Aggressively clean data for JSON compatibility"""
    #     if isinstance(data, dict):
    #         return {str(k): self._clean_for_json(v) for k, v in data.items()}
    #     elif isinstance(data, (list, tuple)):
    #         return [self._clean_for_json(item) for item in data]
    #     elif data is None:
    #         return None
    #     elif isinstance(data, (bool, int, float, str)):
    #         return data
    #     elif isinstance(data, np.ndarray):
    #         return data.tolist()
    #     elif hasattr(data, 'item'):  # numpy scalar
    #         val = data.item()
    #         if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
    #             return None
    #         return val
    #     else:
    #         # Last resort - convert to string
    #         return str(data)








# import os
# import cv2
# import json
# import queue
# import threading
# import time
# from typing import Dict, Any, Optional
# import numpy as np

# class FrameRecorder:
#     def __init__(self, session_name=None, record_interval=10, buffer_size=500, base_dir="tests/data", test_mode=False):
#         self.record_interval = record_interval
#         self.buffer_size = buffer_size
#         self.base_dir = base_dir
#         self.test_mode = test_mode
#         self.is_recording = False
#         self.frame_counter = 0
#         self.frame_idx = 0
#         self.frame_buffer = []
#         self.state_buffer = []
        
#         if session_name is not None:
#             self.current_session = session_name
#         else:
#             self.current_session = f"session_{int(time.time())}"
#         self.session_dir = None

#     def extract_stats(self, driver_status) -> Dict[str, Any]:
#         """Extract stats organized by model with outputs and derived states"""
        
#         # Helper function to safely get array values
#         def safe_get_array(obj, attr, default):
#             try:
#                 value = getattr(obj, attr, default)
#                 if hasattr(value, 'tolist'):
#                     return value.tolist()
#                 elif isinstance(value, (list, tuple)):
#                     return list(value)
#                 else:
#                     return default
#             except:
#                 return default
        
#         frame_data = {
#             "frame_idx": self.frame_idx,
            
#             # ================================================================
#             # OBJECT DETECTOR
#             # ================================================================
#             "object_detector": {
#                 "outputs": {
#                     "face_boxes": getattr(driver_status, 'face_boxes', []),
#                     "hand_boxes": getattr(driver_status, 'hand_boxes', []),
#                     "body_boxes": getattr(driver_status, 'body_boxes', []),
#                 },
#                 "states": {
#                     "is_face": getattr(driver_status, 'is_face', False),
#                     "recent_face_location": getattr(driver_status, 'recent_face_location', None),
#                     "face_area": float(getattr(driver_status, 'face_area', 0)),
#                     "same_face": getattr(driver_status, 'same_face', False),
#                     "is_hand": getattr(driver_status, 'is_hand', False),
#                     "hand_box": getattr(driver_status, 'hand_box', None)
#                 }
#             },
            
#             # ================================================================
#             # FACE LANDMARK
#             # ================================================================
#             "face_landmark": {
#                 "outputs": {
#                     "landmarks_68": getattr(driver_status, 'facelandmark', [[0.0, 0.0]] * 68)
#                 },
#                 "states": {
#                     "landmark_confidence": getattr(driver_status, 'landmark_confidence', 0.0)
#                 }
#             },
            
#             # ================================================================
#             # HEAD POSE
#             # ================================================================
#             "head_pose": {
#                 "outputs": {
#                     "rotation_matrix": safe_get_array(driver_status, 'rotation_matrix', [[1,0,0],[0,1,0],[0,0,1]]),
#                     "pose_angles": safe_get_array(driver_status, 'head_pose', [0, 0, 0])
#                 },
#                 "states": {
#                     "head_pose": safe_get_array(driver_status, 'head_pose', [0, 0, 0]),
#                     "pitch": float(safe_get_array(driver_status, 'head_pose', [0, 0, 0])[0]),
#                     "yaw": float(safe_get_array(driver_status, 'head_pose', [0, 0, 0])[1]),
#                     "roll": float(safe_get_array(driver_status, 'head_pose', [0, 0, 0])[2]),
#                     "HP_READY": getattr(driver_status, 'HP_READY', False),
#                     "PYRavg": getattr(driver_status, 'PYRavg', [0, 0, 0])
#                 }
#             },
            
#             # ================================================================
#             # GAZE ESTIMATOR
#             # ================================================================
#             "gaze_estimator": {
#                 "outputs": {
#                     "pitch_bins": getattr(driver_status, 'gaze_pitch_bins', [0.0] * 90),
#                     "yaw_bins": getattr(driver_status, 'gaze_yaw_bins', [0.0] * 90),
#                     "gaze_raw": safe_get_array(driver_status, 'gaze_raw', [0, 0])
#                 },
#                 "states": {
#                     "gaze": safe_get_array(driver_status, 'gaze', [0, 0]),
#                     "gaze_pitch": float(safe_get_array(driver_status, 'gaze', [0, 0])[1]) if len(safe_get_array(driver_status, 'gaze', [0, 0])) > 1 else 0.0,
#                     "gaze_yaw": float(safe_get_array(driver_status, 'gaze', [0, 0])[0]) if len(safe_get_array(driver_status, 'gaze', [0, 0])) > 0 else 0.0,
#                     "eyes_open": self._check_eyes_open(driver_status)
#                 }
#             },
            
#             # ================================================================
#             # HAND DETECTOR & CLASSIFIER
#             # ================================================================
#             "hand_detector": {
#                 "outputs": {
#                     "hand_detections": getattr(driver_status, 'hand_detections', []),
#                     "selected_hand_box": getattr(driver_status, 'hand_box', None)
#                 },
#                 "states": {
#                     "hand_box": getattr(driver_status, 'hand_box', None),
#                     "hand_not_detected": getattr(driver_status, 'hand_not_detected', 0)
#                 }
#             },
            
#             "hand_classifier": {
#                 "outputs": {
#                     "gesture_logits": getattr(driver_status, 'gesture_logits', [0.0] * 10),
#                     "gesture_class": getattr(driver_status, 'gesture_class', 6)
#                 },
#                 "states": {
#                     "hand_gesture": str(getattr(driver_status, 'hand_gesture', 'None')),
#                     "hand_gesture_id": self._convert_hand_gesture(getattr(driver_status, 'hand_gesture', 'None')),
#                     "activate_hand_gesture": getattr(driver_status, 'activate_hand_gesture', False),
#                     "_hand_gesture_count": getattr(driver_status, '_hand_gesture_count', 0)
#                 }
#             },
            
#             # ================================================================
#             # PHONE DETECTOR
#             # ================================================================
#             "phone_detector": {
#                 "outputs": {
#                     "phone_detections": getattr(driver_status, 'phone_detections', []),
#                     "phone_boxes": getattr(driver_status, 'phone_boxes', [])
#                 },
#                 "states": {
#                     "phone_counter": getattr(driver_status, 'phone_counter', 0)
#                 }
#             },
            
#             # ================================================================
#             # ACTION DETECTOR
#             # ================================================================
#             "action_detector": {
#                 "outputs": {
#                     "smoking_confidence": float(safe_get_array(driver_status, 'action_val', [0.0, 0.0])[0]),
#                     "phone_confidence": float(safe_get_array(driver_status, 'action_val', [0.0, 0.0])[1]),
#                     "action_raw": safe_get_array(driver_status, 'action_val', [0.0, 0.0])
#                 },
#                 "states": {
#                     "action_val": safe_get_array(driver_status, 'action_val', [0.0, 0.0]),
#                     "smoking": getattr(driver_status, 'smoking', 0),
#                     "usingphone": getattr(driver_status, 'usingphone', 0),
#                     "action_thr": getattr(driver_status, 'action_thr', [0.9, 0.7])
#                 }
#             },
            
#             # ================================================================
#             # FACE ATTRIBUTES
#             # ================================================================
#             "face_info_detector": {
#                 "outputs": {
#                     "gender_age_logits": safe_get_array(driver_status, 'face_info_raw', [0.0] * 5),
#                     "face_info_scores": safe_get_array(driver_status, 'face_info_val', [0.0] * 5)
#                 },
#                 "states": {
#                     "face_info_val": safe_get_array(driver_status, 'face_info_val', [0.0] * 5),
#                     "gender": int(getattr(driver_status, 'gender', 3)),
#                     "age": int(getattr(driver_status, 'age', 3)),
#                     "fix_face_info": getattr(driver_status, 'fix_face_info', False),
#                     "face_info_thr": getattr(driver_status, 'face_info_thr', 0.5)
#                 }
#             },
            
#             "face_acce_detector": {
#                 "outputs": {
#                     "accessory_logits": safe_get_array(driver_status, 'face_acce_raw', [0.0] * 4),
#                     "accessory_scores": safe_get_array(driver_status, 'face_acce_val', [0.0] * 2)
#                 },
#                 "states": {
#                     "face_acce_val": safe_get_array(driver_status, 'face_acce_val', [0.0] * 2),
#                     "glasses": int(getattr(driver_status, 'glasses', 0)),
#                     "mask": int(getattr(driver_status, 'mask', 0)),
#                     "face_acce_thr": getattr(driver_status, 'face_acce_thr', 0.9)
#                 }
#             },
            
#             # ================================================================
#             # BODY KEYPOINT (if enabled)
#             # ================================================================
#             "body_keypoint": {
#                 "outputs": {
#                     "keypoint_heatmaps": getattr(driver_status, 'keypoint_heatmaps', []),
#                     "raw_keypoints": safe_get_array(driver_status, 'raw_bodykeypoints', np.zeros((10, 3)))
#                 },
#                 "states": {
#                     "bodykeypoints": safe_get_array(driver_status, 'bodykeypoints', np.zeros((10, 3))),
#                     "vis_bodykeypoints": safe_get_array(driver_status, 'vis_bodykeypoints', np.zeros((10, 2))),
#                     "activate_body": getattr(driver_status, 'activate_body', False),
#                     "thr_bodykeypoint": getattr(driver_status, 'thr_bodykeypoint', 0.2)
#                 }
#             },
            
#             # ================================================================
#             # COMPUTED METRICS (derived from model outputs)
#             # ================================================================
#             "eye_metrics": {
#                 "outputs": {
#                     "raw_ear": float(getattr(driver_status, 'EAR_raw', 0.0)),
#                     "ear_history": getattr(driver_status, 'EARs', [])
#                 },
#                 "states": {
#                     "EAR": float(getattr(driver_status, 'EAR', 0.0)),
#                     "EAR_thres": getattr(driver_status, 'EAR_thres', 0.4),
#                     "microsleep": getattr(driver_status, 'microsleep', False),
#                     "sleep": getattr(driver_status, 'sleep', False),
#                     "drowsy_val": float(getattr(driver_status, 'drowsy_val', 0.0)),
#                     "ear_open_threshold": getattr(driver_status, 'ear_open_threshold', 0.2)
#                 }
#             },
            
#             "mouth_metrics": {
#                 "outputs": {
#                     "raw_lar": float(getattr(driver_status, 'LAR_raw', 0.0))
#                 },
#                 "states": {
#                     "LAR": float(getattr(driver_status, 'LAR', 0.0)),
#                     "LAR_thresh": getattr(driver_status, 'LAR_thresh', 0.4),
#                     "yawning": getattr(driver_status, 'yawning', False),
#                     "_yawn_count": getattr(driver_status, '_yawn_count', 0),
#                     "yawn_consec_frames": getattr(driver_status, 'yawn_consec_frames', 10)
#                 }
#             },
            
#             "head_gestures": {
#                 "outputs": {
#                     "talking_detection": getattr(driver_status, 'talking_raw', False),
#                     "nodding_detection": getattr(driver_status, 'nodding_raw', False),
#                     "shaking_detection": getattr(driver_status, 'shaking_raw', False)
#                 },
#                 "states": {
#                     "talking": getattr(driver_status, 'talking', False),
#                     "nodding": getattr(driver_status, 'nodding', False),
#                     "shaking": getattr(driver_status, 'shaking', False)
#                 }
#             },
            
#             "distraction_metrics": {
#                 "outputs": {
#                     "distraction_angle": getattr(driver_status, 'distraction_angle', 0.0),
#                     "distraction_history": getattr(driver_status, 'distracted_status', [])
#                 },
#                 "states": {
#                     "distracted": getattr(driver_status, 'distracted', False),
#                     "long_distracted": getattr(driver_status, 'long_distracted', False),
#                     "distracted_val": float(getattr(driver_status, 'distracted_val', 0.0)),
#                     "distracted_angle_thres": getattr(driver_status, 'distracted_angle_thres', 40),
#                     "distraction_thres": getattr(driver_status, 'distraction_thres', 0.6),
#                     "long_distracted_thre": getattr(driver_status, 'long_distracted_thre', 0.6)
#                 }
#             },
            
#             # ================================================================
#             # OVERALL SYSTEM STATE
#             # ================================================================
#             "system": {
#                 "outputs": {
#                     "processing_fps": getattr(driver_status, 'total_frame_value', 0),
#                     "frame_timestamp": int(time.time())
#                 },
#                 "states": {
#                     "ID": str(getattr(driver_status, 'ID', 'Unknown')),
#                     "driver_id": str(getattr(driver_status, 'ID', 'Unknown')),
#                     "same_face": getattr(driver_status, 'same_face', False),
#                     "session_name": getattr(self, 'current_session', 'unknown')
#                 }
#             }
#         }
        
#         return frame_data

#     def _check_eyes_open(self, driver_status):
#         """Check if eyes are open using EAR or eye state"""
#         try:
#             if hasattr(driver_status, 'is_eye_open') and callable(getattr(driver_status, 'is_eye_open')):
#                 return driver_status.is_eye_open()
#             else:
#                 ear = getattr(driver_status, 'EAR', 0.0)
#                 ear_threshold = getattr(driver_status, 'ear_open_threshold', 0.2)
#                 return ear > ear_threshold
#         except:
#             return None

#     def _convert_hand_gesture(self, hand_gesture_str: str) -> int:
#         """Convert hand gesture string to integer"""
#         hand_gesture_map = {
#             'Zero': 0, 'One': 1, 'Two': 2, 'Three': 3, 
#             'Four': 4, 'Five': 5, 'None': 6
#         }
#         return hand_gesture_map.get(str(hand_gesture_str), 6)

#     def record(self, frame, driver_status):
#         """Record frame and driver status"""
#         if not self.is_recording or self.frame_counter % self.record_interval != 0:
#             self.frame_counter += 1
#             return False
        
#         if len(self.frame_buffer) >= self.buffer_size:
#             print("Buffer full, dropping frame")
#             self.frame_counter += 1
#             return False
        
#         # Extract structured stats
#         frame_stats = self.extract_stats(driver_status)
        
#         if not self.test_mode:
#             self.frame_buffer.append((frame.copy(), self.frame_idx))
#         self.state_buffer.append(frame_stats)
        
#         self.frame_counter += 1
#         self.frame_idx += 1
#         return True

#     def start_recording(self):
#         if self.is_recording:
#             print("Already recording")
#             return
        
#         self.create_data_dir()
#         self.frame_counter = 0
#         self.frame_idx = 0
#         self.frame_buffer.clear()
#         self.state_buffer.clear()
#         self.is_recording = True
#         print(f"Recording started for session: {self.current_session}")

#     def stop_recording(self):
#         if not self.is_recording:
#             print("Not currently recording")
#             return
        
#         self.is_recording = False
#         self._save_all()
#         self.frame_buffer.clear()
#         self.state_buffer.clear()
        
#         if self.test_mode:
#             print(f"Test session({self.current_session}) stats saved in {self.session_dir}")
#         else:
#             print(f"Session({self.current_session}) saved {self.frame_idx} frames to {self.session_dir}")

#     def create_data_dir(self):
#         """Create directory structure for recording session"""
#         import os
        
#         if not os.path.exists(self.base_dir):
#             os.makedirs(self.base_dir, exist_ok=True)
        
#         self.session_dir = os.path.join(self.base_dir, self.current_session)
#         if not self.test_mode:
#             frames_dir = os.path.join(self.session_dir, "frames")
#             os.makedirs(frames_dir, exist_ok=True)
#         stats_dir = os.path.join(self.session_dir, "stats")
#         os.makedirs(stats_dir, exist_ok=True)
        
#         print(f"Created recording session: {self.current_session}")
#         print(f"Data will be saved in: {self.session_dir}")
        
#         return self.session_dir

#     def _save_all(self):
#         """Save all buffered data"""
#         try:
#             import cv2
            
#             if not self.test_mode:
#                 frames_dir = os.path.join(self.session_dir, "frames")
#                 print(f"Saving {len(self.frame_buffer)} frames...")
                
#                 for i, (frame, frame_idx) in enumerate(self.frame_buffer):
#                     if self.frame_idx > 200 and frame_idx % 50 == 0:
#                         print(f"{frame_idx}/{self.frame_idx} saved")
#                     frame_filename = os.path.join(frames_dir, f"{self.current_session}_{frame_idx:05d}_frame.png")
#                     success = cv2.imwrite(frame_filename, frame)
#                     if not success:
#                         print(f"Warning: Failed to save frame {frame_idx}")
            
#             # Save stats as indexed dictionary
#             self._save_all_stats()
#             print("All data saved successfully!")
            
#         except Exception as e:
#             print(f"Error saving recording data: {e}")

#     def _save_all_stats(self):
#         """Save stats indexed by frame_idx for easy access"""
#         stats_dir = os.path.join(self.session_dir, "stats")
        
#         # Create indexed dictionary: {frame_idx: {model: {outputs/states: ...}}}
#         indexed_stats = {}
#         for frame_data in self.state_buffer:
#             frame_idx = frame_data['frame_idx']
#             # Remove frame_idx from the data since it's now the key
#             frame_stats = {k: v for k, v in frame_data.items() if k != 'frame_idx'}
#             indexed_stats[frame_idx] = frame_stats
        
#         # Save indexed stats
#         indexed_filename = os.path.join(stats_dir, f"{self.current_session}_all_stats.json")
#         with open(indexed_filename, 'w') as f:
#             json.dump(indexed_stats, f, separators=(',', ':'), default=str, indent=2)
        
#         print(f"Saved indexed stats to {indexed_filename}")

# # ================================================================
# # USAGE EXAMPLE
# # ================================================================

# class StatsComparator:
#     """Compare stats between ground truth and test results"""
    
#     def __init__(self):
#         pass
    
#     def load_indexed_stats(self, file_path):
#         """Load indexed stats from JSON file"""
#         with open(file_path, 'r') as f:
#             return json.load(f)
    
#     def compare_model_outputs(self, gt_stats, test_stats, model_name, frame_idx):
#         """Compare specific model outputs for a frame"""
#         gt_frame = gt_stats.get(str(frame_idx), {})
#         test_frame = test_stats.get(str(frame_idx), {})
        
#         gt_model = gt_frame.get(model_name, {})
#         test_model = test_frame.get(model_name, {})
        
#         return {
#             'outputs': self._compare_dict(gt_model.get('outputs', {}), test_model.get('outputs', {})),
#             'states': self._compare_dict(gt_model.get('states', {}), test_model.get('states', {}))
#         }
    
#     def _compare_dict(self, dict1, dict2):
#         """Compare two dictionaries"""
#         comparison = {}
#         all_keys = set(dict1.keys()) | set(dict2.keys())
        
#         for key in all_keys:
#             val1 = dict1.get(key)
#             val2 = dict2.get(key)
            
#             if isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
#                 # Compare arrays/lists
#                 comparison[key] = self._compare_arrays(val1, val2)
#             elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
#                 # Compare numbers
#                 comparison[key] = {
#                     'gt': val1,
#                     'test': val2,
#                     'diff': abs(val1 - val2),
#                     'match': abs(val1 - val2) < 0.001
#                 }
#             else:
#                 # Exact comparison
#                 comparison[key] = {
#                     'gt': val1,
#                     'test': val2,
#                     'match': val1 == val2
#                 }
        
#         return comparison
    
#     def _compare_arrays(self, arr1, arr2):
#         """Compare two arrays"""
#         if len(arr1) != len(arr2):
#             return {'match': False, 'reason': 'different_lengths'}
        
#         try:
#             import numpy as np
#             diff = np.array(arr1) - np.array(arr2)
#             mse = np.mean(diff**2)
#             return {
#                 'gt': arr1,
#                 'test': arr2, 
#                 'mse': float(mse),
#                 'match': mse < 0.001
#             }
#         except:
#             return {
#                 'gt': arr1,
#                 'test': arr2,
#                 'match': arr1 == arr2
#             }

# Example usage:
"""
# Recording
recorder = FrameRecorder(session_name="test_session", record_interval=1)
recorder.start_recording()

for frame in frames:
    result = icms_process(frame, frame.shape[0] * frame.shape[1])
    recorder.record(frame, icms_process.driver_status)

recorder.stop_recording()

# Comparison
comparator = StatsComparator()
gt_stats = comparator.load_indexed_stats("tests/data/ground_truth/ground_truth_indexed_stats.json")
test_stats = comparator.load_indexed_stats("tests/data/test_session/stats/test_session_indexed_stats.json")

# Compare specific model for specific frame
result = comparator.compare_model_outputs(gt_stats, test_stats, "face_landmark", frame_idx=5)
print(f"Face landmark outputs match: {result['outputs']['landmarks_68']['match']}")
print(f"Face landmark states match: {result['states']['landmark_confidence']['match']}")

# Access specific values
gt_head_pose = gt_stats[frame_idx]["head_pose"]["states"]["pitch"]
test_head_pose = test_stats[frame_idx]["head_pose"]["states"]["pitch"]
"""
# import os
# import cv2
# import json
# import queue
# import threading
# import time
# from typing import Dict, Any, Optional
# import numpy as np

# class FrameRecorder():
#     def __init__(self,
#                  session_name: Optional[str] = None,
#                  record_interval: int = 10,
#                  buffer_size: int = 500,
#                  base_dir: str = "tests/data",
#                  test_mode: bool = False
#     ):
#         """
#         Initialize FrameRecorder
#         Args:
#             session_name (str): Name for the recording session, uses timestamp if None
#             record_interval (int): Interval in frames to record
#             buffer_size (int): Maximum number of frames to keep in memory
#             base_dir (str): Base directory to save recorded frames
#         """
        
#         self.record_interval = record_interval
#         self.buffer_size = buffer_size
#         self.base_dir = base_dir
#         self.test_mode = test_mode
        
#         # Recording state
#         self.is_recording = False
#         self.frame_counter = 0
#         self.frame_idx = 0
        
#         # Buffers for frames and states
#         self.frame_buffer = []
#         self.state_buffer = []
        
#         # Session info
#         if session_name is not None:
#             self.current_session = session_name
#         else:
#             self.current_session = f"session_{int(time.time())}"
#         self.session_dir = None
        
#     def create_data_dir(self) -> str:
#         """Create directory structure for recording session"""
#         # Directory structure:
#         # tests/data/
#         #   └── session_name/
#         #       ├── frames/
#         #       └── stats/
    
#         # Create base directory if not exists
#         if not os.path.exists(self.base_dir):
#             os.makedirs(self.base_dir, exist_ok=True)
        
#         # Create directories for the current session
#         self.session_dir = os.path.join(self.base_dir, self.current_session)
#         if not self.test_mode:
#             frames_dir = os.path.join(self.session_dir, "frames")
#             os.makedirs(frames_dir, exist_ok=True)
#         stats_dir = os.path.join(self.session_dir, "stats")
#         os.makedirs(stats_dir, exist_ok=True)
        
#         # Diagnostic message
#         print(f"Created recording session: {self.current_session}")
#         print(f"Data will be saved in: {self.session_dir}")
        
#         return self.session_dir
    
#     def extract_stats(self, driver_status) -> Dict[str, Any]:
#         """Extract comprehensive driver status organized by model outputs"""
#         try:
#             data = {
#                 # =================================================================
#                 # OVERALL SYSTEM STATE (Top-level status indicators)
#                 # =================================================================
#                 "overall": {
#                     "frame_index": self.frame_idx,
#                     "timestamp": int(time.time()),
#                     "session_name": getattr(self, 'current_session', 'unknown'),
                    
#                     # High-level detection states
#                     "is_face_detected": getattr(driver_status, 'is_face', False),
#                     "is_hand_detected": getattr(driver_status, 'is_hand', False),
#                     "is_body_detected": getattr(driver_status, 'activate_body', False),
                    
#                     # Final safety decisions
#                     "microsleep": getattr(driver_status, 'microsleep', False),
#                     "sleep": getattr(driver_status, 'sleep', False),
#                     "yawning": getattr(driver_status, 'yawning', False),
#                     "distracted": getattr(driver_status, 'distracted', False),
#                     "long_distracted": getattr(driver_status, 'long_distracted', False),
#                     "smoking": getattr(driver_status, 'smoking', 0),
#                     "using_phone": getattr(driver_status, 'usingphone', 0),
                    
#                     # Computed metrics
#                     "drowsiness_score": float(getattr(driver_status, 'drowsy_val', 0.0)),
#                     "distraction_score": float(getattr(driver_status, 'distracted_val', 0.0)),
                    
#                     # Identity and attributes
#                     "driver_id": str(getattr(driver_status, 'ID', 'Unknown')),
#                     "gender": int(getattr(driver_status, 'gender', 3)),
#                     "age": int(getattr(driver_status, 'age', 3)),
#                     "glasses": int(getattr(driver_status, 'glasses', 0)),
#                     "mask": int(getattr(driver_status, 'mask', 0)),
#                 },
                
#                 # =================================================================
#                 # MODEL-SPECIFIC OUTPUTS (Organized by model)
#                 # =================================================================
#                 "models": {
#                     # Object Detection Model
#                     "object_detector": {
#                         "face_boxes": self._extract_detection_boxes(driver_status, 'face_boxes'),
#                         "body_boxes": self._extract_detection_boxes(driver_status, 'body_boxes'),
#                         "hand_boxes": self._extract_detection_boxes(driver_status, 'hand_boxes'),
#                         "phone_boxes": self._extract_detection_boxes(driver_status, 'phone_boxes'),
#                         "selected_face_box": getattr(driver_status, 'recent_face_location', None),
#                         "face_area": float(getattr(driver_status, 'face_area', 0)),
#                         "detection_metadata": {
#                             "confidence_threshold": getattr(driver_status, 'confidence_threshold', 0.1),
#                             "nms_threshold": getattr(driver_status, 'nms_threshold', 0.3),
#                             "same_face_tracking": getattr(driver_status, 'same_face', False),
#                         }
#                     },
                    
#                     # Face Landmark Model  
#                     "face_landmark": {
#                         "landmarks_68": getattr(driver_status, 'facelandmark', [[0.0, 0.0]] * 68),
#                         "eye_landmarks": {
#                             "left_eye": self._get_landmark_subset(driver_status, 36, 42),
#                             "right_eye": self._get_landmark_subset(driver_status, 42, 48),
#                             "mouth": self._get_landmark_subset(driver_status, 60, 68),
#                         },
#                         "landmark_confidence": getattr(driver_status, 'landmark_confidence', 0.0),
#                     },
                    
#                     # Head Pose Model
#                     "head_pose": {
#                         "pose_angles": self._safe_array_extract(driver_status, 'head_pose', [0, 0, 0]),
#                         "pitch": float(self._safe_array_extract(driver_status, 'head_pose', [0, 0, 0])[0]),
#                         "yaw": float(self._safe_array_extract(driver_status, 'head_pose', [0, 0, 0])[1]),
#                         "roll": float(self._safe_array_extract(driver_status, 'head_pose', [0, 0, 0])[2]),
#                         "pose_ready": getattr(driver_status, 'HP_READY', False),
#                         "pose_baseline": getattr(driver_status, 'PYRavg', [0, 0, 0]),
#                     },
                    
#                     # Gaze Estimation Model
#                     "gaze_estimator": {
#                         "gaze_direction": self._safe_array_extract(driver_status, 'gaze', [0, 0]),
#                         "gaze_pitch": float(self._safe_array_extract(driver_status, 'gaze', [0, 0])[1]) if len(self._safe_array_extract(driver_status, 'gaze', [0, 0])) > 1 else 0.0,
#                         "gaze_yaw": float(self._safe_array_extract(driver_status, 'gaze', [0, 0])[0]) if len(self._safe_array_extract(driver_status, 'gaze', [0, 0])) > 0 else 0.0,
#                         "eye_state": {
#                             "left_eye_open": getattr(driver_status, 'left_eye_open', None),
#                             "right_eye_open": getattr(driver_status, 'right_eye_open', None),
#                             "eyes_open": self._check_eyes_open(driver_status),
#                         }
#                     },
                    
#                     # Hand Detection & Classification
#                     "hand_detector": {
#                         "hand_boxes": self._extract_detection_boxes(driver_status, 'hand_boxes'),
#                         "selected_hand_box": getattr(driver_status, 'hand_box', None),
#                         "hand_gesture": str(getattr(driver_status, 'hand_gesture', 'None')),
#                         "hand_gesture_id": self._convert_hand_gesture(getattr(driver_status, 'hand_gesture', 'None')),
#                         "gesture_active": getattr(driver_status, 'activate_hand_gesture', False),
#                         "gesture_confidence": getattr(driver_status, 'gesture_confidence', 0.0),
#                     },
                    
#                     # Action Recognition (Phone/Smoking)
#                     "action_detector": {
#                         "action_scores": self._safe_array_extract(driver_status, 'action_val', [0.0, 0.0]),
#                         "smoking_confidence": float(self._safe_array_extract(driver_status, 'action_val', [0.0, 0.0])[0]),
#                         "phone_confidence": float(self._safe_array_extract(driver_status, 'action_val', [0.0, 0.0])[1]),
#                         "action_thresholds": getattr(driver_status, 'action_thr', [0.9, 0.7]),
#                     },
                    
#                     # Face Attribute Models
#                     "face_attributes": {
#                         "age_gender_scores": self._safe_array_extract(driver_status, 'face_info_val', [0.0] * 5),
#                         "accessory_scores": self._safe_array_extract(driver_status, 'face_acce_val', [0.0] * 2),
#                         "attribute_confidence": {
#                             "age_confidence": getattr(driver_status, 'age_confidence', 0.0),
#                             "gender_confidence": getattr(driver_status, 'gender_confidence', 0.0),
#                             "glasses_confidence": getattr(driver_status, 'glasses_confidence', 0.0),
#                             "mask_confidence": getattr(driver_status, 'mask_confidence', 0.0),
#                         },
#                         "attributes_fixed": getattr(driver_status, 'fix_face_info', False),
#                     },
                    
#                     # Body Pose Model (if enabled)
#                     "body_pose": {
#                         "keypoints_3d": self._safe_array_extract(driver_status, 'bodykeypoints', np.zeros((10, 3))),
#                         "keypoints_2d": self._safe_array_extract(driver_status, 'vis_bodykeypoints', np.zeros((10, 2))),
#                         "body_active": getattr(driver_status, 'activate_body', False),
#                         "keypoint_threshold": getattr(driver_status, 'thr_bodykeypoint', 0.2),
#                     },
#                 },
                
#                 # =================================================================
#                 # COMPUTED METRICS (Derived from model outputs)
#                 # =================================================================
#                 "metrics": {
#                     # Eye-based metrics
#                     "eye_aspect_ratio": float(getattr(driver_status, 'EAR', 0.0)),
#                     "lip_aspect_ratio": float(getattr(driver_status, 'LAR', 0.0)),
#                     "ear_threshold": getattr(driver_status, 'EAR_thres', 0.4),
#                     "lar_threshold": getattr(driver_status, 'LAR_thresh', 0.4),
                    
#                     # Head gesture detection
#                     "head_gestures": {
#                         "talking": getattr(driver_status, 'talking', False),
#                         "nodding": getattr(driver_status, 'nodding', False),
#                         "shaking": getattr(driver_status, 'shaking', False),
#                     },
                    
#                     # Distraction metrics
#                     "distraction_angle_threshold": getattr(driver_status, 'distracted_angle_thres', 40),
#                     "distraction_threshold": getattr(driver_status, 'distraction_thres', 0.6),
                    
#                     # Algorithm state
#                     "processing_state": {
#                         "head_pose_calibrated": getattr(driver_status, 'HP_READY', False),
#                         "face_tracking_active": getattr(driver_status, 'same_face', False),
#                         "face_id_fixed": getattr(driver_status, 'fix_face_info', False),
#                     }
#                 }
#             }
            
#             # Convert numpy arrays to lists for JSON serialization
#             data = self._serialize_nested_arrays(data)
#             return data
            
#         except Exception as e:
#             print(f"Error extracting driver data: {e}")
#             return {
#                 'overall': {
#                     'frame_index': self.frame_idx,
#                     'timestamp': int(time.time()),
#                     'error': str(e)
#                 },
#                 'models': {},
#                 'metrics': {}
#             }

#     def _safe_array_extract(self, obj, attr_name, default):
#         """Safely extract array attributes with fallback"""
#         try:
#             value = getattr(obj, attr_name, default)
#             if hasattr(value, 'tolist'):
#                 return value.tolist()
#             elif isinstance(value, (list, tuple)):
#                 return list(value)
#             else:
#                 return default
#         except:
#             return default
    
#     def _get_landmark_subset(self, driver_status, start_idx, end_idx):
#         """Extract specific landmark points safely"""
#         try:
#             landmarks = getattr(driver_status, 'facelandmark', [[0.0, 0.0]] * 68)
#             if len(landmarks) >= end_idx:
#                 return landmarks[start_idx:end_idx]
#             else:
#                 return [[0.0, 0.0]] * (end_idx - start_idx)
#         except:
#             return [[0.0, 0.0]] * (end_idx - start_idx)
    
#     def _check_eyes_open(self, driver_status):
#         """Check if eyes are open using EAR or eye state"""
#         try:
#             if hasattr(driver_status, 'is_eye_open') and callable(getattr(driver_status, 'is_eye_open')):
#                 return driver_status.is_eye_open()
#             else:
#                 # Fallback: use EAR threshold
#                 ear = getattr(driver_status, 'EAR', 0.0)
#                 ear_threshold = getattr(driver_status, 'ear_open_threshold', 0.2)
#                 return ear > ear_threshold
#         except:
#             return None
        
#     def _extract_detection_boxes(self, obj, attr_name):
#         """Extract detection boxes with full metadata"""
#         try:
#             boxes = getattr(obj, attr_name, [])
#             if not boxes:
#                 return []
            
#             formatted_boxes = []
#             for box in boxes:
#                 if isinstance(box, (list, tuple)) and len(box) >= 4:
#                     formatted_boxes.append({
#                         'x1': float(box[0]),
#                         'y1': float(box[1]),
#                         'x2': float(box[2]), 
#                         'y2': float(box[3]),
#                         'confidence': float(box[4]) if len(box) > 4 else 1.0,
#                         'class_id': int(box[5]) if len(box) > 5 else 0,
#                         'area': float((box[2] - box[0]) * (box[3] - box[1])),
#                         'center': [float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)]
#                     })
#             return formatted_boxes
            
#         except Exception as e:
#             print(f"Error extracting detection boxes for {attr_name}: {e}")
#             return []

#     def _serialize_nested_arrays(self, data):
#         """Recursively convert numpy arrays in nested dictionaries"""
#         if isinstance(data, dict):
#             return {key: self._serialize_nested_arrays(value) for key, value in data.items()}
#         elif isinstance(data, list):
#             return [self._serialize_nested_arrays(item) for item in data]
#         elif isinstance(data, np.ndarray):
#             return data.tolist()
#         elif isinstance(data, (np.integer, np.floating)):
#             return data.item()
#         else:
#             return data

#     def _convert_hand_gesture(self, hand_gesture_str: str) -> int:
#         """Convert hand gesture string to integer"""
#         hand_gesture_map = {
#             'Zero': 0, 'One': 1, 'Two': 2, 'Three': 3, 
#             'Four': 4, 'Five': 5, 'None': 6
#         }
#         return hand_gesture_map.get(str(hand_gesture_str), 6)

#     def record(self, frame: np.ndarray, driver_status) -> bool:
#         """Record frame and driver status if conditions are met"""
#         if not self.is_recording:
#             return False
        
#         # Exit if not in record interval
#         if self.frame_counter % self.record_interval != 0:
#             self.frame_counter += 1
#             return False
        
#         # Check buffer size before operation
#         if len(self.frame_buffer) >= self.buffer_size:
#             print("Buffer full, dropping frame")
#             self.frame_counter += 1
#             return False
        
#         # Extract data using new method
#         data = self.extract_stats(driver_status)
        
#         self.frame_buffer.append((frame.copy(), self.frame_idx))
#         self.state_buffer.append(data)
        
#         self.frame_counter += 1
#         self.frame_idx += 1
#         return True
        
#     def start_recording(self):
#         if self.is_recording:
#             print("Already recording")
#             return
        
#         self.create_data_dir()
        
#         self.frame_counter = 0
#         self.frame_idx = 0
        
#         # Clear buffers
#         self.frame_buffer.clear()
#         self.state_buffer.clear()
        
#         self.is_recording = True
#         print(f"Recording started for session: {self.current_session}")
        
#     def stop_recording(self):
#         if not self.is_recording:
#             print("Not currently recording")
#             return
        
#         self.is_recording = False
        
#         # Save all recordings at once
#         self._save_all()
        
#         # Clear buffers
#         self.frame_buffer.clear()
#         self.state_buffer.clear()
        
#         if self.test_mode:
#             print(f"Test session({self.current_session} stats saved in {self.session_dir})")
#         else:
#             print(f"Session({self.current_session}) saved {self.frame_idx} frames to {self.session_dir}")
        
#     def _save_all(self):
#         """Save all buffered data at once"""
#         try:
#             if not self.test_mode:
#                 frames_dir = os.path.join(self.session_dir, "frames")
#             stats_dir = os.path.join(self.session_dir, "stats")
            
#             if not self.test_mode:
#                 show_progress = True if self.frame_idx > 200 else False
#                 print(f"Saving {len(self.frame_buffer)} frames...")
                
#                 for i, (frame, frame_idx) in enumerate(self.frame_buffer):
#                     if show_progress:
#                         if frame_idx % 50 == 0:
#                             print(f"{frame_idx}/{self.frame_idx} saved")
#                     frame_filename = os.path.join(frames_dir, f"{self.current_session}_{frame_idx:05d}_frame.png")
#                     success = cv2.imwrite(frame_filename, frame)
#                     if not success:
#                         print(f"Warning: Failed to save frame {frame_idx}")
            
#             print("Saving data...")
#             self.save_stats(stats_dir)
#             print("All data saved successfully!")
            
#         except Exception as e:
#             print(f"Error saving recording data: {e}")
            
#     def save_stats(self, stats_dir, all_in_one=True):
#         """Save extracted data in one file or separate JSON files"""
#         if all_in_one:
#             # Save all data in one JSON file
#             batch_filename = os.path.join(stats_dir, f"{self.current_session}_all_data.json")
#             with open(batch_filename, 'w') as f:
#                 json.dump(self.state_buffer, f, separators=(',', ':'), default=str)
#             print(f"Saved all data to {batch_filename}")
#         else:
#             # Save each frame as individual file
#             for data in self.state_buffer:
#                 frame_idx = data['overall']['frame_index']
#                 data_filename = os.path.join(stats_dir, f"{self.current_session}_{frame_idx:05d}_data.json")
#                 with open(data_filename, 'w') as f:
#                     json.dump(data, f, separators=(',', ':'), default=str)
#             print(f"Saved {len(self.state_buffer)} individual data files")
    