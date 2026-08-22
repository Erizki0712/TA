
import csv
import json
import math
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pynmea2
import serial
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, simpledialog
from tkintermapview import TkinterMapView

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "v8s_320.onnx"
VIDEO_PATH = "vid1.mp4"

IMG_SIZE = 320
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
SAVE_INTERVAL = 2.0

CLASS_NAMES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes",
]

GPS_PORT = "/dev/ttyAMA0"
GPS_BAUDRATE = 9600

SESSION_ROOT = Path("sessions")
SESSION_ROOT.mkdir(exist_ok=True)

OUT_DIR = Path(".")
DEFAULT_FRAME_CSV = "recorded_frames.csv"
DEFAULT_GPS_CSV = "gps_log.csv"
DEFAULT_DET_CSV = "detections.csv"
SESSION_META = "session.json"

# =========================================================
# HELPERS
# =========================================================
def ensure_csv(path: Path, header: list[str]) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def safe_float(value, default=None):
    try:
        if value in (None, "", "N/A"):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=None):
    try:
        if value in (None, "", "N/A"):
            return default
        return int(float(value))
    except Exception:
        return default


def format_time(ts: float | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def meters_to_lat_deg(meters: float) -> float:
    return meters / 111320.0


def meters_to_lng_deg(meters: float, lat_deg: float) -> float:
    denom = 111320.0 * max(math.cos(math.radians(lat_deg)), 1e-8)
    return meters / denom


def sanitize_folder_name(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_", " ") else "_" for c in name.strip())
    safe = "_".join(safe.split())
    return safe or "session"


def unique_session_dir(base_name: str) -> Path:
    base = sanitize_folder_name(base_name)
    candidate = SESSION_ROOT / base
    idx = 2
    while candidate.exists():
        candidate = SESSION_ROOT / f"{base}_{idx}"
        idx += 1
    return candidate


def preprocess(frame: np.ndarray) -> np.ndarray:
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def nms(boxes, scores):
    if not boxes:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
    if idxs is None or len(idxs) == 0:
        return []
    return idxs.flatten().tolist()


def parse_gsv_signal(raw_sentence: str):
    """
    Parse $GxGSV. Returns:
        satellites_in_view (int|None)
        avg_snr (float|None)
        max_snr (float|None)
    """
    try:
        fields = raw_sentence.strip().split(",")
        if len(fields) < 4:
            return None, None, None

        satellites_in_view = safe_int(fields[3], None)
        snrs = []
        for idx in range(7, len(fields), 4):
            token = fields[idx].split("*")[0].strip()
            if token.isdigit():
                snrs.append(int(token))

        avg_snr = float(sum(snrs) / len(snrs)) if snrs else None
        max_snr = max(snrs) if snrs else None
        return satellites_in_view, avg_snr, max_snr
    except Exception:
        return None, None, None


def nmea_type(raw: str) -> str:
    if not raw.startswith("$") or len(raw) < 6:
        return "UNKNOWN"
    return raw.split(",")[0][1:]


def read_csv_dicts(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# =========================================================
# SESSION MANAGER
# =========================================================
class SessionManager:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(exist_ok=True)

    def list_sessions(self):
        sessions = []
        for p in sorted(self.root_dir.iterdir(), key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True):
            if not p.is_dir():
                continue
            meta = self.load_meta(p)
            sessions.append((p, meta))
        return sessions

    def load_meta(self, session_dir: Path):
        meta_path = session_dir / SESSION_META
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # fallback from filesystem
        status = "new"
        if (session_dir / DEFAULT_FRAME_CSV).exists() or (session_dir / DEFAULT_GPS_CSV).exists():
            status = "recorded"
        if (session_dir / DEFAULT_DET_CSV).exists():
            status = "inferred"

        return {
            "session_name": session_dir.name,
            "folder_name": session_dir.name,
            "created_at": None,
            "status": status,
            "source_video": VIDEO_PATH,
            "inference_done": (session_dir / DEFAULT_DET_CSV).exists(),
            "recording_done": (session_dir / DEFAULT_FRAME_CSV).exists() or (session_dir / DEFAULT_GPS_CSV).exists(),
        }

    def save_meta(self, session_dir: Path, meta: dict):
        meta_path = session_dir / SESSION_META
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def create_session(self, name: str):
        session_dir = unique_session_dir(name)
        session_dir.mkdir(parents=True, exist_ok=False)

        created_at = datetime.now().isoformat(timespec="seconds")
        meta = {
            "session_name": name.strip(),
            "folder_name": session_dir.name,
            "created_at": created_at,
            "status": "recording",
            "source_video": VIDEO_PATH,
            "inference_done": False,
            "recording_done": False,
        }
        self.save_meta(session_dir, meta)

        ensure_csv(
            session_dir / DEFAULT_FRAME_CSV,
            ["frame_index", "frame_timestamp"],
        )
        ensure_csv(
            session_dir / DEFAULT_GPS_CSV,
            [
                "timestamp", "lat", "lng", "signal_dbhz", "num_sats", "fix_quality",
                "hdop", "pdop", "vdop", "altitude", "geoid_sep",
                "speed_knots", "speed_kmh", "course_deg", "gps_mode", "gps_pattern"
            ],
        )
        ensure_csv(
            session_dir / DEFAULT_DET_CSV,
            [
                "frame_index",
                "frame_timestamp",
                "frame_image",
                "label",
                "confidence",
                "gps_before_timestamp",
                "gps_after_timestamp",
                "interp_lat",
                "interp_lng",
                "gps_before_lat",
                "gps_before_lng",
                "gps_after_lat",
                "gps_after_lng",
                "gps_before_signal_dbhz",
                "gps_after_signal_dbhz",
                "gps_mode",
                "gps_pattern",
            ],
        )
        return session_dir, meta

    def update_status(self, session_dir: Path, status: str, **kwargs):
        meta = self.load_meta(session_dir)
        meta["status"] = status
        if status == "inferred":
            meta["inference_done"] = True
        if status == "recorded":
            meta["recording_done"] = True
        meta.update(kwargs)
        self.save_meta(session_dir, meta)
        return meta

    def load_session_payload(self, session_dir: Path):
        meta = self.load_meta(session_dir)
        frames = read_csv_dicts(session_dir / DEFAULT_FRAME_CSV)
        gps = read_csv_dicts(session_dir / DEFAULT_GPS_CSV)
        dets = read_csv_dicts(session_dir / DEFAULT_DET_CSV)
        return meta, frames, gps, dets


# =========================================================
# APPLICATION
# =========================================================
class DetectionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Detection Logger")
        self.root.geometry("1400x900")

        self.session_manager = SessionManager(SESSION_ROOT)

        # -------- state --------
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.stop_record_event = threading.Event()
        self.stop_infer_event = threading.Event()
        self.recording_active = False
        self.inference_active = False
        self.playback_thread = None
        self.infer_thread = None
        self.frame_display = None

        self.current_lat = -6.2
        self.current_lng = 106.8
        self.follow_gps = True

        self.gps_mode = "ONLINE"   # ONLINE / OFFLINE
        self.offline_pattern = "STRAIGHT"  # STRAIGHT / CIRCLE
        self.offline_speed_mps = 5.0
        self.offline_heading_deg = 45.0
        self.offline_circle_radius_m = 30.0
        self.offline_center_lat = self.current_lat
        self.offline_center_lng = self.current_lng
        self.offline_angle_rad = 0.0
        self.offline_last_ts = time.time()
        self.offline_straight_lat = self.current_lat
        self.offline_straight_lng = self.current_lng

        self.latest_signal_dbhz = None
        self.latest_num_sats = None
        self.latest_fix_quality = None
        self.latest_hdop = None
        self.latest_pdop = None
        self.latest_vdop = None
        self.latest_altitude = None
        self.latest_geoid_sep = None
        self.latest_speed_knots = None
        self.latest_speed_kmh = None
        self.latest_course_deg = None
        self.latest_gll_status = None
        self.latest_rmc_status = None
        self.latest_satellites_in_view = None
        self.latest_max_snr = None

        self.gps_state = {
            "mode": self.gps_mode,
            "pattern": self.offline_pattern,
            "last_sentence_type": "",
            "last_raw_sentence": "",
            "lat": self.current_lat,
            "lng": self.current_lng,
            "signal_dbhz": None,
            "num_sats": None,
            "fix_quality": None,
            "hdop": None,
            "pdop": None,
            "vdop": None,
            "altitude": None,
            "geoid_sep": None,
            "speed_knots": None,
            "speed_kmh": None,
            "course_deg": None,
            "rmc_status": None,
            "gll_status": None,
            "gsv_satellites_in_view": None,
            "gsv_avg_snr": None,
            "gsv_max_snr": None,
            "gsa_fix_type": None,
        }

        self.nmea_tail = deque(maxlen=10)

        # session/runtime
        self.selected_session_dir = None
        self.current_session_dir = None
        self.current_session_meta = None
        self.session_frames = []
        self.session_gps = []
        self.session_detections = []
        self.session_frame_count = 0

        self.damage_markers = []
        self.session_route_marker = None

        self._build_ui()
        self._load_model()

        threading.Thread(target=self.gps_thread, daemon=True).start()
        self.update_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # =====================================================
    # UI
    # =====================================================

    def _build_ui(self):
        # Root layout: fixed left sidebar, expandable map area
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=0, minsize=460)
        self.root.grid_columnconfigure(1, weight=1)

        # =====================================================
        # LEFT SIDEBAR
        # =====================================================
        self.sidebar_outer = tk.Frame(self.root, bg="white", width=460, height=900)
        self.sidebar_outer.grid(row=0, column=0, sticky="nsew")
        self.sidebar_outer.grid_propagate(False)
        self.sidebar_outer.grid_columnconfigure(0, weight=1)

        # =====================================================
        # RIGHT MAP AREA
        # =====================================================
        self.map_container = tk.Frame(self.root, bg="white")
        self.map_container.grid(row=0, column=1, sticky="nsew")
        self.map_container.grid_rowconfigure(0, weight=1)
        self.map_container.grid_columnconfigure(0, weight=1)

        self.map_widget = TkinterMapView(self.map_container)
        self.map_widget.pack(fill="both", expand=True)
        self.map_widget.set_position(self.current_lat, self.current_lng)
        self.zoom_level = 15
        self.map_widget.set_zoom(self.zoom_level)

        self.live_marker = self.map_widget.set_marker(self.current_lat, self.current_lng, text="GPS Live")
        self.damage_markers = []

        # =====================================================
        # SESSION & CONTROL
        # =====================================================
        self.control_frame = tk.LabelFrame(self.sidebar_outer, text="Session & Control", bg="white")
        self.control_frame.grid(row=0, column=0, padx=8, pady=8, sticky="nwe")
        for c in range(4):
            self.control_frame.grid_columnconfigure(c, weight=1)

        session_list_container = tk.Frame(self.control_frame, bg="white")
        session_list_container.grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky="we")
        session_list_container.grid_columnconfigure(0, weight=1)
        session_list_container.grid_rowconfigure(0, weight=1)

        self.session_listbox = tk.Listbox(session_list_container, height=7, width=44, font=("Arial", 10))
        self.session_listbox.grid(row=0, column=0, sticky="nsew")

        self.session_list_vscroll = tk.Scrollbar(session_list_container, orient="vertical", command=self.session_listbox.yview)
        self.session_list_vscroll.grid(row=0, column=1, sticky="ns")

        self.session_list_hscroll = tk.Scrollbar(session_list_container, orient="horizontal", command=self.session_listbox.xview)
        self.session_list_hscroll.grid(row=1, column=0, sticky="we")

        self.session_listbox.configure(yscrollcommand=self.session_list_vscroll.set, xscrollcommand=self.session_list_hscroll.set)
        self.session_listbox.bind("<<ListboxSelect>>", self.on_session_select)

        btn_opts = dict(width=10, padx=2, pady=2, font=("Arial", 9))
        self.new_btn = tk.Button(self.control_frame, text="New Session", command=self.new_session, **btn_opts)
        self.new_btn.grid(row=1, column=0, padx=3, pady=3, sticky="we")

        self.load_btn = tk.Button(self.control_frame, text="Load Session", command=self.load_selected_session, **btn_opts)
        self.load_btn.grid(row=1, column=1, padx=3, pady=3, sticky="we")

        self.refresh_btn = tk.Button(self.control_frame, text="Refresh", command=self.refresh_sessions, **btn_opts)
        self.refresh_btn.grid(row=1, column=2, padx=3, pady=3, sticky="we")

        self.center_btn = tk.Button(self.control_frame, text="Recenter", command=self.center_map, **btn_opts)
        self.center_btn.grid(row=1, column=3, padx=3, pady=3, sticky="we")

        self.start_record_btn = tk.Button(self.control_frame, text="Start Recording", command=self.start_recording_from_selection, **btn_opts)
        self.start_record_btn.grid(row=2, column=0, padx=3, pady=3, sticky="we")

        self.stop_record_btn = tk.Button(self.control_frame, text="Stop Recording", command=self.stop_recording, **btn_opts)
        self.stop_record_btn.grid(row=2, column=1, padx=3, pady=3, sticky="we")

        self.start_infer_btn = tk.Button(self.control_frame, text="Start Inference", command=self.start_inference, **btn_opts)
        self.start_infer_btn.grid(row=2, column=2, padx=3, pady=3, sticky="we")

        self.replot_btn = tk.Button(self.control_frame, text="Show Map", command=self.show_current_session_map, **btn_opts)
        self.replot_btn.grid(row=2, column=3, padx=3, pady=3, sticky="we")

        self.status_var = tk.StringVar(value="Status: Idle")
        self.status_label = tk.Label(self.control_frame, textvariable=self.status_var, bg="white", anchor="w", justify="left", wraplength=420)
        self.status_label.grid(row=3, column=0, columnspan=4, padx=5, pady=2, sticky="we")

        self.session_var = tk.StringVar(value="Session: -")
        self.session_label = tk.Label(self.control_frame, textvariable=self.session_var, bg="white", anchor="w", justify="left", wraplength=420)
        self.session_label.grid(row=4, column=0, columnspan=4, padx=5, pady=2, sticky="we")

        self.info_var = tk.StringVar(value="GPS: N/A")
        self.info_label = tk.Label(self.control_frame, textvariable=self.info_var, bg="white", anchor="w", justify="left", wraplength=420)
        self.info_label.grid(row=5, column=0, columnspan=4, padx=5, pady=2, sticky="we")

        # =====================================================
        # GPS MODE / DUMMY CONTROL
        # =====================================================
        mode_frame = tk.LabelFrame(self.sidebar_outer, text="GPS Mode", bg="white")
        mode_frame.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="we")
        for c in range(4):
            mode_frame.grid_columnconfigure(c, weight=1)

        self.gps_mode_var = tk.StringVar(value="ONLINE")
        self.offline_pattern_var = tk.StringVar(value="STRAIGHT")
        self.speed_var = tk.DoubleVar(value=5.0)
        self.heading_var = tk.DoubleVar(value=45.0)
        self.radius_var = tk.DoubleVar(value=30.0)

        tk.Radiobutton(mode_frame, text="Online", variable=self.gps_mode_var, value="ONLINE", bg="white", command=self.on_mode_change).grid(row=0, column=0, sticky="w", padx=4, pady=2)
        tk.Radiobutton(mode_frame, text="Offline", variable=self.gps_mode_var, value="OFFLINE", bg="white", command=self.on_mode_change).grid(row=0, column=1, sticky="w", padx=4, pady=2)
        tk.Radiobutton(mode_frame, text="Straight", variable=self.offline_pattern_var, value="STRAIGHT", bg="white", command=self.on_mode_change).grid(row=1, column=0, sticky="w", padx=4, pady=2)
        tk.Radiobutton(mode_frame, text="Circle", variable=self.offline_pattern_var, value="CIRCLE", bg="white", command=self.on_mode_change).grid(row=1, column=1, sticky="w", padx=4, pady=2)

        tk.Label(mode_frame, text="Speed m/s", bg="white").grid(row=2, column=0, sticky="w", padx=4)
        tk.Spinbox(mode_frame, from_=0.5, to=30.0, increment=0.5, textvariable=self.speed_var, width=8, command=self.on_mode_change).grid(row=2, column=1, sticky="w", padx=4)
        tk.Label(mode_frame, text="Heading", bg="white").grid(row=2, column=2, sticky="w", padx=4)
        tk.Spinbox(mode_frame, from_=0, to=359, increment=5, textvariable=self.heading_var, width=8, command=self.on_mode_change).grid(row=2, column=3, sticky="w", padx=4)
        tk.Label(mode_frame, text="Radius", bg="white").grid(row=3, column=0, sticky="w", padx=4)
        tk.Spinbox(mode_frame, from_=5, to=200, increment=5, textvariable=self.radius_var, width=8, command=self.on_mode_change).grid(row=3, column=1, sticky="w", padx=4)

        self.mode_info_var = tk.StringVar(value="Mode: ONLINE | Pattern: STRAIGHT")
        tk.Label(mode_frame, textvariable=self.mode_info_var, bg="white", justify="left", wraplength=420).grid(row=4, column=0, columnspan=4, sticky="w", padx=4, pady=4)

        # =====================================================
        # GPS DEBUG / NMEA
        # =====================================================
        self.gps_debug_frame = tk.LabelFrame(self.sidebar_outer, text="GPS Debug / NMEA", bg="white")
        self.gps_debug_frame.grid(row=2, column=0, padx=8, pady=(0, 8), sticky="nsew")
        self.sidebar_outer.grid_rowconfigure(2, weight=1)

        self.gps_debug_text = tk.Text(self.gps_debug_frame, width=48, height=18, wrap="word")
        self.gps_debug_text.pack(side="left", fill="both", expand=True)
        self.gps_debug_text.configure(state="disabled")

        scroll = tk.Scrollbar(self.gps_debug_frame, command=self.gps_debug_text.yview)
        scroll.pack(side="right", fill="y")
        self.gps_debug_text.configure(yscrollcommand=scroll.set)

        # =====================================================
        # MAP OVERLAYS / CONTROLS
        # =====================================================
        self.zoom_panel = tk.Frame(self.map_container, bg="white", bd=2, relief="raised")
        self.zoom_panel.place(relx=1.0, x=-12, y=12, anchor="ne")

        zoom_btn_style = dict(width=5, height=1, font=("Arial", 16, "bold"), bg="#f2f2f2", fg="black", relief="raised", bd=1, padx=2, pady=2)
        tk.Button(self.zoom_panel, text="+", command=self.zoom_in, **zoom_btn_style).pack(fill="x", padx=2, pady=(2, 1))
        tk.Button(self.zoom_panel, text="-", command=self.zoom_out, **zoom_btn_style).pack(fill="x", padx=2, pady=1)
        tk.Button(self.zoom_panel, text="Center", command=self.center_map, **zoom_btn_style).pack(fill="x", padx=2, pady=(1, 2))

        self.camera_frame = tk.Label(self.map_container, bg="black", bd=2, relief="solid")
        self.camera_frame.place(relx=1.0, rely=1.0, anchor="se", width=360, height=240)

        self.preview_var = tk.StringVar(value="Preview: -")
        self.preview_label = tk.Label(self.map_container, textvariable=self.preview_var, bg="white", bd=1, relief="solid", anchor="w")
        self.preview_label.place(relx=1.0, rely=1.0, x=-360, y=-250, anchor="sw", width=360, height=24)

        self.refresh_sessions()
    def _load_model(self):
        if not Path(MODEL_PATH).exists():
            print(f"[MODEL ERROR] File model tidak ditemukan: {MODEL_PATH}")
            self.session = None
            return

        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        providers = ["CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(MODEL_PATH, so, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            print("[MODEL] ONNX loaded:", MODEL_PATH)
        except Exception as e:
            print("[MODEL ERROR]", e)
            self.session = None

    def on_mode_change(self):
        self.gps_mode = self.gps_mode_var.get()
        self.offline_pattern = self.offline_pattern_var.get()
        self.offline_speed_mps = float(self.speed_var.get())
        self.offline_heading_deg = float(self.heading_var.get())
        self.offline_circle_radius_m = max(5.0, float(self.radius_var.get()))

        if self.gps_mode == "OFFLINE":
            self.offline_center_lat = self.current_lat
            self.offline_center_lng = self.current_lng
            self.offline_straight_lat = self.current_lat
            self.offline_straight_lng = self.current_lng
            self.offline_angle_rad = 0.0
            self.offline_last_ts = time.time()

        self.nmea_tail.appendleft(f"[MODE] {self.gps_mode} | PATH={self.offline_pattern} | speed={self.offline_speed_mps:.1f} m/s")
        self.update_mode_info()

    def update_mode_info(self):
        self.mode_info_var.set(f"Mode: {self.gps_mode} | Pattern: {self.offline_pattern} | Speed: {self.offline_speed_mps:.1f} m/s")

    def zoom_in(self):
        self.zoom_level = min(self.zoom_level + 1, 20)
        self.map_widget.set_zoom(self.zoom_level)

    def zoom_out(self):
        self.zoom_level = max(self.zoom_level - 1, 1)
        self.map_widget.set_zoom(self.zoom_level)

    def center_map(self):
        self.follow_gps = True
        self.map_widget.set_position(self.current_lat, self.current_lng)
        self.map_widget.set_zoom(self.zoom_level)
        print(f"[MAP] Center -> {self.current_lat:.6f}, {self.current_lng:.6f}")
        self._set_preview_text(f"Center map -> {self.current_lat:.6f}, {self.current_lng:.6f}")

    def _set_status(self, text):
        self.root.after(0, self.status_var.set, f"Status: {text}")

    def _set_session_text(self, text):
        self.root.after(0, self.session_var.set, text)

    def _set_preview_text(self, text):
        self.root.after(0, self.preview_var.set, text)

    def _update_info(self):
        signal_txt = "N/A" if self.latest_signal_dbhz is None else f"{self.latest_signal_dbhz:.1f} dB-Hz"
        sats_txt = "N/A" if self.latest_num_sats is None else str(self.latest_num_sats)
        fix_txt = "N/A" if self.latest_fix_quality is None else str(self.latest_fix_quality)
        hdop_txt = "N/A" if self.latest_hdop is None else f"{self.latest_hdop:.2f}"
        pdop_txt = "N/A" if self.latest_pdop is None else f"{self.latest_pdop:.2f}"
        vdop_txt = "N/A" if self.latest_vdop is None else f"{self.latest_vdop:.2f}"
        alt_txt = "N/A" if self.latest_altitude is None else f"{self.latest_altitude:.1f} m"
        speed_txt = "N/A"
        if self.latest_speed_knots is not None:
            if self.latest_speed_kmh is not None:
                speed_txt = f"{self.latest_speed_knots:.2f} kn / {self.latest_speed_kmh:.2f} km/h"
            else:
                speed_txt = f"{self.latest_speed_knots:.2f} kn"
        course_txt = "N/A" if self.latest_course_deg is None else f"{self.latest_course_deg:.1f}°"

        txt = (
            f"Mode: {self.gps_mode} | Pattern: {self.offline_pattern}\n"
            f"GPS: {self.current_lat:.6f}, {self.current_lng:.6f}\n"
            f"Signal: {signal_txt} | Sats: {sats_txt} | Fix: {fix_txt}\n"
            f"HDOP/PDOP/VDOP: {hdop_txt} / {pdop_txt} / {vdop_txt}\n"
            f"Altitude: {alt_txt} | Speed: {speed_txt} | Course: {course_txt}"
        )
        self.info_var.set(txt)

    def _refresh_gps_debug_text(self):
        with self.lock:
            state = dict(self.gps_state)
            tail = list(self.nmea_tail)

        lines = [
            f"Mode        : {state.get('mode', 'N/A')}",
            f"Pattern     : {state.get('pattern', 'N/A')}",
            f"Sentence    : {state.get('last_sentence_type', 'N/A')}",
            f"Raw         : {state.get('last_raw_sentence', 'N/A')}",
            f"Lat/Lng     : {state.get('lat', 0.0):.6f}, {state.get('lng', 0.0):.6f}",
            f"Signal      : {self.latest_signal_dbhz if self.latest_signal_dbhz is not None else 'N/A'} dB-Hz",
            f"Sat in view : {state.get('gsv_satellites_in_view', 'N/A')}",
            f"Fix quality : {state.get('fix_quality', 'N/A')}",
            f"HDOP/PDOP/VDOP : {state.get('hdop', 'N/A')} / {state.get('pdop', 'N/A')} / {state.get('vdop', 'N/A')}",
            f"Altitude    : {state.get('altitude', 'N/A')}",
            f"Speed kn/kmh: {state.get('speed_knots', 'N/A')} / {state.get('speed_kmh', 'N/A')}",
            f"Course      : {state.get('course_deg', 'N/A')}",
            "",
            "Recent NMEA / Debug:",
        ]
        lines.extend(tail if tail else ["(no data yet)"])

        self.gps_debug_text.configure(state="normal")
        self.gps_debug_text.delete("1.0", tk.END)
        self.gps_debug_text.insert(tk.END, "\n".join(lines))
        self.gps_debug_text.configure(state="disabled")

    def refresh_sessions(self):
        self.session_listbox.delete(0, tk.END)
        self.session_items = []

        sessions = self.session_manager.list_sessions()
        for session_dir, meta in sessions:
            status = meta.get("status", "new")
            inferred = "✓" if meta.get("inference_done") else " "
            recorded = "✓" if meta.get("recording_done") else " "
            created = meta.get("created_at", "")
            label = f"{session_dir.name} | rec:{recorded} inf:{inferred} | {status} | {created}"
            self.session_listbox.insert(tk.END, label)
            self.session_items.append((session_dir, meta))

        if not sessions:
            self._set_session_text("Session: -")
            self.current_session_dir = None
            self.current_session_meta = None
            self._clear_session_overlays()
            self._set_status("Idle")
            return

        # auto-select first session if none selected
        if self.session_listbox.curselection() == ():
            self.session_listbox.selection_set(0)
        self.on_session_select()

    def on_session_select(self, event=None):
        sel = self.session_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(getattr(self, "session_items", [])):
            return

        session_dir, meta = self.session_items[idx]
        self.selected_session_dir = session_dir

        info = (
            f"Session: {session_dir.name}\n"
            f"Name: {meta.get('session_name', session_dir.name)}\n"
            f"Status: {meta.get('status', 'new')}\n"
            f"Created: {meta.get('created_at', '-')}\n"
            f"Inference: {'done' if meta.get('inference_done') else 'not yet'}"
        )
        self._set_session_text(info)

        # if inferred, show map automatically
        if meta.get("inference_done"):
            self.load_session_to_map(session_dir)
            self.start_infer_btn.config(state="disabled")
        else:
            self.start_infer_btn.config(state="normal")

    def new_session(self):
        if self.recording_active or self.inference_active:
            messagebox.showwarning("Busy", "Sesi sedang berjalan. Hentikan dulu sebelum membuat sesi baru.")
            return

        name = simpledialog.askstring("Nama Sesi", "Masukkan nama sesi perekaman:")
        if not name:
            return

        session_dir, meta = self.session_manager.create_session(name)
        self.current_session_dir = session_dir
        self.current_session_meta = meta
        self.selected_session_dir = session_dir
        self.follow_gps = True
        self._set_session_text(
            f"Session: {session_dir.name}\nName: {name}\nStatus: recording\nCreated: {meta.get('created_at')}"
        )
        self.refresh_sessions()
        # re-select newest/current
        for i, (sd, _) in enumerate(self.session_items):
            if sd == session_dir:
                self.session_listbox.selection_clear(0, tk.END)
                self.session_listbox.selection_set(i)
                self.session_listbox.see(i)
                break

        self.start_recording_thread()

    def load_selected_session(self):
        if self.recording_active or self.inference_active:
            messagebox.showwarning("Busy", "Sesi sedang berjalan. Hentikan dulu sebelum memuat sesi lain.")
            return

        if self.selected_session_dir is None:
            messagebox.showinfo("Session", "Pilih sesi dari daftar terlebih dahulu.")
            return

        session_dir = self.selected_session_dir
        meta = self.session_manager.load_meta(session_dir)
        self.current_session_dir = session_dir
        self.current_session_meta = meta

        self._set_session_text(
            f"Session: {session_dir.name}\nName: {meta.get('session_name', session_dir.name)}\nStatus: {meta.get('status', '-')}\nCreated: {meta.get('created_at', '-')}"
        )
        self.follow_gps = False
        self._clear_session_overlays()

        if meta.get("inference_done") and (session_dir / DEFAULT_DET_CSV).exists():
            self.load_session_to_map(session_dir)
            self.start_infer_btn.config(state="disabled")
        else:
            # show GPS log / status only; inference can be started
            self.show_session_status_only(session_dir)
            self.start_infer_btn.config(state="normal")
            messagebox.showinfo(
                "Session Loaded",
                "Sesi dimuat. Jika inference belum selesai, tekan tombol Start Inference."
            )

    def show_current_session_map(self):
        if self.current_session_dir and (self.current_session_dir / DEFAULT_DET_CSV).exists():
            self.load_session_to_map(self.current_session_dir)
        elif self.current_session_dir and (self.current_session_dir / DEFAULT_GPS_CSV).exists():
            self.show_session_status_only(self.current_session_dir)
        else:
            messagebox.showinfo("Map", "Belum ada data sesi untuk ditampilkan.")

    def start_recording_from_selection(self):
        # kept for compatibility: if user selected a session that is brand-new/new, allow recording
        if self.current_session_dir is None:
            messagebox.showinfo("Recording", "Buat sesi baru terlebih dahulu.")
            return
        meta = self.session_manager.load_meta(self.current_session_dir)
        if meta.get("status") == "inferred":
            messagebox.showinfo("Recording", "Sesi ini sudah selesai inferensi. Buat sesi baru untuk perekaman baru.")
            return
        if self.recording_active:
            return
        self.start_recording_thread()

    def stop_recording(self):
        if not self.recording_active:
            messagebox.showinfo("Recording", "Tidak ada perekaman yang sedang berjalan.")
            return
        self.stop_record_event.set()
        self._set_status("Stopping recording...")
        self._set_preview_text("Stopping recording...")

    def start_inference(self):
        if self.inference_active:
            return
        if self.current_session_dir is None:
            messagebox.showinfo("Inference", "Pilih atau buat sesi terlebih dahulu.")
            return

        meta = self.session_manager.load_meta(self.current_session_dir)
        if not (self.current_session_dir / DEFAULT_FRAME_CSV).exists():
            messagebox.showwarning("Inference", "Data frame belum ada. Rekam sesi terlebih dahulu.")
            return

        if not messagebox.askyesno("Konfirmasi Inferensi", "Mulai inferensi untuk sesi ini?"):
            return

        self.start_infer_btn.config(state="disabled")
        self.infer_thread = threading.Thread(target=self.run_inference_for_current_session, daemon=True)
        self.infer_thread.start()

    # =====================================================
    # GPS UPDATE
    # =====================================================
    def gps_thread(self):
        print("[GPS] Thread started")
        ser = None
        last_serial_try = 0.0

        while not self.shutdown_event.is_set():
            try:
                if self.gps_mode == "OFFLINE":
                    if ser is not None:
                        try:
                            ser.close()
                        except Exception:
                            pass
                        ser = None
                    self._offline_step()
                    time.sleep(1.0)
                    continue

                # ONLINE mode
                if ser is None:
                    if time.time() - last_serial_try < 2.0:
                        time.sleep(0.1)
                        continue
                    last_serial_try = time.time()
                    try:
                        ser = serial.Serial(GPS_PORT, baudrate=GPS_BAUDRATE, timeout=1)
                        print(f"[GPS] Connected to {GPS_PORT} @ {GPS_BAUDRATE}")
                    except Exception as e:
                        print("[GPS ERROR - SERIAL]", e)
                        self.nmea_tail.appendleft("[GPS] serial open failed; waiting...")
                        self.root.after(0, self._refresh_gps_debug_text)
                        time.sleep(1.0)
                        continue

                raw = ser.readline()
                if not raw:
                    continue

                data = raw.decode("ascii", errors="ignore").strip()
                if not data.startswith("$"):
                    continue

                stype = nmea_type(data)

                if stype in ("GPGSV", "GNGSV", "GSV"):
                    sats_in_view, avg_snr, max_snr = parse_gsv_signal(data)
                    self.latest_satellites_in_view = sats_in_view
                    self.latest_signal_dbhz = avg_snr
                    self.latest_max_snr = max_snr
                    with self.lock:
                        self.gps_state["gsv_satellites_in_view"] = sats_in_view
                        self.gps_state["gsv_avg_snr"] = avg_snr
                        self.gps_state["gsv_max_snr"] = max_snr
                        self.gps_state["signal_dbhz"] = avg_snr
                    print(f"[GPS GSV] sats_in_view={sats_in_view} avg_snr={avg_snr} max_snr={max_snr}")

                try:
                    msg = pynmea2.parse(data)
                except Exception:
                    self._append_debug_sentence(data)
                    continue

                self._update_state_from_sentence(data, msg=msg)

            except Exception as e:
                print("[GPS ERROR]", e)
                self.nmea_tail.appendleft(f"[GPS ERROR] {e}")
                self.root.after(0, self._refresh_gps_debug_text)
                time.sleep(0.2)

        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def _append_debug_sentence(self, sentence: str):
        self.nmea_tail.appendleft(sentence[:180])
        self.root.after(0, self._refresh_gps_debug_text)

    def _update_state_from_sentence(self, raw: str, msg=None):
        stype = nmea_type(raw)
        now = time.time()

        with self.lock:
            self.gps_state["last_sentence_type"] = stype
            self.gps_state["last_raw_sentence"] = raw
            self.gps_state["mode"] = self.gps_mode
            self.gps_state["pattern"] = self.offline_pattern

        if stype in ("GPRMC", "GNRMC", "RMC"):
            try:
                if msg is None:
                    msg = pynmea2.parse(raw)
                self.latest_rmc_status = getattr(msg, "status", None)
                self.latest_speed_knots = safe_float(getattr(msg, "spd_over_grnd", None), self.latest_speed_knots)
                self.latest_course_deg = safe_float(getattr(msg, "true_course", None), self.latest_course_deg)

                with self.lock:
                    self.gps_state["rmc_status"] = self.latest_rmc_status
                    self.gps_state["speed_knots"] = self.latest_speed_knots
                    self.gps_state["course_deg"] = self.latest_course_deg

                if getattr(msg, "status", "V") == "A":
                    lat = safe_float(getattr(msg, "latitude", None), None)
                    lng = safe_float(getattr(msg, "longitude", None), None)
                    if lat not in (None, 0.0) and lng not in (None, 0.0):
                        self.current_lat = lat
                        self.current_lng = lng
                        with self.lock:
                            self.gps_state["lat"] = lat
                            self.gps_state["lng"] = lng
                    print(f"[GPS RMC] status={self.latest_rmc_status} lat={self.current_lat:.6f} lng={self.current_lng:.6f}")
                else:
                    print(f"[GPS RMC] no fix | status={self.latest_rmc_status}")
            except Exception as e:
                print("[GPS RMC ERROR]", e)

        elif stype in ("GPVTG", "GNVTG", "VTG"):
            try:
                if msg is None:
                    msg = pynmea2.parse(raw)
                self.latest_course_deg = safe_float(getattr(msg, "true_track", None), self.latest_course_deg)
                self.latest_speed_knots = safe_float(getattr(msg, "spd_over_grnd_kts", None), self.latest_speed_knots)
                self.latest_speed_kmh = safe_float(getattr(msg, "spd_over_grnd_kmph", None), self.latest_speed_kmh)
                with self.lock:
                    self.gps_state["speed_knots"] = self.latest_speed_knots
                    self.gps_state["speed_kmh"] = self.latest_speed_kmh
                    self.gps_state["course_deg"] = self.latest_course_deg
                print(f"[GPS VTG] speed={self.latest_speed_knots} kn | course={self.latest_course_deg}")
            except Exception as e:
                print("[GPS VTG ERROR]", e)

        elif stype in ("GPGGA", "GNGGA", "GGA"):
            try:
                if msg is None:
                    msg = pynmea2.parse(raw)
                self.latest_fix_quality = safe_int(getattr(msg, "gps_qual", None), None)
                self.latest_num_sats = safe_int(getattr(msg, "num_sats", None), None)
                self.latest_hdop = safe_float(getattr(msg, "horizontal_dil", None), None)
                self.latest_altitude = safe_float(getattr(msg, "altitude", None), None)
                self.latest_geoid_sep = safe_float(getattr(msg, "geo_sep", None), None)

                lat = safe_float(getattr(msg, "latitude", None), None)
                lng = safe_float(getattr(msg, "longitude", None), None)
                if lat not in (None, 0.0) and lng not in (None, 0.0):
                    self.current_lat = lat
                    self.current_lng = lng
                    with self.lock:
                        self.gps_state["lat"] = lat
                        self.gps_state["lng"] = lng

                with self.lock:
                    self.gps_state["fix_quality"] = self.latest_fix_quality
                    self.gps_state["num_sats"] = self.latest_num_sats
                    self.gps_state["hdop"] = self.latest_hdop
                    self.gps_state["altitude"] = self.latest_altitude
                    self.gps_state["geoid_sep"] = self.latest_geoid_sep

                print(
                    f"[GPS GGA] fix={self.latest_fix_quality} sats={self.latest_num_sats} "
                    f"hdop={self.latest_hdop} alt={self.latest_altitude} "
                    f"lat={self.current_lat:.6f} lng={self.current_lng:.6f}"
                )
            except Exception as e:
                print("[GPS GGA ERROR]", e)

        elif stype in ("GPGSA", "GNGSA", "GSA"):
            try:
                if msg is None:
                    msg = pynmea2.parse(raw)
                fix_type = safe_int(getattr(msg, "mode_fix_type", None), None)
                self.latest_pdop = safe_float(getattr(msg, "pdop", None), None)
                self.latest_hdop = safe_float(getattr(msg, "hdop", None), None)
                self.latest_vdop = safe_float(getattr(msg, "vdop", None), None)

                with self.lock:
                    self.gps_state["gsa_fix_type"] = fix_type
                    self.gps_state["pdop"] = self.latest_pdop
                    self.gps_state["hdop"] = self.latest_hdop
                    self.gps_state["vdop"] = self.latest_vdop

                print(f"[GPS GSA] fix_type={fix_type} pdop={self.latest_pdop} hdop={self.latest_hdop} vdop={self.latest_vdop}")
            except Exception as e:
                print("[GPS GSA ERROR]", e)

        elif stype in ("GPGSV", "GNGSV", "GSV"):
            try:
                sats_in_view, avg_snr, max_snr = parse_gsv_signal(raw)
                self.latest_satellites_in_view = sats_in_view
                self.latest_signal_dbhz = avg_snr
                self.latest_max_snr = max_snr
                with self.lock:
                    self.gps_state["gsv_satellites_in_view"] = sats_in_view
                    self.gps_state["gsv_avg_snr"] = avg_snr
                    self.gps_state["gsv_max_snr"] = max_snr
                    self.gps_state["signal_dbhz"] = avg_snr
                print(f"[GPS GSV] sats_in_view={sats_in_view} avg_snr={avg_snr} max_snr={max_snr}")
            except Exception as e:
                print("[GPS GSV ERROR]", e)

        elif stype in ("GPGLL", "GNGLL", "GLL"):
            try:
                if msg is None:
                    msg = pynmea2.parse(raw)
                self.latest_gll_status = getattr(msg, "status", None)
                lat = safe_float(getattr(msg, "latitude", None), None)
                lng = safe_float(getattr(msg, "longitude", None), None)
                if lat not in (None, 0.0) and lng not in (None, 0.0):
                    self.current_lat = lat
                    self.current_lng = lng
                    with self.lock:
                        self.gps_state["lat"] = lat
                        self.gps_state["lng"] = lng
                with self.lock:
                    self.gps_state["gll_status"] = self.latest_gll_status
                print(f"[GPS GLL] status={self.latest_gll_status} lat={self.current_lat:.6f} lng={self.current_lng:.6f}")
            except Exception as e:
                print("[GPS GLL ERROR]", e)

        self._record_live_gps(now)
        self.root.after(0, self.update_live_marker, self.current_lat, self.current_lng)
        self.nmea_tail.appendleft(raw)
        self.root.after(0, self._update_info)
        self.root.after(0, self._refresh_gps_debug_text)

    def _record_live_gps(self, now: float):
        with self.lock:
            self.gps_state["lat"] = self.current_lat
            self.gps_state["lng"] = self.current_lng
            self.gps_state["signal_dbhz"] = self.latest_signal_dbhz
            self.gps_state["num_sats"] = self.latest_num_sats
            self.gps_state["fix_quality"] = self.latest_fix_quality
            self.gps_state["hdop"] = self.latest_hdop
            self.gps_state["pdop"] = self.latest_pdop
            self.gps_state["vdop"] = self.latest_vdop
            self.gps_state["altitude"] = self.latest_altitude
            self.gps_state["geoid_sep"] = self.latest_geoid_sep
            self.gps_state["speed_knots"] = self.latest_speed_knots
            self.gps_state["speed_kmh"] = self.latest_speed_kmh
            self.gps_state["course_deg"] = self.latest_course_deg

        if self.recording_active and self.current_session_dir is not None:
            gps_csv = self.current_session_dir / DEFAULT_GPS_CSV
            with gps_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    f"{now:.3f}",
                    f"{self.current_lat:.6f}",
                    f"{self.current_lng:.6f}",
                    "" if self.latest_signal_dbhz is None else f"{self.latest_signal_dbhz:.1f}",
                    "" if self.latest_num_sats is None else str(self.latest_num_sats),
                    "" if self.latest_fix_quality is None else str(self.latest_fix_quality),
                    "" if self.latest_hdop is None else f"{self.latest_hdop:.2f}",
                    "" if self.latest_pdop is None else f"{self.latest_pdop:.2f}",
                    "" if self.latest_vdop is None else f"{self.latest_vdop:.2f}",
                    "" if self.latest_altitude is None else f"{self.latest_altitude:.2f}",
                    "" if self.latest_geoid_sep is None else f"{self.latest_geoid_sep:.2f}",
                    "" if self.latest_speed_knots is None else f"{self.latest_speed_knots:.2f}",
                    "" if self.latest_speed_kmh is None else f"{self.latest_speed_kmh:.2f}",
                    "" if self.latest_course_deg is None else f"{self.latest_course_deg:.2f}",
                    self.gps_mode,
                    self.offline_pattern,
                ])

    def _offline_step(self):
        now = time.time()
        dt = max(0.05, now - self.offline_last_ts)
        self.offline_last_ts = now

        # synthetic but plausible signal metrics
        base_sig = 42.0
        wobble = 4.0 * math.sin(now / 7.0)
        self.latest_signal_dbhz = max(25.0, min(55.0, base_sig + wobble))
        self.latest_num_sats = 9 + int(abs(2 * math.sin(now / 11.0)))
        self.latest_fix_quality = 1
        self.latest_hdop = 0.9 + 0.15 * abs(math.sin(now / 5.0))
        self.latest_pdop = 1.2 + 0.2 * abs(math.cos(now / 6.0))
        self.latest_vdop = 1.5 + 0.25 * abs(math.sin(now / 4.0))
        self.latest_altitude = 12.0 + 0.5 * math.sin(now / 13.0)
        self.latest_geoid_sep = 45.0 + 0.2 * math.cos(now / 9.0)
        self.latest_speed_knots = self.offline_speed_mps * 1.94384
        self.latest_speed_kmh = self.offline_speed_mps * 3.6

        if self.offline_pattern == "CIRCLE":
            r = self.offline_circle_radius_m
            omega = self.offline_speed_mps / max(r, 1e-6)
            self.offline_angle_rad = (self.offline_angle_rad + omega * dt) % (2 * math.pi)
            east_m = r * math.cos(self.offline_angle_rad)
            north_m = r * math.sin(self.offline_angle_rad)
            self.current_lat = self.offline_center_lat + meters_to_lat_deg(north_m)
            self.current_lng = self.offline_center_lng + meters_to_lng_deg(east_m, self.offline_center_lat)
            self.latest_course_deg = (math.degrees(self.offline_angle_rad) + 90.0) % 360.0
        else:
            heading = math.radians(self.offline_heading_deg)
            dn = self.offline_speed_mps * dt * math.cos(heading)
            de = self.offline_speed_mps * dt * math.sin(heading)
            self.offline_straight_lat += meters_to_lat_deg(dn)
            self.offline_straight_lng += meters_to_lng_deg(de, self.offline_straight_lat)
            self.current_lat = self.offline_straight_lat
            self.current_lng = self.offline_straight_lng
            self.latest_course_deg = self.offline_heading_deg % 360.0

        self.latest_rmc_status = "A"
        self.latest_gll_status = "A"
        self.latest_satellites_in_view = self.latest_num_sats + 3
        self.latest_max_snr = self.latest_signal_dbhz + 3.0

        with self.lock:
            self.gps_state["mode"] = self.gps_mode
            self.gps_state["pattern"] = self.offline_pattern
            self.gps_state["last_sentence_type"] = "SIM"
            self.gps_state["last_raw_sentence"] = f"SIM {self.offline_pattern}"
            self.gps_state["lat"] = self.current_lat
            self.gps_state["lng"] = self.current_lng
            self.gps_state["signal_dbhz"] = self.latest_signal_dbhz
            self.gps_state["num_sats"] = self.latest_num_sats
            self.gps_state["fix_quality"] = self.latest_fix_quality
            self.gps_state["hdop"] = self.latest_hdop
            self.gps_state["pdop"] = self.latest_pdop
            self.gps_state["vdop"] = self.latest_vdop
            self.gps_state["altitude"] = self.latest_altitude
            self.gps_state["geoid_sep"] = self.latest_geoid_sep
            self.gps_state["speed_knots"] = self.latest_speed_knots
            self.gps_state["speed_kmh"] = self.latest_speed_kmh
            self.gps_state["course_deg"] = self.latest_course_deg
            self.gps_state["rmc_status"] = self.latest_rmc_status
            self.gps_state["gll_status"] = self.latest_gll_status
            self.gps_state["gsv_satellites_in_view"] = self.latest_satellites_in_view
            self.gps_state["gsv_avg_snr"] = self.latest_signal_dbhz
            self.gps_state["gsv_max_snr"] = self.latest_max_snr
            self.gps_state["gsa_fix_type"] = 3

        debug_lines = [
            f"[SIM RMC] ts={now:.3f} status=A lat={self.current_lat:.6f} lng={self.current_lng:.6f} speed={self.latest_speed_knots:.2f}kn course={self.latest_course_deg:.1f}",
            f"[SIM VTG] speed={self.latest_speed_knots:.2f}kn/{self.latest_speed_kmh:.2f}kmh course={self.latest_course_deg:.1f}",
            f"[SIM GGA] fix={self.latest_fix_quality} sats={self.latest_num_sats} hdop={self.latest_hdop:.2f} alt={self.latest_altitude:.2f}",
            f"[SIM GSA] fix_type=3 pdop={self.latest_pdop:.2f} hdop={self.latest_hdop:.2f} vdop={self.latest_vdop:.2f}",
            f"[SIM GSV] sats_in_view={self.latest_satellites_in_view} avg_snr={self.latest_signal_dbhz:.1f} max_snr={self.latest_max_snr:.1f}",
            f"[SIM GLL] status=A lat={self.current_lat:.6f} lng={self.current_lng:.6f}",
        ]
        for line in reversed(debug_lines):
            self.nmea_tail.appendleft(line)

        print(
            f"[GPS OFFLINE-{self.offline_pattern}] ts={now:.3f} "
            f"signal={self.latest_signal_dbhz:.1f} dB-Hz sats={self.latest_num_sats} "
            f"fix={self.latest_fix_quality} lat={self.current_lat:.6f} lng={self.current_lng:.6f}"
        )

        self._record_live_gps(now)
        self.root.after(0, self.update_live_marker, self.current_lat, self.current_lng)
        self.root.after(0, self._update_info)
        self.root.after(0, self._refresh_gps_debug_text)

    def offline_step(self):
        """Backward-compatible alias for the offline GPS simulator."""
        return self._offline_step()

    # =====================================================
    # SESSION RECORDING
    # =====================================================
    def start_recording_thread(self):
        if self.playback_thread and self.playback_thread.is_alive():
            return
        if self.current_session_dir is None:
            messagebox.showinfo("Recording", "Buat atau pilih sesi terlebih dahulu.")
            return
        if self.recording_active:
            return
        self.stop_record_event.clear()
        self.playback_thread = threading.Thread(target=self.record_session, daemon=True)
        self.playback_thread.start()


    def _display_frame(self, frame_bgr, is_rgb=False):
        """Update camera preview safely from a worker thread."""
        if frame_bgr is None:
            return
        try:
            if not is_rgb:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame_bgr
            frame_rgb = cv2.resize(frame_rgb, (360, 240))
            with self.lock:
                self.frame_display = frame_rgb
        except Exception as e:
            print("[DISPLAY ERROR]", e)


    def record_session(self):
        if self.current_session_dir is None:
            return

        self.follow_gps = True
        self.recording_active = True
        self.inference_active = False
        self._set_status("Recording video + GPS")
        self._set_preview_text(f"Recording: {self.current_session_dir.name}")

        self.session_manager.update_status(self.current_session_dir, "recording", recording_done=False, inference_done=False)
        self.refresh_sessions()

        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            self.recording_active = False
            self._set_status("Video open failed")
            self._set_preview_text("Preview: video open failed")
            messagebox.showerror("Video", f"Tidak bisa membuka video: {VIDEO_PATH}")
            return

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 1:
            src_fps = 24.0
        frame_delay = 1.0 / src_fps
        print(f"[VIDEO] Opened {VIDEO_PATH}, source_fps={src_fps:.2f}")

        frame_csv = self.current_session_dir / DEFAULT_FRAME_CSV
        frame_count = 0

        try:
            while not self.shutdown_event.is_set():
                if self.stop_record_event.is_set():
                    print("[CTRL] Stop recording requested.")
                    break

                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("[VIDEO] End of video reached.")
                    break

                frame_ts = time.time()
                with self.lock:
                    self.session_frame_count += 1

                with frame_csv.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([frame_count, f"{frame_ts:.3f}"])

                preview = frame.copy()
                cv2.rectangle(preview, (0, 0), (preview.shape[1], 40), (0, 0, 0), -1)
                cv2.putText(
                    preview,
                    f"RECORD | session={self.current_session_dir.name} | frame={frame_count} | ts={frame_ts:.3f}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                self._display_frame(preview)
                frame_count += 1

                elapsed = time.time() - t0
                sleep_time = max(0.0, frame_delay - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as e:
            print("[RECORD ERROR]", e)
            self.nmea_tail.appendleft(f"[RECORD ERROR] {e}")
            self.root.after(0, self._refresh_gps_debug_text)
        finally:
            try:
                cap.release()
            except Exception:
                pass

            self.recording_active = False
            self.stop_record_event.clear()

            self.session_manager.update_status(
                self.current_session_dir,
                "recorded",
                recording_done=True,
                inference_done=False,
                frames_recorded=frame_count,
                source_video=VIDEO_PATH,
                last_update=datetime.now().isoformat(timespec="seconds"),
            )
            self.refresh_sessions()

            self._set_status("Recording finished. Press Start Inference to process this session.")
            self._set_preview_text(f"Recorded: {self.current_session_dir.name} | frames={frame_count}")
            self.start_infer_btn.config(state="normal")

            if frame_count == 0:
                self._set_preview_text(f"Recorded: {self.current_session_dir.name} | frames=0")

            messagebox.showinfo(
                "Recording selesai",
                f"Rekaman sesi '{self.current_session_dir.name}' selesai.\nTekan Start Inference untuk memproses data."
            )


    # =====================================================
    # INFERENCE
    # =====================================================
    def start_inference_for_session(self, session_dir: Path):
        if self.session is None:
            messagebox.showwarning("Model", "Model ONNX belum siap.")
            return
        if self.inference_active:
            return

        meta = self.session_manager.load_meta(session_dir)
        if not messagebox.askyesno("Konfirmasi Inferensi", f"Mulai inferensi untuk sesi '{session_dir.name}'?"):
            return

        self.stop_infer_event.clear()
        self.infer_thread = threading.Thread(target=self.run_inference_for_current_session, daemon=True)
        self.infer_thread.start()


    def run_inference_for_current_session(self):
        if self.current_session_dir is None:
            self.start_infer_btn.config(state="normal")
            return

        session_dir = self.current_session_dir
        meta = self.session_manager.load_meta(session_dir)
        frames = read_csv_dicts(session_dir / DEFAULT_FRAME_CSV)
        gps_rows = read_csv_dicts(session_dir / DEFAULT_GPS_CSV)

        if not frames:
            messagebox.showwarning("Inference", "Data frame perekaman tidak ditemukan.")
            self.start_infer_btn.config(state="normal")
            return
        if len(gps_rows) < 2:
            messagebox.showwarning("Inference", "Data GPS tidak cukup untuk interpolasi.")
            self.start_infer_btn.config(state="normal")
            return

        frame_timestamps = []
        for row in frames:
            ts = safe_float(row.get("frame_timestamp"), None)
            if ts is not None:
                frame_timestamps.append(ts)

        gps_obs = []
        for row in gps_rows:
            gps_obs.append({
                "timestamp": safe_float(row.get("timestamp"), None),
                "lat": safe_float(row.get("lat"), None),
                "lng": safe_float(row.get("lng"), None),
                "signal_dbhz": safe_float(row.get("signal_dbhz"), None),
                "num_sats": safe_int(row.get("num_sats"), None),
                "fix_quality": safe_int(row.get("fix_quality"), None),
                "hdop": safe_float(row.get("hdop"), None),
                "pdop": safe_float(row.get("pdop"), None),
                "vdop": safe_float(row.get("vdop"), None),
            })
        gps_obs = [g for g in gps_obs if g["timestamp"] is not None and g["lat"] is not None and g["lng"] is not None]
        gps_obs.sort(key=lambda x: x["timestamp"])

        if not gps_obs:
            messagebox.showwarning("Inference", "GPS observation invalid.")
            self.start_infer_btn.config(state="normal")
            return

        self.inference_active = True
        self._set_status("Inference running")
        self._set_preview_text(f"Inference: {session_dir.name}")
        print(f"[INFER] Session {session_dir.name}: frames={len(frame_timestamps)} gps={len(gps_obs)}")

        out_csv = session_dir / DEFAULT_DET_CSV
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "frame_index",
                "frame_timestamp",
                "frame_image",
                "label",
                "confidence",
                "gps_before_timestamp",
                "gps_after_timestamp",
                "interp_lat",
                "interp_lng",
                "gps_before_lat",
                "gps_before_lng",
                "gps_after_lat",
                "gps_after_lng",
                "gps_before_signal_dbhz",
                "gps_after_signal_dbhz",
                "gps_mode",
                "gps_pattern",
            ])

        video_source = meta.get("source_video", VIDEO_PATH)
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            self.inference_active = False
            self.start_infer_btn.config(state="normal")
            messagebox.showerror("Video", f"Tidak bisa membuka video sumber: {video_source}")
            return

        detections_for_map = []
        frame_idx = 0

        try:
            while frame_idx < len(frame_timestamps):
                if self.stop_infer_event.is_set() or self.shutdown_event.is_set():
                    print("[CTRL] Inference stopped.")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_ts = frame_timestamps[frame_idx]
                annotated = self.infer_single_frame(
                    frame.copy(),
                    frame_idx,
                    frame_ts,
                    gps_obs,
                    out_csv,
                    detections_for_map,
                )
                self._display_frame(annotated)
                frame_idx += 1
        except Exception as e:
            print("[INFER ERROR]", e)
            self.nmea_tail.appendleft(f"[INFER ERROR] {e}")
            self.root.after(0, self._refresh_gps_debug_text)
        finally:
            try:
                cap.release()
            except Exception:
                pass

            self.session_manager.update_status(
                session_dir,
                "inferred",
                inference_done=True,
                recording_done=True,
                detections=len(detections_for_map),
                last_inference=datetime.now().isoformat(timespec="seconds"),
            )
            self.refresh_sessions()

            self.inference_active = False
            self.start_infer_btn.config(state="normal")
            self._set_status(f"Inference finished. detections={len(detections_for_map)}")
            self._set_preview_text(f"Inference done: {session_dir.name} | det={len(detections_for_map)}")
            print(f"[INFER] Done. detections={len(detections_for_map)}")
            self.load_session_to_map(session_dir)

            messagebox.showinfo(
                "Inference selesai",
                f"Inference sesi '{session_dir.name}' selesai.\nJumlah deteksi: {len(detections_for_map)}"
            )


    def _save_detection_frame(self, frame, session_dir: Path, detection_id: int) -> str | None:
        """Save a persistent copy of the detected frame inside the session folder."""
        try:
            frame_dir = session_dir / "detection_frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            filename = f"detection_{detection_id:04d}.jpg"
            path = frame_dir / filename
            if cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                return str(path.relative_to(session_dir))
        except Exception as e:
            print(f"[FRAME SAVE ERROR] {e}")
        return None

    def _marker_click_command(self, detection: dict):
        return lambda _marker=None, d=detection: self.show_detection_popup(d)

    def show_detection_popup(self, detection: dict):
        """Display the stored detection frame and metadata in a popup."""
        popup = tk.Toplevel(self.root)
        popup.title(f"Deteksi - {detection.get('label', 'Damage')}")
        popup.geometry("780x650")
        popup.transient(self.root)

        outer = tk.Frame(popup, bg="white")
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        image_path = detection.get("image_path")
        img = None
        if image_path:
            try:
                img = Image.open(image_path)
                img.thumbnail((740, 500), Image.Resampling.LANCZOS)
            except Exception as e:
                print(f"[POPUP IMAGE ERROR] {e}")

        if img is not None:
            imgtk = ImageTk.PhotoImage(img)
            image_label = tk.Label(outer, image=imgtk, bg="black")
            image_label.image = imgtk
            image_label.pack(fill="both", expand=True, pady=(0, 10))
        else:
            image_label = tk.Label(
                outer,
                text="Gambar frame tidak tersedia",
                bg="black",
                fg="white",
                height=20,
            )
            image_label.pack(fill="both", expand=True, pady=(0, 10))

        info = (
            f"Jenis kerusakan : {detection.get('label', '-') }\n"
            f"Confidence      : {detection.get('confidence', '-') }\n"
            f"Timestamp frame : {detection.get('frame_timestamp', '-') }\n"
            f"Latitude        : {detection.get('interp_lat', '-') }\n"
            f"Longitude       : {detection.get('interp_lng', '-') }\n"
            f"Koordinat       : ({detection.get('interp_lat', '-')}, {detection.get('interp_lng', '-')})"
        )
        tk.Label(
            outer,
            text=info,
            bg="white",
            justify="left",
            anchor="w",
            font=("Arial", 11),
        ).pack(fill="x")

        tk.Button(outer, text="Tutup", command=popup.destroy, width=10).pack(pady=(10, 0))

    def infer_single_frame(self, frame, frame_idx, frame_ts, gps_obs, out_csv: Path, detections_for_map):
        original = frame.copy()
        h, w = frame.shape[:2]

        if self.session is None:
            return self._annotate_phase(original, "NO MODEL", frame_ts, (0, 0, 255))

        img_input = preprocess(frame)
        try:
            outputs = self.session.run(None, {self.input_name: img_input})
        except Exception as e:
            print("[INFER ERROR]", e)
            return self._annotate_phase(original, "INFER ERROR", frame_ts, (0, 0, 255))

        preds = np.squeeze(outputs[0]).T
        boxes, scores, class_ids = [], [], []

        for pred in preds:
            if pred.shape[0] < 5:
                continue

            class_scores = pred[4:]
            conf = float(np.max(class_scores))
            if conf > CONF_THRESHOLD:
                class_id = int(np.argmax(class_scores))
                cx, cy, bw, bh = pred[:4]

                x = int((cx - bw / 2) * w / IMG_SIZE)
                y = int((cy - bh / 2) * h / IMG_SIZE)
                width = int(bw * w / IMG_SIZE)
                height = int(bh * h / IMG_SIZE)

                boxes.append([x, y, width, height])
                scores.append(conf)
                class_ids.append(class_id)

        indices = nms(boxes, scores)
        canvas = self._annotate_phase(original, f"INFERENCE | frame={frame_idx}", frame_ts, (255, 255, 0))

        if not indices:
            return canvas

        for i in indices:
            label = CLASS_NAMES[class_ids[i]] if class_ids[i] < len(CLASS_NAMES) else f"Class {class_ids[i]}"
            confidence = scores[i]
            x, y, width, height = boxes[i]
            x2 = x + width
            y2 = y + height

            cv2.rectangle(canvas, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(canvas, f"{label} {confidence:.2f}", (max(0, x), max(25, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

            g0, g1 = self.get_surrounding_gps_from_list(frame_ts, gps_obs)
            if g0 is None or g1 is None:
                print(f"[DETECTION] frame_ts={frame_ts:.3f} label={label} conf={confidence:.2f}")
                print("  [WARN] GPS before/after not found.")
                continue

            interp_lat, interp_lng, alpha = self.interpolate_coord(frame_ts, g0, g1)

            sig0 = "N/A" if g0["signal_dbhz"] is None else f"{g0['signal_dbhz']:.1f} dB-Hz"
            sig1 = "N/A" if g1["signal_dbhz"] is None else f"{g1['signal_dbhz']:.1f} dB-Hz"

            print("\n[DETECTION]")
            print(f"  frame_ts   = {frame_ts:.3f}")
            print(f"  label      = {label}")
            print(f"  confidence = {confidence:.2f}")
            print("[GPS BEFORE]")
            print(f"  ts={g0['timestamp']:.3f} | lat={g0['lat']:.6f} | lng={g0['lng']:.6f} | signal={sig0}")
            print("[GPS AFTER]")
            print(f"  ts={g1['timestamp']:.3f} | lat={g1['lat']:.6f} | lng={g1['lng']:.6f} | signal={sig1}")
            print(f"[INTERP] alpha={alpha:.4f} -> lat={interp_lat:.6f}, lng={interp_lng:.6f}")

            detection_id = len(detections_for_map) + 1

            cv2.putText(canvas, f"INTERP: {interp_lat:.6f}, {interp_lng:.6f}", (10, canvas.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"GPS: {g0['timestamp']:.3f} -> {g1['timestamp']:.3f}", (10, canvas.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

            image_rel = self._save_detection_frame(canvas, self.current_session_dir, detection_id)
            image_abs = str(self.current_session_dir / image_rel) if image_rel else None

            with out_csv.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    frame_idx,
                    f"{frame_ts:.3f}",
                    image_rel or "",
                    label,
                    f"{confidence:.4f}",
                    f"{g0['timestamp']:.3f}",
                    f"{g1['timestamp']:.3f}",
                    f"{interp_lat:.6f}",
                    f"{interp_lng:.6f}",
                    f"{g0['lat']:.6f}",
                    f"{g0['lng']:.6f}",
                    f"{g1['lat']:.6f}",
                    f"{g1['lng']:.6f}",
                    "" if g0["signal_dbhz"] is None else f"{g0['signal_dbhz']:.1f}",
                    "" if g1["signal_dbhz"] is None else f"{g1['signal_dbhz']:.1f}",
                    self.gps_mode,
                    self.offline_pattern,
                ])

            detection = {
                "detection_id": detection_id,
                "frame_index": frame_idx,
                "frame_timestamp": f"{frame_ts:.3f}",
                "label": label,
                "confidence": f"{confidence:.2f}",
                "interp_lat": f"{interp_lat:.6f}",
                "interp_lng": f"{interp_lng:.6f}",
                "image_path": image_abs,
            }
            detections_for_map.append(detection)
        return canvas

    def get_surrounding_gps_from_list(self, frame_ts, gps_obs):
        if len(gps_obs) < 2:
            return None, None
        before = [g for g in gps_obs if g["timestamp"] <= frame_ts]
        after = [g for g in gps_obs if g["timestamp"] >= frame_ts]
        if not before or not after:
            return None, None
        return before[-1], after[0]

    def interpolate_coord(self, frame_ts, g0, g1):
        t0 = g0["timestamp"]
        t1 = g1["timestamp"]
        if t1 == t0:
            return g0["lat"], g0["lng"], 0.0
        alpha = (frame_ts - t0) / (t1 - t0)
        interp_lat = g0["lat"] + alpha * (g1["lat"] - g0["lat"])
        interp_lng = g0["lng"] + alpha * (g1["lng"] - g0["lng"])
        return interp_lat, interp_lng, alpha

    def _annotate_phase(self, frame, text, ts=None, color=(0, 255, 0)):
        canvas = frame.copy()
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (0, 0, 0), -1)
        label = text if ts is None else f"{text} | ts={ts:.3f}"
        cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return canvas

    # =====================================================
    # MAP DRAWING
    # =====================================================
    def _clear_session_overlays(self):
        """Remove all map overlays from the current session."""
        for marker in list(getattr(self, "damage_markers", [])):
            try:
                marker.delete()
            except Exception:
                pass
        if hasattr(self, "damage_markers"):
            self.damage_markers.clear()

        marker = getattr(self, "session_route_marker", None)
        if marker is not None:
            try:
                marker.delete()
            except Exception:
                pass
        self.session_route_marker = None
    def show_session_status_only(self, session_dir: Path):
        meta, frames, gps, dets = self.session_manager.load_session_payload(session_dir)
        if gps:
            first = gps[0]
            lat = safe_float(first.get("lat"), self.current_lat) or self.current_lat
            lng = safe_float(first.get("lng"), self.current_lng) or self.current_lng
            self.map_widget.set_position(lat, lng)
            self.map_widget.set_zoom(self.zoom_level)
        self._clear_session_overlays()
        self.follow_gps = False
        self._set_preview_text(f"Session loaded: {session_dir.name} | no inference yet")

    def load_session_to_map(self, session_dir: Path):
        meta, frames, gps_rows, det_rows = self.session_manager.load_session_payload(session_dir)
        self._clear_session_overlays()
        self.follow_gps = False

        points = []
        for row in gps_rows:
            lat = safe_float(row.get("lat"), None)
            lng = safe_float(row.get("lng"), None)
            if lat is not None and lng is not None:
                points.append((lat, lng))

        if points:
            center = points[0]
            self.map_widget.set_position(center[0], center[1])
            self.map_widget.set_zoom(self.zoom_level)

        # show damage markers
        damage_points = []
        for idx, row in enumerate(det_rows):
            lat = safe_float(row.get("interp_lat"), None)
            lng = safe_float(row.get("interp_lng"), None)
            label = row.get("label", "Damage")
            if lat is None or lng is None:
                continue
            damage_points.append((lat, lng))

            image_rel = row.get("frame_image", "").strip()
            image_abs = None
            if image_rel:
                candidate = session_dir / image_rel
                if candidate.exists():
                    image_abs = str(candidate)

            detection = {
                "detection_id": idx + 1,
                "frame_index": row.get("frame_index", "-"),
                "frame_timestamp": row.get("frame_timestamp", "-"),
                "label": label,
                "confidence": row.get("confidence", "-"),
                "interp_lat": f"{lat:.6f}",
                "interp_lng": f"{lng:.6f}",
                "image_path": image_abs,
            }
            marker = self.map_widget.set_marker(
                lat,
                lng,
                text="",
                command=self._marker_click_command(detection),
            )
            self.damage_markers.append(marker)

        if damage_points:
            self.map_widget.set_position(damage_points[0][0], damage_points[0][1])
            self.map_widget.set_zoom(self.zoom_level)
            self._set_preview_text(f"Loaded inferred session: {session_dir.name} | detections={len(damage_points)}")
        else:
            self._set_preview_text(f"Loaded session: {session_dir.name} | no detections")

    # =====================================================
    # UI LOOP
    # =====================================================
    def update_live_marker(self, lat, lng):
        try:
            self.live_marker.set_position(lat, lng)
            if self.follow_gps:
                self.map_widget.set_position(lat, lng)
        except Exception as e:
            print("[MAP ERROR]", e)

    def update_ui(self):
        with self.lock:
            frame = self.frame_display.copy() if self.frame_display is not None else None

        if frame is not None:
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_frame.imgtk = imgtk
            self.camera_frame.configure(image=imgtk)

        self.root.after(30, self.update_ui)

    # =====================================================
    # CLEANUP
    # =====================================================
    def on_close(self):
        self.shutdown_event.set()
        self.stop_record_event.set()
        self.stop_infer_event.set()
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = DetectionApp()
    app.run()
