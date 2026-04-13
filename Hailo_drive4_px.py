from pathlib import Path
import sys
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import os
import numpy as np
import cv2
import hailo

from hailo_apps.hailo_app_python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# Motor control (all wheels)
sys.path.append("/home/koen/my_motodriver_env")
import all_wheels_control_Z4

# logging
import csv
import time

LOG_FILE = "/home/koen/my_motodriver_env/run5_px.csv"

# ===================== Control parameters =====================

# Velocity limits
VX_MAX = 5
WZ_MAX = 5
vy_enable = 0

# Debounce / detection
MIN_STABLE_FRAMES = 2
STABLE_INV_MAX = 5
MAX_LOST_FRAMES = 20
CONF_THRESH = 0.4

# =========== IBVS parameters (Size/Distance control) ===========
# ===================== Image resolution (must match pipeline) =====================
# Set IMG_WIDTH and IMG_HEIGHT to match your Hailo pipeline output resolution.
IMG_WIDTH  = 1280   # pixels  (native camera resolution — recalibrated with 1280×720)
IMG_HEIGHT = 720    # pixels

F_PX     = 1458.42               # ← RECALIBRATE: run Experiment_lambda_cz.py with 1280×720 data
LAMBDA_V = F_PX / IMG_HEIGHT     # normalised λ_v = f_px / H  (used in interaction matrix)
OBJ_HEIGHT = 0.23
D_C = 0.09

KI_MU = 0.1
KI_ELL = 0.001
MAX_INT_U = 13.0
MAX_INT_ELL = 13

# Pixel-scale integrator clamp limits — scaled proportionally to normalized limits
# (MAX_INT * IMG_SIZE ensures equivalent clamping effect on the control output)
MAX_INT_U_PX   = MAX_INT_U   * IMG_WIDTH
MAX_INT_ELL_PX = MAX_INT_ELL * IMG_HEIGHT

# Control gains (proportional) — UNCHANGED from normalized version.
# Explanation: J_vis scales by IMG size, J_inv scales by 1/IMG size, errors scale
# by IMG size → the products J_inv @ Kp @ e_px and J_inv @ Ki @ int_px are
# numerically identical to the normalized case.
K_MU = 1
K_ELL = 0.28
Kp_IBVS = np.diag([K_MU, K_ELL]).astype(np.float32)
Ki_IBVS = np.diag([KI_MU, KI_ELL]).astype(np.float32)

# Desired setpoints
MU_DES     = 0.0
ELL_DES    = 0.732        # normalized (kept as reference for Z_DES_REF)
ELL_DES_PX = ELL_DES * IMG_HEIGHT   # target bbox height  [px]
MU_DES_PX  = 0.0                    # target x-offset from centre  [px]


def compute_distance_from_apparent_size(ell_px, f_px=F_PX, obj_height=OBJ_HEIGHT):
    """Z = f_px * H_obj / ell_px  (pixel-space depth estimate)."""
    if ell_px < 1e-6:
        return 10.0
    return (f_px * obj_height) / ell_px


def interaction_matrix_mu_distance(mu_px, ell_px, cz, f_v_px=0.0, f_px=F_PX):
    """Interaction matrix in pixel coordinates.

    mu_px  : x-offset from image centre  [px]
    ell_px : bounding-box height          [px]
    f_v_px : y-offset from image centre  [px]
    f_px   : pixel-scale focal length  (= LAMBDA_V * IMG_HEIGHT)
    """
    # Row 1: mu_dot
    J_mu = np.array([
        -f_px / cz,
        0.0,
        mu_px / cz,
        (mu_px * f_v_px) / f_px,
        -(f_px + (mu_px**2) / f_px),
        f_v_px
    ], dtype=np.float32)

    # Row 2: ell_dot (apparent-size variation)
    J_ell = np.array([
        0.0,
        0.0,
        ell_px / cz,
        0,
        -(ell_px * mu_px) / f_px,
        0.0
    ], dtype=np.float32)

    return np.vstack([J_mu, J_ell])


def geometric_jacobian_camera_to_robot(phi=0.0, d_c=D_C):
    J_c_linear = np.array([
        [0.0, vy_enable, d_c],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ], dtype=np.float32)

    J_c_angular = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float32)

    return np.vstack([J_c_linear, J_c_angular])


# Pre-compute constants
_J_C       = geometric_jacobian_camera_to_robot()
_Z_DES_REF = compute_distance_from_apparent_size(ELL_DES_PX)


def compute_visual_jacobian_Jvis(mu_px, ell_px, cz, f_v_px=0.0):
    return interaction_matrix_mu_distance(mu_px, ell_px, cz, f_v_px) @ _J_C


def fmt_float(x, nd=4):
    if x is None:
        return ""
    try:
        if np.isnan(x):
            return ""
    except Exception:
        pass
    return f"{float(x):.{nd}f}"


class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()

        self.stable_inv = 0
        self.no_bottle_frames = 0
        self.last_bottle_x = None
        self.last_bottle_y = None
        self.last_bbox_height = None

        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_wz = 0.0
        self.current_Z = 0.0

        self.integral_mu = 0.0
        self.integral_ell = 0.0
        self.last_update_time = time.time()

        self.start_time = time.time()
        self.log_file = open(LOG_FILE, "w", newline="")
        self.logger = csv.writer(self.log_file)

        # ---------------- Header info (metadata) ----------------
        self.logger.writerow(["# IBVS Control Log - ELL CONTROL [mu, ell] — PIXEL COORDINATES"])
        self.logger.writerow([f"# LOG_FILE={os.path.basename(LOG_FILE)}"])
        self.logger.writerow([f"# DATE_UNIX={time.time():.3f}"])

        # Resolution
        self.logger.writerow([f"# IMG_WIDTH={IMG_WIDTH}"])
        self.logger.writerow([f"# IMG_HEIGHT={IMG_HEIGHT}"])

        # Setpoints (pixel-space; mu and ell columns in CSV are in pixels)
        self.logger.writerow([f"# MU_DES={MU_DES_PX}"])
        self.logger.writerow([f"# ELL_DES={ELL_DES_PX:.4f}"])      # pixels
        self.logger.writerow([f"# ELL_DES_NORM={ELL_DES}"])         # normalized reference
        self.logger.writerow([f"# Z_DES_REF={_Z_DES_REF:.6f}"])

        # Camera model
        self.logger.writerow([f"# LAMBDA_V={LAMBDA_V}"])
        self.logger.writerow([f"# F_PX={F_PX:.4f}"])
        self.logger.writerow([f"# OBJ_HEIGHT={OBJ_HEIGHT}"])
        self.logger.writerow([f"# D_C={D_C}"])

        # Gains
        self.logger.writerow([f"# K_MU={K_MU}"])
        self.logger.writerow([f"# K_ELL={K_ELL}"])
        self.logger.writerow([f"# KI_MU={KI_MU}"])
        self.logger.writerow([f"# KI_ELL={KI_ELL}"])
        self.logger.writerow([f"# Kp=diag({K_MU},{K_ELL})"])
        self.logger.writerow([f"# Ki=diag({KI_MU},{KI_ELL})"])

        # Integrator limits (pixel-scale)
        self.logger.writerow([f"# MAX_INT_U={MAX_INT_U_PX:.1f}"])
        self.logger.writerow([f"# MAX_INT_ELL={MAX_INT_ELL_PX:.1f}"])

        # Velocity limits
        self.logger.writerow([f"# VX_MAX={VX_MAX}"])
        self.logger.writerow([f"# WZ_MAX={WZ_MAX}"])
        self.logger.writerow([f"# VY_ENABLE={vy_enable}"])

        # Detection settings
        self.logger.writerow([f"# CONF_THRESH={CONF_THRESH}"])
        self.logger.writerow([f"# MIN_STABLE_FRAMES={MIN_STABLE_FRAMES}"])
        self.logger.writerow([f"# STABLE_INV_MAX={STABLE_INV_MAX}"])
        self.logger.writerow([f"# MAX_LOST_FRAMES={MAX_LOST_FRAMES}"])

        self.logger.writerow([])

        # Data header — mu and ell are in pixels
        self.logger.writerow([
            "t",
            "mu", "Z_est", "ell",
            "mu_des", "Z_des_ref", "ell_des",
            "e_mu", "e_ell",
            "z_mu", "z_ell",          # integral state (for H_d reconstruction)
            "vx_cmd", "vy_cmd", "wz_cmd",
            "x_center_px", "y_center_px", "bbox_height_px",
            "bottle_seen", "stable",
            "rpm1_t", "rpm2_t", "rpm3_t", "rpm4_t",
            "rpm1", "rpm2", "rpm3", "rpm4",
            "duty1", "duty2", "duty3", "duty4"
        ])

        all_wheels_control_Z4.init_all_wheels()
        all_wheels_control_Z4.start_control_loop()


def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    fmt, width, height = get_caps_from_pad(pad)
    # Always use IMG_WIDTH/IMG_HEIGHT so bbox pixel coords match the calibration space.
    # Hailo bbox coords are normalised to the inference input (640×640), not the display frame.
    img_w = IMG_WIDTH
    img_h = IMG_HEIGHT

    frame = None
    if user_data.use_frame and fmt is not None:
        frame = get_numpy_from_buffer(buffer, fmt, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    bottle_seen    = False
    detection_count = 0
    x_center_px    = None
    y_center_px    = None
    bbox_height_px = None

    for det in detections:
        label = det.get_label()
        conf = det.get_confidence()
        if label == "bottle" and conf > CONF_THRESH:
            bottle_seen = True
            detection_count += 1
            bbox = det.get_bbox()
            x_center_px    = 0.5 * (bbox.xmin() + bbox.xmax()) * img_w
            y_center_px    = 0.5 * (bbox.ymin() + bbox.ymax()) * img_h
            bbox_height_px = (bbox.ymax() - bbox.ymin()) * img_h

            user_data.last_bottle_x    = x_center_px
            user_data.last_bottle_y    = y_center_px
            user_data.last_bbox_height = bbox_height_px
            break

    # Stability Logic
    if bottle_seen:
        user_data.stable_inv = min(user_data.stable_inv + 1, STABLE_INV_MAX)
        user_data.no_bottle_frames = 0
    else:
        user_data.stable_inv = max(user_data.stable_inv - 1, 0)
        user_data.no_bottle_frames += 1

    stable_detected = int(user_data.stable_inv >= MIN_STABLE_FRAMES)

    # Init variables
    mu     = np.nan
    ell    = np.nan
    cz_est = np.nan
    e_mu   = np.nan
    e_ell  = np.nan

    vx_cmd = user_data.current_vx
    vy_cmd = user_data.current_vy
    wz_cmd = user_data.current_wz

    now = time.time()
    dt = now - user_data.last_update_time
    user_data.last_update_time = now
    if dt > 0.1: dt = 0.1

    have_measurement = (x_center_px is not None) and (bbox_height_px is not None)

    if stable_detected and have_measurement:
        # 1. Feature Extraction (pixel coordinates)
        mu  = float(x_center_px - img_w / 2)   # x-offset from image centre  [px]
        f_v = float(y_center_px - img_h / 2)   # y-offset from image centre  [px]
        ell = float(bbox_height_px)              # bounding-box height          [px]

        # 2. Distance estimation
        cz_est = float(compute_distance_from_apparent_size(ell))
        user_data.current_Z = cz_est

        # 3. Error (pixel-space setpoints)
        e_mu  = float(mu  - MU_DES_PX)
        e_ell = float(ell - ELL_DES_PX)

        e = np.array([[e_mu], [e_ell]], dtype=np.float32)

        # 4. Integral
        user_data.integral_mu  += e_mu  * dt
        user_data.integral_ell += e_ell * dt

        user_data.integral_mu  = max(min(user_data.integral_mu,  MAX_INT_U_PX),   -MAX_INT_U_PX)
        user_data.integral_ell = max(min(user_data.integral_ell, MAX_INT_ELL_PX), -MAX_INT_ELL_PX)

        e_int = np.array([[user_data.integral_mu], [user_data.integral_ell]], dtype=np.float32)

        # 5. Control
        J_vis = compute_visual_jacobian_Jvis(mu, ell, cz_est, f_v)
        J_inv = np.linalg.pinv(J_vis)

        control_signal = (Kp_IBVS @ e) + (Ki_IBVS @ e_int)
        v_cmd = -J_inv @ control_signal

        vx_cmd = float(v_cmd[0, 0])
        vy_cmd = vy_enable * float(v_cmd[1, 0])
        wz_cmd = float(v_cmd[2, 0])

        vx_cmd = max(min(vx_cmd, VX_MAX), -VX_MAX)
        vy_cmd = max(min(vy_cmd, VX_MAX), -VX_MAX)
        wz_cmd = max(min(wz_cmd, WZ_MAX), -WZ_MAX)

        string_to_print += (
            f"IBVS: μ={mu:+.1f}px, ℓ={ell:.1f}px (des={ELL_DES_PX:.1f}px)\n"
            f"      e_μ={e_mu:+.1f}px, e_ℓ={e_ell:+.1f}px\n"
            f"      v=[{vx_cmd:+.2f}, {vy_cmd:+.2f}, {wz_cmd:+.2f}]\n"
        )

    elif stable_detected and not have_measurement:
        string_to_print += "HOLD\n"
    else:
        vx_cmd = 0.0
        vy_cmd = 0.0
        wz_cmd = 0.0
        string_to_print += "STOP\n"
        user_data.integral_mu  = 0.0
        user_data.integral_ell = 0.0

    user_data.current_vx = vx_cmd
    user_data.current_vy = vy_cmd
    user_data.current_wz = wz_cmd
    all_wheels_control_Z4.set_twist(vx_cmd, vy_cmd, wz_cmd)

    # --- Overlay ---
    if user_data.use_frame and frame is not None:
        cv2.putText(frame, f"l={ell:.1f}px des={ELL_DES_PX:.1f}px", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"e_l={e_ell:.1f}px" if not np.isnan(e_ell) else "e_l=---",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    # --- Logging ---
    t = time.time() - user_data.start_time
    z_des_ref = _Z_DES_REF

    rpm1_t, rpm2_t, rpm3_t, rpm4_t = all_wheels_control_Z4.get_target_rpms()
    rpm1, rpm2, rpm3, rpm4 = all_wheels_control_Z4.get_current_rpms()
    duty1, duty2, duty3, duty4 = all_wheels_control_Z4.get_current_duties()

    user_data.logger.writerow([
        f"{t:.3f}",
        fmt_float(mu), fmt_float(cz_est), fmt_float(ell),
        fmt_float(MU_DES_PX), fmt_float(z_des_ref), fmt_float(ELL_DES_PX),
        fmt_float(e_mu), fmt_float(e_ell),
        fmt_float(user_data.integral_mu), fmt_float(user_data.integral_ell),
        fmt_float(vx_cmd), fmt_float(vy_cmd), fmt_float(wz_cmd),
        fmt_float(x_center_px), fmt_float(y_center_px), fmt_float(bbox_height_px),
        "1" if bottle_seen else "0", str(stable_detected),
        f"{rpm1_t:.2f}", f"{rpm2_t:.2f}", f"{rpm3_t:.2f}", f"{rpm4_t:.2f}",
        f"{rpm1:.2f}", f"{rpm2:.2f}", f"{rpm3:.2f}", f"{rpm4:.2f}",
        f"{duty1:.2f}", f"{duty2:.2f}", f"{duty3:.2f}", f"{duty4:.2f}"
    ])

    if user_data.get_count() % 30 == 0:
        user_data.log_file.flush()

    print(string_to_print, end="")
    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)

    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)

    try:
        app.run()
    finally:
        all_wheels_control_Z4.cleanup_all()
        user_data.log_file.close()
