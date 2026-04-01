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

sys.path.append("/home/koen/my_motodriver_env")
import all_wheels_control_Z4

import csv
import time

LOG_FILE = "/home/koen/my_motodriver_env/runpH5_4_px.csv"

# ===================== Velocity limits =====================
VX_MAX    = 4
WZ_MAX    = 5
vy_enable = 0

# ===================== Detection / debounce =====================
MIN_STABLE_FRAMES = 2
STABLE_INV_MAX    = 5
MAX_LOST_FRAMES   = 20
CONF_THRESH       = 0.4

# ===================== Camera / geometry =====================
LAMBDA_U   = 1.906
LAMBDA_V   = 1.906
OBJ_HEIGHT = 0.23
D_C        = 0.09

# ===================== Image resolution (must match pipeline) =====================
# Set IMG_WIDTH and IMG_HEIGHT to match your Hailo pipeline output resolution.
IMG_WIDTH  = 640   # pixels
IMG_HEIGHT = 640   # pixels

# Pixel-scale camera parameters (derived — equivalent to focal lengths f_x, f_y)
LAMBDA_U_PX = LAMBDA_U * IMG_WIDTH    # ≈ f_x  [px]
LAMBDA_V_PX = LAMBDA_V * IMG_HEIGHT   # ≈ f_y  [px]

# ===================== Setpoints =====================
MU_DES     = 0.0
ELL_DES    = 0.732        # normalized (kept as reference for Z_DES_REF)
ELL_DES_PX = ELL_DES * IMG_HEIGHT   # target bbox height  [px]
MU_DES_PX  = 0.0                    # target x-offset from centre  [px]

# =======================================================================
# ZONE 1 — PORT-HAMILTONIAN PARAMETER BLOCK
# =======================================================================
#
# Port-Hamiltonian IBVS — Muñoz-Arias, Ito & Scherpen (2025)
# Velocity-level specialisation (flat floor, M=I, V≡0, Γ≈0).
#
# Closed-loop Hamiltonian (Lyapunov function, eq. 30):
#   H(eσ, z) = ½ eσᵀ Kp eσ  +  ½ zᵀ Ki⁻¹ z  ≥ 0
#
# Control law (eq. 59, velocity-level reduction):
#   v_cmd = −J†  (Kp·eσ + Ki·z)        ← proportional + integral  (terms III+IV)
#           − J† Kd J_vis v_prev         ← damping injection        (term II)
#
# Leaky integrator with back-calculation anti-windup (eq. 62 + standard AW):
#   ż = eσ − λ_leak·z + K_aw · J_vis·(v_sat − v_unsat)
#
# Pseudo-inverse: damped least-squares (eq. 19):
#   J† = Jᵀ (J Jᵀ + λ²I)⁻¹
#
# NOTE on pixel-space gains:
#   The Kp, Ki, Kd matrices are UNCHANGED from the normalized version.
#   J_vis scales by IMG size, J_inv by 1/IMG size, and eσ by IMG size,
#   so all three products J†·Kp·eσ, J†·Ki·z, J†·Kd·J·v_prev remain
#   numerically identical to the normalized control output.
#   The integrator clamp limits (MAX_INT) are scaled by IMG size to
#   preserve the same effective saturation threshold.

# --- Inertia / design matrices (velocity level: both identity) ---
Md = np.eye(2, dtype=np.float32)
M  = np.eye(2, dtype=np.float32)

# --- Proportional gain (eq. 59, term III) ---
K_MU  = 1.0
K_ELL = 0.28
Kp_PH = np.diag([K_MU, K_ELL]).astype(np.float32)

# --- Integral gain (eq. 59, term IV) ---
KI_MU  = 0.1
KI_ELL = 0.001
Ki_PH  = np.diag([KI_MU, KI_ELL]).astype(np.float32)
Ki_inv = np.diag([
    1.0 / KI_MU  if KI_MU  > 1e-12 else 0.0,
    1.0 / KI_ELL if KI_ELL > 1e-12 else 0.0,
]).astype(np.float32)   # used in H; 0 when gain is zero → H_int = 0

# --- Damping injection (eq. 59, term II) ---
KD_MU  = 0.1
KD_ELL = 0.1
Kd_PH  = np.diag([KD_MU, KD_ELL]).astype(np.float32)

# --- Leaky integrator coefficient (eq. 62) ---
LEAK = 0.03

# --- Anti-windup back-calculation weight ---
K_AW = 0.3

# --- Hard-clamp limits (pixel-scale) ---
MAX_INT_U   = 13.0
MAX_INT_ELL = 13.0
MAX_INT_U_PX   = MAX_INT_U   * IMG_WIDTH
MAX_INT_ELL_PX = MAX_INT_ELL * IMG_HEIGHT

# --- Damped least-squares regularisation (eq. 19) ---
LAMBDA_DLS = 0.01

# --- Integral enable/disable ---
# Set ENABLE_INTEGRAL = False to zero the integral term entirely (z_int stays at 0).
# This reduces the PH controller to proportional + damping only, which allows a
# direct apples-to-apples comparison with Hailo_drive4 (P + I but no damping).
# For the fairest comparison set KD_MU = KD_ELL = 0 as well.
ENABLE_INTEGRAL = True

# =======================================================================
# END ZONE 1
# =======================================================================


# ---- Helper functions ----

def compute_distance_from_apparent_size(ell_px, lambda_u_px=LAMBDA_U_PX, obj_height=OBJ_HEIGHT):
    """Z = lambda_u_px * H_obj / ell_px  (pixel-space version)."""
    if ell_px < 1e-6:
        return 10.0
    return (lambda_u_px * obj_height) / ell_px


def interaction_matrix_mu_distance(mu_px, ell_px, cz, f_v_px=0.0,
                                    lambda_u=LAMBDA_U_PX, lambda_v=LAMBDA_V_PX):
    """Interaction matrix in pixel coordinates.

    mu_px   : x-offset from image centre  [px]
    ell_px  : bounding-box height          [px]
    f_v_px  : y-offset from image centre  [px]
    lambda_u, lambda_v : pixel-scale focal lengths  (= LAMBDA_U/V * IMG_W/H)
    """
    J_mu = np.array([
        -lambda_u / cz,
        0.0,
        mu_px / cz,
        (mu_px * f_v_px) / lambda_v,
        -(lambda_u + (mu_px**2) / lambda_u),
        (lambda_v * f_v_px) / lambda_u
    ], dtype=np.float32)

    J_ell = np.array([
        0.0,
        0.0,
        ell_px / cz,
        0.0,
        -(ell_px * mu_px) / lambda_u,
        0.0
    ], dtype=np.float32)

    return np.vstack([J_mu, J_ell])


def geometric_jacobian_camera_to_robot(d_c=D_C):
    J_c_linear = np.array([
        [0.0, vy_enable, d_c],
        [0.0, 0.0,       0.0],
        [1.0, 0.0,       0.0]
    ], dtype=np.float32)

    J_c_angular = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float32)

    return np.vstack([J_c_linear, J_c_angular])


_J_C       = geometric_jacobian_camera_to_robot()
_Z_DES_REF = compute_distance_from_apparent_size(ELL_DES_PX)


def compute_visual_jacobian_Jvis(mu_px, ell_px, cz, f_v_px=0.0):
    return interaction_matrix_mu_distance(mu_px, ell_px, cz, f_v_px) @ _J_C


# =======================================================================
# ZONE 2 — Damped least-squares pseudo-inverse (eq. 19)
# =======================================================================
def compute_dls_pseudoinverse(J, lam=LAMBDA_DLS):
    """J† = Jᵀ (J Jᵀ + λ²I)⁻¹"""
    JJt = J @ J.T
    reg = (lam ** 2) * np.eye(JJt.shape[0], dtype=np.float32)
    try:
        return J.T @ np.linalg.inv(JJt + reg)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(J)


# =======================================================================
# ZONE 3 — Closed-loop Hamiltonian / Lyapunov monitor (eq. 30)
# =======================================================================
def compute_hamiltonian(e_sigma, z_int):
    """
    H = ½ eσᵀ Kp eσ  +  ½ zᵀ Ki⁻¹ z
    H ≥ 0 always. ΔH ≤ 0 is the passivity certificate.
    Note: H values are larger in pixel-space (eσ scaled by IMG size)
    but the passivity property (ΔH ≤ 0) is unaffected.
    """
    H_pot = 0.5 * float(e_sigma.T @ Kp_PH @ e_sigma)
    H_int = 0.5 * float(z_int.T @ Ki_inv @ z_int)
    return H_pot + H_int


def fmt_float(x, nd=4):
    if x is None:
        return ""
    try:
        if np.isnan(x):
            return ""
    except Exception:
        pass
    return f"{float(x):.{nd}f}"


# =======================================================================
# ZONE 4a — STATE VARIABLES
# =======================================================================
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()

        self.stable_inv       = 0
        self.no_bottle_frames = 0
        self.last_bottle_x    = None
        self.last_bottle_y    = None
        self.last_bbox_height = None

        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_wz = 0.0
        self.current_Z  = 0.0

        # PH state variables
        self.z_int  = np.zeros((2, 1), dtype=np.float32)   # leaky integral state z̄
        self.v_prev = np.zeros((3, 1), dtype=np.float32)   # previous v_cmd for damping term
        self.H_prev = 0.0                                   # previous H for ΔH logging

        self.last_update_time = time.time()
        self.start_time       = time.time()
        self.log_file         = open(LOG_FILE, "w", newline="")
        self.logger           = csv.writer(self.log_file)

        # Header metadata
        self.logger.writerow(["# Port-Hamiltonian IBVS — Muñoz-Arias, Ito & Scherpen (2025) — PIXEL COORDINATES"])
        self.logger.writerow([f"# LOG_FILE={os.path.basename(LOG_FILE)}"])
        self.logger.writerow([f"# DATE_UNIX={time.time():.3f}"])

        # Resolution
        self.logger.writerow([f"# IMG_WIDTH={IMG_WIDTH}"])
        self.logger.writerow([f"# IMG_HEIGHT={IMG_HEIGHT}"])

        # Setpoints
        self.logger.writerow([f"# MU_DES={MU_DES_PX}"])
        self.logger.writerow([f"# ELL_DES={ELL_DES_PX:.4f}"])      # pixels
        self.logger.writerow([f"# ELL_DES_NORM={ELL_DES}"])         # normalized reference
        self.logger.writerow([f"# Z_DES_REF={_Z_DES_REF:.6f}"])

        # Camera model
        self.logger.writerow([f"# LAMBDA_U={LAMBDA_U}"])
        self.logger.writerow([f"# LAMBDA_V={LAMBDA_V}"])
        self.logger.writerow([f"# LAMBDA_U_PX={LAMBDA_U_PX:.4f}"])
        self.logger.writerow([f"# LAMBDA_V_PX={LAMBDA_V_PX:.4f}"])
        self.logger.writerow([f"# OBJ_HEIGHT={OBJ_HEIGHT}"])
        self.logger.writerow([f"# D_C={D_C}"])

        # Gains
        self.logger.writerow([f"# K_MU={K_MU}"])
        self.logger.writerow([f"# K_ELL={K_ELL}"])
        self.logger.writerow([f"# KI_MU={KI_MU}"])
        self.logger.writerow([f"# KI_ELL={KI_ELL}"])
        self.logger.writerow([f"# KD_MU={KD_MU}"])
        self.logger.writerow([f"# KD_ELL={KD_ELL}"])
        self.logger.writerow([f"# LEAK={LEAK}"])
        self.logger.writerow([f"# K_AW={K_AW}"])
        self.logger.writerow([f"# LAMBDA_DLS={LAMBDA_DLS}"])
        self.logger.writerow([f"# ENABLE_INTEGRAL={int(ENABLE_INTEGRAL)}"])
        self.logger.writerow([f"# MAX_INT_U={MAX_INT_U_PX:.1f}"])
        self.logger.writerow([f"# MAX_INT_ELL={MAX_INT_ELL_PX:.1f}"])
        self.logger.writerow([f"# VX_MAX={VX_MAX}"])
        self.logger.writerow([f"# WZ_MAX={WZ_MAX}"])
        self.logger.writerow([f"# VY_ENABLE={vy_enable}"])
        self.logger.writerow([f"# CONF_THRESH={CONF_THRESH}"])
        self.logger.writerow([f"# MIN_STABLE_FRAMES={MIN_STABLE_FRAMES}"])
        self.logger.writerow([])

        # Data header — mu and ell are in pixels
        self.logger.writerow([
            "t",
            "mu", "Z_est", "ell",
            "mu_des", "Z_des_ref", "ell_des",
            "e_mu", "e_ell",
            "z_mu", "z_ell",
            "H", "dH",
            "aw_mu", "aw_ell",
            "sat_active",
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
    img_w = width  if width  else IMG_WIDTH
    img_h = height if height else IMG_HEIGHT

    frame = None
    if user_data.use_frame and fmt is not None:
        frame = get_numpy_from_buffer(buffer, fmt, width, height)

    roi        = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # ---- Detection ----
    bottle_seen    = False
    detection_count = 0
    x_center_px    = None
    y_center_px    = None
    bbox_height_px = None

    for det in detections:
        label = det.get_label()
        conf  = det.get_confidence()
        if label == "bottle" and conf > CONF_THRESH:
            bottle_seen    = True
            detection_count += 1
            bbox           = det.get_bbox()
            x_center_px    = 0.5 * (bbox.xmin() + bbox.xmax()) * img_w
            y_center_px    = 0.5 * (bbox.ymin() + bbox.ymax()) * img_h
            bbox_height_px = (bbox.ymax() - bbox.ymin()) * img_h
            user_data.last_bottle_x    = x_center_px
            user_data.last_bottle_y    = y_center_px
            user_data.last_bbox_height = bbox_height_px
            break

    # ---- Stability logic ----
    if bottle_seen:
        user_data.stable_inv       = min(user_data.stable_inv + 1, STABLE_INV_MAX)
        user_data.no_bottle_frames = 0
    else:
        user_data.stable_inv       = max(user_data.stable_inv - 1, 0)
        user_data.no_bottle_frames += 1

    stable_detected = int(user_data.stable_inv >= MIN_STABLE_FRAMES)

    # ---- Init frame variables ----
    mu     = np.nan
    ell    = np.nan
    cz_est = np.nan
    e_mu   = np.nan
    e_ell  = np.nan
    H      = np.nan
    dH     = np.nan
    aw_mu  = np.nan
    aw_ell = np.nan
    sat_active = 0

    vx_cmd = user_data.current_vx
    vy_cmd = user_data.current_vy
    wz_cmd = user_data.current_wz

    now = time.time()
    dt  = now - user_data.last_update_time
    user_data.last_update_time = now
    if dt > 0.1:
        dt = 0.1

    have_measurement = (x_center_px is not None) and (bbox_height_px is not None)

    if stable_detected and have_measurement:

        # ================================================================
        # ZONE 4b — PORT-HAMILTONIAN CONTROL LAW (PIXEL COORDINATES)
        # ================================================================

        # -- Step 1: Feature extraction (pixel coordinates) --
        mu  = float(x_center_px - img_w / 2)   # x-offset from image centre  [px]
        f_v = float(y_center_px - img_h / 2)   # y-offset from image centre  [px]
        ell = float(bbox_height_px)              # bounding-box height          [px]

        cz_est = float(compute_distance_from_apparent_size(ell))
        user_data.current_Z = cz_est

        e_mu    = float(mu  - MU_DES_PX)
        e_ell   = float(ell - ELL_DES_PX)
        e_sigma = np.array([[e_mu], [e_ell]], dtype=np.float32)

        # -- Step 2: Visual Jacobian --
        J_vis = compute_visual_jacobian_Jvis(mu, ell, cz_est, f_v)   # 2×3
        J_inv = compute_dls_pseudoinverse(J_vis, lam=LAMBDA_DLS)     # 3×2

        # -- Step 3: Leaky integrator update (eq. 62) --
        #   ż = eσ − λ_leak·z
        # Skipped when ENABLE_INTEGRAL=False; z_int stays at zero throughout.
        if ENABLE_INTEGRAL:
            z_dot = e_sigma - LEAK * user_data.z_int
            user_data.z_int += dt * z_dot

        # -- Step 4: Control law (eq. 59) — implicit damping --
        #   (I + J† Kd J_vis) v = −J† (Kp·eσ + Ki·z)
        if ENABLE_INTEGRAL:
            grad_H = Kp_PH @ e_sigma + Ki_PH @ user_data.z_int  # 2×1
        else:
            grad_H = Kp_PH @ e_sigma                             # 2×1  (Ki·z = 0)
        v_rhs   = -J_inv @ grad_H                               # 3×1
        D_mat   = J_inv @ Kd_PH @ J_vis                         # 3×3
        A_mat   = np.eye(3, dtype=np.float32) + D_mat           # 3×3
        try:
            v_unsat = np.linalg.solve(A_mat, v_rhs)             # 3×1 implicit solution
        except np.linalg.LinAlgError:
            v_unsat = v_rhs                                      # fallback: no damping

        vx_unsat = float(v_unsat[0, 0])
        vy_unsat = vy_enable * float(v_unsat[1, 0])
        wz_unsat = float(v_unsat[2, 0])

        # Saturate
        vx_cmd = float(np.clip(vx_unsat, -VX_MAX, VX_MAX))
        vy_cmd = float(np.clip(vy_unsat, -VX_MAX, VX_MAX))
        wz_cmd = float(np.clip(wz_unsat, -WZ_MAX, WZ_MAX))

        # -- Step 5: Back-calculation anti-windup --
        v_sat_vec   = np.array([[vx_cmd],   [vy_cmd],   [wz_cmd]],   dtype=np.float32)
        v_unsat_vec = np.array([[vx_unsat], [vy_unsat], [wz_unsat]], dtype=np.float32)
        delta_v     = v_sat_vec - v_unsat_vec                        # 3×1

        sat_active = int(np.any(np.abs(delta_v) > 1e-6))

        if ENABLE_INTEGRAL:
            aw_correction = K_AW * (J_vis @ delta_v)                # 2×1
            user_data.z_int += aw_correction
            aw_mu  = float(aw_correction[0, 0])
            aw_ell = float(aw_correction[1, 0])
            # Hard-clamp safety backstop (pixel-scale limits)
            user_data.z_int[0, 0] = np.clip(user_data.z_int[0, 0], -MAX_INT_U_PX,   MAX_INT_U_PX)
            user_data.z_int[1, 0] = np.clip(user_data.z_int[1, 0], -MAX_INT_ELL_PX, MAX_INT_ELL_PX)
        else:
            aw_mu  = 0.0
            aw_ell = 0.0

        # -- Step 6: Hamiltonian monitor (eq. 30) --
        H  = compute_hamiltonian(e_sigma, user_data.z_int)
        dH = H - user_data.H_prev
        user_data.H_prev = H

        # v_prev no longer used (implicit damping solves for v directly)

        # ================================================================
        # END ZONE 4b
        # ================================================================

        string_to_print += (
            f"pH-IBVS: μ={mu:+.1f}px, ℓ={ell:.1f}px (des={ELL_DES_PX:.1f}px)\n"
            f"         e=[{e_mu:+.1f}, {e_ell:+.1f}]px  "
            f"z=[{user_data.z_int[0,0]:+.1f}, {user_data.z_int[1,0]:+.1f}]px\n"
            f"         H={H:.4f}  ΔH={dH:+.5f}  sat={sat_active}"
            f"  aw=[{aw_mu:+.4f},{aw_ell:+.4f}]\n"
            f"         v=[{vx_cmd:+.2f}, {vy_cmd:+.2f}, {wz_cmd:+.2f}]\n"
        )

    elif stable_detected and not have_measurement:
        string_to_print += "HOLD\n"

    else:
        vx_cmd = 0.0
        vy_cmd = 0.0
        wz_cmd = 0.0
        string_to_print += "STOP\n"

        # Reset PH state
        user_data.z_int  = np.zeros((2, 1), dtype=np.float32)
        user_data.v_prev = np.zeros((3, 1), dtype=np.float32)
        user_data.H_prev = 0.0

    # ---- Apply commands ----
    user_data.current_vx = vx_cmd
    user_data.current_vy = vy_cmd
    user_data.current_wz = wz_cmd
    all_wheels_control_Z4.set_twist(vx_cmd, vy_cmd, wz_cmd)

    # ---- Overlay ----
    if user_data.use_frame and frame is not None:
        cv2.putText(frame, f"l={ell:.1f}px des={ELL_DES_PX:.1f}px", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"e_l={e_ell:.1f}px" if not np.isnan(e_ell) else "e_l=---",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"H={H:.2f}" if not np.isnan(H) else "H=---",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"AW={'ON' if sat_active else 'off'}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 100, 255) if sat_active else (180, 180, 180), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    # ---- Logging ----
    t         = time.time() - user_data.start_time
    z_des_ref = _Z_DES_REF

    rpm1_t, rpm2_t, rpm3_t, rpm4_t = all_wheels_control_Z4.get_target_rpms()
    rpm1,   rpm2,   rpm3,   rpm4   = all_wheels_control_Z4.get_current_rpms()
    duty1,  duty2,  duty3,  duty4  = all_wheels_control_Z4.get_current_duties()

    user_data.logger.writerow([
        f"{t:.3f}",
        fmt_float(mu), fmt_float(cz_est), fmt_float(ell),
        fmt_float(MU_DES_PX), fmt_float(z_des_ref), fmt_float(ELL_DES_PX),
        fmt_float(e_mu), fmt_float(e_ell),
        fmt_float(user_data.z_int[0, 0]), fmt_float(user_data.z_int[1, 0]),
        fmt_float(H), fmt_float(dH),
        fmt_float(aw_mu), fmt_float(aw_ell),
        str(sat_active),
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
    env_file     = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)

    user_data = user_app_callback_class()
    app       = GStreamerDetectionApp(app_callback, user_data)

    try:
        app.run()
    finally:
        all_wheels_control_Z4.cleanup_all()
        user_data.log_file.close()
