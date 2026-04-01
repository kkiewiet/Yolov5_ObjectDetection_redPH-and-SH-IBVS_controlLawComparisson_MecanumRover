# all_wheels_control_Z4.py
import RPi.GPIO as GPIO
import time
import threading
import math

# =========================
# Pin assignments (paper wheel numbering)
#   wheel 1: FRONT LEFT
#   wheel 2: BACK  LEFT
#   wheel 3: FRONT RIGHT
#   wheel 4: BACK  RIGHT
# =========================

# --- BACK wheels ---
BACK_L_IN1, BACK_L_IN2 = 23, 24          # wheel 2 dir pins
BACK_L_EN = 12                           # wheel 2 PWM
BACK_L_ENC_A, BACK_L_ENC_B = 7, 8        # wheel 2 encoder

BACK_R_IN3, BACK_R_IN4 = 5, 6            # wheel 4 dir pins
BACK_R_EN = 25                           # wheel 4 PWM
BACK_R_ENC_A, BACK_R_ENC_B = 9, 10       # wheel 4 encoder

# --- FRONT wheels ---
FRONT_L_IN3, FRONT_L_IN4 = 16, 26        # wheel 1 dir pins
FRONT_L_EN = 20                          # wheel 1 PWM
FRONT_L_ENC_A, FRONT_L_ENC_B = 18, 22    # wheel 1 encoder

FRONT_R_IN1, FRONT_R_IN2 = 21, 19        # wheel 3 dir pins
FRONT_R_EN = 11                          # wheel 3 PWM
FRONT_R_ENC_A, FRONT_R_ENC_B = 27, 17    # wheel 3 encoder

# =========================
# Control parameters (EXPORTED FOR LOGGING)
# =========================
PWM_FREQ = 1000
CPR      = 333

# Per-wheel PI gains (EXPORTED for parameter logging)
# Kp [% duty / RPM]      — proportional: immediate bounded response to error
# Ki [% duty / (RPM·s)]  — integral:     eliminates steady-state error
#
# Tuning notes:
#   Start with Ki=0, tune Kp until step response is fast without oscillation.
#   Then increase Ki gradually until steady-state error disappears.
#   MAX_INT_DUTY limits the integral's maximum contribution (anti-windup).
#
# The old incremental P controller (duty += 0.04 * error) behaved like a
# pure I-controller with Ki_eff = 0.04 / CONTROL_DT = 2.0 %/(RPM·s) and
# no proportional term — causing slow startup and windup during stall.
Kp_FL = 0.5    # wheel 1
Ki_FL = 0.2    # wheel 1
Kp_BL = 0.5    # wheel 2
Ki_BL = 0.2    # wheel 2
Kp_FR = 0.5    # wheel 3
Ki_FR = 0.2    # wheel 3
Kp_BR = 0.5    # wheel 4
Ki_BR = 0.2    # wheel 4

# Maximum duty contribution from the integral term [%].
# Prevents integrator windup when the motor stalls or output saturates.
MAX_INT_DUTY = 80.0

WHEEL_RADIUS = 0.06   # [m]
HALF_LENGTH  = 0.093  # l
HALF_WIDTH   = 0.087  # w

MAX_RPM    = 140.0
CONTROL_DT = 0.02     # 50 Hz control loop

# =========================
# Global state
# =========================
_vx_cmd = 0.0
_vy_cmd = 0.0
_wz_cmd = 0.0

_running = False
_thread  = None

pwm_w1 = pwm_w2 = pwm_w3 = pwm_w4 = None

# Thread-safe storage for current wheel states
_lock = threading.Lock()
_rpm_targets = [0.0, 0.0, 0.0, 0.0]  # [rpm1_t, rpm2_t, rpm3_t, rpm4_t]
_rpm_actual  = [0.0, 0.0, 0.0, 0.0]  # [rpm1,   rpm2,   rpm3,   rpm4  ]
_duties      = [0.0, 0.0, 0.0, 0.0]  # [duty1,  duty2,  duty3,  duty4 ]


# =========================
# Encoder
# =========================
class Encoder:
    def __init__(self, A, B, mode=GPIO.BCM):
        GPIO.setmode(mode)
        GPIO.setup(A, GPIO.IN)
        GPIO.setup(B, GPIO.IN)
        self.A = A
        self.B = B
        self.pos = 0
        self.state = 0
        self.last_time = time.time()
        self.speed = 0.0
        if GPIO.input(A): self.state |= 1
        if GPIO.input(B): self.state |= 2
        GPIO.add_event_detect(A, GPIO.BOTH, callback=self._update)
        GPIO.add_event_detect(B, GPIO.BOTH, callback=self._update)

    def _update(self, channel=None):
        current_time = time.time()
        state = self.state & 3
        if GPIO.input(self.A): state |= 4
        if GPIO.input(self.B): state |= 8
        self.state = state >> 2
        old_pos = self.pos
        if state in (1, 7, 8, 14):    self.pos += 1
        elif state in (2, 4, 11, 13): self.pos -= 1
        elif state in (3, 12):        self.pos += 2
        elif state in (6, 9):         self.pos -= 2
        delta_pos  = self.pos - old_pos
        delta_time = current_time - self.last_time
        if delta_time > 0:
            self.speed = delta_pos / delta_time
        self.last_time = current_time

    def read(self):
        return self.pos


# =========================
# Direction helpers
# =========================
def set_dir_from_sign_w1(sign):
    if sign >= 0:
        GPIO.output(FRONT_L_IN3, GPIO.HIGH)
        GPIO.output(FRONT_L_IN4, GPIO.LOW)
    else:
        GPIO.output(FRONT_L_IN3, GPIO.LOW)
        GPIO.output(FRONT_L_IN4, GPIO.HIGH)

def set_dir_from_sign_w2(sign):
    if sign >= 0:
        GPIO.output(BACK_L_IN1, GPIO.HIGH)
        GPIO.output(BACK_L_IN2, GPIO.LOW)
    else:
        GPIO.output(BACK_L_IN1, GPIO.LOW)
        GPIO.output(BACK_L_IN2, GPIO.HIGH)

def set_dir_from_sign_w3(sign):
    if sign >= 0:
        GPIO.output(FRONT_R_IN1, GPIO.HIGH)
        GPIO.output(FRONT_R_IN2, GPIO.LOW)
    else:
        GPIO.output(FRONT_R_IN1, GPIO.LOW)
        GPIO.output(FRONT_R_IN2, GPIO.HIGH)

def set_dir_from_sign_w4(sign):
    if sign >= 0:
        GPIO.output(BACK_R_IN3, GPIO.HIGH)
        GPIO.output(BACK_R_IN4, GPIO.LOW)
    else:
        GPIO.output(BACK_R_IN3, GPIO.LOW)
        GPIO.output(BACK_R_IN4, GPIO.HIGH)

def set_coast_all():
    GPIO.output(BACK_L_IN1,  GPIO.LOW)
    GPIO.output(BACK_L_IN2,  GPIO.LOW)
    GPIO.output(BACK_R_IN3,  GPIO.LOW)
    GPIO.output(BACK_R_IN4,  GPIO.LOW)
    GPIO.output(FRONT_L_IN3, GPIO.LOW)
    GPIO.output(FRONT_L_IN4, GPIO.LOW)
    GPIO.output(FRONT_R_IN1, GPIO.LOW)
    GPIO.output(FRONT_R_IN2, GPIO.LOW)


# =========================
# Mecanum kinematics
# =========================
def twist_to_wheel_rpm(vx, vy, wz):
    r      = WHEEL_RADIUS
    lw     = HALF_LENGTH + HALF_WIDTH
    factor = 60.0 / (2.0 * math.pi)

    # Corrected standard mecanum (O-type)
    u1 = (vx - vy - lw * wz) / r  # FL
    u2 = (vx + vy - lw * wz) / r  # BL
    u3 = (vx + vy + lw * wz) / r  # FR
    u4 = (vx - vy + lw * wz) / r  # BR

    return u1 * factor, u2 * factor, u3 * factor, u4 * factor


def clamp_rpm(rpm):
    return max(-MAX_RPM, min(MAX_RPM, rpm))


# =========================
# Positional PI duty controller with anti-windup
# =========================
def _update_duty_pi(rpm_meas, rpm_target, integral, Kp, Ki, dt):
    """
    Positional PI controller for wheel duty cycle.

    Returns (duty, new_integral).

    When rpm_target ≈ 0 the motor coasts (duty=0) and the integral is reset
    so there is no windup carry-over on the next move command.

    Anti-windup: the integral state is clamped so that its duty contribution
    (Ki * integral) never exceeds ±MAX_INT_DUTY [%].  This prevents the
    integrator from over-charging during stall or hard acceleration, which
    was the root cause of the original 'blow-up at the beginning'.
    """
    if abs(rpm_target) < 1e-3:
        return 0.0, 0.0   # coast + reset integral

    error = abs(rpm_target) - abs(rpm_meas)

    # Integrate with anti-windup clamp
    integral_new = integral + error * dt
    if Ki > 1e-9:
        integral_new = max(-MAX_INT_DUTY / Ki, min(MAX_INT_DUTY / Ki, integral_new))

    duty = max(0.0, min(100.0, Kp * error + Ki * integral_new))
    return duty, integral_new


# =========================
# Public API
# =========================
def init_all_wheels():
    global pwm_w1, pwm_w2, pwm_w3, pwm_w4

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # Setup wheel 2 (BACK LEFT)
    GPIO.setup(BACK_L_IN1, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(BACK_L_IN2, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(BACK_L_EN,  GPIO.OUT)
    pwm_w2 = GPIO.PWM(BACK_L_EN, PWM_FREQ)

    # Setup wheel 4 (BACK RIGHT)
    GPIO.setup(BACK_R_IN3, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(BACK_R_IN4, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(BACK_R_EN,  GPIO.OUT)
    pwm_w4 = GPIO.PWM(BACK_R_EN, PWM_FREQ)

    # Setup wheel 1 (FRONT LEFT)
    GPIO.setup(FRONT_L_IN3, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(FRONT_L_IN4, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(FRONT_L_EN,  GPIO.OUT)
    pwm_w1 = GPIO.PWM(FRONT_L_EN, PWM_FREQ)

    # Setup wheel 3 (FRONT RIGHT)
    GPIO.setup(FRONT_R_IN1, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(FRONT_R_IN2, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(FRONT_R_EN,  GPIO.OUT)
    pwm_w3 = GPIO.PWM(FRONT_R_EN, PWM_FREQ)

    pwm_w1.start(0.0)
    pwm_w2.start(0.0)
    pwm_w3.start(0.0)
    pwm_w4.start(0.0)

    set_dir_from_sign_w1(+1)
    set_dir_from_sign_w2(+1)
    set_dir_from_sign_w3(+1)
    set_dir_from_sign_w4(+1)


def set_twist(vx, vy, wz):
    global _vx_cmd, _vy_cmd, _wz_cmd
    _vx_cmd = float(vx)
    _vy_cmd = float(vy)
    _wz_cmd = float(wz)


def set_forward_rpm(rpm):
    vx = float(rpm) * (2.0 * math.pi * WHEEL_RADIUS) / 60.0
    set_twist(vx, 0.0, 0.0)


def set_rotate_rpm(rot_rpm):
    wz = (float(rot_rpm) * 2.0 * math.pi * WHEEL_RADIUS) / (60.0 * (HALF_LENGTH + HALF_WIDTH))
    set_twist(0.0, 0.0, wz)


def set_target_rpm(rpm):
    set_forward_rpm(rpm)


def start_control_loop():
    global _running, _thread
    if _running:
        return
    _running = True
    _thread = threading.Thread(target=_control_loop, daemon=True)
    _thread.start()


def stop_control_loop():
    global _running, _thread
    _running = False
    if _thread is not None:
        _thread.join()
        _thread = None


def cleanup_all():
    stop_control_loop()
    try:
        if pwm_w1: pwm_w1.stop()
        if pwm_w2: pwm_w2.stop()
        if pwm_w3: pwm_w3.stop()
        if pwm_w4: pwm_w4.stop()
    except Exception:
        pass
    set_coast_all()
    GPIO.cleanup()


# =========================
# GETTER FUNCTIONS (for IBVS logging)
# =========================
def get_target_rpms():
    """Returns (rpm1_t, rpm2_t, rpm3_t, rpm4_t)"""
    with _lock:
        return tuple(_rpm_targets)


def get_current_rpms():
    """Returns (rpm1, rpm2, rpm3, rpm4)"""
    with _lock:
        return tuple(_rpm_actual)


def get_current_duties():
    """Returns (duty1, duty2, duty3, duty4)"""
    with _lock:
        return tuple(_duties)


# =========================
# Background control loop
# =========================
def _control_loop():
    global _rpm_targets, _rpm_actual, _duties

    enc_w1 = Encoder(FRONT_L_ENC_A, FRONT_L_ENC_B)
    enc_w2 = Encoder(BACK_L_ENC_A,  BACK_L_ENC_B)
    enc_w3 = Encoder(FRONT_R_ENC_A, FRONT_R_ENC_B)
    enc_w4 = Encoder(BACK_R_ENC_A,  BACK_R_ENC_B)

    duty1 = duty2 = duty3 = duty4 = 0.0
    integral1 = integral2 = integral3 = integral4 = 0.0

    last_pos1 = enc_w1.read()
    last_pos2 = enc_w2.read()
    last_pos3 = enc_w3.read()
    last_pos4 = enc_w4.read()
    last_t = time.monotonic()

    try:
        while _running:
            vx = _vx_cmd
            vy = _vy_cmd
            wz = _wz_cmd

            # 1) Compute and clamp wheel targets
            rpm1_t, rpm2_t, rpm3_t, rpm4_t = twist_to_wheel_rpm(vx, vy, wz)
            rpm1_t = clamp_rpm(rpm1_t)
            rpm2_t = clamp_rpm(rpm2_t)
            rpm3_t = clamp_rpm(rpm3_t)
            rpm4_t = clamp_rpm(rpm4_t)

            # 2) Measure elapsed time
            now = time.monotonic()
            dt  = now - last_t
            if dt <= 0:
                dt = CONTROL_DT

            # 3) Read encoders → RPM
            pos1 = enc_w1.read()
            pos2 = enc_w2.read()
            pos3 = enc_w3.read()
            pos4 = enc_w4.read()

            rpm1 = (pos1 - last_pos1) * 60.0 / (CPR * dt)
            rpm2 = (pos2 - last_pos2) * 60.0 / (CPR * dt)
            rpm3 = (pos3 - last_pos3) * 60.0 / (CPR * dt)
            rpm4 = (pos4 - last_pos4) * 60.0 / (CPR * dt)

            last_pos1, last_pos2, last_pos3, last_pos4 = pos1, pos2, pos3, pos4
            last_t = now

            # 4) PI controller — returns (duty, new_integral)
            duty1, integral1 = _update_duty_pi(rpm1, rpm1_t, integral1, Kp_FL, Ki_FL, dt)
            duty2, integral2 = _update_duty_pi(rpm2, rpm2_t, integral2, Kp_BL, Ki_BL, dt)
            duty3, integral3 = _update_duty_pi(rpm3, rpm3_t, integral3, Kp_FR, Ki_FR, dt)
            duty4, integral4 = _update_duty_pi(rpm4, rpm4_t, integral4, Kp_BR, Ki_BR, dt)

            # 5) Update shared state (thread-safe)
            with _lock:
                _rpm_targets = [rpm1_t, rpm2_t, rpm3_t, rpm4_t]
                _rpm_actual  = [rpm1,   rpm2,   rpm3,   rpm4  ]
                _duties      = [duty1,  duty2,  duty3,  duty4 ]

            # 6) Set motor directions and apply PWM
            set_dir_from_sign_w1(rpm1_t)
            set_dir_from_sign_w2(rpm2_t)
            set_dir_from_sign_w3(rpm3_t)
            set_dir_from_sign_w4(rpm4_t)

            pwm_w1.ChangeDutyCycle(duty1)
            pwm_w2.ChangeDutyCycle(duty2)
            pwm_w3.ChangeDutyCycle(duty3)
            pwm_w4.ChangeDutyCycle(duty4)

            time.sleep(CONTROL_DT)

    finally:
        pwm_w1.ChangeDutyCycle(0)
        pwm_w2.ChangeDutyCycle(0)
        pwm_w3.ChangeDutyCycle(0)
        pwm_w4.ChangeDutyCycle(0)
        set_coast_all()
