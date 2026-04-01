# Yolov5_ObjectDetection_redPH-and-SH-IBVS_controlLawComparisson_MecanumRover
Experimental testbed for port-Hamiltonian image-based visual servoing (pH-IBVS) on a mecanum rover. Includes SH PI-IBVS and pH-IBVS control laws, YOLOv5 vision pipeline on Hailo-8L, and encoder-based inner-loop wheel control. Companion code to Kiewiet (2026).
Open-source implementation of an IBVS-controlled mecanum rover comparing 
classical Sugiura–Hashimoto PI-IBVS against a velocity-level port-Hamiltonian 
IBVS controller. Built on a Raspberry Pi 5 with Hailo-8L NPU for real-time 
YOLOv5 object detection at 30 Hz and quadrature encoder feedback at 50 Hz.



## Repository Structure

### Core Control Files

| File | Description |
|------|-------------|
| `Hailo_drive4_px.py` | **SH PI-IBVS controller.** Runs the classical Sugiura–Hashimoto control law (Eq. 11). Acquires frames via GStreamer, runs YOLOv5 inference on the Hailo-8L, extracts bounding-box features (μ, ℓ) in pixel coordinates, computes the visual Jacobian J_vis = J_f · J_c, and outputs body twist commands through the Moore–Penrose pseudo-inverse. Includes diagonal PI gains with hard-clamp anti-windup on the integrator. |
| `hailo_drive_Z2PH1_px.py` | **pH-IBVS controller.** Implements the velocity-level port-Hamiltonian control law (Eq. 14) from Muñoz-Arias et al. (2025). Extends the SH law with: damping injection via implicit solve (I + J†·K_d·J_vis)·ξ_b = b, leaky integrator (ż = e − λ·z), back-calculation anti-windup, and real-time Hamiltonian monitoring H_d = ½eᵀK_p·e + ½zᵀK_i⁻¹z. Uses damped least-squares pseudo-inverse. |
| `all_wheels_control_Z4.py` | **Inner-loop wheel controller (50 Hz daemon thread).** Translates body twist commands (v_x, v_y, ω_z) into individual wheel RPM targets via mecanum inverse kinematics. Reads quadrature encoders (333 CPR) and regulates each wheel independently with a positional PI controller with anti-windup. Exposes thread-safe getters for RPM targets, measured RPMs, and duty cycles for CSV logging. |

### Experiment Scripts

| File | Description |
|------|-------------|
| `experiment_gains.py` | **Gain sweep automation.** Iterates over predefined proportional and integral gain configurations, launches the SH controller for each, collects CSV logs, and names output files by gain set. Used for Sections VIII-B (P-sweep) and VIII-B (PI-sweep) of the paper. |
| `experiment_controllers.py` | **Controller comparison automation.** Runs matched-gain experiments for the SH and pH controllers back-to-back, with configurable number of repetitions per controller. Used for Section VIII-E (comprehensive comparison). |

### Shared Components

All control files share:
- **Vision pipeline**: GStreamer → Hailo-8L YOLOv5 inference → bounding-box feature extraction (μ, ℓ, Ẑ)
- **Interaction matrix**: `interaction_matrix_mu_distance()` computes J_f (Eq. 8), `geometric_jacobian_camera_to_robot()` computes J_c (Eq. 9), composed as J_vis = J_f · J_c (Eq. 10)
- **Depth model**: `compute_distance_from_apparent_size()` implements Ẑ = λ_v · h_obj / ℓ (Eq. 6)
- **Logging**: All runs produce timestamped CSV files with features, errors, integral states, commanded velocities, wheel RPMs, and duty cycles


Developed at the University of Groningen in collaboration with Aichi Prefectural University.
