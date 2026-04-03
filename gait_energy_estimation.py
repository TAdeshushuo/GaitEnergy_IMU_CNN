"""
Gait Energy Estimation - Mechanical Energy & Power Calculation
Corresponds to Section 3.3 of the Manuscript:
Energy Parsing Framework for Lower Extremity & Oscillation Energy

GitHub: https://github.com/TAdeshushuo/GaitEnergy_IMU_CNN

This module includes:
1. Lower Extremity Mechanical Energy (LE)
2. Oscillation Energy (OE)
3. Kinetic Energy (KE)
4. Segment-level Power (Foot, Shank, Thigh)
5. Mechanical Efficiency Calculation
6. Energy Decomposition & Normalization

NOTE:
This is a structural implementation for academic demonstration only.
The full dataset, real-time calibration, and motion capture alignment
are not publicly available. Therefore, this code CANNOT BE RUN directly.
"""

import numpy as np
import math
from scipy.signal import butter, filtfilt

# =============================================================================
# Fixed Anthropometric & Biomechanical Parameters (Standard in Gait Analysis)
# =============================================================================
DELTA_T = 0.01
GRAVITY = 9.81
PI = 3.141592653589793

# Body segment parameters (normalized to body mass & height)
M_FOOT = 0.0147
M_CALF = 0.0435
M_THIGH = 0.1027
M_TORSO = 0.5801

# Moment of inertia & length coefficients
I_FOOT = 0.00023
I_CALF = 0.00347
I_THIGH = 0.00762

# Oscillation Energy Empirical Coefficients (from biomechanical model)
C_VTE = 11.9
C_MLE = 6.0
C_APE = 2.4

# =============================================================================
# Filtering Utilities (Standard Signal Processing)
# =============================================================================
def butterworth_filter(data, cutoff=25, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def outlier_filter(signal, threshold=3.5):
    mean = np.mean(signal)
    std = np.std(signal)
    filtered = np.copy(signal)
    upper = mean + threshold * std
    lower = mean - threshold * std
    filtered[filtered > upper] = mean
    filtered[filtered < lower] = mean
    return filtered

# =============================================================================
# Velocity & Angular Acceleration Calculation
# =============================================================================
def compute_linear_velocity(acceleration, dt=DELTA_T, threshold=0.1):
    vel = np.zeros_like(acceleration)
    for i in range(1, len(acceleration)):
        if abs(acceleration[i]) < threshold:
            vel[i] = 0.0
        else:
            vel[i] = vel[i-1] + acceleration[i] * dt
    return vel

def compute_angular_acceleration(gyro, dt=DELTA_T):
    ang_acc = []
    if len(gyro) >= 5:
        for i in range(5, len(gyro)):
            val = (-gyro[i-5] + 8*gyro[i-4] - 8*gyro[i-2] + gyro[i-1]) / (12 * dt)
            ang_acc.append(val)
    ang_acc = np.array(ang_acc)
    ang_acc = np.pad(ang_acc, (5, 0), mode='constant')
    return ang_acc

# =============================================================================
# Segment Mechanical Power Calculation (Core of Energy Estimation)
# =============================================================================
def compute_segment_power(
    vel_body, vel_global, gyro, ang_acc,
    segment_weight, segment_inertia,
    direction_factor
):
    """
    Unified energy parsing for lower extremity segments.
    Three energy components:
    1. Translational Kinetic Power
    2. Rotational Power
    3. Gravitational Potential Related Power
    """
    vel_body_mag = np.linalg.norm(vel_body, axis=0)
    vel_global_mag = np.linalg.norm(vel_global, axis=0)
    gyro_mag = np.linalg.norm(gyro, axis=0)
    ang_acc_mag = np.linalg.norm(ang_acc, axis=0)

    # Translational power
    p_trans = segment_weight * vel_global_mag

    # Rotational power
    rad_gyro = gyro_mag * PI / 180.0
    rad_ang_acc = ang_acc_mag * PI / 180.0
    p_rot = segment_inertia * rad_gyro * rad_ang_acc

    # Gravitational & direction-related power
    p_grav = segment_weight * vel_body_mag * direction_factor

    # Total segment power
    p_total = p_trans + p_rot + p_grav

    return p_total, p_trans, p_rot, p_grav

# =============================================================================
# Oscillation Energy (OE) Calculation
# =============================================================================
def compute_oscillation_energy(vertical_vel, forward_vel, lateral_vel, walking_speed):
    """
    Oscillation energy from vertical, lateral, and forward movement fluctuations.
    Empirical biomechanical model used in the manuscript.
    """
    vte = C_VTE * walking_speed * (np.max(vertical_vel) - np.min(vertical_vel))
    mle = C_MLE * walking_speed * (np.max(lateral_vel) - np.min(lateral_vel))
    ape = C_APE * walking_speed * (np.max(forward_vel) - np.min(forward_vel))
    oe_total = vte + mle + ape
    return oe_total, vte, mle, ape

# =============================================================================
# Kinetic Energy (KE) Calculation
# =============================================================================
def compute_kinetic_energy(body_mass, walking_speed):
    return 0.5 * body_mass * (walking_speed ** 2) * 1.25

# =============================================================================
# Direction Factor (Motion Phase Detection)
# =============================================================================
def compute_direction_factor(body_vel_x):
    factor = np.zeros_like(body_vel_x)
    if len(body_vel_x) == 0:
        return factor
    first_non_zero = body_vel_x[body_vel_x != 0]
    if len(first_non_zero) > 0:
        base = first_non_zero[0]
        for i in range(len(body_vel_x)):
            if body_vel_x[i] == 0:
                factor[i] = 0
            else:
                factor[i] = 1.0 if (body_vel_x[i] / base > 0) else -1.0
    return factor

# =============================================================================
# Full Gait Energy Estimation Pipeline
# =============================================================================
def gait_energy_estimation_pipeline(processed_imu_data, body_weight, body_height, walking_speed):
    """
    Main pipeline for full gait energy estimation.
    Input: calibrated IMU data (from preprocessing module)
    Output: comprehensive energy metrics
    """
    # --------------------------
    # Extract IMU signals
    # --------------------------
    imu1 = processed_imu_data['IMU1']
    imu2 = processed_imu_data['IMU2']
    imu3 = processed_imu_data['IMU3']
    imu4 = processed_imu_data['IMU4']

    # --------------------------
    # Velocity & Angular Acceleration
    # --------------------------
    v1 = np.array([imu1['vel_x'], imu1['vel_y'], imu1['vel_z']])
    v2 = np.array([imu2['vel_x'], imu2['vel_y'], imu2['vel_z']])
    v3 = np.array([imu3['vel_x'], imu3['vel_y'], imu3['vel_z']])
    v4 = np.array([imu4['vel_x'], imu4['vel_y'], imu4['vel_z']])

    w1 = np.array([imu1['gyro_x'], imu1['gyro_y'], imu1['gyro_z']])
    w2 = np.array([imu2['gyro_x'], imu2['gyro_y'], imu2['gyro_z']])
    w3 = np.array([imu3['gyro_x'], imu3['gyro_y'], imu3['gyro_z']])
    w4 = np.array([imu4['gyro_x'], imu4['gyro_y'], imu4['gyro_z']])

    wa1 = compute_angular_acceleration(w1[0])
    wa2 = compute_angular_acceleration(w2[0])
    wa3 = compute_angular_acceleration(w3[0])

    # --------------------------
    # Body Segment Parameters
    # --------------------------
    foot_weight = GRAVITY * body_weight * M_FOOT
    calf_weight = GRAVITY * body_weight * M_CALF
    thigh_weight = GRAVITY * body_weight * M_THIGH

    foot_inertia = I_FOOT * body_weight
    calf_inertia = I_CALF * body_weight
    thigh_inertia = I_THIGH * body_weight

    # --------------------------
    # Direction Factors
    # --------------------------
    dir_foot = compute_direction_factor(imu1['body_x'])
    dir_calf = compute_direction_factor(imu2['body_x'])
    dir_thigh = compute_direction_factor(imu3['body_x'])

    # --------------------------
    # Segment Energy Calculation
    # --------------------------
    p_foot, pft_v, pft_r, pft_g = compute_segment_power(
        np.array([imu1['body_x'], imu1['body_y'], imu1['body_z']]),
        v1, w1, wa1,
        foot_weight, foot_inertia, dir_foot
    )

    p_calf, pcf_v, pcf_r, pcf_g = compute_segment_power(
        np.array([imu2['body_x'], imu2['body_y'], imu2['body_z']]),
        v2, w2, wa2,
        calf_weight, calf_inertia, dir_calf
    )

    p_thigh, pth_v, pth_r, pth_g = compute_segment_power(
        np.array([imu3['body_x'], imu3['body_y'], imu3['body_z']]),
        v3, w3, wa3,
        thigh_weight, thigh_inertia, dir_thigh
    )

    # --------------------------
    # Total Lower Extremity Energy
    # --------------------------
    max_len = max(len(p_foot), len(p_calf), len(p_thigh))
    p_foot = np.pad(p_foot, (0, max_len - len(p_foot)))
    p_calf = np.pad(p_calf, (0, max_len - len(p_calf)))
    p_thigh = np.pad(p_thigh, (0, max_len - len(p_thigh)))

    le_total_power = p_foot + p_calf + p_thigh

    foot_energy = np.mean(pft_v + pft_r + pft_g) / 2.0
    calf_energy = np.mean(pcf_v + pcf_r + pcf_g) / 2.0
    thigh_energy = np.mean(pth_v + pth_r + pth_g) / 2.0
    le_total_energy = foot_energy + calf_energy + thigh_energy

    # --------------------------
    # Oscillation & Kinetic Energy
    # --------------------------
    oe_total, vte, mle, ape = compute_oscillation_energy(
        imu4['vel_z'], imu4['vel_x'], imu4['vel_y'], walking_speed
    )
    ke_total = compute_kinetic_energy(body_weight, walking_speed)

    # --------------------------
    # Mechanical Efficiency
    # --------------------------
    total_input_energy = ke_total + oe_total + le_total_energy
    mechanical_efficiency = (ke_total / total_input_energy) * 100.0 if total_input_energy != 0 else 0.0

    # --------------------------
    # Return all energy metrics
    # --------------------------
    return {
        "OE": oe_total, "VTE": vte, "MLE": mle, "APE": ape,
        "KE": ke_total, "LE": le_total_energy,
        "Foot_E": foot_energy, "Calf_E": calf_energy, "Thigh_E": thigh_energy,
        "LE_KE_Ratio": le_total_energy / ke_total if ke_total != 0 else 0,
        "OE_KE_Ratio": oe_total / ke_total if ke_total != 0 else 0,
        "Mechanical_Efficiency": mechanical_efficiency
    }

# =============================================================================
# Main Entry (Structural Demo Only)
# =============================================================================
if __name__ == "__main__":
    print("=" * 90)
    print("GAIT ENERGY ESTIMATION FRAMEWORK")
    print("Lower Extremity Energy & Oscillation Energy Calculation")
    print("Corresponds to Section 3.3 of the Manuscript")
    print("=" * 90)
    print("\nThis script implements the full energy decomposition model.")
    print("This is a structural academic demonstration only.")
    print("No executable dataset or runtime environment is provided.\n")