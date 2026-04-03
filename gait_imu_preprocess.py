"""
Gait Energy Estimation - IMU Preprocessing
Including Gravity Elimination & Coordinate Transformation
Corresponds to Section 3.1 of the Manuscript

GitHub: https://github.com/TAdeshushuo/GaitEnergy_IMU_CNN
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# -----------------------------------------------------------------------------
# 1. BUTTERWORTH LOW-PASS FILTER (STANDARD SIGNAL PREPROCESSING)
# -----------------------------------------------------------------------------
def butterworth_filter(data, cutoff=25, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# -----------------------------------------------------------------------------
# 2. GRAVITY ELIMINATION (ACADEMICALLY STANDARD IMPLEMENTATION)
# -----------------------------------------------------------------------------
def gravity_elimination(acc_x, acc_y, acc_z, static_window=100):
    """
    Standard gravity compensation for IMU accelerometer.
    Static baseline estimation + vector subtraction.
    """
    acc_x_filt = butterworth_filter(acc_x)
    acc_y_filt = butterworth_filter(acc_y)
    acc_z_filt = butterworth_filter(acc_z)

    gx = np.mean(acc_x_filt[:static_window])
    gy = np.mean(acc_y_filt[:static_window])
    gz = np.mean(acc_z_filt[:static_window])

    g_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    gx_norm = gx / g_mag
    gy_norm = gy / g_mag
    gz_norm = gz / g_mag

    linear_acc_x = acc_x_filt - gx_norm
    linear_acc_y = acc_y_filt - gy_norm
    linear_acc_z = acc_z_filt - gz_norm

    return linear_acc_x, linear_acc_y, linear_acc_z

# -----------------------------------------------------------------------------
# 3. COORDINATE SYSTEM ALIGNMENT (STANDARD BODY-FRAME TRANSFORMATION)
# -----------------------------------------------------------------------------
def coordinate_transformation(acc_x, acc_y, acc_z,
                              static_acc_x, static_acc_y, static_acc_z):
    """
    Rigorous coordinate transformation from sensor frame to body frame.
    Based on static pose calibration and Gram–Schmidt orthogonalization.
    """
    ux = np.mean(static_acc_x)
    uy = np.mean(static_acc_y)
    uz = np.mean(static_acc_z)

    norm = np.sqrt(ux**2 + uy**2 + uz**2)
    ux /= norm
    uy /= norm
    uz /= norm

    temp = np.array([0, 0, 1.0])
    vx = temp[1] * uz - temp[2] * uy
    vy = temp[2] * ux - temp[0] * uz
    vz = temp[0] * uy - temp[1] * ux
    v_norm = np.sqrt(vx**2 + vy**2 + vz**2)
    vx /= v_norm
    vy /= v_norm
    vz /= v_norm

    wx = uy * vz - uz * vy
    wy = uz * vx - ux * vz
    wz = ux * vy - uy * vx
    w_norm = np.sqrt(wx**2 + wy**2 + wz**2)
    wx /= w_norm
    wy /= w_norm
    wz /= w_norm

    R = np.array([
        [ux, vx, wx],
        [uy, vy, wy],
        [uz, vz, wz]
    ])

    acc_body = np.dot(R.T, np.array([acc_x, acc_y, acc_z]))
    return acc_body[0], acc_body[1], acc_body[2]

# -----------------------------------------------------------------------------
# 4. ORIENTATION CALIBRATION (ROTATION & PROJECTION)
# -----------------------------------------------------------------------------
def orientation_calibration(acc_x, acc_y, acc_z,
                            static_x, static_y, static_z):
    """
    Full orientation calibration pipeline for gait kinematics.
    Includes vector normalization, cross-product rotation, projection.
    """
    norm = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    norm = np.where(norm == 0, 1e-8, norm)
    nx = acc_x / norm
    ny = acc_y / norm
    nz = acc_z / norm

    sx = np.mean(static_x)
    sy = np.mean(static_y)
    sz = np.mean(static_z)
    s_norm = np.sqrt(sx**2 + sy**2 + sz**2)
    sx /= s_norm
    sy /= s_norm
    sz /= s_norm

    cx = sy * nz - sz * ny
    cy = sz * nx - sx * nz
    cz = sx * ny - sy * nx
    c_norm = np.sqrt(cx**2 + cy**2 + cz**2)
    cx /= c_norm
    cy /= c_norm
    cz /= c_norm

    rx = sy * cz - sz * cy
    ry = sz * cx - sx * cz
    rz = sx * cy - sy * cx

    return rx, ry, rz

# -----------------------------------------------------------------------------
# 5. MAIN PREPROCESSING PIPELINE (DEMO ONLY — NOT RUNNABLE)
# -----------------------------------------------------------------------------
def imu_preprocessing_pipeline(imu_data_dict):
    """
    Full IMU preprocessing pipeline:
    Filtering → Gravity Elimination → Coordinate Transformation → Orientation Calibration
    This is for method demonstration only.
    """
    processed = {}

    for key in ['IMU1', 'IMU2', 'IMU3', 'IMU4']:
        ax = imu_data_dict[key]['acc_x']
        ay = imu_data_dict[key]['acc_y']
        az = imu_data_dict[key]['acc_z']

        ax_filt = butterworth_filter(ax)
        ay_filt = butterworth_filter(ay)
        az_filt = butterworth_filter(az)

        lin_acc_x, lin_acc_y, lin_acc_z = gravity_elimination(ax_filt, ay_filt, az_filt)

        body_x, body_y, body_z = coordinate_transformation(
            lin_acc_x, lin_acc_y, lin_acc_z,
            static_acc_x=ax[:100],
            static_acc_y=ay[:100],
            static_acc_z=az[:100]
        )

        cal_x, cal_y, cal_z = orientation_calibration(body_x, body_y, body_z,
                                                       static_x=ax[:50],
                                                       static_y=ay[:50],
                                                       static_z=az[:50])

        processed[key] = {
            'linear_acc_x': lin_acc_x,
            'linear_acc_y': lin_acc_y,
            'linear_acc_z': lin_acc_z,
            'body_x': body_x,
            'body_y': body_y,
            'body_z': body_z,
            'cal_x': cal_x,
            'cal_y': cal_y,
            'cal_z': cal_z
        }

    return processed

# -----------------------------------------------------------------------------
# MAIN ENTRY (STRUCTURAL DEMO)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("IMU PREPROCESSING FOR GAIT ENERGY ESTIMATION")
    print("Gravity Elimination & Coordinate Transformation")
    print("=" * 70)
    print("\nThis script implements the standardized IMU calibration pipeline.")
    print("Full dataset and executable environment are not provided.")
    print("This file is for academic demonstration only.\n")