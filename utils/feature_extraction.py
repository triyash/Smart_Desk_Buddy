import numpy as np

def extract_features(buffer):

    ax = [d["ax"] for d in buffer]
    ay = [d["ay"] for d in buffer]
    az = [d["az"] for d in buffer]

    imu_vector = np.sqrt(
        np.array(ax)**2 +
        np.array(ay)**2 +
        np.array(az)**2
    )

    imu_mean = np.mean(imu_vector)
    imu_std = np.std(imu_vector)

    # 🔥 Calculate posture angles
    pitch = np.arctan2(np.array(ax), np.sqrt(np.array(ay)**2 + np.array(az)**2))
    roll = np.arctan2(np.array(ay), np.sqrt(np.array(ax)**2 + np.array(az)**2))

    pitch_mean = np.mean(pitch)
    roll_mean = np.mean(roll)

    return [imu_mean, imu_std, pitch_mean, roll_mean]