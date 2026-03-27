import numpy as np

def extract_features(buffer):

    ax = [d["ax"] for d in buffer]
    ay = [d["ay"] for d in buffer]
    az = [d["az"] for d in buffer]
    hr = [d["heart_rate"] for d in buffer]

    imu_vector = np.sqrt(
        np.array(ax)**2 +
        np.array(ay)**2 +
        np.array(az)**2
    )

    imu_mean = np.mean(imu_vector)
    imu_std = np.std(imu_vector)

    hr_mean = np.mean(hr)
    hr_std = np.std(hr)

    return [imu_mean, imu_std, hr_mean, hr_std]