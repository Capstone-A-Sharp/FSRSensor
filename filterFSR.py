import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Arduino과 연결된 시리얼 포트 설정
ser = serial.Serial('/dev/cu.usbmodem1401', 115200)  # 포트를 'COM14'로 설정
numRows, numCols = 16, 16

# 시간화 설정
plt.style.use('dark_background')
fig, ax = plt.subplots()
# 'inferno' 색상 맵과 bilinear 보감을 사용해 더 부들고 명확하게 표현
img = ax.imshow(np.zeros((numRows, numCols)), cmap='inferno', interpolation='bilinear', vmin=0, vmax=100)
plt.colorbar(img, ax=ax, label="Pressure Intensity")

def kalman_filter(data, process_variance=1e-5, measurement_variance=0.1**2):
    n_iter = len(data)
    xhat = np.zeros(n_iter)      # estimated value of the pressure
    P = np.zeros(n_iter)         # error estimate
    xhatminus = np.zeros(n_iter) # predicted value of the pressure
    Pminus = np.zeros(n_iter)    # predicted error estimate
    K = np.zeros(n_iter)         # Kalman gain

    xhat[0] = data[0]
    P[0] = 1.0

    for k in range(1, n_iter):
        # Prediction step
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + process_variance

        # Update step
        K[k] = Pminus[k] / (Pminus[k] + measurement_variance)
        xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat

def apply_filters(pressure_matrix):
    # Kalman Filter and Low Pass Filter (Uniform)
    filtered_matrix = np.zeros_like(pressure_matrix)
    for i in range(pressure_matrix.shape[0]):
        # Applying Kalman filter
        row_filtered = kalman_filter(pressure_matrix[i])
        # Applying low-pass filter
        padded_row = np.pad(row_filtered, (1,), mode='edge')
        filtered_matrix[i] = np.convolve(padded_row, np.ones(3)/3, mode='valid')
    return filtered_matrix

def update_data(*args):
    global ser
    pressure_matrix = np.zeros((numRows, numCols))
    row = 0

    # 데이터 읽기
    while row < numRows:
        line = ser.readline().decode().strip()
        if line == "END":
            break
        values = line.split(',')
        if len(values) == numCols:
            # 시리얼에서 읽은 데이터 출력
            print(f"Row {row}: {values}")
            pressure_matrix[row] = list(map(int, values))
            row += 1

    # 모든 행의 데이터가 준비된 경우에만 업데이트
    if row == numRows:
        # Apply filters
        filtered_matrix = apply_filters(pressure_matrix)
        img.set_data(filtered_matrix)
    return [img]

# 애니메이션으로 실시간 업데이트 설정
ani = animation.FuncAnimation(fig, update_data, interval=100, blit=False)  # 딩레이 100ms로 설정
plt.title("Real-Time Pressure Sensor Visualization (16x16)")
plt.show()