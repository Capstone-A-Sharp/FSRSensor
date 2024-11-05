import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Arduino과 연결된 시린얼 폴트 설정
ser = serial.Serial('/dev/cu.usbmodem1301', 115200)  # 폴트를 '/dev/cu.usbmodem1401'로 설정
numRows, numCols = 16, 16

# 시간과 설정
plt.style.use('dark_background')
fig, ax = plt.subplots()

img = ax.imshow(np.zeros((numRows, numCols)), cmap='inferno', interpolation='bilinear', vmin=0, vmax=100)
plt.colorbar(img, ax=ax, label="Pressure Intensity")

# HPF 활용 기능을 포함한 데이터 가공 함수
def apply_hpf(data, threshold=10):
    return np.where(data > threshold, data, 0)

# 보드로부터 압력센서 받기
def read_data():
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
            pressure_matrix[row] = list(map(int, values))
            row += 1

    return pressure_matrix if row == numRows else None

# 평균값 터미널에 출력
# 원래는 각 행마다 평균값을 구해서 해당 평균값을 단순 대조하는 방향으로 로직을 설계하려 했음
# TODO: 숫자로 제어 알고리즘 설계
def apply_avg_filter(filtered_matrix):
    avg_values = np.mean(filtered_matrix, axis=1)
    return avg_values

def get_filtered_matrix(pressure_matrix):
    # HPF 적용
    filtered_matrix = apply_hpf(pressure_matrix)
    # 제곱 & 10 나누기 적용
    amplified_matrix = np.square(filtered_matrix) / 10
    
    # 평균 필터 적용
    avg_values = apply_avg_filter(amplified_matrix)
    return amplified_matrix, avg_values

# 그래프 출력
def update_image(amplified_matrix):
    # print(amplified_matrix)
    img.set_data(amplified_matrix)
    return [img]

# 데이터 최종 처리
def update_data(*args):
    pressure_matrix = read_data()
    if pressure_matrix is not None:
        amplified_matrix, avg_values = get_filtered_matrix(pressure_matrix)
        return amplified_matrix, avg_values
    return None, None

# 제어 알고리즘 함수
def control_algorithm():
    amplified_matrix, avg_values = update_data()
    print("Amplified Matrix:")
    print(amplified_matrix)
    print("Average Values:")
    print(avg_values)
    if avg_values is not None:
        if avg_values[6] > 15 and avg_values[7] > 15 and avg_values[8] > 15:
            print("속도 증가")
        else:
            print("속도 감소")
    return amplified_matrix

# 애니메이션으로 실시간 업데이트 설정
def update_wrapper(*args):
    amplified_matrix = control_algorithm()
    if amplified_matrix is not None:
        return update_image(amplified_matrix)
    return [img]

ani = animation.FuncAnimation(fig, update_wrapper, interval=100, blit=False)  # 들레이 100ms로 설정
plt.title("Real-Time Pressure Sensor Visualization with HPF and Averaging Filter (16x16)")
plt.show()
