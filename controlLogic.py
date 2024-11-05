import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Arduino과 연결 설정
ser = serial.Serial('/dev/cu.usbmodem1401', 115200)  # 올바른 시리얼 포트를 설정
numRows, numCols = 16, 16

# 초기 속도 설정
speed = 0
max_speed = 80  # 속도의 최댓값 설정
alpha = 0.8  # 부드러운 속도 변화를 위한 가중치 설정

# 시간 및 설정
plt.style.use('dark_background')
fig, ax = plt.subplots()

img = ax.imshow(np.zeros((numRows, numCols)), cmap='inferno', interpolation='bilinear', vmin=0, vmax=100)
plt.colorbar(img, ax=ax, label="Pressure Intensity")

# HPF 데이터 처리 함수
def apply_hpf(data, threshold=10):
    return np.where(data > threshold, data, 0)

# 보드에서 압력 센서 데이터 읽기
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

# 평균 필터 적용 함수
def apply_avg_filter(filtered_matrix):
    avg_values = np.mean(filtered_matrix, axis=1)
    return avg_values

# 필터링된 행렬 가져오기
def get_filtered_matrix(pressure_matrix):
    # HPF 적용
    filtered_matrix = apply_hpf(pressure_matrix)
    # 행렬 증편
    amplified_matrix = np.square(filtered_matrix) / 10
    
    # 평균 필터 적용
    avg_values = apply_avg_filter(amplified_matrix)
    return amplified_matrix, avg_values

# 플로트에서 이미지 업데이트 함수
def update_image(amplified_matrix):
    img.set_data(amplified_matrix)
    return [img]

# 데이터 업데이트 함수
def update_data(*args):
    pressure_matrix = read_data()
    if pressure_matrix is not None:
        amplified_matrix, avg_values = get_filtered_matrix(pressure_matrix)
        return amplified_matrix, avg_values
    return None, None

# 제어 알고리즘 함수
# 제어 알고리즘 함수 수정
def control_algorithm():
    global speed, ser
    amplified_matrix, avg_values = update_data()
    if amplified_matrix is not None and avg_values is not None:
        # 미르고 있는지 당기고 있는지 상태 판단
        push_sum = avg_values[6] + avg_values[7] + avg_values[8]  # 손바닥 부분
        pull_sum = avg_values[0] + avg_values[1] + avg_values[2]  # 손가락 부분

        # 손바닥 부분에 특정값 이상이 들어올 때 
        if avg_values[7] >= 20 and avg_values[8] >= 20:
            if push_sum > pull_sum:
                acceleration = np.dot(avg_values, [0.05] * len(avg_values))  # 가속도 크기 조정
                target_speed = min(max_speed, speed + acceleration)  # 목표 속도 계산
                speed = alpha * speed + (1 - alpha) * target_speed  # 부드러운 속도 변화 적용
            else:
                # 유지 상태
                pass
        else:
            deceleration = np.dot(avg_values, [0.1] * len(avg_values))  # 감속도 크기 조정 (감속도가 더 크게)
            target_speed = max(0, speed - deceleration)  # 목표 속도 계산
            speed = alpha * speed + (1 - alpha) * target_speed  # 부드러운 속도 변화 적용

        # Arduino에 속도 값 전송
        speed_int = int(speed) 
        ser.write(f"{speed_int}\n".encode())  
        print(f"속도: {speed:.2f}")

    return amplified_matrix


# 애니메이션 업데이트 함수
def update_wrapper(*args):
    amplified_matrix = control_algorithm()
    if amplified_matrix is not None:
        return update_image(amplified_matrix)
    return [img]

# 실시간 애니메이션 업데이트 설정
ani = animation.FuncAnimation(fig, update_wrapper, interval=100, blit=False, save_count=100)  # 딜레이를 100ms로 설정
plt.title("Sensor (16x16)")
plt.show()
