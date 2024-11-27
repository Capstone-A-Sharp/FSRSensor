import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# Arduino과 연결 설정
ser = serial.Serial('COM7', 115200)  # 메가
uno_ser = serial.Serial('COM11', 115200) # 우노
numRows, numCols = 16, 16

data_log = pd.DataFrame(columns=['push_sum', 'pull_sum', 'speed'])

# 초기 속도 설정
speed = 0
max_speed = 35  # 속도의 최대값 설정
alpha = 0.8  # 부들리한 속도 변화를 위한 각중치 설정

# 시간 및 설정
plt.style.use('dark_background')
fig, ax = plt.subplots()

img = ax.imshow(np.zeros((numRows, numCols)), cmap='inferno', interpolation='bilinear', vmin=0, vmax=100)
plt.colorbar(img, ax=ax, label="Pressure Intensity")

# HPF 데이터 처리 함수3
def apply_hpf(data, threshold=20):
    return np.where(data > threshold, data, 0)

# 보드에서 압력 선서 데이터 읽기
def read_data():
    global ser
    pressure_matrix1 = np.zeros((numRows, numCols))
    pressure_matrix2 = np.zeros((numRows, numCols))

    row = 0

    sensor1_active = False
    sensor2_active = False

    # 데이터 읽기
    for sensor in range(2):  # 0: Sensor 1, 1: Sensor 2
        row = 0
        while row < numRows:
            line = ser.readline().decode().strip()
            # print(line)
            # 데이터 시작을 확인하는 문자열에 대한 체크
            if "Pressure Sensor 1 Data:" in line:
                sensor1_active = True
                #print("Reading Pressure Sensor 1 Data...")
                continue
            elif "Pressure Sensor 2 Data:" in line:
                sensor2_active = True
                print("Reading Pressure Sensor 2 Data...")
                continue
            
            # 데이터 처리
            if sensor1_active and sensor == 0:
                if line == "END":
                    break
                values = line.split(',')
                if len(values) == numCols:
                    #print(f"Sensor 1 Row {row}: {values}")
                    pressure_matrix1[row] = list(map(int, values))
                    row += 1

            elif sensor2_active and sensor == 1:
                if line == "END":
                    break
                values = line.split(',')
                if len(values) == numCols:
                    # print(f"Sensor 2 Row {row}: {values}")
                    pressure_matrix2[row] = list(map(int, values))
                    row += 1

    return pressure_matrix2

# 평균 필터 적용 함수
def apply_avg_filter(filtered_matrix):
    avg_values = np.mean(filtered_matrix, axis=1)
    return avg_values

# 필터링된 행렬 가져오기
def get_filtered_matrix(pressure_matrix1):
    # HPF 적용
    filtered_matrix = apply_hpf(pressure_matrix1)
    # 행렬 증평
    amplified_matrix = np.square(filtered_matrix) / 30
    
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
def control_algorithm():
    global speed, ser, uno_ser
    amplified_matrix, avg_values = update_data()
    if amplified_matrix is not None and avg_values is not None:
        # 미르고 있는지 당기고 있는지 상태 판단
        push_sum = sum(avg_values[7:])  # 손바닥 번들
        pull_sum = sum(avg_values[:7])  # 손가락 번들

        # 손바닥 번들에 특정값 이상이 들어올 때
        if speed == 0 and push_sum > 300:
            speed = float(speed) + 20  # 초기값 20 제시

                
        if speed >= max_speed and push_sum > 250:
            speed = max_speed
        elif speed < max_speed and push_sum > 250 :
            speed = float(speed) * 1.1
        elif push_sum < 100:  # 손 안 댔을 때나 가볍게 쥘 때
            speed = 0
        elif pull_sum < 120 and push_sum>100 and push_sum < 250:  # 힘 풀었을 때 압력 (가볍게 쥐었을 때 포함)
            speed = float(speed) * 0.8
        
        # Arduino에 속도 값 전송
        speed_int = int(speed)
        # ser.write(f"{speed_int}\n".encode()) 
        #uno_ser.write(f"{speed_int}\n".encode())  # 우노로 속도 전송
        
        if uno_ser.is_open:  # 우노와 연결이 열려 있는지 확인
            uno_ser.write(f"{speed_int}\n".encode())  # 속도 값을 전송
            print(f"Sent to UNO: {speed_int}")
        
        
        print(f"push_sum: {push_sum:.2f}, pull_sum: {pull_sum:.2f}, speed: {speed:.2f}")
        
        # for i in range(len(avg_values)):
        #     print(f"avg_values[{i}]: {avg_values[i]:.2f}")
    
    return amplified_matrix


# 애니메이션 업데이트 함수
def update_wrapper(*args):
    amplified_matrix = control_algorithm()
    if amplified_matrix is not None:
        return update_image(amplified_matrix)
    return [img]

# 실시간 애니메이션 업데이트 설정
ani = animation.FuncAnimation(fig, update_wrapper, interval=100, blit=False, save_count=100)  # 디레이를 100ms로 설정
plt.title("Sensor (16x16)")
plt.show()