import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import threading
import time

# Arduino과 연결 설정
ser = serial.Serial('COM7', 115200)  # 올바른 포트를 설정
nano_ser = serial.Serial('COM12', 9600)
numRows, numCols = 16, 16

# 초기 속도 설정
speed = 0
max_speed = 80  # 속도의 최대값 설정
alpha = 0.8  # 부드러운 속도 변화를 위한 가중치 설정

# 데이터 기록을 위한 DataFrame 생성
data_log = pd.DataFrame(columns=['push_sum', 'pull_sum', 'speed', 'nano_data'])

# nano_data를 지속적으로 업데이트하기 위한 전역 변수
nano_data = ""

# 시간 및 설정
plt.style.use('dark_background')
fig, ax = plt.subplots()

img = ax.imshow(np.zeros((numRows, numCols)), cmap='inferno', interpolation='bilinear', vmin=0, vmax=100)
plt.colorbar(img, ax=ax, label="Pressure Intensity")

# HPF 데이터 처리 함수
def apply_hpf(data, threshold=20):
    return np.where(data > threshold, data, 0)

# nano_data를 읽고 업데이트하는 함수
def read_nano_data():
    global nano_data
    while True:
        try:
            nano_data = nano_ser.readline().decode().strip()
            # nano_data 출력
            print(f"Nano Data: {nano_data}")
        except Exception as e:
            print(f"Error reading nano_data: {e}")
        #time.sleep(0.01)  

# 보드에서 압력 센서 데이터 읽기
def read_data():
    global ser
    pressure_matrix1 = np.zeros((numRows, numCols))
    row = 0

    sensor1_active = False

    # 데이터 읽기
    while row < numRows:
        line = ser.readline().decode().strip()
        if "Pressure Sensor 1 Data:" in line:
            sensor1_active = True
            continue
        
        if sensor1_active:
            if line == "END":
                break
            values = line.split(',')
            if len(values) == numCols:
                pressure_matrix1[row] = list(map(int, values))
                row += 1

    return pressure_matrix1

# 평균 필터 적용 함수
def apply_avg_filter(filtered_matrix):
    avg_values = np.mean(filtered_matrix, axis=1)
    return avg_values

# 필터링된 행렬 가져오기
def get_filtered_matrix(pressure_matrix1):
    # HPF 적용
    filtered_matrix = apply_hpf(pressure_matrix1)
    # 행렬 증폭
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
    global speed, ser, data_log, nano_data
    amplified_matrix, avg_values = update_data()
    
    if amplified_matrix is not None and avg_values is not None:
        # 미르고 있는지 당기고 있는지 상태 판단
        push_sum = avg_values[0] + avg_values[1] + avg_values[2] + avg_values[3] + avg_values[4] + avg_values[5] + avg_values[6] # 손바닥 번들
        pull_sum = avg_values[7] + avg_values[8] + avg_values[9] +avg_values[10] + avg_values[11] + avg_values[12] + avg_values[13] # 손가락 번들

        # 손바닥 번들에 특정값 이상이 들어올 때
    if speed == 0 and push_sum > 300:
         speed = float(nano_data) + 50
         
    elif push_sum > 250 and pull_sum > 60 and pull_sum < 140:
        speed = float(nano_data) * 1.1
        
    elif push_sum < 130:
        speed = 0
        
    elif pull_sum > 150:
        speed = float(nano_data) * 0.8
        
        # Arduino에 속도 값 전송
    speed_int = int(speed) 
    print(f"속도: {speed:.2f}\n")

        # push_sum, pull_sum, speed, nano_data 값을 DataFrame에 추가
    new_data = pd.DataFrame({'push_sum': [push_sum], 'pull_sum': [pull_sum], 'speed': [speed], 'nano_data': [nano_data]})
    data_log = pd.concat([data_log, new_data], ignore_index=True)

    return amplified_matrix



# 애니메이션 업데이트 함수
def update_wrapper(*args):
    amplified_matrix = control_algorithm()
    if amplified_matrix is not None:
        return update_image(amplified_matrix)
    return [img]

# nano_data 읽는 스레드를 시작
nano_thread = threading.Thread(target=read_nano_data, daemon=True)
nano_thread.start()

# 애니메이션 설정
ani = animation.FuncAnimation(fig, update_wrapper, interval=100, blit=False, save_count=100)  # 딜레이를 100ms로 설정
plt.title("Sensor (16x16)")
plt.show()

# 프로그램 종료 시 엑셀 파일로 저장
data_log.to_excel('pressure_data_log.xlsx', index=False)