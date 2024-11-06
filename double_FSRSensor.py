import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Arduino와 연결된 시리얼 포트 설정
ser = serial.Serial('/dev/cu.usbmodem1301', 115200)  # 적절한 포트로 설정
numRows, numCols = 16, 16

# 시각화 설정
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 두 개의 서브플롯 생성
img1 = ax1.imshow(np.zeros((numRows, numCols)), cmap='inferno', interpolation='bilinear', vmin=0, vmax=100)
img2 = ax2.imshow(np.zeros((numRows, numCols)), cmap='inferno', interpolation='bilinear', vmin=0, vmax=100)

ax1.set_title("Real-Time Pressure Sensor 1 Visualization")
# ax2.set_title("Real-Time Pressure Sensor 2 Visualization")
plt.colorbar(img1, ax=ax1, label="Pressure Intensity")
# plt.colorbar(img2, ax=ax2, label="Pressure Intensity")

def update_data(*args):
    global ser
    pressure_matrix1 = np.zeros((numRows, numCols))
    pressure_matrix2 = np.zeros((numRows, numCols))
    
    sensor1_active = False
    sensor2_active = False
    
    # 데이터 읽기
    for sensor in range(2):  # 0: Sensor 1, 1: Sensor 2
        row = 0
        while row < numRows:
            line = ser.readline().decode().strip()
            # 데이터 시작을 확인하는 문자열에 대한 체크
            if "Pressure Sensor 1 Data:" in line:
                sensor1_active = True
                print("Reading Pressure Sensor 1 Data...")
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
                    print(f"Sensor 1 Row {row}: {values}")
                    pressure_matrix1[row] = list(map(int, values))
                    row += 1

            elif sensor2_active and sensor == 1:
                if line == "END":
                    break
                values = line.split(',')
                if len(values) == numCols:
                    print(f"Sensor 2 Row {row}: {values}")
                    pressure_matrix2[row] = list(map(int, values))
                    row += 1

    # 실시간 데이터 업데이트
    img1.set_data(pressure_matrix1)
    # img2.set_data(pressure_matrix2)
    
    return [img1, img2]

# 애니메이션으로 실시간 업데이트 설정
ani = animation.FuncAnimation(fig, update_data, interval=100, blit=False)  # 딜레이 100ms로 설정
plt.show()
