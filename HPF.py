import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Arduino과 연결된 시리얼 폴트 설정
ser = serial.Serial('/dev/cu.usbmodem1401', 115200)  # 폴트를 '/dev/cu.usbmodem1401'로 설정
numRows, numCols = 16, 16

# 시간과 설정
plt.style.use('dark_background')
fig, ax = plt.subplots()

# 'inferno' 색상 맵과 bilinear 보간을 사용해 더 불량고 명확하게 표현
img = ax.imshow(np.zeros((numRows, numCols)), cmap='inferno', interpolation='bilinear', vmin=0, vmax=100)
plt.colorbar(img, ax=ax, label="Pressure Intensity")

# HPF 활용 기능을 포함한 데이터 가공 함수
def apply_hpf(data, threshold=20):
    return np.where(data > threshold, data, 0)

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
            # print(f"Row {row}: {values}")
            pressure_matrix[row] = list(map(int, values))
            row += 1

    # 모든 행의 데이터가 준비된 경우에만 업데이트
    if row == numRows:
        # HPF 적용
        filtered_matrix = apply_hpf(pressure_matrix)
        print(filtered_matrix)
        img.set_data(filtered_matrix)
    return [img]



# 애니먼이션으로 실시간 업데이트 설정
ani = animation.FuncAnimation(fig, update_data, interval=100, blit=False)  # 딩레이 100ms로 설정
plt.title("Real-Time Pressure Sensor Visualization with HPF (16x16)")
plt.show()
