import torch
import torch.nn as nn
import torch.optim as optim

# 1. 데이터 로드 (임의의 입력 데이터 생성 및 정규화)
raw_input_data = torch.tensor([
    [
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0., 50., 24., 34., 39.,  0., 38.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0., 92.,  0.,  0.,  0.,  0.,  0., 25.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 23.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 23.,  0., 37., 28.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 22., 28., 46.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 26.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 23.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0., 21.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 71., 46.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 36., 31.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 23., 59.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 52., 72.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]
    ]
], dtype=torch.float32)

input_data = raw_input_data / raw_input_data.max()  # 0에서 1 사이로 정규화
input_data = input_data.repeat(64, 1, 1)  # 배치 크기 64로 확장

target = torch.randn(64, dtype=torch.float32)  # 타겟값: 속도 변화량 (실수형), 크기: (배치 크기 64)

# 2. 신경망 정의
class VelocityPredictionNet(nn.Module):
    def __init__(self):
        super(VelocityPredictionNet, self).__init__()
        self.fc1 = nn.Linear(16*16, 128)  # 입력: 16*16, 출력: 128
        self.fc2 = nn.Linear(128, 64)     # 입력: 128, 출력: 64
        self.fc3 = nn.Linear(64, 1)       # 입력: 64, 출력: 1 (속도 변화량)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 16*16)  # 입력 이미지를 1차원으로 펼침
        x = self.fc1(x)  # 첫 번째 완전연결 계층
        x = self.activation(x)  # 활성화 함수 적용 (ReLU)
        x = self.fc2(x)  # 두 번째 완전연결 계층
        x = self.activation(x)  # 활성화 함수 적용 (ReLU)
        x = self.fc3(x)  # 세 번째 완전연결 계층 (출력층)
        return x

# 모델 생성
model = VelocityPredictionNet()

# 3. 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()  # 손실 함수: 평균 제곱 오차 (MSE)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 옵티마이저: Adam, 학습률 0.001로 변경

# 4. 학습 루프 정의
epochs = 1000  # 학습 반복 횟수
target = target.view(-1, 1)  # 타겟값 형태 맞추기

for epoch in range(epochs):
    # 순전파
    output = model(input_data)
    loss = criterion(output, target)
    
    # 역전파 및 가중치 업데이트
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 100번째 에포크마다 손실 출력
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
