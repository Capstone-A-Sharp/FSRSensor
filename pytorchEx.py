import torch
import torch.nn as nn
import torch.optim as optim

# 1. 데이터 로드 (임의의 입력 데이터 생성)
input_data = torch.randint(0, 256, (64, 28*28), dtype=torch.float32)  # 입력값: 0에서 255 사이의 임의의 정수, 크기: (배치 크기 64, 28*28)
target = torch.ones(64, dtype=torch.long)  # 타겟값: 1.0, 크기: (배치 크기 64)

# 2. 신경망 정의
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # 입력: 28*28, 출력: 128
        self.fc2 = nn.Linear(128, 10)     # 입력: 128, 출력: 10 (숫자 0-9)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)  # 입력 이미지를 1차원으로 펼침
        print("Input to the first layer:", x.detach().cpu().numpy())
        
        x = self.fc1(x)  # 첫 번째 완전연결 계층
        print("Output of the first linear layer:", x.detach().cpu().numpy())
        
        x = self.activation(x)  # 활성화 함수 적용 (ReLU)
        print("Output after activation (ReLU):", x.detach().cpu().numpy())
        
        x = self.fc2(x)  # 두 번째 완전연결 계층
        print("Output of the second linear layer:", x.detach().cpu().numpy())
        return x

# 모델 생성
model = SimpleNet()

# 3. 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 순전파 및 역전파 과정 출력
# 순전파 과정 출력
print("\n--- Forward Propagation ---")
output = model(input_data)

# 손실 계산
loss = criterion(output, target)
print("\nLoss:", loss.item())

# 역전파 과정 출력
print("\n--- Backward Propagation ---")
optimizer.zero_grad()  # 기울기 초기화
loss.backward()  # 역전파 수행

# 각 레이어의 기울기 확인
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Gradient for {name}:")
        print(param.grad.detach().cpu().numpy())

# 가중치 업데이트
optimizer.step()
