# Alphabet-recognition

## 2020년 2학기 전산수학 기말 프로젝트

데이터 수집 및 전처리
- 알파벳 5개(J, O, E, Y, N)**를 선정하여 각각 25개씩, 총 125개의 이미지를 제작
- 이미지 데이터를 다음과 같은 과정으로 전처리:
    1. 이미지를 **흑백(grayscale)**으로 변환 후 불러오기
    2. 30×30 크기의 픽셀 데이터를 **1차원 배열(900차원 벡터)**로 변환
    3. 해당 알파벳의 레이블을 0~25 사이 숫자로 변환 (예: J=9, O=14, E=4, Y=24, N=13)
    4. 최종 데이터(input.npz)로 저장
       
머신러닝 모델
- 입력층(Input Layer): 900개의 뉴런 (각 픽셀에 해당)
- 은닉층(Hidden Layer): 40개의 뉴런
- 출력층(Output Layer): 26개의 뉴런 (A~Z까지 총 26개 알파벳을 표현)
  
학습 과정
1. input.npz 파일을 불러와 학습 데이터를 준비
2. 신경망 가중치 초기화 (랜덤 값 설정)
3. 순전파(forward propagation): 입력 데이터를 바탕으로 예측 수행
4. 오차(Error) 계산: 평균 제곱 오차(MSE)를 사용하여 모델의 성능 평가
5. 경사하강법을 이용한 가중치 업데이트
6. 일정 오차 이하로 수렴할 때까지 반복 학습
7. 학습 완료 후 가중치(learning.npz) 저장
   
테스트 과정
1. 새로운 손글씨 알파벳 이미지를 불러오기 (test1.png, test2.png 등)
2. 학습된 모델(learning.npz)을 사용하여 예측 수행
3. 신경망을 거친 후 출력층의 가장 높은 확률 값을 가진 뉴런의 인덱스를 알파벳으로 변환
4. 결과를 시각적으로 출력하여 예측 성능 확인
