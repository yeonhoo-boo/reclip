import pandas as pd
import random

# 설정값
lectures = ['L1', 'L2', 'L3', 'L4', 'L5']
sections = list(range(1, 11))
samples_per_section = 5

data = []

for lec in lectures:
    for sec in sections:
        for i in range(samples_per_section):
            label_rand = random.random()
            
            # 클래스 분포 결정 (0: 44%, 1: 38%, 2: 18%)
            if label_rand < 0.44:
                label = 0
                is_outlier = random.random() < 0.20 # 유형 A 이상치
                if is_outlier:
                    rewind, pause, scrap = random.randint(5, 8), random.randint(4, 7), 1
                else:
                    rewind, pause, scrap = random.randint(0, 1), random.randint(0, 2), 0
            
            elif label_rand < 0.82:
                label = 1
                rewind, pause, scrap = random.randint(2, 3), random.randint(2, 4), random.randint(0, 1)
            
            else:
                label = 2
                is_outlier = random.random() < 0.20 # 유형 B 이상치
                if is_outlier:
                    rewind, pause, scrap = 0, 0, 0
                else:
                    rewind, pause, scrap = random.randint(4, 9), random.randint(5, 10), random.randint(1, 3)
            
            data.append([lec, sec, rewind, pause, scrap, label])

df_generated = pd.DataFrame(data, columns=[
    'ID',             # 강의
    'Section',     # 구간
    'rewind',   # 되감기 횟수
    'pause',    # 멈춤 횟수
    'scrap',    # 스크랩 횟수
    'menual score'    # 정답 레이블 
])

# 엑셀 파일 저장
df_generated.to_excel('강의_데이터_250.xlsx', index=False)
