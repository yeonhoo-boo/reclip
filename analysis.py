# 그래프 !
import pandas as pd
import matplotlib.pyplot as plt
import os

# 한글 폰트 설정 (맥)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# CSV 읽기 (폴더 안에 survey_result.csv 있으므로 상대 경로 가능)
csv_file = 'survey_result.csv'
df = pd.read_csv(csv_file, encoding='utf-8-sig')

# 컬럼 확인
print("설문 문항:", df.columns.tolist())

# 저장 폴더 만들기 (이미지 저장용)
img_folder = 'survey_graphs'
os.makedirs(img_folder, exist_ok=True)

# 모든 문항 반복
for col in df.columns:
    counts = df[col].value_counts()

    # 막대그래프
    plt.figure(figsize=(8,5))
    counts.plot(kind='bar', color='skyblue')
    plt.title(f'{col} - 막대그래프')
    plt.xlabel('응답')
    plt.ylabel('응답 수')
    plt.tight_layout()
    bar_file = os.path.join(img_folder, f'{col}_bar.png')
    plt.savefig(bar_file, dpi=300)
    plt.close()

    # 원그래프
    plt.figure(figsize=(6,6))
    counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.ylabel('')
    plt.title(f'{col} - 원그래프')
    plt.tight_layout()
    pie_file = os.path.join(img_folder, f'{col}_pie.png')
    plt.savefig(pie_file, dpi=300)
    plt.close()

    print(f"{col} 그래프 저장 완료: {bar_file}, {pie_file}")

print("\n모든 설문 문항 그래프 저장 완료! 폴더 확인:", img_folder)
