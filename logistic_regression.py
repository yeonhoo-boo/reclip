import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score)
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드 및 탐색
print("="*70)
print("로지스틱 회귀 모델 - 학생 난이도 진단 시스템")
print("="*70)

print("1단계: 데이터 로드 및 탐색")
print("-"*70)

df = pd.read_excel('강의_데이터_250.xlsx')

print(f"\n 데이터 로드 완료")
print(f"총 샘플 수: {len(df)}")
print(f"총 칼럼 수: {len(df.columns)}")
print(f"강의 종류: {df['ID'].unique().tolist()}")
print(f"강의별 구간 수: {df.groupby('ID').size().to_dict()}")

# 데이터 샘플 확인
print(f"\n 데이터 샘플 :")
print(df.head(10))

print(f"\n 데이터 정보:")
print(df.info())

print(f"\n 기본 통계:")
print(df.describe())

# 결측값 확인
print(f"\n 결측값 확인:")
print(df.isnull().sum())

# 난이도 분포
print(f"\n 난이도 분포:")
difficulty_dist = df['menual score'].value_counts().sort_index()
print(difficulty_dist)
print(f"0 (이해함): {difficulty_dist.get(0, 0)}개")
print(f"1 (보통): {difficulty_dist.get(1, 0)}개")
print(f"2 (어려움): {difficulty_dist.get(2, 0)}개")

# 데이터 전처리 및 분할

print("\n" + "="*70)
print("2단계: 데이터 전처리 및 분할")
print("-"*70)

# feature 선택: rewind, pause, scrap
X = df[['rewind', 'pause', 'scrap']]
y = df['menual score']

print(f"\n feature 및 label 선택 완료")
print(f"입력 (X): rewind, pause, scrap")
print(f"출력 (y): menual score (0/1/2)")
print(f"\n X 모양: {X.shape}")
print(f" y 모양: {y.shape}")

# 입력 데이터 통계
print(f"\n입력 통계:")
print(X.describe())

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n 학습/테스트 데이터 분할 완료")
print(f"학습 데이터: {len(X_train)}개 ({len(X_train)/len(df)*100:.1f}%)")
print(f"테스트 데이터: {len(X_test)}개 ({len(X_test)/len(df)*100:.1f}%)")

print(f"\n 학습 데이터의 난이도 분포:")
print(y_train.value_counts().sort_index())

print(f"\n 테스트 데이터의 난이도 분포:")
print(y_test.value_counts().sort_index())

# 특성 스케일링 (표준화)
print("\n" + "="*70)
print("3단계: 특성 스케일링 (StandardScaler)")
print("-"*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n특성 스케일링 완료")
print(f"   방법: StandardScaler (평균=0, 표준편차=1)")
print(f"\n 스케일링 전 X_train 통계:")
print(f"   평균: {X_train.mean().to_dict()}")
print(f"   표준편차: {X_train.std().to_dict()}")

print(f"\n스케일링 후 X_train_scaled 통계:")
scaler_df = pd.DataFrame(X_train_scaled, columns=['rewind', 'pause', 'scrap'])
print(f"   평균: {scaler_df.mean().to_dict()}")
print(f"   표준편차: {scaler_df.std().to_dict()}")

# 로지스틱 회귀 모델 학습
print("\n" + "="*70)
print("4단계: 로지스틱 회귀 모델 학습")
print("-"*70)

# 모델 생성 및 학습
lr_model = LogisticRegression(
    max_iter=2000,           # 최대 반복 횟수
    random_state=42,         # 재현성을 위한 시드값
    solver='lbfgs'          # 최적화 알고리즘
)

print(f"\n 모델 파라미터:")
print(f"   max_iter: 2000")
print(f"   solver: lbfgs")
print(f"   random_state: 42")

lr_model.fit(X_train_scaled, y_train)
print(f"모델 학습 완료!")

# 모델 파라미터 확인
print(f"\n 학습된 모델의 파라미터:")
print(f"   클래스: {lr_model.classes_}")
print(f"   가중치(계수):\n{lr_model.coef_}")
print(f"   절편:\n{lr_model.intercept_}")

# 모델 성능 평가
print("\n" + "="*70)
print("5단계: 모델 성능 평가 (테스트 데이터)")
print("-"*70)

# 예측
y_pred_train = lr_model.predict(X_train_scaled)
y_pred_test = lr_model.predict(X_test_scaled)

# 정확도
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\n정확도 (Accuracy):")
print(f"   학습 데이터: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"   테스트 데이터: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# 정밀도, 재현율, F1-Score (가중평균)
precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')

print(f"\n분류 성능 메트릭:")
print(f"   정밀도 (Precision): {precision:.4f}")
print(f"   재현율 (Recall): {recall:.4f}")
print(f"   F1-Score: {f1:.4f}")

# 상세 분류 보고서
print(f"\n상세 분류 보고서 (클래스별):")
print(classification_report(y_test, y_pred_test, 
                           target_names=['이해함(0)', '보통(1)', '어려움(2)']))

# 혼동 행렬
cm = confusion_matrix(y_test, y_pred_test)
print(f"\n혼동 행렬 (Confusion Matrix):")
print(f"                 예측: 이해  예측: 보통  예측: 어려움")
for i, row in enumerate(cm):
    label_name = ['실제: 이해', '실제: 보통', '실제: 어려움'][i]
    print(f"{label_name}    {row[0]:3d}       {row[1]:3d}        {row[2]:3d}")

# 교차 검증
cv_scores = cross_val_score(
    LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs'),
    X_train_scaled, y_train, cv=5, scoring='f1_weighted'
)
print(f"\n교차 검증 (5-Fold):")
print(f"   각 폴드 F1-Score: {cv_scores}")
print(f"   평균 F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 성능 시각화
print("\n" + "="*70)
print(" 6단계: 성능 시각화")
print("-"*70)

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(10, 5))
fig.suptitle('로지스틱 회귀 모델 성능 분석', fontsize=16, fontweight='bold')

# 1) 혼동 행렬 히트맵
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['이해(0)', '보통(1)', '어려움(2)'],
            yticklabels=['이해(0)', '보통(1)', '어려움(2)'],
            cbar_kws={'label': '개수'})
ax1.set_title('혼동 행렬 (Confusion Matrix)', fontweight='bold')
ax1.set_ylabel('실제 값')
ax1.set_xlabel('예측 값')

# 2) 성능 메트릭 비교
ax2 = axes[0, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
scores = [test_accuracy, precision, recall, f1]
colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
bars = ax2.bar(metrics, scores, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Score', fontweight='bold')
ax2.set_title('성능 메트릭 비교', fontweight='bold')
ax2.set_ylim([0, 1])
ax2.grid(axis='y', alpha=0.3)
for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

# 3) 학습 vs 테스트 정확도
ax3 = axes[1, 0]
accuracies = [train_accuracy, test_accuracy]
labels = ['학습 데이터', '테스트 데이터']
colors_acc = ['#45B7D1', '#FF6B6B']
bars = ax3.bar(labels, accuracies, color=colors_acc, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Accuracy', fontweight='bold')
ax3.set_title('학습 vs 테스트 정확도', fontweight='bold')
ax3.set_ylim([0, 1])
ax3.grid(axis='y', alpha=0.3)
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

# 4) 클래스별 재현율
ax4 = axes[1, 1]
recall_per_class = recall_score(y_test, y_pred_test, average=None)
class_labels = ['이해함(0)', '보통(1)', '어려움(2)']
colors_class = ['#50C878', '#FFD700', '#FF6B6B']
bars = ax4.bar(class_labels, recall_per_class, color=colors_class, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Recall', fontweight='bold')
ax4.set_title('클래스별 재현율 (Recall)', fontweight='bold')
ax4.set_ylim([0, 1])
ax4.grid(axis='y', alpha=0.3)
for bar, rec in zip(bars, recall_per_class):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{rec:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('logistic_regression_performance.png', dpi=300, bbox_inches='tight')
print(f"\n성능 시각화 저장: logistic_regression_performance.png")
plt.show()

# 새 강의에서 난이도 진단 (Inference)
print("\n" + "="*70)
print("7단계: 새 강의에서 난이도 진단")
print("-"*70)

# 새로운 학생의 행동 데이터 (예시)
new_student_data = pd.DataFrame({
    'rewind': [2, 5, 1, 3, 4, 0, 6, 2, 1, 5],
    'pause': [1, 3, 0, 2, 2, 0, 3, 1, 1, 4],
    'scrap': [0, 1, 1, 0, 2, 1, 1, 1, 0, 2]
})

print(f"\n새 강의 새 학생의 행동 데이터:")
print(new_student_data)

# 스케일링
new_student_scaled = scaler.transform(new_student_data)

# 예측
predictions = lr_model.predict(new_student_scaled)
probabilities = lr_model.predict_proba(new_student_scaled)

print(f"\n" + "="*70)
print(" 진단 결과 (구간별)")
print("="*70)

difficulty_labels = {
    0: '이해함',
    1: '보통',
    2: '어려움'
}

actions = {
    0: 'X',
    1: '복습 권장',
    2: '복습 필수'
}

for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
    confidence = probs[pred] * 100
    label = difficulty_labels[pred]
    action = actions[pred]
    
    print(f"\n구간 {i+1}")
    print(f"   행동: 되감기={new_student_data.iloc[i]['rewind']}, "
          f"멈춤={new_student_data.iloc[i]['pause']}, "
          f"스크랩={new_student_data.iloc[i]['scrap']}")
    print(f"   진단: {label}")
    print(f"   확신도: {confidence:.2f}%")
    print(f"   추천: {action}")
    print(f"   확률분포: 이해(0)={probs[0]:.3f}, 보통(1)={probs[1]:.3f}, 어려움(2)={probs[2]:.3f}")
