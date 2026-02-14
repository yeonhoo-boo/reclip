import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score)
import warnings
warnings.filterwarnings('ignore')

# 1단계: 데이터 로드 및 탐색
print("="*70)
print("Random Forest 모델 - 학생 난이도 진단 시스템")
print("="*70)

print("\n1단계: 데이터 로드 및 탐색")
print("-"*70)

df = pd.read_excel('강의_데이터_250.xlsx')

print(f"\n데이터 로드 완료")
print(f"   총 샘플 수: {len(df)}")
print(f"   총 칼럼 수: {len(df.columns)}")
print(f"   강의 종류: {df['ID'].unique().tolist()}")
print(f"   강의별 구간 수: {df.groupby('ID').size().to_dict()}")

print(f"\n데이터 샘플 (처음 10개):")
print(df.head(10))

print(f"\n기본 통계:")
print(df.describe())

print(f"\n결측값 확인:")
print(df.isnull().sum())

print(f"\n난이도 분포:")
difficulty_dist = df['menual score'].value_counts().sort_index()
print(difficulty_dist)
print(f"   0 (이해함): {difficulty_dist.get(0, 0)}개")
print(f"   1 (보통): {difficulty_dist.get(1, 0)}개")
print(f"   2 (어려움): {difficulty_dist.get(2, 0)}개")

# 2단계: 데이터 전처리 및 분할
print("\n" + "="*70)
print("2단계: 데이터 전처리 및 분할")
print("-"*70)

X = df[['rewind', 'pause', 'scrap']]
y = 2 - df['menual score']  

print(f"\n피처 및 라벨 선택 완료")
print(f"   입력 피처 (X): rewind, pause, scrap")
print(f"   출력 라벨 (y): menual score (0/1/2)")
print(f"\n   X 모양: {X.shape}")
print(f"   y 모양: {y.shape}")

print(f"\n입력 피처 통계:")
print(X.describe())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n학습/테스트 데이터 분할 완료")
print(f"   학습 데이터: {len(X_train)}개 ({len(X_train)/len(df)*100:.1f}%)")
print(f"   테스트 데이터: {len(X_test)}개 ({len(X_test)/len(df)*100:.1f}%)")

print(f"\n학습 데이터의 난이도 분포:")
print(y_train.value_counts().sort_index())

print(f"\n테스트 데이터의 난이도 분포:")
print(y_test.value_counts().sort_index())

# 3단계: Random Forest 모델 학습
print("\n" + "="*70)
print("3단계: Random Forest 모델 학습")
print("-"*70)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print(f"\n모델 파라미터:")
print(f"   n_estimators: 100 ")
print(f"   max_depth: 5")
print(f"   min_samples_split: 3")
print(f"   min_samples_leaf: 2")
print(f"   random_state: 42")

rf_model.fit(X_train, y_train)
print(f"모델 학습 완료!")

print(f"\n학습된 랜덤 포레스트의 특성:")
print(f"   의사결정 나무 개수: {rf_model.n_estimators}")
print(f"   각 나무의 최대 깊이: {rf_model.max_depth}")

# 4단계: 모델 성능 평가
print("\n" + "="*70)
print("4단계: 모델 성능 평가 (테스트 데이터)")
print("-"*70)

y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\n정확도 (Accuracy):")
print(f"   학습 데이터: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"   테스트 데이터: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')

print(f"\n분류 성능 메트릭:")
print(f"   정밀도 (Precision): {precision:.4f}")
print(f"   재현율 (Recall): {recall:.4f}")
print(f"   F1-Score: {f1:.4f}")

print(f"\n상세 분류 보고서 (클래스별):")
print(classification_report(y_test, y_pred_test, 
                           target_names=['이해함(0)', '보통(1)', '어려움(2)']))

cm = confusion_matrix(y_test, y_pred_test)
print(f"\n혼동 행렬 (Confusion Matrix):")
print(f"                 예측: 이해  예측: 보통  예측: 어려움")
for i, row in enumerate(cm):
    label_name = ['실제: 이해', '실제: 보통', '실제: 어려움'][i]
    print(f"{label_name}    {row[0]:3d}       {row[1]:3d}        {row[2]:3d}")

cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    X_train, y_train, cv=5, scoring='f1_weighted'
)
print(f"\n교차 검증 (5-Fold):")
print(f"   각 폴드 F1-Score: {cv_scores}")
print(f"   평균 F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n피처 중요도 (Feature Importance):")
print(feature_importance)

# 5단계: 성능 시각화

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

print("\n" + "="*70)
print("5단계: 성능 시각화")
print("-"*70)

fig, axes = plt.subplots(2, 2, figsize=(10, 5))
fig.suptitle('Random Forest 모델 성능 분석', fontsize=16, fontweight='bold')

# 1) 혼동 행렬
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

# 4) 피처 중요도
ax4 = axes[1, 1]
features = feature_importance['feature'].values
importances = feature_importance['importance'].values
colors_feat = ['#50C878', '#FFD700', '#FF6B6B']
bars = ax4.barh(features, importances, color=colors_feat, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Importance', fontweight='bold')
ax4.set_title('피처 중요도 (Feature Importance)', fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
for bar, imp in zip(bars, importances):
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2.,
            f'{imp:.4f}', ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('random_forest_performance.png', dpi=300, bbox_inches='tight')
print(f"\n성능 시각화 저장: random_forest_performance.png")
plt.show()

# 6단계: OOB 성능 평가 (Out-of-Bag)
print("\n" + "="*70)
print("6단계: OOB 성능 평가 (Out-of-Bag)")
print("-"*70)

oob_rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42,
    oob_score=True,
    n_jobs=-1
)
oob_rf_model.fit(X_train, y_train)

print(f"\nOut-of-Bag (OOB) 정확도: {oob_rf_model.oob_score_:.4f}")
print(f"   OOB는 각 나무가 학습에 사용하지 않은 샘플로 평가한 점수입니다.")
print(f"   추가 테스트 데이터 없이도 모델 성능을 추정할 수 있습니다.")

# 7단계: 새 강의에서 난이도 진단
print("\n" + "="*70)
print("7단계: 새 강의에서 난이도 진단")
print("-"*70)

new_student_data = pd.DataFrame({
    'rewind': [2, 5, 1, 3, 4, 0, 6, 2, 1, 5],
    'pause': [1, 3, 0, 2, 2, 0, 3, 1, 1, 4],
    'scrap': [0, 1, 1, 0, 2, 1, 1, 1, 0, 2]
})

print(f"\n새 강의 새 학생의 행동 데이터:")
print(new_student_data)

predictions = rf_model.predict(new_student_data)
probabilities = rf_model.predict_proba(new_student_data)

print(f"\n" + "="*70)
print("진단 결과 (구간별)")
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
