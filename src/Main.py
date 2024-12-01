# 가장 많이 나오는 체스 오프닝을 분석하는 코드
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 윈도우의 경우, 맑은 고딕 폰트 사용
fontprop = fm.FontProperties(fname=font_path)
plt.rc('font', family=fontprop.get_name())

# 경로 및 파일 설정
file_path = r"C:\Users\user\github\Pandas-Project\data\chess.csv"

# 파일 확인
if not os.path.exists(file_path):
    print(f"Error: 파일이 경로에 없습니다. ({file_path})")
else:
    # CSV 파일 읽기
    data = pd.read_csv(file_path)

    # 오프닝 이름에서 기본 오프닝과 Variation 분리
    data[['base_opening', 'variation']] = data['opening_name'].str.split(': ', n=1, expand=True)

    # 가장 많이 등장하는 기본 오프닝 1개 선택
    most_common_opening = data['base_opening'].value_counts().idxmax()

    # 해당 기본 오프닝에 대한 Variation 빈도수 계산 및 상위 5개 출력
    variation_counts = data[data['base_opening'] == most_common_opening]['variation'].value_counts()
    print(f"\n{most_common_opening}의 상위 5개의 Variation:")
    print(variation_counts.head())

    # 각 Variation별 승률 계산
    variation_win_rates = data[data['base_opening'] == most_common_opening].groupby('variation')['winner'].value_counts(normalize=True).unstack().fillna(0)
    print("\nGroupBy 변수 요약:")
    print(variation_win_rates.describe())
    print("\nGroupBy 변수 합계:")
    print(variation_win_rates.sum())
    print("\nGroupBy 변수 최대값:")
    print(variation_win_rates.max())

    # 각 승리 유형별 평균값 출력
    print(f"\n백 승리의 평균 승률: {variation_win_rates['white'].mean():.2f}")
    print(f"흑 승리의 평균 승률: {variation_win_rates['black'].mean():.2f}")
    print(f"무승부의 평균 승률: {variation_win_rates['draw'].mean():.2f}")

    # 상위 5개의 Variation에 대한 승률 출력
    for variation in variation_counts.head().index:
        win_rate = variation_win_rates.loc[variation]
        print(f"\n{most_common_opening} - {variation}의 승률:")
        print(win_rate)

    # 상위 5개 Variation 중 가장 접전인 Variation 선택 (승률 차이가 가장 적은 Variation)
    closest_variation = variation_win_rates.loc[variation_counts.head().index].apply(lambda x: abs(x.get('white', 0) - x.get('black', 0)), axis=1).idxmin()

    # 각 Variation별 turns에 따른 승률 예측을 위한 데이터 준비
    variation_data = data[(data['base_opening'] == most_common_opening) & (data['variation'] == closest_variation)]

    # 특징과 타겟 설정 (turns와 승자)
    X = variation_data[['turns']]
    y = variation_data['winner'].apply(lambda x: 1 if x == 'white' else 0)  # white 승리: 1, black 승리: 0

    # 학습과 테스트 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Gradient Boosting Classifier 모델 학습
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 테스트 데이터로 예측 수행
    y_pred = model.predict(X_test)

    # 예측 결과 평가
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n모델 정확도: {accuracy:.2f}")
    print("분류 리포트:\n", classification_report(y_test, y_pred))

    # 새로운 turns 값에 따른 승률 예측
    turns_range = pd.DataFrame({'turns': np.arange(1, 151)})  # DataFrame으로 변환하여 feature name 유지
    win_probabilities = model.predict_proba(turns_range)[:, 1]  # white 승리 확률

    # 예측 결과 스무딩 처리
    smoothed_win_probabilities = pd.Series(win_probabilities).rolling(window=10, min_periods=1).mean()

    # 예측 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(turns_range['turns'], smoothed_win_probabilities, label=f'Predicted Win Rate ({closest_variation})', color='b')
    plt.xlabel('Number of Turns', fontproperties=fontprop)
    plt.ylabel('Predicted White Win Rate', fontproperties=fontprop)
    plt.title(f'{most_common_opening} - {closest_variation}의 Turns에 따른 예측 승률 변화', fontproperties=fontprop)
    plt.legend(prop=fontprop)
    plt.grid()
    plt.tight_layout()
    plt.show()

    # 상위 5개의 Variation에 대한 승률 시각화
    plt.figure(figsize=(10, 6))
    for variation in variation_counts.head().index:
        win_rate = variation_win_rates.loc[variation]
        plt.bar(variation, win_rate.get('white', 0), label=f'{variation} - White Win Rate', alpha=0.6)
        plt.bar(variation, win_rate.get('black', 0), bottom=win_rate.get('white', 0), label=f'{variation} - Black Win Rate', alpha=0.6)
    plt.xlabel('Variation', fontproperties=fontprop)
    plt.ylabel('Win Rate', fontproperties=fontprop)
    plt.title(f'{most_common_opening}의 상위 5개 Variation 별 승률', fontproperties=fontprop)
    plt.legend(prop=fontprop)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
