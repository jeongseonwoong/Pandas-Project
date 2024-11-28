import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 경로 및 파일 설정
file_path = r"C:\Users\user\github\Pandas-Project\data\chess.csv"

# 파일 확인
if not os.path.exists(file_path):
    print(f"Error: 파일이 경로에 없습니다. ({file_path})")
else:
    # CSV 파일 읽기
    data = pd.read_csv(file_path)

    # 흑과 백에서 각각 승률이 가장 높은 오프닝 분석
    print("\n흑과 백에서 각각 승률이 가장 높은 오프닝 분석")

    # 오프닝별, 대결 상대별 승리 횟수 계산
    opening_win_counts = data.groupby(['opening_name', 'winner']).size().unstack(fill_value=0)

    # 각 오프닝별 승률 계산 (백 승률, 블랙 승률)
    opening_win_counts['total_games'] = opening_win_counts.sum(axis=1)
    opening_win_counts['white_win_rate'] = opening_win_counts.get('white', 0) / opening_win_counts['total_games']
    opening_win_counts['black_win_rate'] = opening_win_counts.get('black', 0) / opening_win_counts['total_games']

    # 총 대국 횟수가 적은 경우 필터링
    min_games_threshold = 200
    opening_win_counts = opening_win_counts[opening_win_counts['total_games'] >= min_games_threshold]

    # 백에서 승률이 가장 높은 오프닝 찾기
    highest_white_win_opening = opening_win_counts['white_win_rate'].idxmax()
    highest_white_win_rate = opening_win_counts.loc[highest_white_win_opening, 'white_win_rate']
    print(f"\n백에서 승률이 가장 높은 오프닝: {highest_white_win_opening} (승률: {highest_white_win_rate:.2f})")

    # 흑에서 승률이 가장 높은 오프닝 찾기
    highest_black_win_opening = opening_win_counts['black_win_rate'].idxmax()
    highest_black_win_rate = opening_win_counts.loc[highest_black_win_opening, 'black_win_rate']
    print(f"흑에서 승률이 가장 높은 오프닝: {highest_black_win_opening} (승률: {highest_black_win_rate:.2f})")

    # 백에서 승률이 가장 높은 오프닝의 moves 별 승률 분석
    highest_white_opening_data = data[data['opening_name'] == highest_white_win_opening]
    highest_white_opening_data = highest_white_opening_data.copy()
    highest_white_opening_data['move_count'] = highest_white_opening_data['moves'].apply(lambda x: len(x.split()))
    highest_white_opening_data = highest_white_opening_data.sort_values(by='move_count')
    highest_white_opening_data['cumulative_white_wins'] = highest_white_opening_data['winner'].eq('white').cumsum()
    highest_white_opening_data['total_games'] = range(1, len(highest_white_opening_data) + 1)
    highest_white_opening_data['white_win_rate'] = highest_white_opening_data['cumulative_white_wins'] / \
                                                   highest_white_opening_data['total_games']

    # 백 승률 변화 선그래프로 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(highest_white_opening_data['move_count'], highest_white_opening_data['white_win_rate'], label='White Win Rate',
             color='b')
    plt.title(f'Win Rate Changes Over Moves for {highest_white_win_opening}')
    plt.xlabel('Number of Moves')
    plt.ylabel('White Win Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 흑에서 승률이 가장 높은 오프닝의 moves 별 승률 분석
    highest_black_opening_data = data[data['opening_name'] == highest_black_win_opening]
    highest_black_opening_data = highest_black_opening_data.copy()
    highest_black_opening_data['move_count'] = highest_black_opening_data['moves'].apply(lambda x: len(x.split()))
    highest_black_opening_data = highest_black_opening_data.sort_values(by='move_count')
    highest_black_opening_data['cumulative_black_wins'] = highest_black_opening_data['winner'].eq('black').cumsum()
    highest_black_opening_data['total_games'] = range(1, len(highest_black_opening_data) + 1)
    highest_black_opening_data['black_win_rate'] = highest_black_opening_data['cumulative_black_wins'] / \
                                                   highest_black_opening_data['total_games']

    # 흑 승률 변화 선그래프로 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(highest_black_opening_data['move_count'], highest_black_opening_data['black_win_rate'], label='Black Win Rate',
             color='r')
    plt.title(f'Win Rate Changes Over Moves for {highest_black_win_opening}')
    plt.xlabel('Number of Moves')
    plt.ylabel('Black Win Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 백과 흑에서 가장 승률이 높은 오프닝의 대결 데이터 추출
    matchup_data = data[(data['opening_name'] == highest_white_win_opening) | (data['opening_name'] == highest_black_win_opening)]
    matchup_data = matchup_data.copy()
    matchup_data['move_count'] = matchup_data['moves'].apply(lambda x: len(x.split()))

    # 머신러닝 모델 학습 및 평가 - moves에 따른 흑과 백의 승률 예측
    print("\n머신러닝 모델 학습 및 평가 - 흑과 백의 승률 예측")

    # 레이블 인코딩 (백 승리, 흑 승리, 무승부)
    le = LabelEncoder()
    matchup_data['winner_encoded'] = le.fit_transform(matchup_data['winner'])

    # 특징 및 타겟 변수 설정
    feature_columns = ['move_count']
    X = matchup_data[feature_columns]
    y = matchup_data['winner_encoded']

    # 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 랜덤 포레스트 모델 학습
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 모델 예측 및 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"모델 정확도: {accuracy:.2f}")
    print("분류 리포트:\n", classification_report(y_test, y_pred))

    # moves에 따른 흑과 백의 승률 예측 시각화
    X_full = pd.DataFrame({'move_count': range(1, 200)})  # 1부터 100까지의 move_count
    y_pred_full = model.predict_proba(X_full)

    plt.figure(figsize=(10, 6))
    plt.plot(X_full['move_count'], pd.Series(y_pred_full[:, le.transform(['white'])[0]]).rolling(window=10).mean(), label='Predicted White Win Rate (Smoothed)', color='b')
    plt.plot(X_full['move_count'], pd.Series(y_pred_full[:, le.transform(['black'])[0]]).rolling(window=10).mean(), label='Predicted Black Win Rate (Smoothed)', color='r')
    plt.plot(X_full['move_count'], pd.Series(y_pred_full[:, le.transform(['draw'])[0]]).rolling(window=10).mean(), label='Predicted Draw Rate (Smoothed)', color='g')
    plt.title(f'Predicted Win Rate Changes Over Moves for {highest_white_win_opening} vs {highest_black_win_opening}')
    plt.xlabel('Number of Moves')
    plt.ylabel('Predicted Win Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()
