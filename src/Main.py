import os
import pandas as pd

# 경로 및 파일 설정
file_path = r"C:\Users\user\github\Pandas-Project\data\chess.csv"

# 파일 확인
if not os.path.exists(file_path):
    print(f"Error: 파일이 경로에 없습니다. ({file_path})")
else:
    # CSV 파일 읽기
    data = pd.read_csv(file_path)

    # 데이터 확인
    print("데이터셋의 첫 5개 행:")
    print(data.head())

    # 데이터 정보 확인
    print("\n데이터셋 정보:")
    print(data.info())

    # 통계 요약
    print("\n데이터 통계 요약:")
    print(data.describe())

    # 특정 열의 값 분포 확인 (예: 'winner' 열이 있다고 가정)
    if 'winner' in data.columns:
        print("\n'Winner' 열의 값 분포:")
        print(data['winner'].value_counts())

    # 결측값 확인
    print("\n결측값 확인:")
    print(data.isnull().sum())

    # 가장 승률이 접전인 오프닝 분석 (white_id와 black_id를 고려하여 승률 계산)
    if 'opening_name' in data.columns and 'winner' in data.columns and 'white_id' in data.columns and 'black_id' in data.columns:
        print("\n상대방을 고려한 가장 승률이 접전인 오프닝 분석:")

        # 오프닝별, 대결 상대별 승리 횟수 계산
        opening_win_counts = data.groupby(['opening_name', 'winner']).size().unstack(fill_value=0)

        # 각 오프닝별 승률 계산 (백 승률, 무승부, 블랙 승률)
        opening_win_counts['total_games'] = opening_win_counts.sum(axis=1)
        opening_win_counts['white_win_rate'] = opening_win_counts.get('white', 0) / opening_win_counts['total_games']
        opening_win_counts['black_win_rate'] = opening_win_counts.get('black', 0) / opening_win_counts['total_games']
        opening_win_counts['draw_rate'] = opening_win_counts.get('draw', 0) / opening_win_counts['total_games']

        # 총 대국 횟수가 적은 경우 필터링
        min_games_threshold = 10
        opening_win_counts = opening_win_counts[opening_win_counts['total_games'] >= min_games_threshold]

        # 승률 차이가 가장 적은 (가장 접전인) 오프닝 찾기
        opening_win_counts['win_rate_diff'] = abs(opening_win_counts['white_win_rate'] - opening_win_counts['black_win_rate'])
        closest_openings = opening_win_counts.nsmallest(5, 'win_rate_diff')
        most_closet_opening_name = opening_win_counts.nsmallest(1, 'win_rate_diff').index[0]

        # 상위 5개 오프닝 승률 차이가 적은 결과 표시
        print("\n상대방을 고려한 가장 승률이 접전인 오프닝 상위 5개:")
        pd.set_option('display.max_columns', None)  # 모든 열 출력
        pd.set_option('display.width', None)  # 모든 내용 출력
        print(closest_openings[['white_win_rate', 'black_win_rate', 'draw_rate', 'total_games', 'win_rate_diff']])

        # 가장 승률이 접전인 오프닝에서 백과 흑의 가장 빠른 승리 찾기
        print("\n최고로 승률이 접전인 오프닝에서 백과 흑의 가장 빠른 승리:")
        most_closet_opening_data = data[data['opening_name'] == most_closet_opening_name]
        white_wins = most_closet_opening_data[most_closet_opening_data['winner'] == 'white']
        black_wins = most_closet_opening_data[most_closet_opening_data['winner'] == 'black']

