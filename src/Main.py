import os
import pandas as pd

# 경로 및 파일 설정
path = r"C:\Users\user\github\Pandas-Project\data"
file_name = "chess.csv"  # 파일 이름 변경
file_path = os.path.join(path, file_name)

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

    #오프닝
    print("오프닝 이름과 오프닝 Eco")
