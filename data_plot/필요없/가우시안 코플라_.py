import pandas as pd                       # pandas 불러오기: 데이터프레임 처리용
import numpy as np                        # numpy 불러오기: 수치연산용
import matplotlib.pyplot as plt           # matplotlib.pyplot 불러오기: 시각화용
from scipy import stats                   # scipy.stats 불러오기: 통계함수 사용
from scipy.interpolate import PchipInterpolator, UnivariateSpline  # 보간함수와 스플라인 사용
import io                                 # io.StringIO 사용을 위해 불러오기
from textwrap import dedent               # 여러줄 문자열 정리용 dedent
from collections import OrderedDict       # 순서 보장 dict 사용
import pprint                             # pprint로 읽기좋게 출력

# ================== 0. 설정 파라미터 ==================
B_BOOT = 700           # 부트스트랩 반복 횟수
CI_LEVEL = 0.95        # 신뢰구간 수준
band_type = 'mean'     # 'mean' 또는 'predictive' 선택값
grid_points = 220      # 그리드 포인트 개수
grid_mode = 'quantile' # 그리드 생성 방식: 'quantile' 또는 'linear'
quantile_smoothing = True  # 분위수 역함수 스무딩 사용 여부
optional_spline_smoothing = True  # 부트스트랩 결과 스플라인 후처리 여부
smooth_factor = 0.25   # 스플라인 후처리 강도(heuristic)
min_points_copula = 5  # 코플라 추정 최소 표본 수

# ================== 1. 데이터 읽기 ==================
policy_text = """date	Statement	P&B	Policy
2017-02	0.33	0.32	0.24
2017-03	0.4	0.4	0.35
2017-05	0.34	0.36	0.28
2017-06	0.4	0.35	0.33
2017-07	0.35	0.32	0.26
2017-09	0.38	0.36	0.25
2017-11	0.4	0.43	0.41
2017-12	0.58	0.55	0.58
2018-01	0.66	0.65	0.53
2018-03	0.4	0.52	0.58
2018-05	0.44	0.4	0.27
2018-06	0.75	0.7	0.64
2018-08	0.65	0.67	0.65
2018-09	0.79	0.72	0.82
2018-11	0.66	0.65	0.45
2018-12	0.7	0.65	0.55
2019-01	0.48	0.58	0.4
2019-03	0.15	0.18	0.03
2019-05	0.27	0.22	0.05
2019-06	0.2	0.12	0.04
2019-07	0.21	0.2	0.06
2019-09	0.23	0.2	0.06
2019-10	0.2	0.22	0.06
2019-12	0.3	0.25	0.14
2020-01	0.28	0.25	0.08
2020-03x	-0.2	-0.1	-0.22
2020-03y	-0.62	-0.4	-0.66
2020-04	-0.75	-0.45	-0.68
2020-06	-0.66	-0.5	-0.62
2020-07	-0.4	-0.22	-0.25
2020-09	-0.25	-0.1	-0.2
2020-11	-0.25	-0.1	-0.1
2020-12	-0.22	-0.1	-0.18
2021-01	-0.3	-0.15	-0.05
2021-03	0.1	0.2	0.27
2021-04	0.2	0.22	0.35
2021-06	0.4	0.435	0.44
2021-07	0.4	0.45	0.6
2021-09	0.3	0.25	0.2
2021-11	0.3	0.25	0.25
2021-12	0.32	0.27	0.27
2022-01	0.26	0.2	0.15
2022-03	0.1	0.04	0.03
2022-05	-0.15	-0.18	-0.1
2022-06	-0.15	-0.2	-0.15
2022-07	-0.3	-0.4	-0.55
2022-09	-0.25	-0.35	-0.48
2022-11	-0.2	-0.25	-0.45
2022-12	-0.2	-0.25	-0.5
2023-02	-0.1	-0.14	-0.12
2023-03	-0.14	-0.18	-0.35
2023-05	0.1	0.04	-0.2
2023-06	0.1	0.05	-0.12
2023-07	0.1	0.05	-0.02
2023-09	0.18	0.1	-0.12
2023-11	0.15	0.1	-0.15
2023-12	-0.1	-0.08	-0.25
2024-01	0.3	0.2	0
2024-03	0.32	0.29	0.1
2024-05	0.2	0.1	0.02
2024-06	0.3	0.27	0.1
2024-07	0.25	0.2	0.04
2024-09	0.3	0.4	0.2
2024-11	0.3	0.25	0.1
2024-12	0.3	0.32	0.2
2025-01	0.3	0.25	0.05
2025-03	0.22	0.25	0.12
2025-05	0.15	0.1	-0.08
2025-06	0.3	0.22	0.05
2025-07	-0.15	-0.08	-0.2
"""                                        # 정책 데이터 문자열 블록

cpi_text = """Prev_CPI_Value	Next_CPI_Value	FOMC_date
241.432	243.603	2017-02-01
243.603	243.801	2017-03-15
243.801	244.733	2017-05-03
244.733	244.955	2017-06-14
244.955	244.786	2017-07-26
245.519	246.819	2017-09-20
246.819	246.669	2017-11-01
246.669	246.524	2017-12-13
246.524	247.867	2018-01-31
248.991	249.554	2018-03-21
249.554	251.588	2018-05-02
251.588	251.989	2018-06-13
251.989	252.146	2018-08-01
252.146	252.439	2018-09-26
252.439	252.038	2018-11-08
252.038	251.233	2018-12-19
251.233	251.712	2019-01-30
252.776	254.202	2019-03-20
254.202	256.092	2019-05-01
256.092	256.143	2019-06-19
256.143	256.571	2019-07-31
256.558	256.759	2019-09-18
256.759	257.346	2019-10-30
257.208	256.974	2019-12-11
256.974	257.971	2020-01-29
257.971	258.115	2020-03-03
258.678	258.115	2020-03-15
258.115	256.389	2020-04-29
256.394	257.797	2020-06-10
257.797	259.101	2020-07-29
259.918	260.28	2020-09-16
260.28	260.229	2020-11-05
260.229	260.474	2020-12-16
260.474	261.582	2021-01-27
263.014	264.877	2021-03-17
264.877	267.054	2021-04-28
269.195	271.696	2021-06-16
271.696	273.003	2021-07-28
273.567	274.31	2021-09-22
274.31	277.948	2021-11-03
277.948	278.802	2021-12-15
278.802	281.148	2022-01-26
283.716	287.504	2022-03-16
287.504	292.296	2022-05-04
292.296	296.311	2022-06-15
296.311	296.276	2022-07-27
296.171	296.808	2022-09-21
296.808	297.711	2022-11-02
297.711	296.797	2022-12-14
296.797	300.84	2023-02-01
300.84	301.836	2023-03-22
301.836	304.127	2023-05-03
304.127	305.109	2023-06-14
305.109	305.691	2023-07-26
307.026	307.789	2023-09-20
307.789	307.051	2023-11-01
307.051	306.746	2023-12-13
306.746	308.417	2024-01-31
310.326	312.332	2024-03-20
312.332	314.069	2024-05-01
314.069	314.175	2024-06-12
314.175	314.54	2024-07-31
314.796	315.301	2024-09-18
315.301	315.493	2024-11-07
315.493	315.605	2024-12-18
315.605	317.671	2025-01-29
319.082	319.799	2025-03-19
321.465	322.561	2025-05-18
321.465	322.561	2025-06-18
322.561	323.048	2025-07-30
"""                                        # CPI 데이터 문자열 블록

policy_df_raw = pd.read_csv(io.StringIO(dedent(policy_text)), sep=r"\s+")  # policy 텍스트를 DataFrame으로 읽음
cpi_df = pd.read_csv(io.StringIO(dedent(cpi_text)), sep=r"\s+")            # cpi 텍스트를 DataFrame으로 읽음
cpi_df['FOMC_date'] = pd.to_datetime(cpi_df['FOMC_date'])                   # FOMC_date 열을 datetime 타입으로 변환

# ================== 2. CPI date key ==================
cpi_df['month'] = cpi_df['FOMC_date'].dt.strftime('%Y-%m')  # 연-월 문자열 생성
cpi_df['idx_in_month'] = cpi_df.groupby('month').cumcount()  # 월 내 인덱스(동일 월 순번) 생성

def make_date_key(row):                                             # date_key 생성 함수 정의
    if row['month'] == '2020-03':                                   # 2020-03 특별 처리
        return f"{row['month']}{'x' if row['idx_in_month']==0 else 'y'}"  # 첫 회의 x, 두번째 y
    return row['month'] if row['idx_in_month'] == 0 else f"{row['month']}_{row['idx_in_month']}"  # 기본 키 포맷
cpi_df['date_key'] = cpi_df.apply(make_date_key, axis=1)           # date_key 컬럼 생성

# ================== 3. Policy 전처리 ==================
policy_df = policy_df_raw.copy()                                   # 원본 복사본 생성
policy_df['date'] = policy_df['date'].str.strip()                   # date 문자열 양쪽 공백 제거
has_x = (policy_df['date'] == '2020-03x').any()                     # 2020-03x 존재 여부 체크
has_y = (policy_df['date'] == '2020-03y').any()                     # 2020-03y 존재 여부 체크
has_plain_mar = (policy_df['date'] == '2020-03').any()              # 2020-03(plain) 존재 여부 체크
if has_plain_mar and (has_x or has_y):                              # 서로 섞여있으면 에러
    raise ValueError("2020-03 / 2020-03x / 2020-03y 혼재.")

if has_x and has_y:                                                 # x,y 둘 다 있으면
    method = "B"                                                    # 방법 B 선택
    policy_df['date_key'] = policy_df['date']                       # date_key를 date로 설정
    merged = policy_df.merge(                                       # CPI의 date_key로 필요한 열 병합
        cpi_df[['date_key','Prev_CPI_Value','Next_CPI_Value','FOMC_date']],
        on='date_key', how='left'
    )
elif has_plain_mar:                                                 # plain 2020-03만 있으면
    method = "A"                                                    # 방법 A 선택
    mar_rows = cpi_df[cpi_df['month'] == '2020-03']                  # 2020-03에 해당하는 CPI 행들
    if len(mar_rows) != 2:                                           # 해당 월에 2회의 회의가 있어야 함
        raise ValueError("2020-03 두 회의 못찾음.")
    policy_df['date_key'] = policy_df['date']                       # date_key 설정
    cpi_month = cpi_df.groupby('month', as_index=False).agg({       # 월별 CPI 평균 및 최소 FOMC_date 생성
        'Prev_CPI_Value':'mean','Next_CPI_Value':'mean','FOMC_date':'min'
    }).rename(columns={'month':'date_key'})
    merged = policy_df.merge(                                       # 병합 수행
        cpi_month[['date_key','Prev_CPI_Value','Next_CPI_Value','FOMC_date']],
        on='date_key', how='left'
    )
else:
    raise ValueError("2020-03 표기 인식 실패.")                     # 어떤 형식도 아니면 에러

missing = merged[merged['Prev_CPI_Value'].isna()]                    # Prev_CPI_Value가 없는 행 확인
if not missing.empty:
    print(missing[['date','date_key']])                              # 누락 행 출력
    raise AssertionError("일부 날짜 매핑 실패.")                     # 예외 발생
print(f"Method {method} 적용. 관측수={len(merged)}")                 # 적용한 method와 관측수 출력

# ================== 4. 분석 준비 ==================
pairs = [
    ('Statement', 'Prev_CPI_Value'),
    ('P&B', 'Prev_CPI_Value'),
    ('Policy', 'Prev_CPI_Value'),
    ('Statement', 'Next_CPI_Value'),
    ('P&B', 'Next_CPI_Value'),
    ('Policy', 'Next_CPI_Value'),
]                                                                # 분석할 X-Y 페어 목록
merged['year'] = merged['FOMC_date'].dt.year                     # 연도 컬럼 생성
early_mask = merged['year'] <= 2022                              # 2017-2022 마스크
late_mask  = merged['year'] >= 2023                              # 2023-2025 마스크

# 색상
early_point_color = '#1f77b4'                                     # 초기기간 점 색
late_point_color  = '#d62728'                                     # 후기기간 점 색
subset_colors = {
    '17-22': '#1f77b4',
    '23-25': '#d62728',
    '17-25'  : '#2ca02c'
}                                                                  # 서브셋별 선 색
ci_gray_levels = {
    '17-22': '#555555',
    '23-25': '#888888',
    '17-25' : '#BBBBBB'
}                                                                  # CI 채우기 색(회색계)
ci_gray_alpha = {
    '17-22': 0.30,
    '23-25': 0.30,
    '17-25' : 0.25
}                                                                  # CI 투명도 딕셔너리

alpha = 1 - CI_LEVEL                                               # 유의수준 다시 계산
z_crit = stats.norm.ppf(1 - alpha/2)                               # 정규 임계값 계산

# ================== 5. 보조 함수 ==================
def collapse_duplicate_x(x, y):
    x = np.asarray(x); y = np.asarray(y)                          # numpy 배열로 변환
    mask = ~np.isnan(x) & ~np.isnan(y)                             # NaN이 아닌 관측만 선택
    x = x[mask]; y = y[mask]                                        # 결측치 제거
    if len(x)==0: return x,y                                        # 비어있으면 바로 반환
    ux, inv = np.unique(x, return_inverse=True)                     # 고유 x값과 인덱스 맵 생성
    y_sum = np.zeros_like(ux, dtype=float); cnt = np.zeros_like(ux, dtype=int)  # 합계/카운트 초기화
    for i,v in zip(inv,y):                                          # 각 관측에 대해
        y_sum[i]+=v; cnt[i]+=1                                      # 해당 고유 x에 y 합산 및 카운트
    return ux, y_sum/cnt                                            # 고유 x와 평균 y 반환

def build_smoothed_inverse_cdf(values):
    """단조 cubic (PCHIP) 기반 역CDF quantile 함수 생성"""        # docstring: 함수 목적 설명
    v = np.sort(values)                                             # 값 정렬
    n = len(v)                                                      # 샘플 개수
    ranks = np.arange(1, n+1)                                       # 1..n 순위 생성
    # Blom 보정
    u = (ranks - 0.375) / (n + 0.25)                                # Blom plotting position으로 u 계산
    # 양끝 보정
    u_ext = np.concatenate(([0.0], u, [1.0]))                       # 0과 1을 양끝에 추가
    v_ext = np.concatenate(([v[0]], v, [v[-1]]))                    # 값 배열도 경계값 추가
    return PchipInterpolator(u_ext, v_ext, extrapolate=True)        # PCHIP으로 역CDF 보간함수 반환

def gaussian_copula_single(x, y,
                           grid_points=200,
                           grid_mode='quantile',
                           band_type='mean',
                           quantile_smoothing=True):
    """
    단일 (x,y)에서 가우시안 코플라 평균 및 (mean or predictive) 밴드(분산/표준편차는 여기서 X 의존 없음).
    여기서는 base mean만 (후처리 bootstrap bagging은 별도 함수) 반환.
    """                                                              # 함수 docstring
    x = np.asarray(x); y = np.asarray(y)                            # 입력을 numpy 배열로 변환
    if len(x) < min_points_copula:                                  # 최소 표본 수 확인
        return None                                                  # 작으면 None 반환
    # 순위 변환 (Blom)
    rx = stats.rankdata(x, method='average')                        # x의 순위(동률 시 평균)
    ry = stats.rankdata(y, method='average')                        # y의 순위
    u_x = (rx - 0.375) / (len(x) + 0.25)                            # 순위를 u로 변환(Blom)
    u_y = (ry - 0.375) / (len(y) + 0.25)                            # y도 동일

    z_x = stats.norm.ppf(np.clip(u_x,1e-6,1-1e-6))                   # u->정규 분위수(z), 클리핑으로 안정화
    z_y = stats.norm.ppf(np.clip(u_y,1e-6,1-1e-6))                   # y의 z값
    rho = np.corrcoef(z_x, z_y)[0,1]                                # z들 간의 상관계수 추정
    rho = np.clip(rho, -0.9999, 0.9999)                             # 수치적 안정성 위해 약간 클리핑

    # grid
    if grid_mode == 'quantile':                                     # 그리드 모드가 분위수 기반이면
        q = np.linspace(0,1,grid_points)                            # 0..1 균등한 확률격자 생성
        x_sorted = np.sort(x)                                       # x 정렬
        grid_x = np.quantile(x_sorted, q)                           # 경험적 분위수를 그리드로 사용
        u_grid = q                                                  # 그리드의 u는 q와 동일
    else:
        grid_x = np.linspace(x.min(), x.max(), grid_points)         # x값 구간 균등 그리드 생성
        # ECDF 근사
        x_sorted = np.sort(x)                                       # x 정렬
        ranks = np.searchsorted(x_sorted, grid_x, side='right')     # 각 grid_x의 정렬 내 위치 추정
        ranks = np.clip(ranks,1,len(x))                             # 1..n 으로 제한
        u_grid = (ranks - 0.375)/(len(x)+0.25)                      # 순위를 u로 변환(Blom)

    z_grid = stats.norm.ppf(np.clip(u_grid,1e-6,1-1e-6))             # u_grid -> z_grid 변환
    z_mean = rho * z_grid                                            # 조건부 평균 E[Z_y|Z_x]=rho*z_x
    var_cond = 1 - rho**2                                            # 조건부 분산
    sd_cond = np.sqrt(var_cond)                                      # 조건부 표준편차

    if band_type == 'predictive':                                    # predictive band 선택 시
        z_low = z_mean - z_crit*sd_cond                              # z-space 예측 하한
        z_high= z_mean + z_crit*sd_cond                              # z-space 예측 상한
    elif band_type == 'mean':                                        # mean band 선택 시
        # mean band
        sd_mean = sd_cond / np.sqrt(len(x))                          # 평균의 표준오차로 축소
        z_low = z_mean - z_crit*sd_mean                              # z-space mean CI 하한
        z_high= z_mean + z_crit*sd_mean                              # z-space mean CI 상한
    else:
        raise ValueError("band_type must be 'predictive' or 'mean'")  # 잘못된 band_type 예외

    if quantile_smoothing:                                            # quantile_smoothing 사용 시
        qfun = build_smoothed_inverse_cdf(y)                          # smoothed inverse CDF 생성
        def z_to_y(zv):
            uu = stats.norm.cdf(zv)                                   # z -> u
            return qfun(uu)                                           # u -> y via smoothed quantile
    else:
        y_sorted = np.sort(y)                                         # y 정렬
        n = len(y)                                                    # y 길이
        def z_to_y(zv):
            uu = stats.norm.cdf(zv)                                   # z -> u
            pos = uu*(n+1)                                            # 경험적 위치(pos) 계산
            pos = np.clip(pos,1,n)                                    # pos를 1..n으로 제한
            lo = np.floor(pos).astype(int)                            # 아래 인덱스
            hi = np.ceil(pos).astype(int)                             # 위 인덱스
            w = pos - lo                                              # 보간 가중치
            lo = np.clip(lo,1,n); hi = np.clip(hi,1,n)                # 인덱스 안전화
            return (1-w)*y_sorted[lo-1] + w*y_sorted[hi-1]            # 선형 보간으로 y 반환

    mean_y = z_to_y(z_mean)                                           # z_mean -> y 공간의 평균
    low_y  = z_to_y(z_low)                                            # z_low -> y 공간의 하한
    high_y = z_to_y(z_high)                                           # z_high -> y 공간의 상한
    return dict(grid_x=grid_x, mean=mean_y, lower=low_y, upper=high_y,
                rho=rho, n=len(x))                                    # 결과 딕셔너리 반환

def bootstrap_copula_bagging(x, y,
                             B=500,
                             grid_points=200,
                             grid_mode='quantile',
                             band_type='mean',
                             quantile_smoothing=True,
                             seed=2024):
    """
    부트스트랩: 각 bootstrap에서 가우시안 코플라 mean/band 계산
    - bagged_mean: 모든 부트스트랩 mean들의 평균
    - CI: 각 grid에서 (2.5%, 97.5%) (mean band)
    base(원자료) 결과도 함께 반환 (비교용)
    """                                                                # 함수 docstring
    rng = np.random.default_rng(seed)                                   # 난수 생성기 초기화(재현성)
    base = gaussian_copula_single(x, y,
                                  grid_points=grid_points,
                                  grid_mode=grid_mode,
                                  band_type=band_type,
                                  quantile_smoothing=quantile_smoothing)  # 원자료 기준 추정 수행
    if base is None:
        return None                                                      # 데이터 부족 시 None 반환
    n = len(x)                                                          # 표본 크기
    means = []                                                           # 부트스트랩별 mean 저장 리스트
    lowers = []                                                          # 부트스트랩별 lower 저장 리스트
    uppers = []                                                          # 부트스트랩별 upper 저장 리스트
    for b in range(B):                                                   # B번 반복
        idx = rng.integers(0, n, n)                                      # 복원추출 인덱스 생성
        xb = x[idx]; yb = y[idx]                                         # 부트스트랩 샘플 생성
        xb2, yb2 = collapse_duplicate_x(xb, yb)                         # 중복 x 합치기
        res_b = gaussian_copula_single(xb2, yb2,
                                       grid_points=grid_points,
                                       grid_mode=grid_mode,
                                       band_type=band_type,
                                       quantile_smoothing=quantile_smoothing)  # 부트스트랩 샘플로 코플라 추정
        if res_b is None:
            continue                                                     # 실패하면 건너뜀
        means.append(res_b['mean'])                                      # mean 저장
        lowers.append(res_b['lower'])                                    # lower 저장
        uppers.append(res_b['upper'])                                    # upper 저장
    if len(means) < max(30, B//5):                                       # 유효 반복 수 검사(휴리스틱)
        # 부트스트랩 실패가 많아 신뢰도 낮음
        return dict(**base,
                    bagged_mean=base['mean'],
                    bagged_lower=base['lower'],
                    bagged_upper=base['upper'],
                    B_eff=len(means),
                    warning="부트스트랩 유효 반복 수 적음")              # 기본값 반환 및 경고

    means_arr = np.vstack(means)                                          # (B_eff x grid) 행렬 생성
    lowers_arr= np.vstack(lowers)                                         # lowers 행렬 생성
    uppers_arr= np.vstack(uppers)                                         # uppers 행렬 생성

    bagged_mean = means_arr.mean(axis=0)                                  # 그리드별 부트스트랩 평균 계산
    # mean band CI (mean 자체의 변동) -> means_arr 분위수
    q_low = np.percentile(means_arr, 2.5, axis=0)                         # 2.5% 분위수
    q_high= np.percentile(means_arr,97.5, axis=0)                         # 97.5% 분위수

    # (선택) 추가 약한 후처리 스플라인
    if optional_spline_smoothing:
        # heuristic s: var * len * smooth_factor
        v = np.var(bagged_mean)                                           # bagged_mean 분산 계산
        s_param = v * len(bagged_mean) * smooth_factor                    # 스플라인 s 파라미터 결정
        try:
            sp = UnivariateSpline(base['grid_x'], bagged_mean, s=s_param, k=3)  # cubic spline 적합
            bagged_mean_smooth = sp(base['grid_x'])                       # 스무딩 결과 계산
            # 구간도 약간 스무딩 (너무 과하진 않게 동일 sp 의 residual scale 사용)
            sp_l = UnivariateSpline(base['grid_x'], q_low, s=s_param*0.8, k=3)  # lower 스플라인
            sp_u = UnivariateSpline(base['grid_x'], q_high, s=s_param*0.8, k=3) # upper 스플라인
            q_low = sp_l(base['grid_x'])                                  # 스무딩된 lower로 교체
            q_high= sp_u(base['grid_x'])                                  # 스무딩된 upper로 교체
            bagged_mean = bagged_mean_smooth                              # 스무딩된 평균으로 교체
        except Exception:
            pass                                                           # 실패 시 무시

    return dict(grid_x=base['grid_x'],
                base_mean=base['mean'],
                base_lower=base['lower'],
                base_upper=base['upper'],
                bagged_mean=bagged_mean,
                bagged_lower=q_low,
                bagged_upper=q_high,
                rho=base['rho'],
                n=base['n'],
                B_eff=len(means))                                         # 최종 결과 반환

# ================== 6. 패널 함수 ==================
def add_panel(ax, df, xcol, ycol):
    # 산점도 (전체 점)
    ax.scatter(df.loc[early_mask, xcol], df.loc[early_mask, ycol],
               color=early_point_color, edgecolors='white',
               linewidth=0.5, alpha=0.85, label='2017-2022 points')    # 2017-2022 점들 산점도
    ax.scatter(df.loc[late_mask, xcol], df.loc[late_mask, ycol],
               color=late_point_color, edgecolors='white',
               linewidth=0.5, alpha=0.85, label='2023-2025 points')    # 2023-2025 점들 산점도

    subsets = [
        ('17-22', early_mask),
        ('23-25', late_mask),
        ('17-25', slice(None)),
    ]                                                                # 분석용 서브셋 목록
    info_lines = []                                                   # 요약 텍스트 저장 리스트

    for label_text, mask in subsets:                                  # 각 서브셋 반복
        x_raw = df.loc[mask, xcol].to_numpy()                         # x 원시값 추출
        y_raw = df.loc[mask, ycol].to_numpy()                         # y 원시값 추출
        x_proc, y_proc = collapse_duplicate_x(x_raw, y_raw)           # 중복 x 처리 (평균 y)
        if len(x_proc) < min_points_copula:                           # 표본 부족 시 건너뜀
            continue

        res = bootstrap_copula_bagging(
            x_proc, y_proc,
            B=B_BOOT,
            grid_points=grid_points,
            grid_mode=grid_mode,
            band_type=band_type,
            quantile_smoothing=quantile_smoothing,
            seed=2024
        )                                                              # 부트스트랩 코플라 실행
        if res is None:
            continue                                                   # 결과 없으면 건너뜀

        # CI 채우기 (bagged mean CI)
        ci_color = ci_gray_levels[label_text]                          # CI 색 선택
        ax.fill_between(res['grid_x'], res['bagged_lower'], res['bagged_upper'],
                        color=ci_color, alpha=ci_gray_alpha[label_text], linewidth=0,
                        label=f"{label_text} {int(CI_LEVEL*100)}% Copula (boot mean CI)")  # CI 영역 채우기

        # Bagged mean 곡선
        ax.plot(res['grid_x'], res['bagged_mean'],
                color=subset_colors[label_text], lw=2.2,
                label=f"{label_text} Bagged mean")                    # 부트스트랩 평균 곡선 표시

        # Base mean (참고용 점선)
        ax.plot(res['grid_x'], res['base_mean'],
                color=subset_colors[label_text], lw=1.2, ls='--', alpha=0.65,
                label=f"{label_text} Base mean(single)")                # 원자료 기준 평균 점선 표시

        # 상관/요약
        if len(x_raw) >= 3:
            try:
                r, p = stats.pearsonr(x_raw, y_raw)                    # Pearson 상관 및 p-value 계산
                info_lines.append(
                    f"{label_text} r={r:.2f} p={p:.1e} ρ={res['rho']:.2f} B={res['B_eff']}"
                )                                                    # 요약 문자열 추가
            except Exception:
                pass                                                   # 통계 계산 실패 시 무시

    ax.set_xlabel(xcol)                                              # x축 라벨 설정
    ax.set_ylabel(ycol)                                              # y축 라벨 설정
    if xcol == 'Statement':
        ax.set_title(f"{xcol} vs {ycol}")                                # 서브플롯 제목 설정
    else :
        ax.set_title(f"add_{xcol} vs {ycol}")
    ax.grid(alpha=0.3)                                               # 그리드 표시
    if info_lines:
        ax.text(0.02, 0.98, "\n".join(info_lines),
                transform=ax.transAxes, va='top', ha='left',
                fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.3',
                          fc='white', ec='gray', alpha=0.75))        # 요약 텍스트 박스 표시

# ================== 7. 플로팅 ==================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))                      # 2x3 서브플롯 생성
axes = axes.flatten()                                                # 축 배열 평탄화
for ax, (xcol, ycol) in zip(axes, pairs):                            # 각 페어에 대해 패널 추가
    add_panel(ax, merged, xcol, ycol)

# 범례 중복 제거
handles_all, labels_all = [], []
for ax in axes:
    h,l = ax.get_legend_handles_labels()                              # 각 축의 범례 항목 수집
    handles_all.extend(h); labels_all.extend(l)
uniq = OrderedDict()
for h,l in zip(handles_all, labels_all):
    if l not in uniq:
        uniq[l] = h                                                   # 라벨 중복 제거하여 OrderedDict에 저장

fig.legend(uniq.values(), uniq.keys(),
           loc='upper center', ncol=4,
           frameon=True, fontsize=8,
           bbox_to_anchor=(0.5, 1.02))                               # 통합 범례 표시

title_band = "Mean" if band_type=='mean' else "Predictive"            # 제목용 밴드 문자열 선택
plt.suptitle(
    f"Gaussian Copula (Bagged {title_band} Curve, {int(CI_LEVEL*100)}% CI, B={B_BOOT}, Method {method})",
    fontsize=14, y=0.995
)                                                                     # 전체 타이틀 설정
plt.tight_layout(rect=[0,0,1,0.965])                                  # 레이아웃 정리
plt.show()                                                            # 플롯 출력

# ================== 8. 기간별 상관 요약 ==================
def corr_block(x, y):
    pear = stats.pearsonr(x, y)                                       # Pearson 상관 계산
    spear = stats.spearmanr(x, y)                                     # Spearman 상관 계산
    kend = stats.kendalltau(x, y)                                     # Kendall tau 계산
    return dict(
        n=len(x),
        pearson_r=pear.statistic, pearson_p=pear.pvalue,
        spearman_rho=spear.statistic, spearman_p=spear.pvalue,
        kendall_tau=kend.statistic, kendall_p=kend.pvalue
    )                                                                  # 통계 요약 반환

summary = {}
for xcol, ycol in pairs:
    summary[f"{xcol}~{ycol}_ALL"] = corr_block(merged[xcol], merged[ycol])                     # 전체 기간 상관
    summary[f"{xcol}~{ycol}_2017_2022"] = corr_block(merged.loc[early_mask, xcol],
                                                     merged.loc[early_mask, ycol])             # 2017-2022 상관
    summary[f"{xcol}~{ycol}_2023_2025"] = corr_block(merged.loc[late_mask, xcol],
                                                     merged.loc[late_mask, ycol])             # 2023-2025 상관

print("\n=== Correlation Summary (ALL / 2017-2022 / 2023-2025) ===")                                # 출력 헤더
pprint.pprint(summary)                                                                               # 요약 출력