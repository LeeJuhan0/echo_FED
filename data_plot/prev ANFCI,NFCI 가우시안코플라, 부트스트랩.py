import pandas as pd                       # pandas 불러오기 (데이터프레임 조작)
import numpy as np                        # numpy 불러오기 (수치연산)
import matplotlib.pyplot as plt           # 그래프용 matplotlib pyplot 불러오기
from scipy import stats                   # scipy의 통계 함수들 불러오기
from scipy.interpolate import PchipInterpolator, UnivariateSpline  # 보간/스플라인 함수
import io                                 # 문자열을 파일처럼 다루기 위한 io
from textwrap import dedent               # 여러줄 문자열 정리용 dedent
from collections import OrderedDict       # 순서 보장 딕셔너리 (필요시)
import pprint                             # 사람이 읽기 좋은 형태로 출력

# ================== 0. 설정 파라미터 ==================
B_BOOT = 700                              # 부트스트랩 반복 횟수
CI_LEVEL = 0.95                           # 신뢰구간 수준
band_type = 'mean'                        # 밴드 유형: 'mean' 또는 'predictive'
grid_points = 220                         # 그리드 포인트 수
grid_mode = 'quantile'                    # 그리드 생성 방식: 'quantile' 혹은 다른
quantile_smoothing = True                 # 분위수 스무딩 사용 여부
optional_spline_smoothing = True          # 부트스트랩 결과 스플라인 스무딩 여부
smooth_factor = 0.25                      # 스무딩 강도 조절 파라미터
min_points_copula = 5                     # 코플라 추정 최소 관측치 수
alpha = 1 - CI_LEVEL                      # 유의수준
z_crit = stats.norm.ppf(1 - alpha/2)     # 정규 임계값 (양측)

# ================== 1. 원 데이터 (Policy / CPI) ==================
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
"""                                        # 정책 관련 원시 텍스트 블록 (표 형태 문자열)

cpi_text = """Prev_CPI_Value Next_CPI_Value FOMC_date
241.432 243.603 2017-02-01
243.603 243.801 2017-03-15
243.801 244.733 2017-05-03
244.733 244.955 2017-06-14
244.955 244.786 2017-07-26
245.519 246.819 2017-09-20
246.819 246.669 2017-11-01
246.669 246.524 2017-12-13
246.524 247.867 2018-01-31
248.991 249.554 2018-03-21
249.554 251.588 2018-05-02
251.588 251.989 2018-06-13
251.989 252.146 2018-08-01
252.146 252.439 2018-09-26
252.439 252.038 2018-11-08
252.038 251.233 2018-12-19
251.233 251.712 2019-01-30
252.776 254.202 2019-03-20
254.202 256.092 2019-05-01
256.092 256.143 2019-06-19
256.143 256.571 2019-07-31
256.558 256.759 2019-09-18
256.759 257.346 2019-10-30
257.208 256.974 2019-12-11
256.974 257.971 2020-01-29
257.971 258.115 2020-03-03
258.678 258.115 2020-03-15
258.115 256.389 2020-04-29
256.394 257.797 2020-06-10
257.797 259.101 2020-07-29
259.918 260.28 2020-09-16
260.28 260.229 2020-11-05
260.229 260.474 2020-12-16
260.474 261.582 2021-01-27
263.014 264.877 2021-03-17
264.877 267.054 2021-04-28
269.195 271.696 2021-06-16
271.696 273.003 2021-07-28
273.567 274.31 2021-09-22
274.31 277.948 2021-11-03
277.948 278.802 2021-12-15
278.802 281.148 2022-01-26
283.716 287.504 2022-03-16
287.504 292.296 2022-05-04
292.296 296.311 2022-06-15
296.311 296.276 2022-07-27
296.171 296.808 2022-09-21
296.808 297.711 2022-11-02
297.711 296.797 2022-12-14
296.797 300.84 2023-02-01
300.84 301.836 2023-03-22
301.836 304.127 2023-05-03
304.127 305.109 2023-06-14
305.109 305.691 2023-07-26
307.026 307.789 2023-09-20
307.789 307.051 2023-11-01
307.051 306.746 2023-12-13
306.746 308.417 2024-01-31
310.326 312.332 2024-03-20
312.332 314.069 2024-05-01
314.069 314.175 2024-06-12
314.175 314.54 2024-07-31
314.796 315.301 2024-09-18
315.301 315.493 2024-11-07
315.493 315.605 2024-12-18
315.605 317.671 2025-01-29
319.082 319.799 2025-03-19
321.465 322.561 2025-05-18
321.465 322.561 2025-06-18
322.561 323.048 2025-07-30
"""                                        # CPI 관련 원시 텍스트 블록

policy_df_raw = pd.read_csv(io.StringIO(dedent(policy_text)), sep=r"\s+")  # policy 텍스트를 DataFrame으로 읽기
cpi_df = pd.read_csv(io.StringIO(dedent(cpi_text)), sep=r"\s+")            # cpi 텍스트를 DataFrame으로 읽기
cpi_df['FOMC_date'] = pd.to_datetime(cpi_df['FOMC_date'])                   # FOMC_date 컬럼을 datetime으로 변환

# 2. CPI date key (2020-03 두 회의 구분)
cpi_df['month'] = cpi_df['FOMC_date'].dt.strftime('%Y-%m')                   # 연-월 문자열 생성
cpi_df['idx_in_month'] = cpi_df.groupby('month').cumcount()                 # 월 내 인덱스(같은 달 내 순번)
def make_date_key(row):                                                     # date_key 생성 함수 정의 (2020-03 특수 처리)
    if row['month'] == '2020-03':
        return f"{row['month']}{'x' if row['idx_in_month']==0 else 'y'}"     # 2020-03이면 x/y 구분
    return row['month'] if row['idx_in_month']==0 else f"{row['month']}_{row['idx_in_month']}"  # 기본 키 포맷
cpi_df['date_key'] = cpi_df.apply(make_date_key, axis=1)                    # 함수 적용해 date_key 컬럼 생성

# 3. Policy merge (Method B 가정: 2020-03x / 2020-03y 존재)
policy_df = policy_df_raw.copy()                                            # 원시 policy 복사본 생성
has_x = (policy_df['date']=='2020-03x').any()                                # 2020-03x 존재 여부
has_y = (policy_df['date']=='2020-03y').any()                                # 2020-03y 존재 여부
has_plain = (policy_df['date']=='2020-03').any()                             # 2020-03(plain) 존재 여부
if has_plain and (has_x or has_y):                                           # 서로 섞여있으면 에러
    raise ValueError("2020-03 형식 혼재.")
if has_x and has_y:
    method="B"                                                               # 방법 B 사용 표시
    policy_df['date_key']=policy_df['date']                                  # date_key는 그대로 사용
    merged = policy_df.merge(cpi_df[['date_key','FOMC_date']], on='date_key', how='left')  # 병합
elif has_plain:
    method="A"                                                               # 방법 A 사용 표시
    month_map = cpi_df.groupby('month', as_index=False).agg({'FOMC_date':'min'}).rename(columns={'month':'date_key'})  # 월별 최소 FOMC_date 맵
    policy_df['date_key']=policy_df['date']                                  # date_key 설정
    merged = policy_df.merge(month_map, on='date_key', how='left')           # 병합(월 기준)
else:
    raise ValueError("2020-03 표기 인식 실패.")                              # 어떤 형식도 아니면 에러

if merged['FOMC_date'].isna().any():                                         # 매핑 실패 체크
    print("FOMC_date 매핑 실패 행:\n", merged[merged['FOMC_date'].isna()])    # 실패 행 출력
    raise AssertionError("일부 날짜 매핑 실패")                              # 예외 발생

# 4. 새로운 NFCI / ANFCI 시계열 붙이기
prev_NFCI = [
    -0.5051747427,-0.5012278561,-0.5248851848,-0.5725607943,-0.5893044337,-0.5717574109,-0.6076308315,-0.6149953549,
    -0.6143644059,-0.5140657618,-0.5469950403,-0.5635640586,-0.5719724724,-0.6059188001,-0.5439605423,-0.4351837331,
    -0.5139149293,-0.6038582901,-0.6196408433,-0.5752064269,-0.5813783612,-0.5182709171,-0.5529726217,-0.5755378651,
    -0.5673448594,-0.6286392543,0.04145253481,0.1643774046,-0.3184473367,-0.4635388763,-0.5081334801,-0.5116844951,
    -0.5976715841,-0.6255278526,-0.6411606016,-0.6820721653,-0.699224074,-0.6660163319,-0.6578002505,-0.62632724,
    -0.5400240858,-0.5577510432,-0.3873114785,-0.3317972676,-0.2047625971,-0.1941450165,-0.1420047721,-0.1204636447,
    -0.1863351264,-0.306579769,-0.1723375172,-0.1915992989,-0.2173785466,-0.2666628768,-0.3362841511,-0.2851610288,
    -0.3425432627,-0.3993766902,-0.4414308974,-0.4150108365,-0.4066587716,-0.3659056084,-0.4167244652,-0.453596878,
    -0.4825094617,-0.5086765442,-0.4468250678,-0.404841017,-0.4823948827,-0.5320184188
]                                          # 이전 NFCI 시계열 (리스트)
prev_ANFCI = [
    -0.4116244131,-0.4264627969,-0.4657276594,-0.5263562456,-0.567624621,-0.5050250777,-0.5960663231,-0.6316020939,
    -0.6210717312,-0.4957534277,-0.5193089202,-0.5382515628,-0.5717436054,-0.6352729661,-0.5871509654,-0.4555196331,
    -0.5237359383,-0.6013068435,-0.6228027926,-0.5905090711,-0.5996808333,-0.5659224441,-0.5805069649,-0.6257507225,
    -0.5622399729,-0.607551336,0.06903925894,0.2282937624,-0.4144029943,-0.5464601277,-0.6991041804,-0.5437072596,
    -0.5305703901,-0.5495374118,-0.5881440857,-0.5885676005,-0.6158025896,-0.6158368941,-0.6626167635,-0.6235932035,
    -0.5023109474,-0.5359246677,-0.2755368823,-0.2150508919,-0.08634414817,-0.1648524178,-0.1679423134,-0.1167039561,
    -0.1916614742,-0.3004480292,-0.1306616195,-0.2231477601,-0.2425657858,-0.2710770734,-0.2871016529,-0.2753451041,
    -0.3715241523,-0.4281602687,-0.4853750538,-0.4340774975,-0.4023535485,-0.364958591,-0.4566038506,-0.5078742234,
    -0.5182652175,-0.5261303212,-0.4547258862,-0.4334718001,-0.4792118789,-0.5071333834
]                                          # 이전 ANFCI 시계열 (리스트)
date_str = [
    "02/01/2017","03/15/2017","05/03/2017","06/14/2017","07/26/2017","09/20/2017","11/01/2017","12/13/2017",
    "01/31/2018","03/21/2018","05/02/2018","06/13/2018","08/01/2018","09/26/2018","11/08/2018","12/19/2018",
    "01/30/2019","03/20/2019","05/01/2019","06/19/2019","07/31/2019","09/18/2019","10/30/2019","11/11/2019",
    "12/11/2019","01/29/2020","03/15/2020","04/29/2020","06/10/2020","07/29/2020","09/16/2020","11/05/2020",
    "12/16/2020","01/27/2021","03/17/2021","04/28/2021","06/16/2021","07/28/2021","09/22/2021","11/03/2021",
    "12/15/2021","01/26/2022","03/16/2022","05/04/2022","06/15/2022","07/27/2022","09/21/2022","11/02/2022",
    "12/14/2022","02/01/2023","03/22/2023","05/03/2023","06/14/2023","07/26/2023","09/20/2023","11/01/2023",
    "12/13/2023","01/31/2024","03/20/2024","05/01/2024","06/12/2024","07/31/2024","09/18/2024","11/07/2024",
    "12/18/2024","01/29/2025","03/19/2025","05/07/2025","06/18/2025","07/30/2025"
]                                          # NFCI/ANFCI에 대응하는 날짜 리스트

nfcidf = pd.DataFrame({
    'FOMC_date': pd.to_datetime(date_str, format='%m/%d/%Y', errors='coerce'),  # 날짜를 datetime으로 변환
    'Prev_NFCI': prev_NFCI,                                                     # NFCI 값 할당
    'Prev_ANFCI': prev_ANFCI                                                    # ANFCI 값 할당
})

# Join (Left on merged FOMC_date)
merged = merged.merge(nfcidf, on='FOMC_date', how='left')  # policy와 NFCI 데이터 병합 (왼쪽 조인)

# 누락 제거(필요시 주석 처리)
before = len(merged)                                       # 병합 전 행 수 저장
merged = merged.dropna(subset=['Prev_NFCI','Prev_ANFCI'])  # NFCI/ANFCI가 없는 행 제거
after = len(merged)                                        # 제거 후 행 수 저장
if after < before:
    print(f"NFCI/ANFCI 매칭 실패 {before-after}개 행 제거 (총 {after})")  # 제거된 행 수 출력

# 5. Copula 함수들
def collapse_duplicate_x(x, y):
    x = np.asarray(x); y = np.asarray(y)                  # 입력을 numpy 배열로 변환
    m = ~np.isnan(x) & ~np.isnan(y)                       # x,y 둘다 결측이 아닌 마스크
    x = x[m]; y = y[m]                                    # 결측치 제거
    if len(x)==0: return x,y                              # 데이터 없으면 바로 반환
    ux, inv = np.unique(x, return_inverse=True)           # 고유 x값과 인덱스 반환
    y_sum = np.zeros_like(ux, dtype=float); cnt = np.zeros_like(ux, dtype=int)  # 합계와 카운트 초기화
    for i,v in zip(inv,y):                                # 각 관측치에 대해
        y_sum[i]+=v; cnt[i]+=1                            # 대응하는 고유 x에 y 더하고 카운트 증가
    return ux, y_sum/cnt                                  # 고유 x와 평균 y 반환

def build_smoothed_inverse_cdf(values):
    v = np.sort(values)                                   # 값 정렬
    n = len(v)                                            # 샘플 개수
    ranks = np.arange(1, n+1)                             # 1..n 순위 생성
    u = (ranks - 0.375) / (n + 0.25)                      # Blom 방식의 플로팅 포지션으로 u 계산
    u_ext = np.r_[0.0, u, 1.0]                            # 0과 1을 포함해 확장
    v_ext = np.r_[v[0], v, v[-1]]                         # 경계값을 추가해 값 배열 확장
    return PchipInterpolator(u_ext, v_ext, extrapolate=True)  # PCHIP으로 역CDF 보간함수 반환

def gaussian_copula_single(x, y,
                           grid_points=200,
                           grid_mode='quantile',
                           band_type='mean',
                           quantile_smoothing=True):
    x = np.asarray(x); y = np.asarray(y)                  # 입력을 numpy 배열로 변환
    if len(x) < min_points_copula:                        # 최소 관측치 미만이면
        return None                                       # None 반환 (추정 불가)
    rx = stats.rankdata(x, method='average')              # x에 대한 순위(평균 방식)
    ry = stats.rankdata(y, method='average')              # y에 대한 순위
    u_x = (rx - 0.375)/(len(x)+0.25)                      # 순위를 확률(u)로 변환 (Blom)
    u_y = (ry - 0.375)/(len(y)+0.25)                      # y도 동일
    z_x = stats.norm.ppf(np.clip(u_x,1e-6,1-1e-6))        # u->정규 분위수로 변환(극단값 안정화)
    z_y = stats.norm.ppf(np.clip(u_y,1e-6,1-1e-6))        # y의 z값
    rho = np.corrcoef(z_x, z_y)[0,1]                      # z들 간의 상관(코플라 파라미터)
    rho = np.clip(rho,-0.9999,0.9999)                     # 수치적 안정성 위해 클리핑
    if grid_mode=='quantile':                             # 그리드 모드가 분위수 기반이면
        q = np.linspace(0,1,grid_points)                  # 0..1 균등 확률 격자
        x_sorted = np.sort(x)                             # x 정렬
        grid_x = np.quantile(x_sorted, q)                 # 경험적 분위수로 그리드 x 설정
        u_grid = q                                        # 그리드의 u는 q와 동일
    else:
        grid_x = np.linspace(x.min(), x.max(), grid_points)  # 값 범위 균등 그리드
        x_sorted = np.sort(x)                             # x 정렬
        ranks = np.searchsorted(x_sorted, grid_x, side='right')  # 각 grid_x의 순위 추정
        ranks = np.clip(ranks,1,len(x))                   # 순위를 1..n으로 제한
        u_grid = (ranks - 0.375)/(len(x)+0.25)            # 순위를 u로 변환
    z_grid = stats.norm.ppf(np.clip(u_grid,1e-6,1-1e-6))  # 그리드 u -> z 변환
    z_mean = rho*z_grid                                   # 조건부 정규 평균(E[Z_y|Z_x]=rho*z_x)
    var_cond = 1 - rho**2                                 # 조건부 분산
    sd_cond = np.sqrt(var_cond)                           # 조건부 표준편차
    if band_type=='predictive':                           # 예측 밴드(전체 분포)라면
        z_low = z_mean - z_crit*sd_cond                   # z-space 하한 (예측)
        z_high= z_mean + z_crit*sd_cond                   # z-space 상한 (예측)
    else:
        sd_mean = sd_cond/np.sqrt(len(x))                 # 평균의 표준오차로 조정
        z_low = z_mean - z_crit*sd_mean                   # z-space 하한 (mean CI)
        z_high= z_mean + z_crit*sd_mean                   # z-space 상한 (mean CI)
    if quantile_smoothing:                                # 분위수 스무딩을 사용할 경우
        qfun = build_smoothed_inverse_cdf(y)              # y로부터 스무딩된 역CDF 생성
        def z_to_y(zv):
            return qfun(stats.norm.cdf(zv))              # z->u->역CDF로 y값 반환
    else:
        y_sorted = np.sort(y)                             # y 정렬
        n = len(y)                                       # y 길이
        def z_to_y(zv):
            uu = stats.norm.cdf(zv)                      # z->u
            pos = uu*(n+1)                               # 경험적 위치(pos) 계산
            pos = np.clip(pos,1,n)                       # pos를 1..n 범위로 제한
            lo = np.floor(pos).astype(int)               # 아래 인덱스
            hi = np.ceil(pos).astype(int)                # 위 인덱스
            w = pos - lo                                 # 보간 가중치
            return (1-w)*y_sorted[lo-1] + w*y_sorted[hi-1]  # 선형 보간으로 y 반환
    return dict(
        grid_x=grid_x,                                    # x 그리드 반환
        mean=z_to_y(z_mean),                              # 조건부 평균(y 공간)
        lower=z_to_y(z_low),                              # 신뢰구간 하한(y 공간)
        upper=z_to_y(z_high),                             # 신뢰구간 상한(y 공간)
        rho=rho,                                          # 추정 rho 반환
        n=len(x)                                         # 사용된 샘플 수 반환
    )

def bootstrap_copula_bagging(x, y,
                             B=500,
                             grid_points=200,
                             grid_mode='quantile',
                             band_type='mean',
                             quantile_smoothing=True,
                             seed=2024):
    rng = np.random.default_rng(seed)                     # 재현 가능한 난수 생성기 초기화
    base = gaussian_copula_single(x,y,grid_points,grid_mode,band_type,quantile_smoothing)  # 기준 추정 수행
    if base is None:
        return None                                       # 데이터 부족 시 None 반환
    n = len(x)                                           # 표본 크기
    means=[]                                             # 부트스트랩에서 얻은 mean들을 저장할 리스트
    for b in range(B):                                   # B번 반복
        idx = rng.integers(0,n,n)                        # 복원추출 인덱스 생성
        xb, yb = x[idx], y[idx]                          # 부트스트랩 샘플 추출
        xb2,yb2 = collapse_duplicate_x(xb,yb)            # 중복 x 합치기(평균 y)
        res = gaussian_copula_single(xb2,yb2,grid_points,grid_mode,band_type,quantile_smoothing)  # 부트스트랩 코플라 추정
        if res is None: continue                         # 실패하면 건너뜀
        means.append(res['mean'])                        # 추정된 mean을 리스트에 추가
    if len(means) < max(30,B//5):                        # 유효 부트스트랩 샘플 수 검사 (최소 조건)
        return dict(**base,
                    bagged_mean=base['mean'],
                    bagged_lower=base['lower'],
                    bagged_upper=base['upper'],
                    rho=base['rho'], B_eff=len(means),
                    warning="부트스트랩 샘플 부족")     # 부족하면 기본 결과 반환 및 경고
    means_arr = np.vstack(means)                         # (B_eff x grid) 행렬로 쌓기
    bagged_mean = means_arr.mean(axis=0)                 # 그리드별 부트스트랩 평균
    q_low = np.percentile(means_arr,2.5,axis=0)          # 2.5% 분위수(하한)
    q_high= np.percentile(means_arr,97.5,axis=0)         # 97.5% 분위수(상한)
    if optional_spline_smoothing:                        # 스플라인 스무딩 옵션이 켜져있으면
        v = np.var(bagged_mean)                          # 평균의 분산 계산
        s_param = v*len(bagged_mean)*smooth_factor       # 스플라인 s 파라미터 경험적 계산
        try:
            sp = UnivariateSpline(base['grid_x'], bagged_mean, s=s_param, k=3)  # 평균에 대한 스플라인 적합
            bagged_mean = sp(base['grid_x'])            # 스무딩된 평균으로 대체
            sp_l = UnivariateSpline(base['grid_x'], q_low, s=s_param*0.8, k=3)  # 하한 스플라인
            sp_u = UnivariateSpline(base['grid_x'], q_high,s=s_param*0.8, k=3)  # 상한 스플라인
            q_low = sp_l(base['grid_x']); q_high= sp_u(base['grid_x'])         # 스무딩된 하/상한으로 대체
        except Exception:
            pass                                          # 실패하면 스무딩 건너뜀
    return dict(
        grid_x=base['grid_x'],                           # 그리드 x 반환
        bagged_mean=bagged_mean,                         # 부트스트랩 평균 반환
        bagged_lower=q_low,                              # 부트스트랩 하한 반환
        bagged_upper=q_high,                             # 부트스트랩 상한 반환
        rho=base['rho'],                                 # 기준 rho 반환
        n=base['n'],                                     # 샘플 수 반환
        B_eff=len(means)                                 # 유효 부트스트랩 반복 수 반환
    )

# 6. 페어 정의: (X) × (Y: Prev_NFCI, Prev_ANFCI)
pairs = [
    ('Statement','Prev_NFCI'),                          # Statement vs Prev_NFCI
    ('P&B','Prev_NFCI'),                                # P&B vs Prev_NFCI
    ('Policy','Prev_NFCI'),                             # Policy vs Prev_NFCI
    ('Statement','Prev_ANFCI'),                         # Statement vs Prev_ANFCI
    ('P&B','Prev_ANFCI'),                               # P&B vs Prev_ANFCI
    ('Policy','Prev_ANFCI'),                            # Policy vs Prev_ANFCI
]

# 7. 패널 함수 (전체만)
def add_panel(ax, df, xcol, ycol):
    x_raw = df[xcol].to_numpy()                        # x 원시값 배열
    y_raw = df[ycol].to_numpy()                        # y 원시값 배열
    ax.scatter(x_raw, y_raw, color='#1f77b4', edgecolors='white', linewidth=0.5,
               alpha=0.85, label='points')             # 산점도 그리기
    x_proc, y_proc = collapse_duplicate_x(x_raw, y_raw)  # 중복 x 합치기(평균 y)
    res = bootstrap_copula_bagging(
        x_proc, y_proc,
        B=B_BOOT,
        grid_points=grid_points,
        grid_mode=grid_mode,
        band_type=band_type,
        quantile_smoothing=quantile_smoothing,
        seed=2024
    )                                                  # 부트스트랩 코플라 적용
    if res is not None:
        ax.fill_between(res['grid_x'], res['bagged_lower'], res['bagged_upper'],
                        color='#AAAAAA', alpha=0.35, linewidth=0,
                        label=f"{int(CI_LEVEL*100)}% boot mean CI")  # 부트스트랩 CI 영역 표시
        ax.plot(res['grid_x'], res['bagged_mean'],
                color='#2ca02c', lw=2.2, label='Bagged Copula mean')  # 부트스트랩 평균 곡선 표시
        # 정보 텍스트
        if len(x_raw) >= 3:
            r,p = stats.pearsonr(x_raw,y_raw)           # Pearson 상관과 p-value 계산
            ax.text(0.02,0.98,
                    f"ρ_cop={res['rho']:.2f}\nPearson r={r:.2f} p={p:.1e}\nB={res['B_eff']}",
                    transform=ax.transAxes, va='top', ha='left', fontsize=8.5,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.75))  # 텍스트 박스 추가
    ax.set_xlabel(xcol)                                 # x축 라벨
    ax.set_ylabel(ycol)                                 # y축 라벨
    ax.set_title(f"{xcol} vs {ycol}")                   # 서브플롯 타이틀
    ax.grid(alpha=0.3)                                  # 그리드 표시

# 8. 플로팅
fig, axes = plt.subplots(2,3, figsize=(15,8))          # 2x3 서브플롯 생성
axes = axes.flatten()                                  # 축 배열 평탄화
for ax,(xcol,ycol) in zip(axes,pairs):
    add_panel(ax, merged, xcol, ycol)                  # 각 페어에 대해 패널 추가

# 범례 통합
handles, labels = [], []                                # 범례 핸들/라벨 저장용
for ax in axes:
    h,l = ax.get_legend_handles_labels()                # 각 축의 범례 아이템 수집
    for hh,ll in zip(h,l):
        if ll not in labels:
            handles.append(hh); labels.append(ll)      # 중복 라벨 제거하며 수집

fig.legend(handles, labels, loc='upper center', ncol=3, frameon=True, fontsize=9,
           bbox_to_anchor=(0.5,1.02))                   # 전체 범례 표시
title_band = "Mean" if band_type=='mean' else "Predictive"  # 타이틀에 밴드 종류 표시
plt.suptitle(f"Gaussian Copula (Bagged {title_band}, {int(CI_LEVEL*100)}% CI, NFCI/ANFCI, B={B_BOOT}, Method {method})",
             fontsize=14, y=0.995)                      # 전체 제목 설정
plt.tight_layout(rect=[0,0,1,0.965])                    # 레이아웃 정리
plt.show()                                             # 플롯 출력

# 9. 상관 요약 (전체)
def corr_block(x,y):
    pear = stats.pearsonr(x,y)                          # Pearson 상관
    spear = stats.spearmanr(x,y)                        # Spearman 상관
    kend = stats.kendalltau(x,y)                        # Kendall tau
    return dict(n=len(x),
                pearson_r=pear.statistic, pearson_p=pear.pvalue,
                spearman_rho=spear.statistic, spearman_p=spear.pvalue,
                kendall_tau=kend.statistic, kendall_p=kend.pvalue)  # 통계 요약 반환

summary = {}
for xcol,ycol in pairs:
    summary[f"{xcol}~{ycol}"] = corr_block(merged[xcol], merged[ycol])  # 각 페어의 상관 요약 계산

print("\n=== Correlation Summary (전체) ===")
pprint.pprint(summary)                                   # 요약 출력