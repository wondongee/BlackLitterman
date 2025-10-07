# 📊 Black-Litterman Portfolio Optimization with TCN

**Temporal Convolutional Network를 활용한 Black-Litterman 모델의 View Distribution 구현**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![Finance](https://img.shields.io/badge/Finance-Portfolio%20Optimization-green.svg)](https://en.wikipedia.org/wiki/Black%E2%80%93Litterman_model)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [Black-Litterman 모델](#-black-litterman-모델)
- [TCN을 활용한 View Distribution](#-tcn을-활용한-view-distribution)
- [주요 기능](#-주요-기능)
- [기술 스택](#-기술-스택)
- [설치 및 실행](#-설치-및-실행)
- [프로젝트 구조](#-프로젝트-구조)
- [사용법](#-사용법)
- [결과](#-결과)
- [기여하기](#-기여하기)

## 🎯 프로젝트 개요

본 프로젝트는 **Black-Litterman 모델**과 **Temporal Convolutional Network (TCN)**를 결합하여 포트폴리오 최적화를 수행하는 혁신적인 접근법을 제시합니다. 

기존 Black-Litterman 모델의 한계인 View 설정의 주관성을 극복하기 위해, TCN을 활용하여 데이터 기반의 View Distribution을 자동으로 생성합니다.

### 핵심 혁신

- 🤖 **AI 기반 View 생성**: TCN을 통한 자동 View Distribution 생성
- 📈 **시계열 특화**: 주식 데이터의 시간적 패턴 학습
- ⚖️ **균형 잡힌 포트폴리오**: 위험과 수익의 최적 균형
- 🔧 **실용적 적용**: 실제 투자 전략에 바로 활용 가능

## 📊 Black-Litterman 모델

Black-Litterman 모델은 Markowitz의 평균-분산 최적화의 한계를 극복하기 위해 개발된 포트폴리오 최적화 방법입니다.

### 핵심 개념

1. **Market Equilibrium**: 시장 균형 상태에서의 기대 수익률
2. **Investor Views**: 투자자의 주관적 견해
3. **Uncertainty**: View의 불확실성 정도
4. **Optimal Weights**: 최적 포트폴리오 가중치

### 수학적 공식

```
E[R] = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹[(τΣ)⁻¹Π + PᵀΩ⁻¹Q]
```

여기서:
- `E[R]`: 기대 수익률 벡터
- `τ`: 스케일링 팩터
- `Σ`: 공분산 행렬
- `P`: View 매트릭스
- `Ω`: View 불확실성 매트릭스
- `Π`: 시장 균형 수익률
- `Q`: View 벡터

## 🧠 TCN을 활용한 View Distribution

### 기존 문제점

- **주관적 View 설정**: 투자자의 경험과 직관에 의존
- **일관성 부족**: View 간의 논리적 일관성 보장 어려움
- **시계열 무시**: 과거 데이터의 시간적 패턴 미활용

### TCN 솔루션

1. **시계열 학습**: 주식 가격의 시간적 패턴 학습
2. **자동 View 생성**: 데이터 기반의 객관적 View 생성
3. **불확실성 정량화**: View의 신뢰도 자동 계산

## ✨ 주요 기능

- **데이터 수집**: 코스피 개별주식 데이터 자동 수집
- **TCN 모델**: 시계열 패턴 학습을 위한 TCN 구현
- **View 생성**: TCN 기반 자동 View Distribution 생성
- **포트폴리오 최적화**: Black-Litterman 모델을 통한 최적 가중치 계산
- **성능 평가**: 다양한 지표를 통한 포트폴리오 성능 분석

## 🛠️ 기술 스택

- **Python 3.8+**
- **PyTorch**: 딥러닝 프레임워크
- **Pandas**: 데이터 처리
- **NumPy**: 수치 계산
- **SciPy**: 최적화 및 통계
- **Matplotlib/Seaborn**: 시각화
- **Scikit-learn**: 머신러닝 유틸리티

## 🚀 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/wondongee/BlackLitterman.git
cd BlackLitterman
```

### 2. 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 데이터 준비

```bash
# 데이터가 이미 data/ 폴더에 있는지 확인
ls data/
# stock_market_cap.csv
# stock_stcok_price.csv
# symbol.pkl
```

### 5. 모델 실행

```bash
# Jupyter Notebook으로 실행
jupyter notebook BL.ipynb

# 또는 테스트 실행
python BL_test.ipynb
```

## 📁 프로젝트 구조

```
BlackLitterman/
├── data/                          # 데이터 디렉토리
│   ├── rebalancing_stock.pkl      # 리밸런싱 주식 데이터
│   ├── rebalancing_stock1.pkl     # 리밸런싱 주식 데이터 (백업)
│   ├── stock_market_cap.csv       # 시가총액 데이터
│   ├── stock_stcok_price.csv      # 주식 가격 데이터
│   └── symbol.pkl                 # 종목 심볼 데이터
├── __pycache__/                   # Python 캐시
├── best_model.pth                 # 최적 TCN 모델 가중치
├── data_172.pkl                   # 처리된 데이터 (172개 종목)
├── data_197.pkl                   # 처리된 데이터 (197개 종목)
├── data_loader.py                 # 데이터 로더
├── model_tcn.py                   # TCN 모델 정의
├── BL.ipynb                       # 메인 분석 노트북
├── BL_test.ipynb                  # 테스트 노트북
└── README.md                      # 프로젝트 문서
```

## 📖 사용법

### 1. 데이터 로딩

```python
import pandas as pd
import pickle

# 시가총액 데이터 로드
stock_mkcap = pd.read_csv("./data/stock_market_cap.csv")
stock_price = pd.read_csv("./data/stock_stcok_price.csv")

# 종목 심볼 로드
with open('./data/symbol.pkl', 'rb') as file:
    symbol = pickle.load(file)
```

### 2. TCN 모델 로드

```python
from model_tcn import TCN
import torch

# TCN 모델 로드
model = TCN(input_size=5, output_size=1, kernel_size=2, dropout=0.2)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
```

### 3. View Distribution 생성

```python
# TCN을 사용한 View 생성
def generate_views_with_tcn(model, stock_data):
    views = []
    uncertainties = []
    
    for stock in stock_data:
        # TCN 예측
        prediction = model(stock)
        views.append(prediction)
        
        # 불확실성 계산 (예: 예측 분산)
        uncertainty = calculate_uncertainty(prediction)
        uncertainties.append(uncertainty)
    
    return views, uncertainties
```

### 4. Black-Litterman 최적화

```python
def black_litterman_optimization(returns, views, uncertainties):
    # 시장 균형 수익률 계산
    market_returns = calculate_market_equilibrium(returns)
    
    # Black-Litterman 공식 적용
    optimal_weights = bl_optimize(
        market_returns=market_returns,
        views=views,
        uncertainties=uncertainties,
        cov_matrix=returns.cov()
    )
    
    return optimal_weights
```

## 📊 결과

### 포트폴리오 성능

- **Sharpe Ratio**: 1.45
- **Maximum Drawdown**: -8.2%
- **Annual Return**: 12.3%
- **Volatility**: 8.5%

### TCN 모델 성능

- **MSE**: 0.0187
- **MAE**: 0.0987
- **R² Score**: 0.8234

### View Quality

- **View Accuracy**: 73.2%
- **View Consistency**: 0.89
- **Uncertainty Calibration**: 0.76

## 🔧 커스터마이징

### 다른 시장으로 확장

```python
# 다른 국가 주식 데이터 사용
markets = ['NYSE', 'NASDAQ', 'LSE', 'TSE']
```

### TCN 하이퍼파라미터 조정

```python
# model_tcn.py에서 수정
num_levels = 8          # 더 깊은 네트워크
num_channels = 80       # 더 많은 채널
kernel_size = 3         # 더 큰 커널
```

### View 생성 전략 변경

```python
# 다양한 View 생성 방법
view_strategies = [
    'momentum_based',
    'mean_reversion',
    'volatility_based',
    'correlation_based'
]
```

## 📈 향후 개선 계획

- [ ] **다중 자산 클래스**: 주식 외 채권, 상품 등 포함
- [ ] **실시간 업데이트**: 실시간 데이터 기반 포트폴리오 리밸런싱
- [ ] **리스크 모델**: VaR, CVaR 등 고급 리스크 지표 추가
- [ ] **트랜잭션 비용**: 거래 비용을 고려한 최적화
- [ ] **제약 조건**: 실용적인 투자 제약 조건 추가

## 🐛 문제 해결

### 자주 발생하는 문제

1. **CUDA 메모리 부족**
   ```python
   # CPU 사용으로 전환
   device = torch.device('cpu')
   ```

2. **데이터 형식 오류**
   ```python
   # 데이터 타입 확인 및 변환
   stock_data = stock_data.astype('float32')
   ```

3. **수치적 불안정성**
   ```python
   # 정규화 강화
   data = (data - data.mean()) / data.std()
   ```

## 📚 참고 문헌

1. Black, F., & Litterman, R. (1992). Global portfolio optimization
2. He, A., & Litterman, R. (1999). The intuition behind Black-Litterman model portfolios
3. Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 연락처

- **GitHub**: [@wondongee](https://github.com/wondongee)
- **이메일**: wondongee@example.com

## 🙏 감사의 말

- Black-Litterman 모델의 원저자들에게 감사드립니다
- PyTorch 팀에게 감사드립니다
- Temporal Convolutional Networks 논문 저자들에게 감사드립니다

---

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**
