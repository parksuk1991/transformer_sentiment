# Advanced Portfolio Analysis Module
# 포트폴리오 고급 분석 모듈
# 시계열 분석, 감정 변화 추적, 리스크 분석 등

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class AdvancedPortfolioAnalyzer:
    """포트폴리오 고급 분석 클래스"""
    
    def __init__(self, sentiment_df):
        """
        Parameters:
        -----------
        sentiment_df : DataFrame
            Sentiment, Sentiment_Score, Equity, Date 열을 포함한 DataFrame
        """
        self.df = sentiment_df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
    
    # ==================== 시계열 분석 ====================
    
    def sentiment_momentum(self, window=7):
        """
        감정 모멘텀 계산 (이동평균)
        
        Returns:
        --------
        DataFrame with momentum scores
        """
        df_sorted = self.df.sort_values('Date')
        
        momentum_data = []
        for equity in self.df['Equity'].unique():
            equity_data = df_sorted[df_sorted['Equity'] == equity].copy()
            
            if len(equity_data) > 0:
                equity_data['Momentum'] = (
                    equity_data['Sentiment_Score'].rolling(
                        window=window, min_periods=1
                    ).mean()
                )
                
                # 추세: 현재 vs 이전 기간
                if len(equity_data) > window:
                    equity_data['Trend'] = np.where(
                        equity_data['Momentum'].diff() > 0, 'UP', 'DOWN'
                    )
                else:
                    equity_data['Trend'] = 'NEUTRAL'
                
                momentum_data.append(equity_data)
        
        return pd.concat(momentum_data, ignore_index=True)
    
    def sentiment_volatility(self):
        """
        종목별 감정 변동성 계산
        높은 변동성 = 불안정한 시장 평가
        """
        volatility = self.df.groupby('Equity')['Sentiment_Score'].agg([
            ('Volatility', 'std'),
            ('Mean', 'mean'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Range', lambda x: x.max() - x.min())
        ])
        
        # 변동성 지수화 (0-1)
        volatility['Normalized_Volatility'] = (
            volatility['Volatility'] / volatility['Volatility'].max()
        )
        
        return volatility.round(4)
    
    def sentiment_trend(self):
        """
        종목별 감정 추세 분석 (선형 회귀)
        """
        trend_data = []
        
        df_sorted = self.df.sort_values('Date')
        
        for equity in self.df['Equity'].unique():
            equity_data = df_sorted[df_sorted['Equity'] == equity].copy()
            
            if len(equity_data) > 2:
                # x: 시간 순서, y: 감정 점수
                x = np.arange(len(equity_data))
                y = equity_data['Sentiment_Score'].values
                
                # 선형 회귀
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                # 추세 해석
                if slope > 0.05:
                    trend = "상승 추세"
                elif slope < -0.05:
                    trend = "하락 추세"
                else:
                    trend = "무추세"
                
                trend_data.append({
                    'Equity': equity,
                    'Slope': slope,
                    'Intercept': intercept,
                    'R_squared': r_value**2,
                    'P_value': p_value,
                    'Trend': trend,
                    'Significance': "유의" if p_value < 0.05 else "비유의"
                })
        
        return pd.DataFrame(trend_data).sort_values('Slope', ascending=False)
    
    # ==================== 리스크 분석 ====================
    
    def downside_risk(self, threshold=0):
        """
        하락 리스크 분석
        threshold 이하의 부정적 감정의 심도와 빈도
        """
        risk_metrics = []
        
        for equity in self.df['Equity'].unique():
            equity_data = self.df[self.df['Equity'] == equity]
            
            # 부정적 문서 필터링
            negative_docs = equity_data[equity_data['Sentiment_Score'] < threshold]
            
            risk_metrics.append({
                'Equity': equity,
                'Downside_Frequency': len(negative_docs) / len(equity_data),  # 부정적 비율
                'Avg_Downside_Score': negative_docs['Sentiment_Score'].mean() if len(negative_docs) > 0 else 0,
                'Worst_Score': equity_data['Sentiment_Score'].min(),
                'Risk_Level': (len(negative_docs) / len(equity_data)) * abs(negative_docs['Sentiment_Score'].mean() if len(negative_docs) > 0 else 0)
            })
        
        df_risk = pd.DataFrame(risk_metrics)
        
        # 리스크 레벨링
        df_risk['Risk_Rating'] = pd.cut(
            df_risk['Risk_Level'],
            bins=4,
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return df_risk.sort_values('Risk_Level', ascending=False)
    
    def value_at_risk_sentiment(self, confidence=0.95):
        """
        감정 기반 Value-at-Risk (VaR)
        특정 신뢰도에서의 최악 시나리오
        """
        var_data = []
        
        for equity in self.df['Equity'].unique():
            equity_scores = self.df[self.df['Equity'] == equity]['Sentiment_Score']
            
            # percentile 기반 VaR
            var = np.percentile(equity_scores, (1-confidence)*100)
            
            # Conditional VaR (평균 손실)
            cvar = equity_scores[equity_scores <= var].mean()
            
            var_data.append({
                'Equity': equity,
                'VaR_95%': var,
                'CVaR_95%': cvar,
                'Expected_Loss': abs(cvar) if cvar < 0 else 0
            })
        
        return pd.DataFrame(var_data).sort_values('Expected_Loss', ascending=False)
    
    # ==================== 성과 분석 ====================
    
    def sentiment_return_distribution(self):
        """
        감정 점수 분포 분석 (정규분포 검증)
        """
        distribution_data = []
        
        for equity in self.df['Equity'].unique():
            equity_scores = self.df[self.df['Equity'] == equity]['Sentiment_Score']
            
            # 정규성 검증 (Shapiro-Wilk test)
            if len(equity_scores) > 3:
                stat, p_value = stats.shapiro(equity_scores)
            else:
                stat, p_value = np.nan, np.nan
            
            # 왜도와 첨도
            skewness = stats.skew(equity_scores)
            kurtosis = stats.kurtosis(equity_scores)
            
            distribution_data.append({
                'Equity': equity,
                'Mean': equity_scores.mean(),
                'Std': equity_scores.std(),
                'Skewness': skewness,  # 음수: 왼쪽 편향, 양수: 오른쪽 편향
                'Kurtosis': kurtosis,  # 높음: 극단값 많음
                'Normality_p_value': p_value,
                'Is_Normal': "Yes" if p_value > 0.05 else "No" if not np.isnan(p_value) else "N/A"
            })
        
        return pd.DataFrame(distribution_data)
    
    # ==================== 포트폴리오 상관성 분석 ====================
    
    def correlation_matrix(self):
        """
        종목 간 감정 점수 상관관계
        """
        # 종목별 평균 감정 점수
        pivot_data = self.df.pivot_table(
            values='Sentiment_Score',
            index='Date',
            columns='Equity',
            aggfunc='mean'
        )
        
        # 상관계수 계산
        correlation = pivot_data.corr()
        
        return correlation
    
    def portfolio_diversification(self):
        """
        포트폴리오 다양성 분석
        상관계수가 낮을수록 리스크 분산 효과
        """
        corr_matrix = self.correlation_matrix()
        
        # 평균 상관계수 (대각선 제외)
        mask = np.ones(corr_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        
        avg_correlation = corr_matrix.values[mask].mean()
        
        diversification_score = 1 - avg_correlation
        
        return {
            'Average_Correlation': avg_correlation,
            'Diversification_Score': diversification_score,  # 높을수록 좋음
            'Portfolio_Quality': (
                "Well Diversified" if diversification_score > 0.5 else
                "Moderately Diversified" if diversification_score > 0.3 else
                "Poor Diversification"
            )
        }
    
    # ==================== 종합 점수 ====================
    
    def composite_score(self):
        """
        다양한 지표를 종합한 최종 점수
        
        구성:
        - 기대 수익률 (평균 감정): 40%
        - 리스크 관리: 30%
        - 일관성: 20%
        - 모멘텀: 10%
        """
        
        # 1. 기대 수익률
        returns = self.df.groupby('Equity')['Sentiment_Score'].mean()
        returns_normalized = (returns - returns.min()) / (returns.max() - returns.min())
        
        # 2. 리스크
        risk_analysis = self.downside_risk()
        risk_scores = 1 - (risk_analysis.set_index('Equity')['Risk_Level'] / 
                          risk_analysis['Risk_Level'].max())
        
        # 3. 일관성
        consistency = self.df.groupby('Equity')['Sentiment_Score'].std()
        consistency_scores = 1 - (consistency / consistency.max())
        
        # 4. 모멘텀
        momentum_df = self.sentiment_momentum()
        momentum_scores = momentum_df.groupby('Equity')['Momentum'].mean()
        momentum_normalized = (momentum_scores - momentum_scores.min()) / \
                             (momentum_scores.max() - momentum_scores.min() + 1e-8)
        
        # 종합 점수
        all_equities = self.df['Equity'].unique()
        composite = pd.DataFrame({
            'Equity': all_equities,
            'Return_Score': [returns_normalized.get(e, 0) for e in all_equities],
            'Risk_Score': [risk_scores.get(e, 0) for e in all_equities],
            'Consistency_Score': [consistency_scores.get(e, 0) for e in all_equities],
            'Momentum_Score': [momentum_normalized.get(e, 0) for e in all_equities],
        })
        
        # 최종 합성 점수
        composite['Composite_Score'] = (
            composite['Return_Score'] * 0.4 +
            composite['Risk_Score'] * 0.3 +
            composite['Consistency_Score'] * 0.2 +
            composite['Momentum_Score'] * 0.1
        )
        
        # 등급 부여
        composite['Grade'] = pd.cut(
            composite['Composite_Score'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['F', 'D', 'C', 'B', 'A']
        )
        
        return composite.sort_values('Composite_Score', ascending=False)
    
    def generate_report(self):
        """
        최종 분석 보고서 생성
        """
        report = {
            'volatility': self.sentiment_volatility(),
            'trend': self.sentiment_trend(),
            'downside_risk': self.downside_risk(),
            'var_analysis': self.value_at_risk_sentiment(),
            'distribution': self.sentiment_return_distribution(),
            'diversification': self.portfolio_diversification(),
            'composite': self.composite_score()
        }
        
        return report


# ==================== 사용 예제 ====================

if __name__ == "__main__":
    """
    사용 방법:
    
    1. Streamlit 앱에서 분석 완료 후 CSV 다운로드
    2. 이 스크립트 실행:
    
    python advanced_analysis.py sentiment_analysis_20250106.csv
    """
    
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        df = pd.read_csv(csv_file)
        
        # 고급 분석 실행
        analyzer = AdvancedPortfolioAnalyzer(df)
        
        print("="*80)
        print("포트폴리오 감정 분석 - 고급 분석 보고서")
        print("="*80)
        
        print("\n1. 감정 변동성 (Volatility)")
        print("-" * 80)
        print(analyzer.sentiment_volatility())
        
        print("\n2. 감정 추세 (Trend Analysis)")
        print("-" * 80)
        print(analyzer.sentiment_trend())
        
        print("\n3. 하락 리스크 분석 (Downside Risk)")
        print("-" * 80)
        print(analyzer.downside_risk())
        
        print("\n4. Value-at-Risk 분석")
        print("-" * 80)
        print(analyzer.value_at_risk_sentiment())
        
        print("\n5. 분포 분석 (Distribution)")
        print("-" * 80)
        print(analyzer.sentiment_return_distribution())
        
        print("\n6. 포트폴리오 다양성")
        print("-" * 80)
        diversification = analyzer.portfolio_diversification()
        for key, value in diversification.items():
            print(f"{key}: {value}")
        
        print("\n7. 종합 점수 (Composite Score)")
        print("-" * 80)
        print(analyzer.composite_score())
        
    else:
        print("사용법: python advanced_analysis.py <csv_file>")
        print("예: python advanced_analysis.py sentiment_analysis_20250106.csv")
