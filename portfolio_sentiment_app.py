# Portfolio Sentiment Analysis with Transformer-based NLP
# Streamlit Application for Financial Sentiment Analysis

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 페이지 구성 설정
st.set_page_config(
    page_title="포트폴리오 감정 분석 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일 정의
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
    .neutral {
        color: #6c757d;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ======================== 모델 캐싱 설정 ========================
@st.cache_resource
def load_sentiment_model():
    """
    FinBERT 모델 로드 (금융 텍스트에 최적화된 Transformer 모델)
    FinBERT는 BERT를 금융 데이터로 파인튜닝한 모델로 매우 높은 정확도 제공
    """
    try:
        # FinBERT 모델 시도 (금융 도메인 최적화)
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        sentiment_pipeline = pipeline("sentiment-analysis", 
                                     model=model, 
                                     tokenizer=tokenizer,
                                     device=0 if torch.cuda.is_available() else -1)
        return sentiment_pipeline, "FinBERT (금융 최적화)"
    except Exception as e:
        # 폴백: 표준 BERT 모델
        try:
            sentiment_pipeline = pipeline("sentiment-analysis",
                                         model="distilbert-base-uncased-finetuned-sst-2-english",
                                         device=0 if torch.cuda.is_available() else -1)
            return sentiment_pipeline, "DistilBERT (일반 감정분석)"
        except:
            # 최종 폴백
            sentiment_pipeline = pipeline("sentiment-analysis",
                                         device=0 if torch.cuda.is_available() else -1)
            return sentiment_pipeline, "기본 Transformer 모델"

@st.cache_resource
def load_zero_shot_model():
    """
    Zero-shot classification 모델 로드 (금융 특정 카테고리 분류)
    """
    classifier = pipeline("zero-shot-classification",
                         model="facebook/bart-large-mnli",
                         device=0 if torch.cuda.is_available() else -1)
    return classifier

# ======================== 데이터 처리 함수 ========================

def preprocess_text(text):
    """텍스트 전처리"""
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    # URL 제거
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # 특수 문자 정리
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_length=512):
    """텍스트를 청크로 분할 (모델 토큰 제한 대응)"""
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text[:max_length]]

def analyze_sentiment_batch(texts, sentiment_pipeline, max_batch_size=32):
    """배치 단위 감정 분석"""
    all_sentiments = []
    all_scores = []
    
    for text in texts:
        if not text or len(text.strip()) == 0:
            all_sentiments.append("NEUTRAL")
            all_scores.append(0.0)
            continue
        
        # 텍스트 전처리
        text = preprocess_text(text)
        
        # 긴 텍스트는 청킹
        chunks = chunk_text(text, max_length=512)
        
        chunk_results = []
        for chunk in chunks:
            try:
                result = sentiment_pipeline(chunk, truncation=True, max_length=512)
                chunk_results.append(result)
            except Exception as e:
                st.warning(f"청크 분석 오류: {e}")
                continue
        
        # 청크 결과 집계
        if chunk_results:
            positive_score = sum(1 for r in chunk_results if r[0]['label'] in ['POSITIVE', 'positive'])
            negative_score = sum(1 for r in chunk_results if r[0]['label'] in ['NEGATIVE', 'negative'])
            neutral_score = len(chunk_results) - positive_score - negative_score
            
            scores = [r[0]['score'] for r in chunk_results]
            avg_score = np.mean(scores)
            
            # 최종 감정 결정
            if positive_score > negative_score and positive_score > neutral_score:
                sentiment = "POSITIVE"
                final_score = avg_score
            elif negative_score > positive_score and negative_score > neutral_score:
                sentiment = "NEGATIVE"
                final_score = -avg_score
            else:
                sentiment = "NEUTRAL"
                final_score = 0.0
            
            all_sentiments.append(sentiment)
            all_scores.append(final_score)
        else:
            all_sentiments.append("NEUTRAL")
            all_scores.append(0.0)
    
    return all_sentiments, all_scores

def extract_keywords(text, n_words=10):
    """텍스트에서 주요 키워드 추출"""
    # 전처리
    text = preprocess_text(text.lower())
    
    # 불용어 제거
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'that', 'this', 'as', 'if',
        'it', 'its', 'which', 'who', 'what', 'when', 'where', 'why', 'how',
        'all', 'each', 'every', 'both', 'either', 'neither', 'such', 'same',
        'so', 'than', 'then', 'they', 'them', 'their', 'we', 'us', 'our',
        'you', 'your', 'he', 'him', 'his', 'she', 'her', 'hers', 'i', 'me', 'my', 'mine'
    }
    
    # 단어 추출 및 필터링
    words = re.findall(r'\b[a-z]{3,}\b', text)
    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # 빈도 계산
    word_freq = Counter(filtered_words)
    return word_freq.most_common(n_words)

def calculate_sentiment_metrics(df):
    """감정 분석 메트릭 계산"""
    sentiment_counts = df['Sentiment'].value_counts()
    
    metrics = {
        'total_documents': len(df),
        'positive_count': sentiment_counts.get('POSITIVE', 0),
        'negative_count': sentiment_counts.get('NEGATIVE', 0),
        'neutral_count': sentiment_counts.get('NEUTRAL', 0),
        'positive_ratio': sentiment_counts.get('POSITIVE', 0) / len(df) * 100,
        'negative_ratio': sentiment_counts.get('NEGATIVE', 0) / len(df) * 100,
        'neutral_ratio': sentiment_counts.get('NEUTRAL', 0) / len(df) * 100,
        'avg_sentiment_score': df['Sentiment_Score'].mean(),
        'sentiment_volatility': df['Sentiment_Score'].std(),
    }
    
    return metrics

def calculate_equity_ranking(sentiment_df):
    """종목별 순위 계산 (포트폴리오 선호도 점수)"""
    equity_stats = sentiment_df.groupby('Equity').agg({
        'Sentiment': 'count',
        'Sentiment_Score': ['mean', 'std', 'max', 'min'],
        'Document Title': 'count'
    }).round(4)
    
    equity_stats.columns = ['Total_Docs', 'Avg_Score', 'Score_Std', 'Max_Score', 'Min_Score', 'Reports']
    
    # 종합 점수 계산 (여러 지표 가중합)
    equity_stats['Sentiment_Grade'] = equity_stats['Avg_Score'].apply(
        lambda x: 'A+' if x > 0.8 else ('A' if x > 0.6 else ('B+' if x > 0.4 else 
                  ('B' if x > 0.2 else ('C' if x > -0.2 else ('D' if x > -0.4 else 'F')))))
    )
    
    # 종합 점수: 평균 + 일관성(std 역수) + 샘플 수 정규화
    consistency_score = 1 / (1 + equity_stats['Score_Std'].fillna(0))
    document_weight = equity_stats['Total_Docs'] / equity_stats['Total_Docs'].max()
    
    equity_stats['Portfolio_Score'] = (
        equity_stats['Avg_Score'] * 0.5 + 
        consistency_score * 0.3 + 
        document_weight * 0.2
    )
    
    equity_stats['Investment_Preference'] = equity_stats['Portfolio_Score'].apply(
        lambda x: '강력 추천' if x > 0.6 else ('추천' if x > 0.3 else ('중립' if x > -0.1 else '회피')))
    
    return equity_stats.sort_values('Portfolio_Score', ascending=False)

# ======================== 시각화 함수 ========================

def plot_sentiment_distribution(df):
    """감정 분포 시각화"""
    sentiment_counts = df['Sentiment'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            marker=dict(
                color=['#28a745' if x == 'POSITIVE' else ('#dc3545' if x == 'NEGATIVE' else '#6c757d')
                       for x in sentiment_counts.index]
            ),
            text=sentiment_counts.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="전체 감정 분포",
        xaxis_title="감정 분류",
        yaxis_title="문서 수",
        template="plotly_white",
        height=400
    )
    
    return fig

def plot_equity_sentiment_heatmap(df):
    """종목별 감정 점수 히트맵"""
    pivot_data = df.pivot_table(
        values='Sentiment_Score',
        index='Equity',
        columns='Sentiment',
        aggfunc='count',
        fill_value=0
    )
    
    # 감정별 점수 평균
    pivot_scores = df.pivot_table(
        values='Sentiment_Score',
        index='Equity',
        aggfunc='mean'
    )
    
    fig = px.bar(
        pivot_scores.reset_index(),
        x='Equity',
        y='Sentiment_Score',
        color='Sentiment_Score',
        color_continuous_scale='RdYlGn',
        title="종목별 평균 감정 점수",
        labels={'Sentiment_Score': '감정 점수'},
    )
    
    fig.update_layout(height=400, template="plotly_white")
    return fig

def plot_sentiment_timeline(df):
    """시간대별 감정 추이"""
    df_time = df.copy()
    df_time['Date'] = pd.to_datetime(df_time['Date'], errors='coerce')
    df_time = df_time.dropna(subset=['Date'])
    
    if len(df_time) == 0:
        return None
    
    daily_sentiment = df_time.groupby(df_time['Date'].dt.date).agg({
        'Sentiment_Score': 'mean',
        'Document Title': 'count'
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_sentiment['Date'],
        y=daily_sentiment['Sentiment_Score'],
        mode='lines+markers',
        name='감정 점수',
        line=dict(color='#0066cc', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="시간대별 감정 추이",
        xaxis_title="날짜",
        yaxis_title="감정 점수",
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def plot_wordcloud(text_data, sentiment_filter=None):
    """워드클라우드 생성"""
    if sentiment_filter:
        text_data = text_data[text_data['Sentiment'] == sentiment_filter]
    
    combined_text = ' '.join(text_data['Text'].astype(str).tolist())
    
    if not combined_text.strip():
        return None
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(combined_text)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

def plot_equity_comparison(equity_ranking):
    """종목 비교 차트"""
    top_n = min(10, len(equity_ranking))
    top_equities = equity_ranking.head(top_n)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("포트폴리오 점수", "평균 감정 점수", "점수 일관성 (역수)", "문서 수"),
        specs=[[{}, {}], [{}, {}]]
    )
    
    # Portfolio Score
    fig.add_trace(
        go.Bar(
            x=top_equities.index,
            y=top_equities['Portfolio_Score'],
            name='포트폴리오 점수',
            marker_color='lightblue',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Average Score
    fig.add_trace(
        go.Bar(
            x=top_equities.index,
            y=top_equities['Avg_Score'],
            name='평균 감정',
            marker_color='lightgreen',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Consistency
    fig.add_trace(
        go.Bar(
            x=top_equities.index,
            y=1/(1+top_equities['Score_Std'].fillna(0)),
            name='일관성',
            marker_color='lightyellow',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Document Count
    fig.add_trace(
        go.Bar(
            x=top_equities.index,
            y=top_equities['Total_Docs'],
            name='문서수',
            marker_color='lightcoral',
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, template="plotly_white", showlegend=False)
    return fig

# ======================== Streamlit 메인 앱 ========================

def main():
    st.title("📊 포트폴리오 감정 분석 시스템")
    st.markdown("Transformer 기반 최신 NLP 모델을 활용한 금융 텍스트 감정 분석")
    
    # 사이드바 설정
    st.sidebar.markdown("## ⚙️ 설정")
    
    uploaded_file = st.sidebar.file_uploader(
        "CSV 파일 업로드",
        type=['csv'],
        help="Document Title, Date, Equity, 0-6 열을 포함한 CSV 파일"
    )
    
    analyze_button = st.sidebar.button("📈 감정 분석 실행", key="analyze_main")
    
    if uploaded_file is not None:
        # 파일 로드
        df = pd.read_csv(uploaded_file)
        
        st.sidebar.success("✅ 파일 로드 완료")
        st.sidebar.markdown("---")
        
        # 데이터 미리보기
        with st.sidebar.expander("📋 데이터 정보"):
            st.write(f"총 행 수: {len(df)}")
            st.write(f"컬럼: {', '.join(df.columns.tolist())}")
        
        # 텍스트 컬럼 통합
        text_columns = [col for col in df.columns if col not in ['Document Title', 'Date', 'Equity']]
        df['Text'] = df[text_columns].fillna('').agg(' '.join, axis=1)
        
        if analyze_button or 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
            
            with st.spinner("🔄 모델 로드 중..."):
                sentiment_pipeline, model_name = load_sentiment_model()
            
            st.info(f"✅ 모델: {model_name}")
            
            with st.spinner("⏳ 감정 분석 진행 중... (이 과정은 데이터 크기에 따라 시간이 걸릴 수 있습니다)"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                texts = df['Text'].tolist()
                sentiments, scores = analyze_sentiment_batch(texts, sentiment_pipeline)
                
                df['Sentiment'] = sentiments
                df['Sentiment_Score'] = scores
                
                progress_bar.progress(100)
                status_text.text("✅ 분석 완료!")
            
            st.session_state.analysis_complete = True
            st.session_state.analysis_df = df
        
        if st.session_state.analysis_complete:
            df = st.session_state.analysis_df
            
            st.success("✅ 감정 분석 완료!")
            
            # ==================== 메트릭 대시보드 ====================
            st.markdown("---")
            st.subheader("📊 주요 메트릭")
            
            metrics = calculate_sentiment_metrics(df)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "긍정적 문서",
                    f"{metrics['positive_count']}개",
                    f"{metrics['positive_ratio']:.1f}%",
                    delta_color="normal"
                )
            
            with col2:
                st.metric(
                    "부정적 문서",
                    f"{metrics['negative_count']}개",
                    f"{metrics['negative_ratio']:.1f}%",
                    delta_color="inverse"
                )
            
            with col3:
                st.metric(
                    "중립적 문서",
                    f"{metrics['neutral_count']}개",
                    f"{metrics['neutral_ratio']:.1f}%"
                )
            
            with col4:
                st.metric(
                    "평균 감정 점수",
                    f"{metrics['avg_sentiment_score']:.3f}",
                    f"변동성: {metrics['sentiment_volatility']:.3f}"
                )
            
            # ==================== 시각화 ====================
            st.markdown("---")
            st.subheader("📈 감정 분석 시각화")
            
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "감정 분포", "종목 점수", "시간대 추이", "상위 종목", "워드클라우드", "상세 분석"
            ])
            
            with tab1:
                st.plotly_chart(plot_sentiment_distribution(df), use_container_width=True)
            
            with tab2:
                st.plotly_chart(plot_equity_sentiment_heatmap(df), use_container_width=True)
            
            with tab3:
                timeline_fig = plot_sentiment_timeline(df)
                if timeline_fig:
                    st.plotly_chart(timeline_fig, use_container_width=True)
                else:
                    st.info("날짜 정보가 없어 시간대 추이를 표시할 수 없습니다.")
            
            with tab4:
                equity_ranking = calculate_equity_ranking(df)
                st.plotly_chart(plot_equity_comparison(equity_ranking), use_container_width=True)
            
            with tab5:
                sentiment_filter = st.radio(
                    "감정 선택",
                    options=["전체", "POSITIVE", "NEGATIVE", "NEUTRAL"],
                    horizontal=True
                )
                
                filter_value = None if sentiment_filter == "전체" else sentiment_filter
                wordcloud_fig = plot_wordcloud(df, sentiment_filter=filter_value)
                
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig, use_container_width=True)
                else:
                    st.warning("워드클라우드를 생성할 텍스트가 없습니다.")
            
            with tab6:
                st.markdown("### 📋 상세 분석")
                
                # 종목별 순위
                equity_ranking = calculate_equity_ranking(df)
                
                st.markdown("#### 🏆 종목 순위 및 포트폴리오 평가")
                
                # 예쁜 테이블로 표시
                display_ranking = equity_ranking[['Total_Docs', 'Avg_Score', 'Score_Std', 
                                                  'Portfolio_Score', 'Sentiment_Grade', 
                                                  'Investment_Preference']].copy()
                display_ranking.columns = ['문서수', '평균감정', '점수편차', '포트폴리오점수', '등급', '투자선호도']
                display_ranking = display_ranking.round(4)
                
                st.dataframe(
                    display_ranking,
                    use_container_width=True,
                    height=400
                )
                
                # 종목별 키워드
                st.markdown("#### 🔍 종목별 주요 키워드")
                
                equities = df['Equity'].unique()
                selected_equity = st.selectbox("종목 선택", equities)
                
                equity_data = df[df['Equity'] == selected_equity]
                keywords = extract_keywords(' '.join(equity_data['Text'].astype(str)), n_words=15)
                
                if keywords:
                    keyword_df = pd.DataFrame(keywords, columns=['키워드', '빈도'])
                    
                    fig = px.bar(
                        keyword_df,
                        x='빈도',
                        y='키워드',
                        orientation='h',
                        title=f"{selected_equity} - 주요 키워드",
                        color='빈도',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("추출할 키워드가 없습니다.")
                
                # 감정별 분석
                st.markdown("#### 💭 감정 분류별 상세 통계")
                
                sentiment_detail = df.groupby(['Equity', 'Sentiment']).agg({
                    'Document Title': 'count',
                    'Sentiment_Score': ['mean', 'std']
                }).round(4)
                
                sentiment_detail.columns = ['문서수', '평균점수', '표준편차']
                st.dataframe(sentiment_detail, use_container_width=True)
            
            # ==================== 다운로드 섹션 ====================
            st.markdown("---")
            st.subheader("💾 결과 다운로드")
            
            # 분석 결과 CSV 다운로드
            result_csv = df[['Document Title', 'Date', 'Equity', 'Sentiment', 'Sentiment_Score']].to_csv(index=False)
            st.download_button(
                label="📥 분석 결과 (CSV)",
                data=result_csv,
                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # 종목 순위 다운로드
            ranking_csv = equity_ranking.to_csv()
            st.download_button(
                label="📊 종목 순위 및 점수 (CSV)",
                data=ranking_csv,
                file_name=f"equity_ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👈 왼쪽 사이드바에서 CSV 파일을 업로드하세요.")
        
        st.markdown("---")
        st.subheader("📝 사용 방법")
        st.markdown("""
        1. **파일 업로드**: CSV 파일을 업로드합니다
           - 필수 열: Document Title, Date, Equity
           - 텍스트 열: 0, 1, 2, 3, 4, 5, 6 (자동으로 통합됩니다)
        
        2. **분석 실행**: "감정 분석 실행" 버튼을 클릭합니다
           - 최신 Transformer 모델 (FinBERT) 사용
           - 금융 도메인에 최적화된 감정 분석
        
        3. **결과 확인**: 
           - 📊 감정 분포 및 시각화
           - 🏆 종목별 순위 및 포트폴리오 점수
           - 🔍 주요 키워드 분석
           - 💭 감정별 상세 통계
        
        4. **결과 다운로드**: 분석 결과를 CSV로 저장합니다
        """)
        
        st.markdown("---")
        st.subheader("🤖 사용 모델")
        st.markdown("""
        - **FinBERT**: BERT를 금융 텍스트로 파인튜닝한 최신 모델
        - **Zero-shot Classification**: 사용자 정의 카테고리 분류
        - **Word Cloud**: 감정별 주요 단어 시각화
        
        이 모델들은 NLTK, Vader 등 전통적 방식보다 훨씬 높은 정확도를 제공합니다.
        """)

if __name__ == "__main__":
    main()
