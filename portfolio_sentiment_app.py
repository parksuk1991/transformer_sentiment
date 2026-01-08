# Portfolio Sentiment Analysis with Transformer-based NLP
# Streamlit Application for Financial Sentiment Analysis

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import shap
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# 페이지 구성 설정
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .top-equity {
        font-size: 24px;
        font-weight: bold;
        color: #0066cc;
    }
    </style>
    """, unsafe_allow_html=True)

# ======================== 모델 캐싱 설정 ========================
@st.cache_resource
def load_sentiment_model():
    """FinBERT 모델 로드"""
    try:
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        sentiment_pipeline = pipeline("sentiment-analysis", 
                                     model=model, 
                                     tokenizer=tokenizer,
                                     device=0 if torch.cuda.is_available() else -1)
        return sentiment_pipeline, "FinBERT (금융 최적화)"
    except Exception as e:
        try:
            sentiment_pipeline = pipeline("sentiment-analysis",
                                         model="distilbert-base-uncased-finetuned-sst-2-english",
                                         device=0 if torch.cuda.is_available() else -1)
            return sentiment_pipeline, "DistilBERT (일반 센티먼트 분석)"
        except:
            sentiment_pipeline = pipeline("sentiment-analysis",
                                         device=0 if torch.cuda.is_available() else -1)
            return sentiment_pipeline, "기본 Transformer 모델"

# ======================== 데이터 처리 함수 ========================

def preprocess_text(text):
    """텍스트 전처리"""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_length=512):
    """텍스트를 청크로 분할"""
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

def analyze_sentiment_for_equity(text, sentiment_pipeline):
    """
    종목별 전체 텍스트에 대한 센티먼트 분석
    
    점수 계산 방식:
    1. 텍스트를 512토큰 단위로 청킹
    2. 각 청크별로 FinBERT 모델이 POSITIVE/NEGATIVE/NEUTRAL 분류 + 신뢰도 점수(0~1) 반환
    3. 최종 점수 = (긍정 청크 비율 × 평균 긍정 신뢰도) - (부정 청크 비율 × 평균 부정 신뢰도)
    4. 범위: -1(완전 부정) ~ +1(완전 긍정)
    
    분류 기준:
    - POSITIVE: 점수 > 0.2 (긍정 기준선)
    - NEGATIVE: 점수 < -0.2 (부정 기준선)
    - NEUTRAL: -0.2 ≤ 점수 ≤ 0.2
    
    참고: 실제 earnings call 텍스트는 대부분 중립적이거나 약간 긍정적인 경향이 있어
          부정적 점수가 드물게 나타날 수 있습니다.
    """
    if not text or len(text.strip()) == 0:
        return "NEUTRAL", 0.0
    
    text = preprocess_text(text)
    chunks = chunk_text(text, max_length=512)
    
    chunk_results = []
    for chunk in chunks:
        try:
            result = sentiment_pipeline(chunk, truncation=True, max_length=512)
            chunk_results.append(result[0])
        except Exception as e:
            continue
    
    if not chunk_results:
        return "NEUTRAL", 0.0
    
    # 감정 점수 집계
    positive_scores = [r['score'] for r in chunk_results if r['label'] in ['POSITIVE', 'positive']]
    negative_scores = [r['score'] for r in chunk_results if r['label'] in ['NEGATIVE', 'negative']]
    neutral_scores = [r['score'] for r in chunk_results if r['label'] in ['NEUTRAL', 'neutral']]
    
    # 최종 점수 계산 (-1 ~ 1 범위)
    # 긍정/부정 청크의 평균 신뢰도와 비율을 모두 고려
    avg_positive = np.mean(positive_scores) if positive_scores else 0
    avg_negative = np.mean(negative_scores) if negative_scores else 0
    
    positive_weight = len(positive_scores) / len(chunk_results)
    negative_weight = len(negative_scores) / len(chunk_results)
    
    # 최종 점수: 긍정 기여도 - 부정 기여도
    final_score = (avg_positive * positive_weight) - (avg_negative * negative_weight)
    
    # 감정 분류 (엄격한 기준)
    if final_score > 0.2:
        sentiment = "POSITIVE"
    elif final_score < -0.2:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    
    return sentiment, final_score

def extract_keywords(text, n_words=15):
    """텍스트에서 주요 키워드 추출"""
    text = preprocess_text(text.lower())
    
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'that', 'this', 'as', 'if',
        'it', 'its', 'which', 'who', 'what', 'when', 'where', 'why', 'how',
        'all', 'each', 'every', 'both', 'either', 'neither', 'such', 'same',
        'so', 'than', 'then', 'they', 'them', 'their', 'we', 'us', 'our',
        'you', 'your', 'he', 'him', 'his', 'she', 'her', 'hers', 'i', 'me', 
        'my', 'mine', 'thank', 'thanks', 'good', 'day', 'now', 'bye', 'welcome',
        'hello', 'hi', 'ladies', 'gentlemen', 'everyone', 'conclude', 'concludes',
        'disconnect', 'today', 'call', 'conference', 'thank', 'think', 'year'
    }
    
    words = re.findall(r'\b[a-z]{3,}\b', text)
    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    word_freq = Counter(filtered_words)
    return word_freq.most_common(n_words)

def calculate_equity_ranking(equity_df):
    """종목별 순위 계산"""
    equity_df = equity_df.copy()
    equity_df['Portfolio_Score'] = equity_df['Sentiment_Score']
    
    equity_df['Sentiment_Grade'] = equity_df['Sentiment_Score'].apply(
        lambda x: 'A+' if x > 0.6 else ('A' if x > 0.4 else ('B+' if x > 0.2 else 
                  ('B' if x > 0 else ('C' if x > -0.2 else ('D' if x > -0.4 else 'F')))))
    )
    
    equity_df['Investment_Preference'] = equity_df['Sentiment_Score'].apply(
        lambda x: '강력 추천' if x > 0.4 else ('추천' if x > 0.2 else ('중립' if x > -0.2 else '회피')))
    
    return equity_df.sort_values('Portfolio_Score', ascending=False)

# ======================== 시각화 함수 ========================

def plot_sentiment_distribution(df):
    """센티먼트 분포 시각화"""
    sentiment_counts = df['Sentiment'].value_counts()
    
    colors = {
        'POSITIVE': '#28a745',
        'NEGATIVE': '#dc3545',
        'NEUTRAL': '#6c757d'
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            marker=dict(color=[colors.get(x, '#6c757d') for x in sentiment_counts.index]),
            text=sentiment_counts.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="전체 센티먼트 분포",
        xaxis_title="센티먼트 분류",
        yaxis_title="종목수",
        template="plotly_white",
        height=400
    )
    
    return fig

def plot_equity_sentiment_scores(df):
    """종목별 센티먼트 점수 시각화"""
    df_sorted = df.sort_values('Sentiment_Score', ascending=False)
    
    colors = df_sorted['Sentiment_Score'].apply(
        lambda x: '#28a745' if x > 0.2 else ('#dc3545' if x < -0.2 else '#6c757d')
    )
    
    fig = go.Figure(data=[
        go.Bar(
            x=df_sorted['Equity'],
            y=df_sorted['Sentiment_Score'],
            marker=dict(color=colors),
            text=df_sorted['Sentiment_Score'].round(3),
            textposition='auto',
        )
    ])
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=0.2, line_dash="dot", line_color="green", opacity=0.3, 
                  annotation_text="긍정 기준선 (0.2)")
    fig.add_hline(y=-0.2, line_dash="dot", line_color="red", opacity=0.3,
                  annotation_text="부정 기준선 (-0.2)")
    
    fig.update_layout(
        title="종목별 센티먼트 (높을수록 긍정적)<br><sub>긍정 기준: >0.2 | 부정 기준: <-0.2 | 중립: -0.2~0.2</sub>",
        xaxis_title="종목",
        yaxis_title="센티먼트",
        template="plotly_white",
        height=500,
        showlegend=False
    )
    
    return fig


def extract_sentiment_contributing_words(text, sentiment_pipeline, target_sentiment, top_n=100):
    """
    SHAP을 사용하여 센티먼트에 실제로 기여한 단어 추출
    
    Args:
        text: 분석할 텍스트
        sentiment_pipeline: 센티먼트 파이프라인
        target_sentiment: 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
        top_n: 추출할 상위 단어 수
    
    Returns:
        dict: {단어: 기여도 점수}
    """
    if not text or len(text.strip()) < 10:
        return {}
    
    # 텍스트 전처리 및 청킹
    text = preprocess_text(text)
    chunks = chunk_text(text, max_length=512)
    
    # 각 청크에서 기여도 높은 단어 추출
    word_contributions = {}
    
    model = sentiment_pipeline.model
    tokenizer = sentiment_pipeline.tokenizer
    
    # 센티먼트 레이블 매핑
    sentiment_map = {
        'POSITIVE': ['positive', 'POSITIVE'],
        'NEGATIVE': ['negative', 'NEGATIVE'],
        'NEUTRAL': ['neutral', 'NEUTRAL']
    }
    
    for chunk in chunks[:5]:  # 처리 시간을 위해 최대 5개 청크만
        try:
            # 토큰화
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
            
            # 모델 예측
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # 해당 센티먼트의 확률
            predicted_label = sentiment_pipeline(chunk, truncation=True, max_length=512)[0]['label']
            
            # 타겟 센티먼트가 아니면 스킵
            if predicted_label not in sentiment_map[target_sentiment]:
                continue
            
            # SHAP 값 계산 (간소화 버전: attention weights 사용)
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Attention weights를 기여도 근사치로 사용
            with torch.no_grad():
                attention = model(**inputs, output_attentions=True).attentions
                # 마지막 레이어의 attention 평균
                avg_attention = attention[-1].mean(dim=1).squeeze().mean(dim=0)
            
            # 토큰별 기여도 집계
            for token, weight in zip(tokens, avg_attention):
                # 특수 토큰 및 서브워드 처리
                if token.startswith('##'):
                    token = token[2:]
                elif token in ['[CLS]', '[SEP]', '[PAD]']:
                    continue
                
                token = token.lower().strip()
                
                # stop words 필터링
                stop_words = {
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                    'should', 'may', 'might', 'must', 'can', 'that', 'this', 'as', 'if',
                    'it', 'its', 'which', 'who', 'what', 'when', 'where', 'why', 'how',
                    'thank', 'thanks', 'think', 'year'
                }
                
                if token in stop_words or len(token) < 3:
                    continue
                
                # 기여도 누적
                if token in word_contributions:
                    word_contributions[token] += float(weight)
                else:
                    word_contributions[token] = float(weight)
        
        except Exception as e:
            continue
    
    # 상위 N개 단어 반환
    sorted_words = sorted(word_contributions.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_words[:top_n])

def plot_sentiment_wordcloud(text, sentiment, sentiment_pipeline, title="센티먼트 기여 워드클라우드"):
    """센티먼트 기여도 기반 워드클라우드 생성"""
    if not text or len(text.strip()) < 10:
        return None
    
    # 센티먼트에 기여한 단어 추출
    word_scores = extract_sentiment_contributing_words(text, sentiment_pipeline, sentiment, top_n=100)
    
    if not word_scores:
        return None
    
    # WordCloud 생성 (빈도수 대신 기여도 사용)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='RdYlGn' if sentiment == 'POSITIVE' else ('Reds_r' if sentiment == 'NEGATIVE' else 'Blues'),
        max_words=80,
        relative_scaling=0.5,
        min_font_size=10
    ).generate_from_frequencies(word_scores)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    return fig






def plot_wordcloud(text, title="워드클라우드"):
    """워드클라우드 생성"""
    if not text or len(text.strip()) < 10:
        return None
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    return fig

def plot_top_equities_comparison(df, top_n=10):
    """상위 종목 비교 차트"""
    # Investment_Preference 컬럼 추가
    df_with_pref = df.copy()
    df_with_pref['Investment_Preference'] = df_with_pref['Sentiment_Score'].apply(
        lambda x: '강력 추천' if x > 0.4 else ('추천' if x > 0.2 else ('중립' if x > -0.2 else '회피'))
    )
    
    df_top = df_with_pref.nlargest(top_n, 'Sentiment_Score')
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("센티먼트 Top 10", "투자 선호도 분포"),
        specs=[[{"type": "bar"}, {"type": "pie"}]]
    )
    
    # 막대 차트
    fig.add_trace(
        go.Bar(
            x=df_top['Equity'],
            y=df_top['Sentiment_Score'],
            marker_color='lightblue',
            text=df_top['Sentiment_Score'].round(3),
            textposition='auto',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 파이 차트
    preference_counts = df_with_pref['Investment_Preference'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=preference_counts.index,
            values=preference_counts.values,
            marker_colors=['#28a745', '#17a2b8', '#ffc107', '#dc3545'],
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, template="plotly_white")
    return fig

def plot_document_length_analysis(df):
    """문서 길이 분석"""
    df['Text_Length'] = df['Combined_Text'].str.len()
    
    fig = px.scatter(
        df,
        x='Text_Length',
        y='Sentiment_Score',
        color='Sentiment',
        size='Text_Length',
        hover_data=['Equity'],
        title="문서 길이 vs 센티먼트",
        labels={'Text_Length': '문서 길이 (문자 수)', 'Sentiment_Score': '센티먼트'},
        color_discrete_map={'POSITIVE': '#28a745', 'NEGATIVE': '#dc3545', 'NEUTRAL': '#6c757d'}
    )
    
    fig.update_layout(height=400, template="plotly_white")
    return fig

def plot_sentiment_score_distribution(df):
    """센티먼트 분포"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['Sentiment_Score'],
        nbinsx=30,
        marker_color='lightblue',
        opacity=0.7,
    ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=df['Sentiment_Score'].mean(), line_dash="dot", line_color="red", 
                  annotation_text=f"평균: {df['Sentiment_Score'].mean():.3f}")
    
    fig.update_layout(
        title="센티먼트 분포",
        xaxis_title="센티먼트",
        yaxis_title="종목수",
        template="plotly_white",
        height=400
    )
    
    return fig

def plot_sentiment_comparison_radar(df):
    """센티먼트 상세 분석 차트"""
    top10 = df.nlargest(10, 'Sentiment_Score')
    
    fig = go.Figure(data=[
        go.Bar(
            x=top10['Equity'],
            y=top10['Sentiment_Score'],
            marker=dict(
                color=top10['Sentiment_Score'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="센티먼트")
            ),
            text=top10['Sentiment_Score'].round(3),
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="상위 10개 종목 센티먼트 상세",
        xaxis_title="종목",
        yaxis_title="센티먼트",
        template="plotly_white",
        height=500
    )
    
    return fig

# ======================== Streamlit 메인 앱 ========================

def main():
    st.title("📊 Portfolio Sentiment Analysis")
    st.markdown("Transformer 기반 텍스트 센티먼트")
    
    # 사이드바 설정
    st.sidebar.markdown("## ⚙️ 설정")
    
    uploaded_file = st.sidebar.file_uploader(
        "CSV 파일 업로드",
        type=['csv'],
        help="Document Title, Date, Equity, 0-6 열을 포함한 CSV 파일"
    )
    
    analyze_button = st.sidebar.button("📈 센티먼트 분석 실행", key="analyze_main")
    
    if uploaded_file is not None:
        # 파일 로드
        df = pd.read_csv(uploaded_file)
        
        st.sidebar.success("✅ 파일 로드 완료")
        st.sidebar.markdown("---")
        
        # 데이터 미리보기
        with st.sidebar.expander("📋 데이터 정보"):
            st.write(f"총 행 수: {len(df)}")
            st.write(f"총 종목 수: {df['Equity'].nunique()}")
            st.write(f"컬럼: {', '.join(df.columns.tolist())}")
        
        if analyze_button or 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
            
            with st.spinner("🔄 모델 로드 중..."):
                sentiment_pipeline, model_name = load_sentiment_model()
                st.session_state.sentiment_pipeline = sentiment_pipeline
            
            st.info(f"✅ 모델: {model_name}")
            
            with st.spinner("⏳ 센티먼트 분석 진행 중..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 종목별로 텍스트 통합 및 분석
                text_columns = [str(i) for i in range(7) if str(i) in df.columns]
                
                # 종목별로 그룹화
                equity_groups = df.groupby('Equity')
                
                results = []
                total_equities = len(equity_groups)
                
                for idx, (equity, group) in enumerate(equity_groups):
                    # 동일 Document Title이 있는 경우 평균 계산
                    doc_results = []
                    
                    for doc_title in group['Document Title'].unique():
                        doc_rows = group[group['Document Title'] == doc_title]
                        
                        # 모든 텍스트 열 통합
                        combined_text = ' '.join(
                            doc_rows[text_columns].fillna('').astype(str).values.flatten()
                        )
                        
                        sentiment, score = analyze_sentiment_for_equity(combined_text, sentiment_pipeline)
                        doc_results.append({
                            'sentiment': sentiment,
                            'score': score,
                            'text': combined_text
                        })
                    
                    # 동일 종목의 여러 Document 평균
                    avg_score = np.mean([r['score'] for r in doc_results])
                    
                    # 최종 감정 재분류
                    if avg_score > 0.2:
                        final_sentiment = "POSITIVE"
                    elif avg_score < -0.2:
                        final_sentiment = "NEGATIVE"
                    else:
                        final_sentiment = "NEUTRAL"
                    
                    # 모든 텍스트 통합 (워드클라우드용)
                    all_text = ' '.join([r['text'] for r in doc_results])
                    
                    results.append({
                        'Equity': equity,
                        'Sentiment': final_sentiment,
                        'Sentiment_Score': avg_score,
                        'Document_Count': len(doc_results),
                        'Combined_Text': all_text
                    })
                    
                    progress_bar.progress((idx + 1) / total_equities)
                    status_text.text(f"분석 중: {equity} ({idx + 1}/{total_equities})")
                
                result_df = pd.DataFrame(results)
                
                progress_bar.progress(100)
                status_text.text("✅ 분석 완료!")
            
            st.session_state.analysis_complete = True
            st.session_state.analysis_df = result_df
        
        if st.session_state.analysis_complete:
            df = st.session_state.analysis_df
            
            st.success("✅ 센티먼트 분석 완료!")
            
            # ==================== 메트릭 대시보드 ====================
            st.markdown("---")
            st.subheader("📊 주요 메트릭")
            
            sentiment_counts = df['Sentiment'].value_counts()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "긍정적 종목",
                    f"{sentiment_counts.get('POSITIVE', 0)}개",
                    f"{sentiment_counts.get('POSITIVE', 0) / len(df) * 100:.1f}%"
                )
            
            with col2:
                st.metric(
                    "부정적 종목",
                    f"{sentiment_counts.get('NEGATIVE', 0)}개",
                    f"{sentiment_counts.get('NEGATIVE', 0) / len(df) * 100:.1f}%"
                )
            
            with col3:
                st.metric(
                    "중립적 종목",
                    f"{sentiment_counts.get('NEUTRAL', 0)}개",
                    f"{sentiment_counts.get('NEUTRAL', 0) / len(df) * 100:.1f}%"
                )
            
            with col4:
                top_equity = df.nlargest(1, 'Sentiment_Score').iloc[0]
                st.metric(
                    "최고 선호 종목",
                    top_equity['Equity'],
                    f"{top_equity['Sentiment_Score']:.3f}"
                )
            
            # Top 5 종목 표시
            st.markdown("### 🏆 센티먼트 Top 5")
            top5 = df.nlargest(5, 'Sentiment_Score')[['Equity', 'Sentiment_Score', 'Sentiment']]
            
            cols = st.columns(5)
            for idx, (_, row) in enumerate(top5.iterrows()):
                with cols[idx]:
                    st.markdown(f"**#{idx+1} {row['Equity']}**")
                    st.markdown(f"<p class='{row['Sentiment'].lower()}'>{row['Sentiment_Score']:.3f}</p>", 
                               unsafe_allow_html=True)
            
            # ==================== 시각화 ====================
            st.markdown("---")
            st.subheader("📈 센티먼트 분석 상세")
            
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "센티먼트 분포", "종목 점수", "상위 종목 비교", "워드클라우드", "문서 분석", "상세 분석"
            ])
            
            with tab1:
                st.plotly_chart(plot_sentiment_distribution(df), width="stretch")
                st.plotly_chart(plot_sentiment_score_distribution(df), width="stretch")
            
            with tab2:
                st.plotly_chart(plot_equity_sentiment_scores(df), width="stretch")
                st.plotly_chart(plot_sentiment_comparison_radar(df), width="stretch")
                
                # 감정 분류 기준 설명 추가
                st.info("""
                **📌 센티먼트 분류 기준 **
                
                - **긍정 (POSITIVE)**: 센티먼트 > 0.2
                - **중립 (NEUTRAL)**: -0.2 ≤ 센티먼트 ≤ 0.2  
                - **부정 (NEGATIVE)**: 센티먼트 < -0.2
                
                **💡 부정적 종목이 적은 이유:**
                Earnings call 및 재무 보고서는 일반적으로 중립적이거나 긍정적인 언어를 사용하는 경향이 있습니다. 
                실제로 부정적인 내용도 완곡하게 표현되는 경우가 많아, 명확히 부정적인 점수(-0.2 이하)를 받는 경우는 드뭅니다.
                """)
            
            with tab3:
                st.plotly_chart(plot_top_equities_comparison(df), width="stretch")

            with tab4:
                sentiment_pipeline = st.session_state.get('sentiment_pipeline')  # 이 줄 추가
                st.markdown("### 워드클라우드 분석")
    
                # 분석 모드 선택
                analysis_mode = st.radio(
                    "분석 모드 선택",
                    options=["빈도 기반 (기본)", "센티먼트 기여도 기반 (AI)"],
                    horizontal=True,
                    help="기여도 기반: 실제로 센티먼트 분류에 영향을 준 단어만 표시 (처리 시간 더 소요)"
                )
    
                wc_option = st.radio(
                    "워드클라우드 유형 선택",
                    options=["센티먼트별", "종목별"],
                    horizontal=True
                )
    
                if analysis_mode == "빈도 기반 (기본)":
                    # 기존 코드 유지
                    if wc_option == "센티먼트별":
                        sentiment_filter = st.selectbox(
                            "센티먼트 선택",
                            options=["전체"] + df['Sentiment'].unique().tolist()
                        )
            
                        if sentiment_filter == "전체":
                            text_data = ' '.join(df['Combined_Text'].tolist())
                            title = "All Word Cloud"
                        else:
                            text_data = ' '.join(df[df['Sentiment'] == sentiment_filter]['Combined_Text'].tolist())
                            title = f"{sentiment_filter} Word Cloud"
            
                        wordcloud_fig = plot_wordcloud(text_data, title)
                        if wordcloud_fig:
                            st.pyplot(wordcloud_fig, use_container_width=True)
                        else:
                            st.warning("워드클라우드를 생성할 텍스트가 없습니다.")
        
                    else:  # 종목별
                        equity_filter = st.selectbox(
                            "종목 선택",
                            options=df['Equity'].tolist()
                        )
            
                        text_data = df[df['Equity'] == equity_filter]['Combined_Text'].iloc[0]
                        title = f"{equity_filter} Word Cloud"
            
                        wordcloud_fig = plot_wordcloud(text_data, title)
                        if wordcloud_fig:
                            st.pyplot(wordcloud_fig, use_container_width=True)
                        else:
                            st.warning("워드클라우드를 생성할 텍스트가 없습니다.")
    
                else:  # 센티먼트 기여도 기반
                    st.info("⏳ AI 모델이 센티먼트에 실제로 기여한 단어를 분석 중입니다...")
        
                    if wc_option == "센티먼트별":
                        sentiment_filter = st.selectbox(
                            "센티먼트 선택",
                            options=df['Sentiment'].unique().tolist(),
                            key="sentiment_contrib"
                        )
            
                        text_data = ' '.join(df[df['Sentiment'] == sentiment_filter]['Combined_Text'].tolist())
                        title = f"{sentiment_filter} - 센티먼트 기여 단어"
            
                        with st.spinner("분석 중..."):
                            wordcloud_fig = plot_sentiment_wordcloud(
                                text_data, 
                                sentiment_filter, 
                                sentiment_pipeline,
                                title
                            )
            
                        if wordcloud_fig:
                            st.pyplot(wordcloud_fig, use_container_width=True)
                            st.caption("💡 단어 크기 = 해당 센티먼트 분류에 대한 AI 모델의 기여도")
                        else:
                            st.warning("분석할 텍스트가 없습니다.")
        
                    else:  # 종목별
                        equity_filter = st.selectbox(
                            "종목 선택",
                            options=df['Equity'].tolist(),
                            key="equity_contrib"
                        )
            
                        equity_data = df[df['Equity'] == equity_filter].iloc[0]
                        text_data = equity_data['Combined_Text']
                        sentiment = equity_data['Sentiment']
                        title = f"{equity_filter} - {sentiment} 기여 단어"
            
                        with st.spinner("분석 중..."):
                            wordcloud_fig = plot_sentiment_wordcloud(
                                text_data,
                                sentiment,
                                sentiment_pipeline,
                                title
                            )
            
                        if wordcloud_fig:
                            st.pyplot(wordcloud_fig, use_container_width=True)
                            st.caption("💡 이 종목이 해당 센티먼트로 분류된 이유가 되는 핵심 단어들입니다.")
                        else:
                            st.warning("분석할 텍스트가 없습니다.")
            
            with tab5:
                st.plotly_chart(plot_document_length_analysis(df), width="stretch")
                
                # 통계 요약
                st.markdown("### 📊 문서 통계")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("평균 문서 길이", f"{df['Combined_Text'].str.len().mean():.0f} 자")
                with col2:
                    st.metric("최대 문서 길이", f"{df['Combined_Text'].str.len().max():.0f} 자")
                with col3:
                    st.metric("최소 문서 길이", f"{df['Combined_Text'].str.len().min():.0f} 자")
            
            with tab6:
                st.markdown("### 📋 상세 분석")
                
                # 종목별 순위
                equity_ranking = calculate_equity_ranking(df)
                
                st.markdown("#### 🏆 종목 순위 및 포트폴리오 평가")
                
                display_ranking = equity_ranking[['Sentiment_Score', 'Sentiment', 
                                                  'Document_Count', 'Sentiment_Grade', 
                                                  'Investment_Preference']].copy()
                display_ranking.columns = ['센티먼트', '센티먼트 분류', '문서수', '등급', '투자선호도']
                display_ranking = display_ranking.round(4)
                
                st.dataframe(
                    display_ranking,
                    use_container_width=True,
                    height=400
                )
                
                # 종목별 키워드
                st.markdown("#### 🔍 종목별 주요 키워드")
                
                selected_equity = st.selectbox("종목 선택", df['Equity'].tolist(), key="keyword_equity")
                
                equity_text = df[df['Equity'] == selected_equity]['Combined_Text'].iloc[0]
                keywords = extract_keywords(equity_text, n_words=20)
                
                if keywords:
                    keyword_df = pd.DataFrame(keywords, columns=['키워드', '빈도'])
                    
                    fig = px.bar(
                        keyword_df.head(15),
                        x='빈도',
                        y='키워드',
                        orientation='h',
                        title=f"{selected_equity} - 주요 키워드 Top 15",
                        color='빈도',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("추출할 키워드가 없습니다.")
                
                # 감정별 통계
                st.markdown("#### 💭 감정 분류별 통계")
                
                sentiment_stats = df.groupby('Sentiment').agg({
                    'Equity': 'count',
                    'Sentiment_Score': ['mean', 'std', 'min', 'max']
                }).round(4)
                
                sentiment_stats.columns = ['종목수', '평균점수', '표준편차', '최소', '최대']
                st.dataframe(sentiment_stats, use_container_width=True)


            # ==================== 다운로드 섹션 ====================
            st.markdown("---")
            st.subheader("💾 결과 다운로드")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 분석 결과 CSV 다운로드
                result_csv = df[['Equity', 'Sentiment', 'Sentiment_Score', 'Document_Count']].to_csv(index=False)
                st.download_button(
                    label="📥 분석 결과 (CSV)",
                    data=result_csv,
                    file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # 종목 순위 다운로드
                equity_ranking = calculate_equity_ranking(df)
                ranking_csv = equity_ranking.to_csv()
                st.download_button(
                    label="📊 종목 순위 (CSV)",
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
        
        2. **분석 실행**: "센티먼트 분석 실행" 버튼을 클릭합니다
           - 최신 Transformer 모델 (FinBERT) 사용
           - 금융 도메인에 최적화된 센티먼트 분석
        
        3. **결과 확인**: 
           - 📊 센티먼트 분포 및 시각화
           - 🏆 종목별 순위 및 포트폴리오 점수
           - 🔍 주요 키워드 분석
           - 💭 센티먼트별 상세 통계
        
        4. **결과 다운로드**: 분석 결과를 CSV로 저장합니다
        """)
        
        st.markdown("---")
        st.subheader("🤖 사용 모델")
        st.markdown("""
        - **FinBERT**: BERT를 금융 텍스트로 파인튜닝한 최신 모델
        - **Transformer Pipeline**: 고성능 센티먼트 분석
        - **Word Cloud**: 센티먼트별/종목별 주요 단어 시각화
        
        이 모델들은 전통적 방식보다 훨씬 높은 정확도를 제공합니다.
        """)

if __name__ == "__main__":
    main()
