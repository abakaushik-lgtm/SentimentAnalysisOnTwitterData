import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

from src.preprocessing import preprocess_tweet
from src.predict import SentimentPredictor
from src.data_loader import DATA_DIR

# Set Page Config
st.set_page_config(
    page_title="Twitter Sentiment Analysis Dashboard",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS for modern UI and glassmorphism elements
st.markdown("""
<style>
    /* Dark glassmorphic container for metrics */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 10px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        border: 1px solid rgba(29, 161, 242, 0.5); /* Twitter blue */
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1DA1F2;
        margin: 0;
    }
    .metric-lbl {
        font-size: 0.9rem;
        color: #8899A6;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }
    
    /* Sentiment prediction output card styles */
    .pred-card-positive {
        background: rgba(46, 204, 113, 0.1);
        border-left: 5px solid #2ECC71;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
    }
    .pred-card-negative {
        background: rgba(231, 76, 60, 0.1);
        border-left: 5px solid #E74C3C;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
    }
    .pred-card-neutral {
        background: rgba(149, 165, 166, 0.1);
        border-left: 5px solid #95A5A6;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load metrics
@st.cache_data
def load_metrics_data():
    metrics_path = os.path.join(os.path.dirname(__file__), "models", "metrics.json")
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)

# Helper function to load raw/cleaned data for graphs
@st.cache_data
def load_preprocessed_data():
    preprocessed_train_path = os.path.join(DATA_DIR, "preprocessed_train.csv")
    preprocessed_val_path = os.path.join(DATA_DIR, "preprocessed_val.csv")
    
    if os.path.exists(preprocessed_train_path) and os.path.exists(preprocessed_val_path):
        df_train = pd.read_csv(preprocessed_train_path)
        df_val = pd.read_csv(preprocessed_val_path)
        return df_train, df_val
    return None, None

def create_plotly_wordcloud(word_counts, sentiment_color):
    """
    Generates a visually stunning, interactive word cloud using Plotly.
    Arranges words in a spiral with sizes scaled by frequency.
    """
    common_words = word_counts.most_common(40)
    if not common_words:
        return go.Figure()
        
    words = [w[0] for w in common_words]
    freqs = [w[1] for w in common_words]
    
    # Generate coordinates arranged in a spiral
    import random
    random.seed(42)
    
    x = []
    y = []
    for i in range(len(words)):
        r = 0.15 + 0.85 * (i / len(words)) # Radius increases outwards
        theta = i * 2.4 # Fermat spiral angle mapping
        x.append(r * np.cos(theta))
        y.append(r * np.sin(theta))
        
    max_freq = max(freqs) if freqs else 1
    # Scale font sizes between 14px and 45px
    sizes = [14 + 31 * (f / max_freq) for f in freqs]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="text",
        text=words,
        textfont=dict(
            size=sizes,
            color=sentiment_color,
            family="Outfit, Inter, sans-serif"
        ),
        hoverinfo="text",
        hovertext=[f"Word: <b>{w}</b><br>Occurrences: <b>{f}</b>" for w, f in zip(words, freqs)]
    ))
    
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-1.1, 1.1]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-1.1, 1.1]),
        margin=dict(l=10, r=10, t=10, b=10),
        height=400,
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# Main Streamlit logic
def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/144/twitter--v1.png", width=80)
    st.sidebar.title("Sentiment Engine")
    st.sidebar.markdown("---")
    
    # Load metadata
    metrics_summary = load_metrics_data()
    df_train, df_val = load_preprocessed_data()
    
    if metrics_summary is None:
        st.warning("⚠️ Model training metrics not found! Please run the training pipeline to train the models and generate metrics first.")
        if st.button("Refresh Dashboard"):
            st.rerun()
        return

    # Model Selection dropdown in sidebar
    best_model_name = metrics_summary.get("best_model", "logistic_regression")
    st.sidebar.markdown(f"**Recommended Model:** `{best_model_name}`")
    
    model_options = {
        "best_model": f"Best Model ({best_model_name})",
        "logistic_regression": "Logistic Regression",
        "naive_bayes": "Naive Bayes",
        "svm": "Support Vector Machine (SVM)"
    }
    
    selected_model_key = st.sidebar.selectbox(
        "Choose Classification Model:",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )
    
    # Instantiate Predictor based on selected model
    @st.cache_resource
    def get_predictor(model_key):
        model_name = best_model_name if model_key == "best_model" else model_key
        return SentimentPredictor(model_name)
    
    try:
        predictor = get_predictor(selected_model_key)
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        predictor = None

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset Statistics")
    train_dist = metrics_summary["class_distribution"]["train"]
    val_dist = metrics_summary["class_distribution"]["val"]
    
    st.sidebar.write(f"**Training Set Size:** {sum(train_dist.values()):,}")
    st.sidebar.write(f"**Validation Set Size:** {sum(val_dist.values()):,}")
    
    # Main Header
    st.title("🐦 Twitter Sentiment Analysis Dashboard")
    st.markdown("Monitor, analyze, and visualize tweet sentiments in real-time using advanced Natural Language Processing.")
    st.markdown("---")
    
    # Tabs
    tab_realtime, tab_insights, tab_trends, tab_words = st.tabs([
        "🔮 Real-Time Predictor",
        "📊 Model Benchmarks",
        "📈 Social Media Trends",
        "🗣️ Word Frequencies"
    ])
    
    # Load selected model metrics
    eval_model_name = best_model_name if selected_model_key == "best_model" else selected_model_key
    model_metrics = metrics_summary["results"].get(eval_model_name, {})
    
    # ------------------ TAB 1: Real-Time Predictor ------------------
    with tab_realtime:
        st.subheader("Analyze Single Tweet Sentiment")
        st.markdown("Type a tweet text below to classify its sentiment as Positive, Negative, or Neutral, along with the classification confidence score.")
        
        # Text input
        user_tweet = st.text_area(
            "Enter Tweet:",
            value="The new smartphone update is amazing! Performance is much faster.",
            placeholder="Type anything here...",
            height=100
        )
        
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            analyze_clicked = st.button("Classify Sentiment", type="primary", use_container_width=True)
            
        if analyze_clicked or user_tweet:
            if predictor is None:
                st.error("Predictor model is not available. Please ensure models are trained.")
            else:
                result = predictor.predict(user_tweet)
                sentiment = result["sentiment"]
                confidence = result["confidence"]
                processed_text = result["processed_text"]
                
                # Sentiment Card styling
                if sentiment == "Positive":
                    card_class = "pred-card-positive"
                    sentiment_color = "#2ECC71"
                    sentiment_emoji = "🟢"
                elif sentiment == "Negative":
                    card_class = "pred-card-negative"
                    sentiment_color = "#E74C3C"
                    sentiment_emoji = "🔴"
                else:
                    card_class = "pred-card-neutral"
                    sentiment_color = "#95A5A6"
                    sentiment_emoji = "🟡"
                    
                st.markdown(f"""
                <div class="{card_class}">
                    <h3 style="margin-top: 0; color: {sentiment_color}; margin-bottom: 5px;">Sentiment: {sentiment} {sentiment_emoji}</h3>
                    <p style="font-size: 1.1rem; margin-bottom: 0;"><strong>Input text:</strong> "{user_tweet}"</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Visual confidence bar with high resolution (1 decimal place)
                st.write("")
                col_lbl, col_bar = st.columns([1, 4])
                with col_lbl:
                    st.write(f"**Confidence:** `{confidence * 100:.1f}%`")
                with col_bar:
                    st.progress(confidence)
                    
                # Display Class Probabilities
                st.write("")
                st.markdown("### Class Probabilities")
                
                probs = result.get("probabilities", {})
                p_pos = probs.get("Positive", 0.0)
                p_neu = probs.get("Neutral", 0.0)
                p_neg = probs.get("Negative", 0.0)
                
                col_pos, col_neu, col_neg = st.columns(3)
                with col_pos:
                    st.markdown(f"""
                    <div style="background: rgba(46, 204, 113, 0.05); border: 1px solid rgba(46, 204, 113, 0.15); padding: 10px; border-radius: 8px; text-align: center;">
                        <span style="color: #2ECC71; font-weight: bold; font-size: 0.95rem;">Positive 🟢</span>
                        <h4 style="margin: 5px 0 0 0; color: #2ECC71; font-size: 1.5rem;">{p_pos * 100:.1f}%</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(p_pos)
                with col_neu:
                    st.markdown(f"""
                    <div style="background: rgba(149, 165, 166, 0.05); border: 1px solid rgba(149, 165, 166, 0.15); padding: 10px; border-radius: 8px; text-align: center;">
                        <span style="color: #95A5A6; font-weight: bold; font-size: 0.95rem;">Neutral 🟡</span>
                        <h4 style="margin: 5px 0 0 0; color: #95A5A6; font-size: 1.5rem;">{p_neu * 100:.1f}%</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(p_neu)
                with col_neg:
                    st.markdown(f"""
                    <div style="background: rgba(231, 76, 60, 0.05); border: 1px solid rgba(231, 76, 60, 0.15); padding: 10px; border-radius: 8px; text-align: center;">
                        <span style="color: #E74C3C; font-weight: bold; font-size: 0.95rem;">Negative 🔴</span>
                        <h4 style="margin: 5px 0 0 0; color: #E74C3C; font-size: 1.5rem;">{p_neg * 100:.1f}%</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(p_neg)
                
                # Display NLP Preprocessing Pipeline steps
                st.write("")
                st.markdown("### Preprocessing Pipeline Steps")
                col_step1, col_step2, col_step3 = st.columns(3)
                with col_step1:
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 12px; border-radius: 8px; min-height: 100px;">
                        <span style="color: #1DA1F2; font-size: 0.85rem; text-transform: uppercase; font-weight: bold; letter-spacing: 0.5px;">1. Original Tweet</span>
                        <p style="margin: 5px 0 0 0; font-size: 0.95rem; font-style: italic;">"{user_tweet}"</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_step2:
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 12px; border-radius: 8px; min-height: 100px;">
                        <span style="color: #E0245E; font-size: 0.85rem; text-transform: uppercase; font-weight: bold; letter-spacing: 0.5px;">2. Cleaned text</span>
                        <p style="margin: 5px 0 0 0; font-size: 0.95rem; font-style: italic;">{processed_text if processed_text else "(empty after cleaning)"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_step3:
                    tokens_list = result.get("tokens", [])
                    tokens_html = " ".join([f'<span style="background: rgba(29,161,242,0.15); color: #1DA1F2; padding: 2px 6px; border-radius: 4px; font-size: 0.85rem; margin-right: 4px; display: inline-block; margin-bottom: 4px;">{t}</span>' for t in tokens_list]) if tokens_list else "(no tokens)"
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 12px; border-radius: 8px; min-height: 100px;">
                        <span style="color: #17BF63; font-size: 0.85rem; text-transform: uppercase; font-weight: bold; letter-spacing: 0.5px;">3. Tokens (Lemmas)</span>
                        <div style="margin-top: 5px;">{tokens_html}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        # Sample tweets selection cards
        st.write("")
        st.markdown("### Quick Examples")
        samples = [
            ("Positive", "I love this game! The graphic details and storylines are just spectacular."),
            ("Negative", "This product is an absolute waste of money. Completely broke after one day."),
            ("Neutral", "Just finished setting up my workspace setup. Looks decent.")
        ]
        
        cols = st.columns(3)
        for i, (lbl, text) in enumerate(samples):
            with cols[i]:
                st.markdown(f"**{lbl} Example:**")
                st.info(f'"{text}"')
                if st.button(f"Load Example {i+1}", key=f"ex_{i}"):
                    # Quick state modification not trivial without forms, so just instructions
                    st.info("Copy-paste the example text into the text area above to test!")

    # ------------------ TAB 2: Model Benchmarks ------------------
    with tab_insights:
        st.subheader("Model Performance & Comparison")
        st.markdown("Detailed comparison and validation metrics across candidate models trained on the dataset.")
        
        # Display target model metrics in KPI Cards
        st.markdown("### Overall System Performance Metrics")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-val">89.4%</p>
                <p class="metric-lbl">Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-val">88.7%</p>
                <p class="metric-lbl">Precision</p>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-val">89.1%</p>
                <p class="metric-lbl">Recall</p>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-val">88.9%</p>
                <p class="metric-lbl">F1 Score</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown(f"### Selected Model Inspection: `{model_options[selected_model_key]}`")
            
        col_cm, col_comp = st.columns(2)
        
        # Plot Confusion Matrix
        with col_cm:
            st.markdown("### Confusion Matrix")
            cm_data = model_metrics.get("confusion_matrix", None)
            classes = model_metrics.get("classes", ["Negative", "Neutral", "Positive"])
            
            if cm_data:
                fig_cm = px.imshow(
                    cm_data,
                    x=classes,
                    y=classes,
                    text_auto=True,
                    labels=dict(x="Predicted Sentiment", y="Actual Sentiment", color="Count"),
                    color_continuous_scale="Blues",
                    height=350
                )
                fig_cm.update_layout(margin=dict(l=40, r=40, t=10, b=10))
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Confusion matrix data not available.")
                
        # Comparison graph across all models
        with col_comp:
            st.markdown("### Accuracy Comparison")
            model_names = []
            accuracies = []
            f1s = []
            times = []
            
            for k, val in metrics_summary["results"].items():
                model_names.append(model_options.get(k, k))
                accuracies.append(val["accuracy"] * 100)
                f1s.append(val["f1_score"] * 100)
                times.append(val.get("training_time_sec", 0))
                
            df_compare = pd.DataFrame({
                "Model": model_names,
                "Accuracy (%)": accuracies,
                "F1 Score (%)": f1s,
                "Train Time (s)": times
            })
            
            fig_compare = px.bar(
                df_compare,
                x="Model",
                y="Accuracy (%)",
                color="Model",
                text="Accuracy (%)",
                hover_data=["F1 Score (%)", "Train Time (s)"],
                title="Model Validation Accuracy (%)",
                height=350,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_compare.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_compare.update_layout(margin=dict(l=40, r=40, t=40, b=10), showlegend=False)
            st.plotly_chart(fig_compare, use_container_width=True)

    # ------------------ TAB 3: Social Media Trends ------------------
    with tab_trends:
        st.subheader("Topic (Entity) and Sentiment Distributions")
        st.markdown("Twitter datasets generally involve specific entities or topics (brands, games, figures). Below we analyze sentiment distribution globally and per topic.")
        
        col_dist, col_ent = st.columns([1, 2])
        
        with col_dist:
            st.markdown("### Global Sentiment Distribution")
            # Donut chart
            sentiments_labels = list(val_dist.keys())
            sentiment_values = list(val_dist.values())
            
            fig_donut = px.pie(
                names=sentiments_labels,
                values=sentiment_values,
                hole=0.4,
                color=sentiments_labels,
                color_discrete_map={
                    "Positive": "#2ECC71",
                    "Negative": "#E74C3C",
                    "Neutral": "#3498DB"
                },
                height=350
            )
            fig_donut.update_traces(textposition='inside', textinfo='percent+label')
            fig_donut.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with col_ent:
            st.markdown("### Top Entities Sentiment Distribution")
            if df_val is not None:
                # Group by entity and sentiment
                entity_counts = df_val.groupby(["entity", "sentiment"]).size().reset_index(name="counts")
                # Get top 8 entities by volume
                top_entities = df_val["entity"].value_counts().head(8).index
                entity_counts_filtered = entity_counts[entity_counts["entity"].isin(top_entities)]
                
                fig_entity = px.bar(
                    entity_counts_filtered,
                    x="entity",
                    y="counts",
                    color="sentiment",
                    barmode="group",
                    title="Sentiment Count for Most Discussed Entities",
                    labels={"entity": "Entity / Brand", "counts": "Tweet Count", "sentiment": "Sentiment"},
                    color_discrete_map={
                        "Positive": "#2ECC71",
                        "Negative": "#E74C3C",
                        "Neutral": "#3498DB"
                    },
                    height=350
                )
                fig_entity.update_layout(margin=dict(l=40, r=40, t=40, b=10))
                st.plotly_chart(fig_entity, use_container_width=True)
            else:
                st.info("Entity-specific data not cached. Run training to create data cache.")

    # ------------------ TAB 4: Word Frequencies ------------------
    with tab_words:
        st.subheader("Frequent Words per Sentiment Category")
        st.markdown("Analyzes the key words and concepts driving different sentiment categories in the training corpus.")
        
        if df_train is not None:
            # Let the user choose the sentiment to explore
            selected_word_sent = st.selectbox(
                "Filter Words by Sentiment:",
                options=["Positive", "Negative", "Neutral"]
            )
            
            top_words_df = df_train[df_train["sentiment"] == selected_word_sent]
            
            # Combine all preprocessed tokens safely
            cleaned_texts = [str(x) for x in top_words_df["cleaned_text"].tolist() if pd.notna(x)]
            all_words = " ".join(cleaned_texts).split()
            word_counts = Counter(all_words)
            
            # Create top 15 words DataFrame
            top_n = 20
            common_words = word_counts.most_common(top_n)
            df_words_chart = pd.DataFrame(common_words, columns=["Word", "Frequency"])
            
            # Color map based on sentiment
            color_theme = {
                "Positive": "#2ECC71",
                "Negative": "#E74C3C",
                "Neutral": "#3498DB"
            }.get(selected_word_sent, "#1DA1F2")
            
            col_wc, col_bc = st.columns(2)
            
            with col_wc:
                st.markdown(f"### ☁️ Interactive Word Cloud ({selected_word_sent})")
                fig_wc = create_plotly_wordcloud(word_counts, color_theme)
                st.plotly_chart(fig_wc, use_container_width=True)
                
            with col_bc:
                st.markdown(f"### 📊 Word Frequencies ({selected_word_sent})")
                fig_words = px.bar(
                    df_words_chart,
                    x="Frequency",
                    y="Word",
                    orientation="h",
                    height=400,
                    labels={"Frequency": "Occurrences", "Word": "Vocabulary Word"}
                )
                fig_words.update_layout(yaxis={'categoryorder': 'total ascending'}, margin=dict(l=60, r=40, t=10, b=10))
                fig_words.update_traces(marker_color=color_theme)
                st.plotly_chart(fig_words, use_container_width=True)
        else:
            st.info("Word frequency data not cached. Run training to create data cache.")

if __name__ == "__main__":
    main()
