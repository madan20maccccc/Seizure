
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_toggle import st_toggle_switch
import time

# ------------------------------
# APP CONFIGURATION
# ------------------------------
st.set_page_config(
    page_title="NeuroCluster Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary: #6f42c1;
        --secondary: #20c997;
        --dark: #212529;
        --light: #f8f9fa;
    }
    
    * {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    
    .header {
        color: var(--primary);
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
    }
    
    .stPlotlyChart {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# DATA LOADING
# ------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv('cleaned_seizure_data.csv')
    data['y_binary'] = (data['y'] == 1).astype(int)
    X = data.drop(columns=['y', 'y_binary'])
    y = data['y_binary']
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dimensionality reduction
    pca = PCA(n_components=40, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_scaled, X_pca, y, data

X_scaled, X_pca, y, raw_data = load_data()

# ------------------------------
# SIDEBAR CONTROLS
# ------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.title("NeuroCluster Explorer")
    st.markdown("""
    <div style="margin-bottom: 30px;">
        Interactive exploration of DBSCAN clustering for seizure detection data.
        Adjust parameters and visualize results in real-time.
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("⚙️ Clustering Parameters", expanded=True):
        eps = st.slider(
            "EPS (ε)", 
            min_value=0.1, 
            max_value=5.0, 
            value=3.0, 
            step=0.1,
            help="The maximum distance between two samples for one to be considered in the neighborhood of the other"
        )
        
        min_samples = st.slider(
            "Minimum Samples", 
            min_value=1, 
            max_value=20, 
            value=5, 
            step=1,
            help="The number of samples in a neighborhood for a point to be considered a core point"
        )
        
        advanced = st_toggle_switch(
            "Show Advanced Options",
            default_value=False,
            label_after=False,
            inactive_color="#D3D3D3",
            active_color="#6f42c1"
        )
        
        if advanced:
            algorithm = st.selectbox(
                "Algorithm",
                options=['auto', 'ball_tree', 'kd_tree', 'brute'],
                index=0
            )
            
            metric = st.selectbox(
                "Distance Metric",
                options=['euclidean', 'manhattan', 'cosine'],
                index=0
            )
        else:
            algorithm = 'auto'
            metric = 'euclidean'
    
    with st.expander("📊 Visualization Options"):
        dim_reduction = st.radio(
            "Dimensionality Reduction",
            options=["PCA", "t-SNE"],
            index=0
        )
        
        n_components = st.slider(
            "Visualization Dimensions",
            min_value=2,
            max_value=3,
            value=2,
            step=1
        )
        
        color_theme = st.selectbox(
            "Color Theme",
            options=["Viridis", "Plasma", "Inferno", "Magma", "Cividis"],
            index=0
        )
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; font-size: 0.8em; color: #666;">
        Created with ❤️ by [Your Name]<br>
        Powered by Streamlit
    </div>
    """, unsafe_allow_html=True)

# ------------------------------
# MAIN DASHBOARD
# ------------------------------
st.header("🧠 Seizure Detection Cluster Analysis")
st.markdown("""
<div style="margin-bottom: 30px;">
    Explore how DBSCAN clustering performs on EEG seizure detection data. 
    Adjust parameters in the sidebar to see real-time changes in clustering performance.
</div>
""", unsafe_allow_html=True)

# Initialize session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = []

# ------------------------------
# CLUSTERING EXECUTION
# ------------------------------
def run_dbscan(X, eps, min_samples, algorithm, metric):
    with st.spinner('🏃‍♂️ Running DBSCAN clustering...'):
        start_time = time.time()
        
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            algorithm=algorithm,
            metric=metric
        )
        
        clusters = dbscan.fit_predict(X)
        runtime = time.time() - start_time
        
        # Calculate metrics
        n_clusters = len(np.unique(clusters[clusters != -1]))
        n_noise = np.sum(clusters == -1)
        noise_percentage = n_noise / len(clusters) * 100
        
        # Only calculate silhouette if more than 1 cluster
        if n_clusters > 1:
            silhouette = silhouette_score(X, clusters)
        else:
            silhouette = -1
        
        # Map clusters to true labels for classification metrics
        mapped_labels = np.zeros_like(clusters)
        for cluster in np.unique(clusters):
            mask = clusters == cluster
            mapped_labels[mask] = np.bincount(y[mask]).argmax()
        
        f1 = f1_score(y, mapped_labels)
        
        return {
            'clusters': clusters,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_percentage': noise_percentage,
            'silhouette': silhouette,
            'f1_score': f1,
            'runtime': runtime,
            'params': {
                'eps': eps,
                'min_samples': min_samples,
                'algorithm': algorithm,
                'metric': metric
            }
        }

# Run clustering
result = run_dbscan(X_pca, eps, min_samples, algorithm, metric)

# Store results for comparison
if st.sidebar.button("💾 Save Current Configuration"):
    st.session_state.results.append(result)
    st.success("Configuration saved for comparison!")

# ------------------------------
# METRICS DISPLAY
# ------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Clusters Found",
        value=result['n_clusters'],
        help="Number of clusters identified (excluding noise)"
    )

with col2:
    st.metric(
        label="Noise Points",
        value=f"{result['n_noise']} ({result['noise_percentage']:.1f}%)",
        help="Number and percentage of points classified as noise"
    )

with col3:
    st.metric(
        label="Silhouette Score",
        value=f"{result['silhouette']:.3f}" if result['silhouette'] != -1 else "N/A",
        help="Measure of cluster cohesion and separation (-1 to 1)"
    )

with col4:
    st.metric(
        label="F1 Score",
        value=f"{result['f1_score']:.3f}",
        help="F1 score when mapping clusters to true labels"
    )

with col5:
    st.metric(
        label="Runtime",
        value=f"{result['runtime']:.3f}s",
        help="Time taken to perform clustering"
    )

# Apply styling to metrics
style_metric_cards(
    border_left_color="#6f42c1",
    box_shadow=True,
    border_size_px=3
)

# ------------------------------
# VISUALIZATION
# ------------------------------
st.markdown("## 📊 Cluster Visualization")

# Dimensionality reduction for visualization
@st.cache_data
def reduce_dimensions(X, method="PCA", n_components=2):
    if method == "PCA":
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        reducer = TSNE(n_components=n_components, random_state=42)
    return reducer.fit_transform(X)

X_vis = reduce_dimensions(X_scaled, dim_reduction, n_components)

# Create visualization
tab1, tab2, tab3 = st.tabs(["2D/3D Scatter Plot", "Cluster Distribution", "Confusion Matrix"])

with tab1:
    if n_components == 2:
        fig = px.scatter(
            x=X_vis[:, 0],
            y=X_vis[:, 1],
            color=result['clusters'],
            color_continuous_scale=color_theme.lower(),
            title=f"DBSCAN Clustering (EPS={eps}, Min Samples={min_samples})",
            labels={'color': 'Cluster'},
            hover_name=y.index,
            hover_data={'True Label': y}
        )
        fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(height=600)
    else:
        fig = px.scatter_3d(
            x=X_vis[:, 0],
            y=X_vis[:, 1],
            z=X_vis[:, 2],
            color=result['clusters'],
            color_continuous_scale=color_theme.lower(),
            title=f"DBSCAN Clustering (EPS={eps}, Min Samples={min_samples})",
            labels={'color': 'Cluster'},
            hover_name=y.index,
            hover_data={'True Label': y}
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(height=700)
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Create DataFrame for Plotly
    cluster_df = pd.DataFrame({
        'Cluster': result['clusters'],
        'Type': ['Noise' if x == -1 else f'Cluster {x}' for x in result['clusters']]
    })
    
    # Use categorical coloring for discrete clusters
    fig2 = px.histogram(
        cluster_df,
        x='Cluster',
        nbins=result['n_clusters'] + 1,
        title="Cluster Distribution",
        labels={'count': 'Number of Points'},
        color='Type',
        color_discrete_sequence=px.colors.qualitative.Plotly,
        category_orders={"Type": [f'Cluster {i}' for i in range(result['n_clusters'])] + ['Noise']}
    )
    
    # Customize layout
    fig2.update_layout(
        bargap=0.1,
        height=500,
        xaxis_title='Cluster ID',
        yaxis_title='Count',
        legend_title='Cluster Type'
    )
    
    # Highlight noise differently
    fig2.for_each_trace(
        lambda t: t.update(marker_color='#ff7f0e') if 'Noise' in t.name else ()
    )
    
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    mapped_labels = np.zeros_like(result['clusters'])
    for cluster in np.unique(result['clusters']):
        mask = result['clusters'] == cluster
        mapped_labels[mask] = np.bincount(y[mask]).argmax()
    
    cm = confusion_matrix(y, mapped_labels)
    fig3 = px.imshow(
        cm,
        text_auto=True,
        title="Confusion Matrix (Clusters Mapped to True Labels)",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Cluster 0', 'Cluster 1'],
        y=['Actual 0', 'Actual 1'],
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig3, use_container_width=True)

# ------------------------------
# RESULTS COMPARISON
# ------------------------------
if st.session_state.results:
    st.markdown("## 🔍 Results Comparison")
    
    comparison_data = []
    for i, res in enumerate(st.session_state.results):
        comparison_data.append({
            "Run": i+1,
            "EPS": res['params']['eps'],
            "Min Samples": res['params']['min_samples'],
            "Clusters": res['n_clusters'],
            "Noise %": f"{res['noise_percentage']:.1f}%",
            "Silhouette": f"{res['silhouette']:.3f}" if res['silhouette'] != -1 else "N/A",
            "F1 Score": f"{res['f1_score']:.3f}",
            "Runtime (s)": f"{res['runtime']:.3f}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    with stylable_container(
        key="comparison_table",
        css_styles="""
            {
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                padding: 15px;
                background-color: white;
            }
        """
    ):
        st.dataframe(
            df_comparison.style.background_gradient(cmap='Purples', subset=['F1 Score', 'Silhouette']),
            use_container_width=True,
            height=(len(df_comparison) + 1) * 35 + 3
        )
    
    if st.button("Clear Saved Results"):
        st.session_state.results = []
        st.rerun()

# ------------------------------
# DATA EXPLORER
# ------------------------------
with st.expander("🔍 Explore Raw Data"):
    filtered_df = dataframe_explorer(raw_data)
    st.dataframe(filtered_df, use_container_width=True)
