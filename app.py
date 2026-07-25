import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from scipy.spatial.distance import cdist
import statsmodels.api as sm

st.set_page_config(
    page_title="AI Multi-Tool Hub", 
    page_icon="Multi_task/logo.png", 
    layout="wide"
)

if 'data' not in st.session_state:
    st.session_state['data'] = None

if os.path.exists("Multi_task/logo.png"):
    st.sidebar.image("Multi_task/logo.png", width=120)
st.sidebar.title("🤖 AI Multi-Tools Engine")

menu = st.sidebar.selectbox(
    "Navigation Menu",
    [
        "🏠 Dashboard Home",
        "📈 Linear Regression", 
        "🎯 K-Means Clustering", 
        "🏷️ KNN Classification"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Global Configuration")
test_size_user = st.sidebar.slider("Testing Data Size (%)", min_value=10, max_value=50, value=20, step=5) / 100.0

if menu == "🏠 Dashboard Home":
    st.markdown("<h1 style='text-align: center; color: #4F46E5;'>Automated Advanced ML Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload your dataset to initialize creative data analysis pipelines safely.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Drop your secure CSV file here", type=["csv"])
    
    if uploaded_file is not None:
        st.session_state['data'] = pd.read_csv(uploaded_file)
        st.success("Dataset successfully authenticated and loaded into secure cache.")
            
    if st.session_state['data'] is not None:
        df = st.session_state['data']
        st.markdown("### 📊 Live Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", df.shape[0])
        c2.metric("Total Columns", df.shape[1])
        c3.metric("Missing Cells Detected", df.isna().sum().sum())
        st.dataframe(df.head(10), use_container_width=True)

elif menu == "📈 Linear Regression":
    st.markdown("<h2 style='color: #2563EB;'>📈 Linear Regression Pipeline</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state['data'] is None:
        st.warning("Please upload a safe CSV file from the 'Dashboard Home' section first.")
    else:
        df = st.session_state['data'].copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("Dataset lacks sufficient continuous numeric variables for regression analysis.")
        else:
            y_col = st.selectbox("Select Target Variable (Y)", numeric_cols)
            x_cols = st.multiselect("Select Independent Features (X)", [c for c in numeric_cols if c != y_col])
            
            if x_cols:
                working_df = df[x_cols + [y_col]].dropna()
                X = working_df[x_cols]
                y = working_df[y_col]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_user, random_state=0)
                
                X_train_sm = sm.add_constant(X_train)
                X_test_sm = sm.add_constant(X_test, has_constant='add')
                ols_model = sm.OLS(y_train, X_train_sm).fit()
                preds = ols_model.predict(X_test_sm)
                
                if len(x_cols) == 1:
                    st.markdown("### 🔮 Live Interactive Prediction Calculator")
                    feature_name = x_cols[0]
                    intercept_val = ols_model.params['const']
                    coefficient_val = ols_model.params[feature_name]
                    p_val = ols_model.pvalues[feature_name]
                    
                    user_input_val = st.number_input(f"Set Custom Input Value for {feature_name}:", value=float(X_test[feature_name].mean()))
                    predicted_outcome = (coefficient_val * user_input_val) + intercept_val
                    
                    st.success(f"💡 **Statement:** Agar **{feature_name}** ki value **{user_input_val:.2f}** hogi, toh **{y_col}** ki estimated value **{predicted_outcome:.2f}** hogi!")
                
                if st.button("Execute Detailed Analysis"):
                    st.markdown("### 🏆 Performance Matrix")
                    col1, col2 = st.columns(2)
                    col1.metric("R² Prediction Accuracy", f"{r2_score(y_test, preds):.4f}")
                    col2.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test, preds):.4f}")
                    
                    st.markdown("### 📋 Detailed OLS Regression Statistical Summary")
                    st.text_area("OLS Summary Report Table", str(ols_model.summary()), height=350)
                    
                    if len(x_cols) == 1:
                        st.markdown("### 🔍 Automated Trend Analysis Insights")
                        if coefficient_val > 0:
                            trend_dir = "Positive Upward Trend 📈"
                            trend_desc = f"As **{feature_name}** increases, **{y_col}** also increases dynamically."
                        else:
                            trend_dir = "Negative Downward Trend 📉"
                            trend_desc = f"As **{feature_name}** increases, **{y_col}** decreases proportionally."
                            
                        sig_status = "Statistically Significant ✅" if p_val < 0.05 else "Not Statistically Significant ⚠️"
                        st.info(f"**Detected Relationship:** {trend_dir}\n\n{trend_desc}\n\n**Confidence Level:** {sig_status} (P-value: {p_val:.5f})")
                        
                        st.markdown("### 📊 Continuous Fitting Trend Plot")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        plt.scatter(X_test[feature_name], y_test, color='#2563EB', alpha=0.7, label='Actual Values')
                        plt.plot(X_test[feature_name], preds, color='#EF4444', linewidth=3, label='Optimal Fit Line')
                        plt.xlabel(feature_name)
                        plt.ylabel(y_col)
                        plt.grid(True, linestyle='--', alpha=0.5)
                        plt.legend()
                        st.pyplot(fig)

elif menu == "🎯 K-Means Clustering":
    st.markdown("<h2 style='color: #7C3AED;'>🎯 K-Means Clustering Core Engine</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state['data'] is None:
        st.warning("Please upload a safe CSV file from the 'Dashboard Home' section first.")
    else:
        df = st.session_state['data'].copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        x_cols = st.multiselect("Select Processing Features (Min 2)", numeric_cols)
        
        if x_cols and len(x_cols) >= 2:
            working_df = df[x_cols].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(working_df)
            
            st.markdown("### 📊 Step 1: Evaluating Distortion Curves")
            distortions = []
            max_k = min(10, len(working_df))
            K_range = range(1, max_k + 1)
            
            for k in K_range:
                kmeanModel = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
                distortions.append(sum(np.min(cdist(X_scaled, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X_scaled.shape[0])
                
            fig_elbow, ax_elbow = plt.subplots(figsize=(10, 4))
            plt.plot(K_range, distortions, 'bx-', color='#7C3AED', marker='o', linewidth=2)
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel('Calculated Structural Distortion')
            plt.title('The Elbow Method Optimization Interface')
            plt.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig_elbow)
            
            st.markdown("### ⚙️ Step 2: Final Cluster Optimization")
            k_choice = st.slider("Select Target Hyperparameter (K)", min_value=2, max_value=max_k, value=3)
            
            if st.button("Generate Partition Clusters"):
                kmeans = KMeans(n_clusters=k_choice, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                working_df['Cluster_Output'] = clusters
                st.success(f"Mathematical structural optimization split completed into {k_choice} separate nodes.")
                
                st.markdown("### 🔍 Automated Cluster Trend Interpretation")
                cluster_means = working_df.groupby('Cluster_Output').mean()
                trend_text = "Following data trends were identified within distinct density regions:\n\n"
                for idx, row in cluster_means.iterrows():
                    trend_text += f"* **Group/Cluster {idx}:** Density centers around "
                    trend_text += ", ".join([f"**{col}** = {row[col]:.2f}" for col in x_cols]) + ".\n"
                st.info(trend_text)
                
                fig_cluster, ax_cluster = plt.subplots(figsize=(10, 5))
                sns.scatterplot(x=working_df.iloc[:, 0], y=working_df.iloc[:, 1], hue=working_df['Cluster_Output'], palette='viridis', s=120, alpha=0.8)
                plt.grid(True, linestyle='--', alpha=0.4)
                st.pyplot(fig_cluster)

elif menu == "🏷️ KNN Classification":
    st.markdown("<h2 style='color: #059669;'>🏷️ KNN Classification Core Pipeline</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state['data'] is None:
        st.warning("Please upload a safe CSV file from the 'Dashboard Home' section first.")
    else:
        df = st.session_state['data'].copy()import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from scipy.spatial.distance import cdist
import statsmodels.api as sm

st.set_page_config(
    page_title="AI Multi-Tool Hub", 
    page_icon="Multi_task/logo.png", 
    layout="wide"
)

if 'data' not in st.session_state:
    st.session_state['data'] = None

if os.path.exists("Multi_task/logo.png"):
    st.sidebar.image("Multi_task/logo.png", width=120)
st.sidebar.title("🤖 AI Multi-Tools Engine")

menu = st.sidebar.selectbox(
    "Navigation Menu",
    [
        "🏠 Dashboard Home",
        "📈 Linear Regression", 
        "🎯 K-Means Clustering", 
        "🏷️ KNN Classification"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Global Configuration")
test_size_user = st.sidebar.slider("Testing Data Size (%)", min_value=10, max_value=50, value=20, step=5) / 100.0

if menu == "🏠 Dashboard Home":
    st.markdown("<h1 style='text-align: center; color: #4F46E5;'>Automated Advanced ML Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload your dataset to initialize creative data analysis pipelines safely.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Drop your secure CSV file here", type=["csv"])
    
    if uploaded_file is not None:
        st.session_state['data'] = pd.read_csv(uploaded_file)
        st.success("Dataset successfully authenticated and loaded into secure cache.")
            
    if st.session_state['data'] is not None:
        df = st.session_state['data']
        st.markdown("### 📊 Live Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", df.shape[0])
        c2.metric("Total Columns", df.shape[1])
        c3.metric("Missing Cells Detected", df.isna().sum().sum())
        st.dataframe(df.head(10), use_container_width=True)

elif menu == "📈 Linear Regression":
    st.markdown("<h2 style='color: #2563EB;'>📈 Linear Regression Pipeline</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state['data'] is None:
        st.warning("Please upload a safe CSV file from the 'Dashboard Home' section first.")
    else:
        df = st.session_state['data'].copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("Dataset lacks sufficient continuous numeric variables for regression analysis.")
        else:
            y_col = st.selectbox("Select Target Variable (Y)", numeric_cols)
            x_cols = st.multiselect("Select Independent Features (X)", [c for c in numeric_cols if c != y_col])
            
            if x_cols:
                working_df = df[x_cols + [y_col]].dropna()
                X = working_df[x_cols]
                y = working_df[y_col]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_user, random_state=0)
                
                X_train_sm = sm.add_constant(X_train)
                X_test_sm = sm.add_constant(X_test, has_constant='add')
                ols_model = sm.OLS(y_train, X_train_sm).fit()
                preds = ols_model.predict(X_test_sm)
                
                if len(x_cols) == 1:
                    st.markdown("### 🔮 Live Interactive Prediction Calculator")
                    feature_name = x_cols[0]
                    intercept_val = ols_model.params['const']
                    coefficient_val = ols_model.params[feature_name]
                    p_val = ols_model.pvalues[feature_name]
                    
                    user_input_val = st.number_input(f"Set Custom Input Value for {feature_name}:", value=float(X_test[feature_name].mean()))
                    predicted_outcome = (coefficient_val * user_input_val) + intercept_val
                    
                    st.success(f"💡 **Statement:** Agar **{feature_name}** ki value **{user_input_val:.2f}** hogi, toh **{y_col}** ki estimated value **{predicted_outcome:.2f}** hogi!")
                
                if st.button("Execute Detailed Analysis"):
                    st.markdown("### 🏆 Performance Matrix")
                    col1, col2 = st.columns(2)
                    col1.metric("R² Prediction Accuracy", f"{r2_score(y_test, preds):.4f}")
                    col2.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test, preds):.4f}")
                    
                    st.markdown("### 📋 Detailed OLS Regression Statistical Summary")
                    st.text_area("OLS Summary Report Table", str(ols_model.summary()), height=350)
                    
                    if len(x_cols) == 1:
                        st.markdown("### 🔍 Automated Trend Analysis Insights")
                        if coefficient_val > 0:
                            trend_dir = "Positive Upward Trend 📈"
                            trend_desc = f"As **{feature_name}** increases, **{y_col}** also increases dynamically."
                        else:
                            trend_dir = "Negative Downward Trend 📉"
                            trend_desc = f"As **{feature_name}** increases, **{y_col}** decreases proportionally."
                            
                        sig_status = "Statistically Significant ✅" if p_val < 0.05 else "Not Statistically Significant ⚠️"
                        st.info(f"**Detected Relationship:** {trend_dir}\n\n{trend_desc}\n\n**Confidence Level:** {sig_status} (P-value: {p_val:.5f})")
                        
                        st.markdown("### 📊 Continuous Fitting Trend Plot")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        plt.scatter(X_test[feature_name], y_test, color='#2563EB', alpha=0.7, label='Actual Values')
                        plt.plot(X_test[feature_name], preds, color='#EF4444', linewidth=3, label='Optimal Fit Line')
                        plt.xlabel(feature_name)
                        plt.ylabel(y_col)
                        plt.grid(True, linestyle='--', alpha=0.5)
                        plt.legend()
                        st.pyplot(fig)

elif menu == "🎯 K-Means Clustering":
    st.markdown("<h2 style='color: #7C3AED;'>🎯 K-Means Clustering Core Engine</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state['data'] is None:
        st.warning("Please upload a safe CSV file from the 'Dashboard Home' section first.")
    else:
        df = st.session_state['data'].copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        x_cols = st.multiselect("Select Processing Features (Min 2)", numeric_cols)
        
        if x_cols and len(x_cols) >= 2:
            working_df = df[x_cols].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(working_df)
            
            st.markdown("### 📊 Step 1: Evaluating Distortion Curves")
            distortions = []
            max_k = min(10, len(working_df))
            K_range = range(1, max_k + 1)
            
            for k in K_range:
                kmeanModel = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
                distortions.append(sum(np.min(cdist(X_scaled, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X_scaled.shape[0])
                
            fig_elbow, ax_elbow = plt.subplots(figsize=(10, 4))
            plt.plot(K_range, distortions, 'bx-', color='#7C3AED', marker='o', linewidth=2)
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel('Calculated Structural Distortion')
            plt.title('The Elbow Method Optimization Interface')
            plt.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig_elbow)
            
            st.markdown("### ⚙️ Step 2: Final Cluster Optimization")
            k_choice = st.slider("Select Target Hyperparameter (K)", min_value=2, max_value=max_k, value=3)
            
            if st.button("Generate Partition Clusters"):
                kmeans = KMeans(n_clusters=k_choice, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                working_df['Cluster_Output'] = clusters
                st.success(f"Mathematical structural optimization split completed into {k_choice} separate nodes.")
                
                st.markdown("### 🔍 Automated Cluster Trend Interpretation")
                cluster_means = working_df.groupby('Cluster_Output').mean()
                trend_text = "Following data trends were identified within distinct density regions:\n\n"
                for idx, row in cluster_means.iterrows():
                    trend_text += f"* **Group/Cluster {idx}:** Density centers around "
                    trend_text += ", ".join([f"**{col}** = {row[col]:.2f}" for col in x_cols]) + ".\n"
                st.info(trend_text)
                
                fig_cluster, ax_cluster = plt.subplots(figsize=(10, 5))
                sns.scatterplot(x=working_df.iloc[:, 0], y=working_df.iloc[:, 1], hue=working_df['Cluster_Output'], palette='viridis', s=120, alpha=0.8)
                plt.grid(True, linestyle='--', alpha=0.4)
                st.pyplot(fig_cluster)

elif menu == "🏷️ KNN Classification":
    st.markdown("<h2 style='color: #059669;'>🏷️ KNN Classification Core Pipeline</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state['data'] is None:
        st.warning("Please upload a safe CSV file from the 'Dashboard Home' section first.")
    else:
        df = st.session_state['data'].copy()
