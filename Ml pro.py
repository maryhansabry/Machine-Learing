import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib
from datetime import datetime

st.set_page_config(page_title="Interactive ML Dashboard", layout="wide")

# Ensure a directory exists for saving models
MODEL_DIR = "saved_models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ----------------- Pages -----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["File Upload", "Data Visualization & Preprocessing", "Model Selection", "Model Evaluation"])

# ----------------- Page 1: File Upload -----------------
if page == "File Upload":
    st.title('📂 File Upload')

    uploaded_file = st.file_uploader("Upload your file (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            st.success('✅ File uploaded successfully!')
            for col in df.columns:
                if 'date' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        df['day'] = df[col].dt.day
                        df['month'] = df[col].dt.month
                        df['year'] = df[col].dt.year
                        # Fill NaN values in extracted columns with mode
                        for new_col in ['day', 'month', 'year']:
                            if new_col in df.columns:
                                df[new_col] = df[new_col].fillna(df[new_col].mode()[0] if not df[new_col].mode().empty else 1)
                        st.success(f"Extracted day/month/year from: {col}")
                    except Exception as e:
                        st.warning(f"Failed to convert {col} to datetime: {e}")

            st.subheader('Sample of Uploaded Data:')
            st.dataframe(df.head())

            st.session_state['df_original'] = df.copy()
            st.session_state['df'] = df.copy()
        except Exception as e:
            st.error(f"Error: {e}")

# ----------------- Page 2: Data Visualization & Preprocessing -----------------
if page == "Data Visualization & Preprocessing":
    st.title('📊 Data Visualization & Preprocessing')

    if 'df' in st.session_state:
        df = st.session_state['df']

        # ----------------- Visualization Section -----------------
        st.header('🔵 Data Visualization')

        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        st.write("---")

        chart_type = st.selectbox(
            'Choose chart type:',
            ['Scatter Plot', 'Line Chart', 'Bar Chart', 'Box Plot', 'Histogram']
        )

        if chart_type in ['Scatter Plot', 'Line Chart', 'Bar Chart']:
            x_axis = st.selectbox('Select X-axis column:', options=numerical_cols)
            y_axis = st.selectbox('Select Y-axis column:', options=numerical_cols)
            color_col = st.selectbox('Select a categorical column for color (Optional):', options=[None] + categorical_cols)

        elif chart_type in ['Box Plot', 'Histogram']:
            selected_col = st.selectbox('Select numerical column:', options=numerical_cols)
            color_col = st.selectbox('Select a categorical column for color (Optional):', options=[None] + categorical_cols)

        if st.button('Generate Chart'):
            st.subheader("Generated Chart:")

            if chart_type == 'Scatter Plot':
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col)
            elif chart_type == 'Line Chart':
                fig = px.line(df, x=x_axis, y=y_axis, color=color_col)
            elif chart_type == 'Bar Chart':
                fig = px.bar(df, x=x_axis, y=y_axis, color=color_col)
            elif chart_type == 'Box Plot':
                fig = px.box(df, y=selected_col, color=color_col)
            elif chart_type == 'Histogram':
                fig = px.histogram(df, x=selected_col, color=color_col)

            st.plotly_chart(fig, use_container_width=True)

        st.write("---")

        # ----------------- Preprocessing Section -----------------
        st.header('🟢 Data Preprocessing')

        st.subheader('📌 Data Summary per Column')

        # Skew
        skewness = df.skew(numeric_only=True)
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100

        # Outliers
        outliers = {}
        outliers_percentage = {}
        
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0 or pd.isna(IQR):
                outliers[col] = 0
                outliers_percentage[col] = 0
            else:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outliers[col] = outlier_count
                outliers_percentage[col] = (outlier_count / len(df)) * 100

        unique_values = df[categorical_cols].nunique()

        # Summary table
        summary_table = pd.DataFrame({
            'Missing Values': missing_values,
            'Missing %': missing_percentage.round(2),
            'Outliers Count': pd.Series(outliers),
            'Outliers %': pd.Series(outliers_percentage).round(2),
            'Skewness': skewness.round(2),
            'Unique Values': unique_values
        }).fillna(0)

        st.dataframe(summary_table)
        st.write("---")

        # ----------------- Data Preprocessing using Expander -----------------
        with st.expander('Drop Duplicated Rows'):
            if st.button('Drop Duplicates'):
                df = df.drop_duplicates()
                st.session_state['df'] = df
                st.success('✅ Duplicated rows removed.')
                st.rerun()

        with st.expander('Handle Missing Values'):
            missing_strategy = st.selectbox('Choose missing value strategy:', ['Drop Rows', 'Fill with Mean', 'Fill with Median', 'Fill with Mode'])
            if st.button('Apply Missing Handling'):
                if missing_strategy == 'Drop Rows':
                    df = df.dropna()
                elif missing_strategy == 'Fill with Mean':
                    df = df.fillna(df.mean(numeric_only=True))
                elif missing_strategy == 'Fill with Median':
                    df = df.fillna(df.median(numeric_only=True))
                elif missing_strategy == 'Fill with Mode':
                    df = df.fillna(df.mode().iloc[0])
                st.session_state['df'] = df
                st.success('✅ Missing values handled.')
                st.rerun()

        with st.expander('Handle Outliers'):
            selected_outlier_col = st.selectbox('Select column for outlier handling:', options=["All Columns"] + numerical_cols)
            
            if selected_outlier_col == "All Columns":
                outlier_count = 0
                for col in numerical_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_count += ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                st.info(f"Total outliers in all numerical columns: {outlier_count}")
            else:
                Q1 = df[selected_outlier_col].quantile(0.25)
                Q3 = df[selected_outlier_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = ((df[selected_outlier_col] < lower_bound) | (df[selected_outlier_col] > upper_bound)).sum()
                st.info(f"Number of outliers in {selected_outlier_col}: {outlier_count}")

            outlier_method = st.radio('Choose outlier handling method:', ['Remove Outliers', 'Cap Outliers (IQR)'])

            if st.button('Apply Outlier Handling'):
                if selected_outlier_col == "All Columns":
                    for col in numerical_cols:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        if outlier_method == 'Remove Outliers':
                            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                        elif outlier_method == 'Cap Outliers (IQR)':
                            df[col] = df[col].clip(lower_bound, upper_bound)
                else:
                    Q1 = df[selected_outlier_col].quantile(0.25)
                    Q3 = df[selected_outlier_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    if outlier_method == 'Remove Outliers':
                        df = df[(df[selected_outlier_col] >= lower_bound) & (df[selected_outlier_col] <= upper_bound)]
                    elif outlier_method == 'Cap Outliers (IQR)':
                        df[selected_outlier_col] = df[selected_outlier_col].clip(lower_bound, upper_bound)

                st.session_state['df'] = df
                st.success('✅ Outlier handling applied.')
                st.rerun()

        with st.expander('Encoding Categorical Data'):
            col_to_encode = st.selectbox('Select column to encode:', options=categorical_cols)
            encoding_type = st.radio('Choose encoding method:', ['Label Encoding', 'One Hot Encoding'])
            if st.button('Apply Encoding'):
                if encoding_type == 'Label Encoding':
                    df[col_to_encode] = df[col_to_encode].astype('category').cat.codes
                elif encoding_type == 'One Hot Encoding':
                    df = pd.get_dummies(df, columns=[col_to_encode])
                st.session_state['df'] = df
                st.success('✅ Encoding applied.')
                st.rerun()

        with st.expander('Normalization / Scaling'):
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            col_to_scale = st.selectbox('Select column to scale:', options=["All Columns"] + numeric_cols)
            scaling_type = st.radio('Choose scaling method:', ['Min-Max Scaling', 'Standardization (Z-score)'])

            if st.button('Apply Scaling'):
                if col_to_scale == "All Columns":
                    target_cols = numeric_cols
                else:
                    target_cols = [col_to_scale]

                for col in target_cols:
                    if scaling_type == 'Min-Max Scaling':
                        min_val = df[col].min()
                        max_val = df[col].max()
                        if max_val != min_val:  # Avoid division by zero
                            df[col] = (df[col] - min_val) / (max_val - min_val)
                    elif scaling_type == 'Standardization (Z-score)':
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        if std_val != 0:  # Avoid division by zero
                            df[col] = (df[col] - mean_val) / std_val

                st.success(f'✅ {scaling_type} applied on: {", ".join(target_cols)}')
                st.session_state['df'] = df
                st.rerun()

        with st.expander('Drop a Column'):
            col_to_drop = st.selectbox('Select column to drop:', options=df.columns.tolist())
            if st.button('Drop Selected Column'):
                df = df.drop(columns=[col_to_drop])
                st.session_state['df'] = df
                st.success(f'✅ Column "{col_to_drop}" dropped.')
                st.rerun()

        st.write("---")

        # Correlation Section
        if st.checkbox('Show Correlation Heatmap'):
            st.subheader('Correlation Matrix:')
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        st.write("---")

        # Button to Show Updated Sample
        if st.button('Show Updated Data Sample'):
            st.subheader('Updated Sample After Preprocessing:')
            st.dataframe(df.head())

    else:
        st.warning('⚠️ Please upload a file first from the "File Upload" page.')

# ------------------- Model Selection --------------------
if page == "Model Selection":
    st.title('🚀 Model Training')

    if 'df' in st.session_state:
        df = st.session_state['df']

        # Check if train-test split is already cached
        if 'X_train' not in st.session_state:
            features = ['Latitude', 'Longitude', 'wdir', 'wspd', 'pres', 'day', 'month']
            target = 'tavg'
            missing_cols = [col for col in features + [target] if col not in df.columns]
            if missing_cols:
                st.error(f"Missing columns in dataset: {', '.join(missing_cols)}")
            else:
                # Check for NaN values
                X = df[features]
                y = df[target]
                if X.isnull().any().any() or y.isnull().any():
                    st.error("Dataset contains NaN values. Please handle missing values in the 'Data Visualization & Preprocessing' page.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                    st.write(f"Features: {features}")
                    st.write(f"Target: {target}")
        else:
            X_train = st.session_state['X_train']
            X_test = st.session_state['X_test']
            y_train = st.session_state['y_train']
            y_test = st.session_state['y_test']
            st.write("Retrieved preprocessed data from session state.")

        # Model selection
        st.subheader("Select Models to Train")
        model_choice = st.multiselect(
            "Select Regression Model(s)",
            ["K-Nearest Neighbors (KNN)", "Decision Tree", "Random Forest", "Support Vector Regression (SVR)"]
        )

        st.write(f"Selected models: **{', '.join(model_choice)}**")

        # Store trained models in session state
        if 'trained_models' not in st.session_state:
            st.session_state['trained_models'] = {}

        # Train new models
        if st.button("Train Models"):
            if not model_choice:
                st.warning("⚠️ Please select at least one model to train.")
            elif 'X_train' not in st.session_state:
                st.error("⚠️ Please ensure the dataset is properly preprocessed and contains no NaN values.")
            else:
                for model_name in model_choice:
                    st.subheader(f"Training {model_name}...")
                    if model_name == "K-Nearest Neighbors (KNN)":
                        model = KNeighborsRegressor(n_neighbors=5)
                    elif model_name == "Decision Tree":
                        model = DecisionTreeRegressor(random_state=42)
                    elif model_name == "Random Forest":
                        model = RandomForestRegressor(n_estimators=50, random_state=42)
                    elif model_name == "Support Vector Regression (SVR)":
                        model = SVR(C=1.0, kernel='linear')

                    try:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        st.write(f"**{model_name} Performance:**")
                        st.write(f"Mean Squared Error: {mse:.4f}")
                        st.write(f"Mean Absolute Error: {mae:.4f}")
                        st.write(f"R² Score: {r2:.4f}")

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        model_filename = os.path.join(MODEL_DIR, f"{model_name.replace(' ', '_')}_{timestamp}.pkl")
                        joblib.dump(model, model_filename)
                        st.success(f"✅ {model_name} trained and saved as {model_filename}")

                        st.session_state['trained_models'][model_name] = {
                            'model': model,
                            'mse': mse,
                            'mae': mae,
                            'r2': r2,
                            'filename': model_filename
                        }

                        with open(model_filename, "rb") as f:
                            st.download_button(
                                label=f"Download {model_name} Model",
                                data=f,
                                file_name=os.path.basename(model_filename),
                                mime="application/octet-stream"
                            )
                    except ValueError as e:
                        st.error(f"Error training {model_name}: {e}")

        # Option to load a saved model
        st.subheader("📂 Load a Saved Model")
        uploaded_model = st.file_uploader("Upload a saved model (.pkl)", type=["pkl"])
        if uploaded_model is not None:
            try:
                model = joblib.load(uploaded_model)
                model_name = uploaded_model.name.split('_')[0] + " (Loaded)"
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.success(f"✅ Model loaded successfully!")
                st.write(f"**{model_name} Performance:**")
                st.write(f"Mean Squared Error: {mse:.4f}")
                st.write(f"Mean Absolute Error: {mae:.4f}")
                st.write(f"R² Score: {r2:.4f}")

                st.session_state['trained_models'][model_name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'filename': uploaded_model.name
                }
            except Exception as e:
                st.error(f"Error loading model: {e}")

        # Option to select a previously saved model
        saved_models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
        if saved_models:
            st.subheader("📚 Available Saved Models")
            selected_model = st.selectbox("Select a saved model to load:", saved_models)
            if st.button("Load Selected Model"):
                model_path = os.path.join(MODEL_DIR, selected_model)
                try:
                    model = joblib.load(model_path)
                    model_name = selected_model.split('_')[0] + " (Loaded)"
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.success(f"✅ Loaded {selected_model} successfully!")
                    st.write(f"**{model_name} Performance:**")
                    st.write(f"Mean Squared Error: {mse:.4f}")
                    st.write(f"Mean Absolute Error: {mae:.4f}")
                    st.write(f"R² Score: {r2:.4f}")

                    st.session_state['trained_models'][model_name] = {
                        'model': model,
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'filename': selected_model
                    }
                except Exception as e:
                    st.error(f"Error loading model: {e}")
    else:
        st.warning('⚠️ Please upload a file first from the "File Upload" page.')

# ------------------- Model Evaluation --------------------
if page == "Model Evaluation":
    st.title('📈 Model Evaluation')

    if 'trained_models' in st.session_state and st.session_state['trained_models']:
        st.subheader("Model Performance Comparison")

        # Prepare data for bar plot
        model_names = list(st.session_state['trained_models'].keys())
        mse_values = [st.session_state['trained_models'][name]['mse'] for name in model_names]
        mae_values = [st.session_state['trained_models'][name]['mae'] for name in model_names]
        r2_values = [st.session_state['trained_models'][name]['r2'] for name in model_names]

        # Create DataFrame for plotting
        metrics_df = pd.DataFrame({
            'Model': model_names * 3,
            'Metric': ['MSE'] * len(model_names) + ['MAE'] * len(model_names) + ['R² Score'] * len(model_names),
            'Value': mse_values + mae_values + r2_values
        })

        # Create bar plot similar to the provided image
        fig = px.bar(
            metrics_df,
            x='Model',
            y='Value',
            color='Metric',
            barmode='group',
            title='Comparison of Regression Model Metrics (MSE, MAE, R² Score)',
            labels={'Value': 'Score', 'Model': ''},
            color_discrete_map={'MSE': 'blue', 'MAE': 'orange', 'R² Score': 'green'}
        )
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Score",
            legend_title="",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display detailed metrics
        st.subheader("Detailed Performance")
        for model_name in model_names:
            st.write(f"**{model_name}:**")
            st.write(f"Mean Squared Error: {st.session_state['trained_models'][model_name]['mse']:.4f}")
            st.write(f"Mean Absolute Error: {st.session_state['trained_models'][model_name]['mae']:.4f}")
            st.write(f"R² Score: {st.session_state['trained_models'][model_name]['r2']:.4f}")
            st.write("---")

    else:
        st.warning("⚠️ No trained or loaded models available. Please train or load a model from the 'Model Selection' page first.")