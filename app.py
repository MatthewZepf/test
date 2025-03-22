import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import json

# Load model and data
df = pd.read_csv("grouped_data.csv")
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

with open("trained_feature_cols.json") as f:
    trained_feature_cols = json.load(f)

def predict_and_plot_country(
    df: pd.DataFrame,
    country: str,
    trained_feature_cols: list,
    scaler,
    model,
    impute_strategy="mean"
):
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.impute import SimpleImputer

    df = df.sort_values(["country_name", "year"]).copy()

    # Calculate target if missing
    if "target_polyarchy_15yr_avg" not in df.columns:
        df["target_polyarchy_15yr_avg"] = (
            df.groupby("country_name")["Democratic index"]
              .transform(lambda x: x.shift(-1).rolling(window=15, min_periods=10).mean())
        )

    # Filter by country
    country_df = df[df["country_name"] == country].copy()
    country_df_clean = country_df.dropna(subset=["target_polyarchy_15yr_avg"]).copy()

    if country_df_clean.empty:
        return go.Figure().add_annotation(text=f"No data available for {country}", showarrow=False)

    # Ensure all trained features are in the DataFrame
    for col in trained_feature_cols:
        if col not in country_df_clean.columns:
            country_df_clean[col] = np.nan

    # Select only those columns
    X = country_df_clean[trained_feature_cols]

    # Impute
    imputer = SimpleImputer(strategy=impute_strategy)
    X_imputed = imputer.fit_transform(X)

    # Rebuild DataFrame with only those columns that survived
    surviving_cols = X.columns[~np.all(np.isnan(X.values), axis=0)].tolist()
    X_df = pd.DataFrame(X_imputed, columns=surviving_cols)

    # Add any dropped columns back as 0, and re-order
    for col in trained_feature_cols:
        if col not in X_df.columns:
            X_df[col] = 0

    X_df = X_df[trained_feature_cols]

    # Scale + predict
    X_scaled = scaler.transform(X_df)
    country_df_clean["predicted_polyarchy"] = model.predict(X_scaled)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=country_df_clean["year"], y=country_df_clean["Democratic index"],
        name="Current Polyarchy", mode="lines+markers"
    ))
    fig.add_trace(go.Scatter(
        x=country_df_clean["year"], y=country_df_clean["target_polyarchy_15yr_avg"],
        name="Future 10–15yr Avg", mode="lines+markers", line=dict(dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=country_df_clean["year"], y=country_df_clean["predicted_polyarchy"],
        name="Model Prediction", mode="lines+markers", line=dict(dash="dash")
    ))

    fig.update_layout(
        title=f"{country} Democracy: Actual vs Predicted (10–15 Year Avg)",
        xaxis_title="Year",
        yaxis_title="v2x_polyarchy",
        yaxis_range=[0, 1],
        template="plotly_white"
    )

    return fig



st.title("🗳️ Democracy Predictor Dashboard")

tab1, tab2 = st.tabs(["📈 Model Predictions", "📊 Variable Viewer"])

with tab1:
    country = st.selectbox("Select a country", sorted(df["country_name"].unique()))
    fig = predict_and_plot_country(df, country, trained_feature_cols, scaler, model)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    country = st.selectbox("Select country", sorted(df["country_name"].unique()), key="var_country")
    variables = st.multiselect("Select variable(s) to view", [col for col in df.columns if col not in ["country_name", "year"]])

    if variables:
        subset = df[df["country_name"] == country]
        fig = px.line(subset, x="year", y=variables, title=f"{country}: Variable Trends")
        st.plotly_chart(fig, use_container_width=True)
