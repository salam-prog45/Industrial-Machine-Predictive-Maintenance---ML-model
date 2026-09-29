"""
Industrial Machine Predictive Maintenance — AI Health Monitoring Dashboard
Run with:  streamlit run app.py
"""

import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent

# ------------------------------------------------------------------
# STYLING
# ------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 18px;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #8b949e; }
    .risk-banner {
        padding: 18px 24px;
        border-radius: 12px;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 12px;
        text-align: center;
        letter-spacing: 0.5px;
    }
    .risk-low { background: rgba(46, 160, 67, 0.15); border: 1px solid #2ea043; color: #3fb950; }
    .risk-moderate { background: rgba(210, 153, 34, 0.15); border: 1px solid #d29922; color: #e3b341; }
    .risk-high { background: rgba(219, 109, 40, 0.15); border: 1px solid #db6d28; color: #f0883e; }
    .risk-critical { background: rgba(248, 81, 73, 0.15); border: 1px solid #f85149; color: #ff7b72; }
    .cond-item {
        background-color: #161b22;
        border-left: 3px solid #d29922;
        padding: 8px 14px;
        border-radius: 4px;
        margin-bottom: 6px;
        font-size: 0.92rem;
    }
    section[data-testid="stSidebar"] { background-color: #10151c; }
    h1, h2, h3 { letter-spacing: 0.3px; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# LOAD MODEL + REPORT (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    stack = joblib.load(BASE_DIR / "stack_model.joblib")
    base_models = joblib.load(BASE_DIR / "base_models.joblib")
    with open(BASE_DIR / "model_report.json") as f:
        report = json.load(f)
    return stack, base_models, report


try:
    stack, base_models, report = load_artifacts()
    MODEL_READY = True
except Exception as e:
    MODEL_READY = False
    LOAD_ERROR = str(e)

FRIENDLY_NAMES = {
    "RF": "Random Forest",
    "DT": "Decision Tree",
    "KNN": "K-Nearest Neighbors",
    "LR": "Logistic Regression",
}

def friendly_name(model_name):
    """Return a display name for model keys regardless of capitalization."""
    key = str(model_name).strip().upper()
    return FRIENDLY_NAMES.get(key, str(model_name))


def engineer_features(row: dict) -> pd.DataFrame:
    temp_diff = row["Process temperature [K]"] - row["Air temperature [K]"]
    power = row["Torque [Nm]"] * row["Rotational speed [rpm]"] * (2 * np.pi / 60)
    strain = row["Tool wear [min]"] * row["Torque [Nm]"]
    return pd.DataFrame([{
        "Type": row["Type"],
        "Air temperature [K]": row["Air temperature [K]"],
        "Process temperature [K]": row["Process temperature [K]"],
        "Rotational speed [rpm]": row["Rotational speed [rpm]"],
        "Torque [Nm]": row["Torque [Nm]"],
        "Tool wear [min]": row["Tool wear [min]"],
        "Temp_diff": temp_diff,
        "Power": power,
        "Strain": strain,
    }])


def risk_level(p):
    if p < 20:
        return "LOW RISK", "risk-low"
    elif p < 50:
        return "MODERATE RISK", "risk-moderate"
    elif p < 75:
        return "HIGH RISK", "risk-high"
    return "CRITICAL RISK", "risk-critical"


def get_conditions(air_temp, process_temp, rot_speed, torque, tool_wear, temp_diff, power):
    conditions = []
    if tool_wear >= 200:
        conditions.append("High tool wear detected")
    elif tool_wear >= 150:
        conditions.append("Elevated tool wear detected")
    if torque >= 50:
        conditions.append("High torque detected")
    if rot_speed >= 2000:
        conditions.append("High rotational speed detected")
    elif rot_speed < 1000:
        conditions.append("Low rotational speed detected")
    if temp_diff >= 10:
        conditions.append("Large temperature difference detected")
    if power >= 5000:
        conditions.append("High mechanical power detected")
    return conditions


def maintenance_action(p):
    if p >= 75:
        return ("🚨 CRITICAL", [
            "Schedule immediate inspection",
            "Inspect tool condition",
            "Check torque and rotational behaviour",
            "Check temperature conditions",
            "Do not ignore repeated high-risk predictions",
        ])
    elif p >= 50:
        return ("⚠️ HIGH RISK", [
            "Perform preventive inspection",
            "Monitor tool wear closely",
            "Monitor temperature and torque",
        ])
    elif p >= 20:
        return ("🟡 MODERATE RISK", ["Not critical, but continue monitoring"])
    return ("🟢 LOW RISK", ["Model predicts normal operation", "Continue routine preventive maintenance"])


def run_prediction(row: dict):
    X_new = engineer_features(row)
    final_pred = stack.predict(X_new)[0]
    final_proba = stack.predict_proba(X_new)[0]
    failure_p = final_proba[1] * 100
    normal_p = final_proba[0] * 100

    base_preds, base_probs = {}, {}
    for name, model in base_models.items():
        base_preds[name] = int(model.predict(X_new)[0])
        base_probs[name] = float(model.predict_proba(X_new)[0][1] * 100)

    agreement = sum(base_preds.values()) / len(base_preds) * 100
    health = max(0, min(100, 100 - failure_p))

    return {
        "X": X_new,
        "final_pred": int(final_pred),
        "failure_p": failure_p,
        "normal_p": normal_p,
        "base_preds": base_preds,
        "base_probs": base_probs,
        "agreement": agreement,
        "health": health,
    }


# ------------------------------------------------------------------
# SIDEBAR — INPUT FORM
# ------------------------------------------------------------------
st.sidebar.title("⚙️ Machine Parameters")
st.sidebar.caption("Enter live sensor readings for a single machine")

machine_type = st.sidebar.selectbox("Machine Type", ["L", "M", "H"], help="L = Low, M = Medium, H = High quality variant")
air_temp = st.sidebar.number_input("Air temperature [K]", value=298.5, min_value=280.0, max_value=320.0, step=0.1)
process_temp = st.sidebar.number_input("Process temperature [K]", value=309.0, min_value=290.0, max_value=330.0, step=0.1)
rotational_speed = st.sidebar.number_input("Rotational speed [rpm]", value=1500, min_value=0, max_value=5000, step=10)
torque = st.sidebar.number_input("Torque [Nm]", value=40.0, min_value=0.0, max_value=150.0, step=0.5)
tool_wear = st.sidebar.number_input("Tool wear [min]", value=100, min_value=0, max_value=300, step=1)

predict_btn = st.sidebar.button("🔍 Run Prediction", use_container_width=True, type="primary")

st.sidebar.divider()
st.sidebar.subheader("📁 Batch Prediction")
batch_file = st.sidebar.file_uploader("Upload CSV (same columns as training data)", type=["csv"])
st.sidebar.caption("Columns required: Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]")

st.sidebar.divider()
st.sidebar.caption("Built on a Stacking Ensemble (RF + DT + KNN + LR → Logistic Regression meta-model)")


# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.title("🏭 Industrial Machine Predictive Maintenance")
st.caption("AI Health Monitoring Dashboard — real-time failure risk scoring")

if not MODEL_READY:
    st.error(f"Model artifacts could not be loaded: {LOAD_ERROR}")
    st.stop()

tab_predict, tab_batch, tab_performance, tab_data = st.tabs(
    ["🔍 Live Prediction", "📁 Batch Scoring", "📊 Model Performance", "🗂️ Dataset Overview"]
)

# ------------------------------------------------------------------
# TAB 1 — LIVE PREDICTION
# ------------------------------------------------------------------
with tab_predict:
    if predict_btn or "last_result" in st.session_state:
        if predict_btn:
            row = {
                "Type": machine_type, "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp, "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque, "Tool wear [min]": tool_wear,
            }
            st.session_state["last_result"] = run_prediction(row)
            st.session_state["last_row"] = row

        result = st.session_state["last_result"]
        row = st.session_state["last_row"]
        temp_diff = result["X"]["Temp_diff"].iloc[0]
        power = result["X"]["Power"].iloc[0]
        strain = result["X"]["Strain"].iloc[0]
        level_text, level_class = risk_level(result["failure_p"])

        status_icon = "⚠️ MACHINE FAILURE PREDICTED" if result["final_pred"] == 1 else "✅ MACHINE OPERATING NORMALLY"
        st.markdown(f"<div class='risk-banner {level_class}'>{status_icon} — {level_text}</div>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Health Score", f"{result['health']:.1f}%")
        c2.metric("Failure Probability", f"{result['failure_p']:.1f}%")
        c3.metric("Model Agreement", f"{result['agreement']:.0f}%")
        c4.metric("Power Draw", f"{power:.0f} W")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Prediction Probability")
            fig = go.Figure(go.Bar(
                x=["Normal Operation", "Machine Failure"],
                y=[result["normal_p"], result["failure_p"]],
                marker_color=["#3fb950", "#f85149"],
                text=[f"{result['normal_p']:.1f}%", f"{result['failure_p']:.1f}%"],
                textposition="outside",
            ))
            fig.update_layout(yaxis_range=[0, 100], template="plotly_dark", height=340,
                               margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Base Model Comparison")
            names = [friendly_name(n) for n in result["base_probs"]]
            vals = list(result["base_probs"].values())
            fig2 = go.Figure(go.Bar(
                x=names, y=vals, marker_color="#58a6ff",
                text=[f"{v:.1f}%" for v in vals], textposition="outside",
            ))
            fig2.update_layout(yaxis_range=[0, 100], template="plotly_dark", height=340,
                                yaxis_title="Failure probability (%)", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        with col_right:
            st.subheader("Machine Condition Profile")
            categories = ["Tool Wear", "Temperature Δ", "Speed", "Torque", "Power"]
            scores = [
                min(tool_wear / 250 * 100, 100),
                min(abs(temp_diff) / 15 * 100, 100),
                min(rotational_speed / 2500 * 100, 100),
                min(torque / 80 * 100, 100),
                min(power / 10000 * 100, 100),
            ]
            fig3 = go.Figure()
            fig3.add_trace(go.Scatterpolar(r=scores + [scores[0]], theta=categories + [categories[0]],
                                            fill="toself", line_color="#f0883e"))
            fig3.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                template="plotly_dark", height=340, showlegend=False,
                                margin=dict(l=30, r=30, t=20, b=20))
            st.plotly_chart(fig3, use_container_width=True)

            st.subheader("Machine Profile")
            st.dataframe(pd.DataFrame({
                "Parameter": ["Type", "Air Temp [K]", "Process Temp [K]", "Temp Diff [K]",
                              "Rotational Speed [rpm]", "Torque [Nm]", "Tool Wear [min]",
                              "Power [W]", "Strain"],
                "Value": [row["Type"], f"{row['Air temperature [K]']:.2f}", f"{row['Process temperature [K]']:.2f}",
                          f"{temp_diff:.2f}", row["Rotational speed [rpm]"], f"{row['Torque [Nm]']:.2f}",
                          row["Tool wear [min]"], f"{power:.2f}", f"{strain:.2f}"],
            }), hide_index=True, use_container_width=True)

        st.subheader("⚠️ Condition Analysis")
        conditions = get_conditions(air_temp, process_temp, rotational_speed, torque, tool_wear, temp_diff, power)
        if conditions:
            for c in conditions:
                st.markdown(f"<div class='cond-item'>⚠️ {c}</div>", unsafe_allow_html=True)
        else:
            st.success("No major threshold-based warning indicators detected.")

        st.subheader("🛠️ Recommended Maintenance Action")
        title, actions = maintenance_action(result["failure_p"])
        st.markdown(f"**{title}**")
        for a in actions:
            st.markdown(f"- {a}")

        with st.expander("Model agreement detail"):
            agree_df = pd.DataFrame({
                "Model": [friendly_name(n) for n in result["base_preds"]],
                "Prediction": ["FAILURE" if v == 1 else "NORMAL" for v in result["base_preds"].values()],
                "Failure Probability (%)": [f"{v:.2f}" for v in result["base_probs"].values()],
            })
            st.dataframe(agree_df, hide_index=True, use_container_width=True)
    else:
        st.info("Enter machine parameters in the sidebar and click **Run Prediction** to see results.")

# ------------------------------------------------------------------
# TAB 2 — BATCH SCORING
# ------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch scoring from CSV")
    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            required = ["Type", "Air temperature [K]", "Process temperature [K]",
                        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
            missing = [c for c in required if c not in batch_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                feat_df = batch_df.copy()
                feat_df["Temp_diff"] = feat_df["Process temperature [K]"] - feat_df["Air temperature [K]"]
                feat_df["Power"] = feat_df["Torque [Nm]"] * feat_df["Rotational speed [rpm]"] * (2 * np.pi / 60)
                feat_df["Strain"] = feat_df["Tool wear [min]"] * feat_df["Torque [Nm]"]

                model_input = feat_df[["Type", "Air temperature [K]", "Process temperature [K]",
                                        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
                                        "Temp_diff", "Power", "Strain"]]
                preds = stack.predict(model_input)
                probs = stack.predict_proba(model_input)[:, 1] * 100

                out_df = batch_df.copy()
                out_df["Failure Probability (%)"] = probs.round(2)
                out_df["Prediction"] = np.where(preds == 1, "FAILURE", "NORMAL")
                out_df["Risk Level"] = [risk_level(p)[0] for p in probs]

                c1, c2, c3 = st.columns(3)
                c1.metric("Machines Scored", len(out_df))
                c2.metric("Flagged for Failure", int((preds == 1).sum()))
                c3.metric("Avg Failure Probability", f"{probs.mean():.1f}%")

                fig = px.histogram(out_df, x="Failure Probability (%)", nbins=30,
                                    color="Prediction", color_discrete_map={"FAILURE": "#f85149", "NORMAL": "#3fb950"},
                                    template="plotly_dark")
                fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(out_df, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download scored results (CSV)", out_df.to_csv(index=False),
                                    file_name="scored_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not process file: {e}")
    else:
        st.info("Upload a CSV in the sidebar to score multiple machines at once.")

# ------------------------------------------------------------------
# TAB 3 — MODEL PERFORMANCE
# ------------------------------------------------------------------
with tab_performance:
    st.subheader("Held-out Test Set Performance")
    m = report["metrics"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
    c2.metric("Precision", f"{m['precision']*100:.2f}%")
    c3.metric("Recall", f"{m['recall']*100:.2f}%")
    c4.metric("F1 Score", f"{m['f1']*100:.2f}%")
    c5.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c6.metric("PR-AUC", f"{m['pr_auc']:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Confusion Matrix**")
        cm = np.array(report["confusion_matrix"])
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=["Normal", "Failure"], y=["Normal", "Failure"], template="plotly_dark")
        fig_cm.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.markdown("**ROC Curve**")
        roc = report["roc_curve"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                      line=dict(color="#58a6ff", width=3), name="Stacking Ensemble"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      line=dict(color="#8b949e", dash="dash"), name="Random baseline"))
        fig_roc.update_layout(template="plotly_dark", height=350, xaxis_title="False Positive Rate",
                               yaxis_title="True Positive Rate", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("**Base Model Cross-Validation Comparison (F1 score)**")
    res_df = pd.DataFrame(report["model_results"])
    fig_bm = px.bar(res_df, x="Model", y="CV F1", color="Model", template="plotly_dark",
                     color_discrete_sequence=px.colors.qualitative.Set2, text="CV F1")
    fig_bm.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_bm.update_layout(yaxis_range=[0, 1], height=350, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_bm, use_container_width=True)

    with st.expander("Best hyperparameters per base model"):
        for r in report["model_results"]:
            st.markdown(f"**{friendly_name(r['Model'])}** — CV F1: {r['CV F1']}")
            st.json(r["Best Params"])

# ------------------------------------------------------------------
# TAB 4 — DATASET OVERVIEW
# ------------------------------------------------------------------
with tab_data:
    st.subheader("Training Dataset Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", report["n_samples"])
    balance = report["class_balance"]
    normal_n = balance.get("0", balance.get(0, 0))
    failure_n = balance.get("1", balance.get(1, 0))
    c2.metric("Normal Cases", normal_n)
    c3.metric("Failure Cases", failure_n)

    fig_pie = px.pie(values=[normal_n, failure_n], names=["Normal", "Failure"],
                      color_discrete_sequence=["#3fb950", "#f85149"], template="plotly_dark", hole=0.45)
    fig_pie.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("**Model Features**")
    st.write("Numeric:", ", ".join(report["numeric_columns"]))
    st.write("Categorical:", ", ".join(report["categorical_columns"]))

    st.markdown("**Pipeline Summary**")
    st.markdown("""
    - Outlier capping via IQR clipping on numeric features
    - Ordinal encoding for the categorical `Type` feature
    - Engineered features: `Temp_diff`, `Power`, `Strain`
    - Base learners: Random Forest, Decision Tree, KNN, Logistic Regression (each grid-searched with 5-fold stratified CV)
    - Meta-model: Logistic Regression trained on out-of-fold base predictions (stacking, `passthrough=False`)
    - Identifier columns (`UDI`, `Product ID`) and failure-mode indicator columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) excluded to avoid leakage
    """)

st.divider()
st.caption("Predictive Maintenance Dashboard · Stacking Ensemble Model · For decision support — always confirm critical alerts with manual inspection.")
