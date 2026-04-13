"""
Vendor Delivery Delay Predictor - Streamlit Web Application
===========================================================
Predicts whether a delivery will be delayed based on shipment/order details.
Built from the DataCo Supply Chain Dataset analysis notebook.

Models used: Logistic Regression, Random Forest, XGBoost
The BEST performing model is auto-selected and used for predictions.

Training happens ONLY ONCE on first launch. The trained model + artifacts
are persisted to disk (.pkl) and reloaded instantly on subsequent runs.
"""

import os
import random
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore")

# ── try importing xgboost (optional dependency) ──
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# ───────────────────────────── constants ──────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "DataCoSupplyChainDataset.csv")
ARTIFACTS_PATH = os.path.join(BASE_DIR, "model_artifacts.pkl")

TARGET = "Late_delivery_risk"

# Columns to drop during cleaning (matching notebook)
COLUMNS_TO_DROP_INITIAL = [
    "Customer Email", "Customer Fname", "Customer Id",
    "Customer Lname", "Customer Password", "Customer Street",
    "Customer Zipcode", "Order Zipcode", "Product Card Id",
    "Product Description", "Product Image", "Product Name",
    "Order Id", "Order Item Cardprod Id", "Order Customer Id",
    "Category Name", "Department Name", "Latitude", "Longitude",
    "Customer State",
]

COLUMNS_TO_DROP_FEATURE_SELECTION = [
    "Customer Country", "Customer Segment", "Market",
    "Delivery Status", "Customer City", "Order City",
    "Order Region", "Order State", "Order Status",
]

COLUMNS_TO_DROP_NUMERICAL = ["Order Item Discount Rate", "Product Status", "Order Item Id"]

COLUMNS_TO_DROP_FINAL = [
    "order date (DateOrders)", "shipping date (DateOrders)",
    "ship_day_of_week_name", "order_day_of_week_name",
    "ship_daypart", "order_daypart", "Order Country",
]

SHIPPING_MODE_ORDER = ["Standard Class", "Second Class", "First Class", "Same Day"]
TYPE_VALUES = ["DEBIT", "TRANSFER", "CASH", "PAYMENT"]

DAYPART_MAP = {
    "Early Morning": 0, "Morning": 1, "Noon": 2,
    "Eve": 3, "Night": 4, "Late Night": 5,
}


# ───────────────────────────── helpers ──────────────────────────────

def hour_to_daypart(h: int) -> str:
    if 4 < h <= 8:
        return "Early Morning"
    elif 8 < h <= 12:
        return "Morning"
    elif 12 < h <= 16:
        return "Noon"
    elif 16 < h <= 20:
        return "Eve"
    elif 20 < h <= 24:
        return "Night"
    else:
        return "Late Night"


def hour_to_daypart_n(h: int) -> int:
    return DAYPART_MAP[hour_to_daypart(h)]


# ──────────────────────── data loading ────────────────────────────

@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    """Load the raw CSV dataset."""
    return pd.read_csv(DATA_PATH, encoding="ISO-8859-1")


def _evaluate_model(model, X_test, y_test):
    """Return a dict of metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "confusion_matrix": cm, "report": report,
    }


@st.cache_resource(show_spinner="Training models (one-time only — saved for future runs)...")
def train_and_save_model():
    """
    Replicate notebook preprocessing, train 3 models (Logistic Regression,
    Random Forest, XGBoost), compare them, persist the BEST one.
    Returns artifacts dict.
    """
    # --- load ---
    data = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")

    # --- 1. Drop irrelevant columns ---
    cols_exist = [c for c in COLUMNS_TO_DROP_INITIAL if c in data.columns]
    data.drop(columns=cols_exist, inplace=True, errors="ignore")

    # --- 2. Handle missing values ---
    data.dropna(inplace=True)

    # --- 3. Feature-selection drops ---
    cols_exist = [c for c in COLUMNS_TO_DROP_FEATURE_SELECTION if c in data.columns]
    data.drop(columns=cols_exist, inplace=True, errors="ignore")

    # --- 4. Drop low-variance / non-informative numerical cols ---
    cols_exist = [c for c in COLUMNS_TO_DROP_NUMERICAL if c in data.columns]
    data.drop(columns=cols_exist, inplace=True, errors="ignore")

    # --- 5. Feature engineering (time features) ---
    data["shipping date (DateOrders)"] = pd.to_datetime(data["shipping date (DateOrders)"])
    data["order date (DateOrders)"] = pd.to_datetime(data["order date (DateOrders)"])

    data["Order_to_Shipment_Time"] = (
        (data["shipping date (DateOrders)"] - data["order date (DateOrders)"])
        .dt.total_seconds() / 3600
    ).astype(int)

    data["ship_day_of_week"] = data["shipping date (DateOrders)"].dt.dayofweek
    data["order_day_of_week"] = data["order date (DateOrders)"].dt.dayofweek

    day_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                 4: "Friday", 5: "Saturday", 6: "Sunday"}
    data["ship_day_of_week_name"] = data["ship_day_of_week"].map(day_names)
    data["order_day_of_week_name"] = data["order_day_of_week"].map(day_names)

    data["ship_hour"] = data["shipping date (DateOrders)"].dt.hour
    data["order_hour"] = data["order date (DateOrders)"].dt.hour

    data["ship_daypart"] = data["ship_hour"].apply(hour_to_daypart)
    data["order_daypart"] = data["order_hour"].apply(hour_to_daypart)

    data["ship_daypart_n"] = data["ship_daypart"].map(DAYPART_MAP)
    data["order_daypart_n"] = data["order_daypart"].map(DAYPART_MAP)

    # --- 6. Drop date / redundant categorical cols ---
    cols_exist = [c for c in COLUMNS_TO_DROP_FINAL if c in data.columns]
    data.drop(columns=cols_exist, inplace=True, errors="ignore")

    # --- 7. Split ---
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- 8. Encode categoricals ---
    ord_enc = OrdinalEncoder(categories=[SHIPPING_MODE_ORDER])
    X_train["Shipping Mode"] = ord_enc.fit_transform(X_train[["Shipping Mode"]])
    X_test["Shipping Mode"] = ord_enc.transform(X_test[["Shipping Mode"]])

    le_type = LabelEncoder()
    X_train["Type"] = le_type.fit_transform(X_train["Type"])
    X_test["Type"] = le_type.transform(X_test["Type"])

    # --- 9. Scale ---
    scaler = StandardScaler()
    feature_cols = X_train.columns.tolist()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 10. Train ALL models ---
    models = {}

    # (A) Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train_scaled, y_train)
    models["Logistic Regression"] = lr

    # (B) Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    models["Random Forest"] = rf

    # (C) XGBoost
    if HAS_XGBOOST:
        xgb = XGBClassifier(
            n_estimators=200, max_depth=10, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric="logloss",
            use_label_encoder=False,
        )
        xgb.fit(X_train_scaled, y_train)
        models["XGBoost"] = xgb

    # --- 11. Evaluate ALL models ---
    all_metrics = {}
    for name, model in models.items():
        all_metrics[name] = _evaluate_model(model, X_test_scaled, y_test)

    # --- 12. Pick BEST model by accuracy ---
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["accuracy"])
    best_model = models[best_name]

    # --- 13. Persist ---
    artifacts = {
        "best_model": best_model,
        "best_model_name": best_name,
        "all_models": models,
        "all_metrics": all_metrics,
        "scaler": scaler,
        "ord_enc": ord_enc,
        "le_type": le_type,
        "feature_cols": feature_cols,
        "type_classes": list(le_type.classes_),
    }
    joblib.dump(artifacts, ARTIFACTS_PATH)

    return artifacts


def load_or_train():
    """Load persisted artifacts or train from scratch."""
    if os.path.exists(ARTIFACTS_PATH):
        return joblib.load(ARTIFACTS_PATH)
    return train_and_save_model()


def preprocess_input(user_data: dict, artifacts: dict) -> np.ndarray:
    """Apply the exact same preprocessing pipeline to a single user-input record."""
    df = pd.DataFrame([user_data])
    df["Shipping Mode"] = artifacts["ord_enc"].transform(df[["Shipping Mode"]])
    df["Type"] = artifacts["le_type"].transform(df["Type"])
    df = df[artifacts["feature_cols"]]
    return artifacts["scaler"].transform(df)


# ───────────────────────────── UI ─────────────────────────────────

def build_ui():
    st.set_page_config(page_title="Vendor Delivery Delay Predictor", page_icon="📦", layout="wide")

    # ---------- custom CSS ----------
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }

    .header-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(14px);
        text-align: center;
    }
    .header-card h1 { color: #e0e0ff; font-size: 2.2rem; margin-bottom: .4rem; }
    .header-card p  { color: #a5a5d0; font-size: 1.05rem; }

    .result-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        backdrop-filter: blur(12px);
        margin-top: 1.5rem;
    }
    .result-delayed {
        background: rgba(255,60,60,0.15);
        border: 1px solid rgba(255,60,60,0.4);
    }
    .result-ontime {
        background: rgba(0,220,130,0.15);
        border: 1px solid rgba(0,220,130,0.4);
    }
    .result-card h2 { font-size: 1.8rem; margin: 0; }
    .result-card .prob { font-size: 1rem; color: #c0c0e0; margin-top: .4rem; }

    .metric-box {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: .8rem;
    }
    .metric-box .val { font-size: 1.8rem; color: #7dd3fc; font-weight: 700; }
    .metric-box .lbl { color: #a5a5d0; font-size: .85rem; }
    </style>
    """, unsafe_allow_html=True)

    # ───── header ─────
    st.markdown("""
    <div class="header-card">
        <h1>📦 Vendor Delivery Delay Predictor</h1>
        <p>Enter shipment &amp; order details to predict whether the delivery will be delayed.</p>
    </div>
    """, unsafe_allow_html=True)

    # ───── load model ─────
    artifacts = load_or_train()
    raw_data = load_data()

    best_name = artifacts["best_model_name"]
    all_metrics = artifacts["all_metrics"]
    best_metrics = all_metrics[best_name]

    # ───── sidebar ─────
    with st.sidebar:
        st.markdown("### ⚙️ Active Model")

        st.markdown(f"""
        <div class="metric-box">
            <div class="val">{best_name}</div>
            <div class="lbl">Best Model (auto-selected)</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-box">
            <div class="val">{best_metrics['accuracy']*100:.2f}%</div>
            <div class="lbl">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**Features used:** {len(artifacts['feature_cols'])}")
        st.markdown(f"**Dataset rows:** {len(raw_data):,}")
        st.caption("💾 Model trains once on first launch, then loads instantly from saved `.pkl` file.")

        st.divider()
        show_metrics = st.checkbox("📊 Show Performance Metrics", value=True)
        show_importance = st.checkbox("📈 Show Feature Importance")
        show_data = st.checkbox("🗂️ Show Dataset Preview")
        autofill = st.button("🎲 Autofill Sample Input", use_container_width=True)

    # ───── random sample from real data ─────
    def get_random_sample(data: pd.DataFrame) -> dict:
        """Pick a random row from the raw dataset and derive all needed fields."""
        row = data.sample(n=1, random_state=random.randint(0, 999999)).iloc[0]

        # parse dates for feature engineering
        try:
            ship_dt = pd.to_datetime(row.get("shipping date (DateOrders)", "2018-01-15 12:00:00"))
            order_dt = pd.to_datetime(row.get("order date (DateOrders)", "2018-01-13 10:00:00"))
        except Exception:
            ship_dt = pd.Timestamp("2018-01-15 12:00:00")
            order_dt = pd.Timestamp("2018-01-13 10:00:00")

        ship_hr = int(ship_dt.hour)
        order_hr = int(order_dt.hour)
        ots_hours = max(int((ship_dt - order_dt).total_seconds() / 3600), 0)

        return {
            "Days for shipping (real)": int(row.get("Days for shipping (real)", 3)),
            "Days for shipment (scheduled)": int(row.get("Days for shipment (scheduled)", 4)),
            "Benefit per order": float(row.get("Benefit per order", 20.0)),
            "Sales per customer": float(row.get("Sales per customer", 200.0)),
            "Category Id": int(row.get("Category Id", 73)),
            "Department Id": int(row.get("Department Id", 2)),
            "Order Item Discount": float(row.get("Order Item Discount", 10.0)),
            "Order Item Product Price": float(row.get("Order Item Product Price", 200.0)),
            "Order Item Profit Ratio": float(row.get("Order Item Profit Ratio", 0.1)),
            "Order Item Quantity": int(row.get("Order Item Quantity", 1)),
            "Sales": float(row.get("Sales", 200.0)),
            "Order Item Total": float(row.get("Order Item Total", 190.0)),
            "Order Profit Per Order": float(row.get("Order Profit Per Order", 20.0)),
            "Product Category Id": int(row.get("Product Category Id", 73)),
            "Product Price": float(row.get("Product Price", 200.0)),
            "Type": str(row.get("Type", "DEBIT")),
            "Shipping Mode": str(row.get("Shipping Mode", "Standard Class")),
            "Order_to_Shipment_Time": ots_hours,
            "ship_day_of_week": int(ship_dt.dayofweek),
            "order_day_of_week": int(order_dt.dayofweek),
            "ship_hour": ship_hr,
            "order_hour": order_hr,
            "ship_daypart_n": hour_to_daypart_n(ship_hr),
            "order_daypart_n": hour_to_daypart_n(order_hr),
        }

    if autofill:
        st.session_state["autofill"] = get_random_sample(raw_data)
    af = st.session_state.get("autofill", {})

    # ───── PERFORMANCE METRICS SECTION ─────
    if show_metrics:
        st.markdown("---")
        st.markdown("### 📊 Model Performance Comparison")
        st.caption("All three algorithms were trained and evaluated. The best one is auto-selected.")

        # comparison table
        comp_rows = []
        for name in all_metrics:
            m = all_metrics[name]
            comp_rows.append({
                "Model": name,
                "Accuracy (%)": round(m["accuracy"] * 100, 2),
                "Precision (%)": round(m["precision"] * 100, 2),
                "Recall (%)": round(m["recall"] * 100, 2),
                "F1-Score (%)": round(m["f1"] * 100, 2),
                "Selected": "✅" if name == best_name else "",
            })
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # detailed metrics for best model
        st.markdown(f"#### 🏆 Best Model: **{best_name}**")
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Accuracy", f"{best_metrics['accuracy']*100:.2f}%")
        with mc2:
            st.metric("Precision", f"{best_metrics['precision']*100:.2f}%")
        with mc3:
            st.metric("Recall", f"{best_metrics['recall']*100:.2f}%")
        with mc4:
            st.metric("F1-Score", f"{best_metrics['f1']*100:.2f}%")

        # confusion matrix
        st.markdown("#### Confusion Matrix")
        cm = best_metrics["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: On-Time", "Actual: Delayed"],
            columns=["Predicted: On-Time", "Predicted: Delayed"],
        )
        st.dataframe(cm_df, use_container_width=True)

        # per-class report
        st.markdown("#### Classification Report")
        report = best_metrics["report"]
        report_rows = []
        for label in ["0", "1"]:
            if label in report:
                report_rows.append({
                    "Class": "On-Time (0)" if label == "0" else "Delayed (1)",
                    "Precision": round(report[label]["precision"], 4),
                    "Recall": round(report[label]["recall"], 4),
                    "F1-Score": round(report[label]["f1-score"], 4),
                    "Support": int(report[label]["support"]),
                })
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

    # ───── input form ─────
    st.markdown("---")
    st.markdown("### 🔧 Enter Shipment Details")

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown("#### Shipping & Order Info")
        payment_type = st.selectbox("Payment Type", TYPE_VALUES,
                                    index=TYPE_VALUES.index(af.get("Type", "DEBIT")))
        shipping_mode = st.selectbox("Shipping Mode", SHIPPING_MODE_ORDER,
                                      index=SHIPPING_MODE_ORDER.index(af.get("Shipping Mode", "Standard Class")))
        days_real = st.number_input("Days for Shipping (Real)", 0, 30,
                                    value=af.get("Days for shipping (real)", 3))
        days_scheduled = st.number_input("Days for Shipment (Scheduled)", 0, 30,
                                          value=af.get("Days for shipment (scheduled)", 4))
        shipment_time = st.number_input("Order-to-Shipment Time (hours)", 0, 720,
                                         value=af.get("Order_to_Shipment_Time", 96))
        order_quantity = st.number_input("Order Item Quantity", 1, 100,
                                          value=af.get("Order Item Quantity", 1))

    with col_right:
        st.markdown("#### Financial & Product Info")
        sales = st.number_input("Sales ($)", 0.0, 50000.0, value=float(af.get("Sales", 200.0)), step=10.0)
        product_price = st.number_input("Product Price ($)", 0.0, 50000.0,
                                         value=float(af.get("Product Price", 200.0)), step=10.0)
        benefit = st.number_input("Benefit per Order ($)", -5000.0, 5000.0,
                                   value=float(af.get("Benefit per order", 20.0)), step=5.0)
        discount = st.number_input("Order Item Discount ($)", 0.0, 5000.0,
                                    value=float(af.get("Order Item Discount", 10.0)), step=1.0)
        profit_ratio = st.number_input("Order Item Profit Ratio", -1.0, 1.0,
                                        value=float(af.get("Order Item Profit Ratio", 0.1)), step=0.05)
        category_id = st.number_input("Category Id", 1, 100,
                                       value=af.get("Category Id", 73))

    st.markdown("#### 📅 Time Features")
    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        ship_dow = st.slider("Ship Day of Week (0=Mon)", 0, 6, value=af.get("ship_day_of_week", 3))
    with tc2:
        order_dow = st.slider("Order Day of Week (0=Mon)", 0, 6, value=af.get("order_day_of_week", 5))
    with tc3:
        ship_hr = st.slider("Ship Hour", 0, 23, value=af.get("ship_hour", 12))
    with tc4:
        order_hr = st.slider("Order Hour", 0, 23, value=af.get("order_hour", 10))

    # ───── predict ─────
    predict_btn = st.button("🚀 Predict Delivery Status", type="primary", use_container_width=True)

    if predict_btn:
        user_input = {
            "Type": payment_type,
            "Days for shipping (real)": days_real,
            "Days for shipment (scheduled)": days_scheduled,
            "Benefit per order": benefit,
            "Sales per customer": sales,
            "Category Id": category_id,
            "Department Id": af.get("Department Id", 2),
            "Order Item Discount": discount,
            "Order Item Product Price": product_price,
            "Order Item Profit Ratio": profit_ratio,
            "Order Item Quantity": order_quantity,
            "Sales": sales,
            "Order Item Total": sales - discount,
            "Order Profit Per Order": benefit,
            "Product Category Id": category_id,
            "Product Price": product_price,
            "Shipping Mode": shipping_mode,
            "Order_to_Shipment_Time": shipment_time,
            "ship_day_of_week": ship_dow,
            "order_day_of_week": order_dow,
            "ship_hour": ship_hr,
            "order_hour": order_hr,
            "ship_daypart_n": hour_to_daypart_n(ship_hr),
            "order_daypart_n": hour_to_daypart_n(order_hr),
        }

        X_in = preprocess_input(user_input, artifacts)
        model = artifacts["best_model"]

        prediction = model.predict(X_in)[0]
        proba = model.predict_proba(X_in)[0]

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card result-delayed">
                <h2>⚠️ Delivery Likely <span style="color:#ff5555;">DELAYED</span></h2>
                <p class="prob">Probability of delay: <strong>{proba[1]*100:.1f}%</strong>
                &nbsp;|&nbsp; Predicted by: <strong>{best_name}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-ontime">
                <h2>✅ Delivery Likely <span style="color:#00dc82;">ON-TIME</span></h2>
                <p class="prob">Probability of on-time: <strong>{proba[0]*100:.1f}%</strong>
                &nbsp;|&nbsp; Predicted by: <strong>{best_name}</strong></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Confidence Breakdown")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("On-Time Probability", f"{proba[0]*100:.1f}%")
        with col_b:
            st.metric("Delay Probability", f"{proba[1]*100:.1f}%")

    # ───── Feature Importance ─────
    if show_importance:
        st.markdown("---")
        st.markdown("### 📈 Feature Importance")
        best_model = artifacts["best_model"]
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
            feat_imp = pd.DataFrame({
                "Feature": artifacts["feature_cols"],
                "Importance": importances
            }).sort_values("Importance", ascending=True).tail(15)
            st.bar_chart(feat_imp.set_index("Feature"))
        else:
            st.info("Feature importance is not available for Logistic Regression. "
                     "It uses model coefficients instead.")
            coef = best_model.coef_[0]
            feat_imp = pd.DataFrame({
                "Feature": artifacts["feature_cols"],
                "Coefficient (abs)": np.abs(coef)
            }).sort_values("Coefficient (abs)", ascending=True).tail(15)
            st.bar_chart(feat_imp.set_index("Feature"))

    # ───── Dataset Preview ─────
    if show_data:
        st.markdown("---")
        st.markdown("### 🗂️ Dataset Preview")
        st.dataframe(raw_data.head(100), use_container_width=True)


# ──────────────────────────── main ────────────────────────────────
if __name__ == "__main__":
    build_ui()
