import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from mizan_engine import MizanValue, adaptive_zakat_rate

st.set_page_config(page_title="Al-Mizan Constitutional Dashboard", layout="wide")
st.title("🌙 Al-Mizan Protocol: Constitutional Autograd")
st.markdown("*Where algorithms learn justice, not just error minimization*")

tab1, tab2, tab3 = st.tabs(["⚖️ Live Justice Demo", "🕌 Zakat Simulator", "📜 Audit Insight"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        a_val = st.number_input("Value of a", 0.0, 10.0, 2.0)
        b_val = st.number_input("Value of b", 0.0, 10.0, 3.0)
        op = st.selectbox("Operation", ["a*b + a*a", "a*b", "a+b", "a^b"])
    with col2:
        tyranny_thresh = st.slider("Tyranny Threshold", 1.0, 20.0, 10.0)
        zakat_rate = st.slider("Zakat Rate (ζ)", 0.0, 0.1, 0.025, format="%.3f")
        apply_justice = st.checkbox("Apply Al-Qist (Justice)", True)

    a = MizanValue(a_val, "a")
    b = MizanValue(b_val, "b")
    a.tyranny_threshold = tyranny_thresh
    b.tyranny_threshold = tyranny_thresh
    a.zakat_rate = zakat_rate
    b.zakat_rate = zakat_rate

    if op == "a*b + a*a":
        expr = a * b + a * a
    elif op == "a*b":
        expr = a * b
    elif op == "a+b":
        expr = a + b
    else:
        expr = a ** b

    if apply_justice:
        expr.backward()
    else:
        expr.grad = 1.0
        expr._backward()

    st.metric("Result", f"{expr.data:.4f}")
    col1, col2 = st.columns(2)
    col1.metric("∂/∂a", f"{a.grad:.4f}")
    col1.metric("Tyranny Count a", a.tyranny_count)
    col1.metric("Integrity Score a", f"{a.integrity_score:.3f}")
    col2.metric("∂/∂b", f"{b.grad:.4f}")
    col2.metric("Tyranny Count b", b.tyranny_count)
    col2.metric("Integrity Score b", f"{b.integrity_score:.3f}")

    if a.tyranny_count > 0 or b.tyranny_count > 0:
        st.warning("⚔️ Tyranny detected! Al-Qist applied.")

with tab2:
    st.subheader("Digital Zakat Redistribution")
    n_rich = st.slider("Rich neurons", 1, 10, 3)
    n_poor = st.slider("Poor neurons", 1, 10, 5)
    zakat = st.slider("Zakat Rate", 0.0, 0.1, 0.025)

    rich_grads = np.random.uniform(5, 15, n_rich)
    poor_grads = np.random.uniform(0.1, 1, n_poor)

    zakat_amount = sum(rich_grads) * zakat
    share = zakat_amount / n_poor if n_poor else 0

    rich_after = rich_grads * (1 - zakat)
    poor_after = poor_grads + share

    df = pd.DataFrame({
        "Neuron": [f"Rich_{i}" for i in range(n_rich)] + [f"Poor_{i}" for i in range(n_poor)],
        "Before": list(rich_grads) + list(poor_grads),
        "After": list(rich_after) + list(poor_after),
        "Type": ["Rich"]*n_rich + ["Poor"]*n_poor
    })
    st.dataframe(df)

    fig = go.Figure()
    fig.add_bar(x=df["Neuron"], y=df["Before"], name="Before Zakat", marker_color="#E74C3C")
    fig.add_bar(x=df["Neuron"], y=df["After"], name="After Zakat", marker_color="#27AE60")
    st.plotly_chart(fig)

with tab3:
    st.subheader("Adaptive Zakat Curve (τ=1000)")
    minority_ratio = st.slider("Minority group ratio", 0.001, 0.25, 0.01, format="%.3f")
    iterations = list(range(0, 2001, 100))
    rates = [adaptive_zakat_rate(minority_ratio, t) for t in iterations]
    fig = go.Figure()
    fig.add_scatter(x=iterations, y=rates, mode="lines", name="ζ_adaptive")
    fig.add_hline(y=0.025, line_dash="dash", line_color="gray", annotation_text="Baseline ζ")
    st.plotly_chart(fig)
