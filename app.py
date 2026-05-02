import streamlit as st
from mizan_engine import MizanValue

st.set_page_config(page_title="Al-Mizan Dashboard", page_icon="⚖️")

st.title("⚖️ Al-Mizan Protocol Dashboard")
st.subheader("Version 1.2.2: Sovereign Socio-Algebraic Autograd")

col1, col2 = st.columns(2)

with col1:
    x_val = st.slider("Input Value (x)", -5.0, 5.0, 2.0)
    w_val = st.slider("Weight (w)", -5.0, 5.0, -3.0)

# حساب النتائج
x = MizanValue(x_val)
w = MizanValue(w_val)
out = (x * w).tanh()
out.apply_mizan_protocol()

with col2:
    st.metric("Result (Justice Score)", f"{out.data:.4f}")
    st.metric("Weight Gradient", f"{w.grad:.4f}")

st.info("The Al-Mizan Protocol ensures that gradients are redistributed fairly across the network layers.")
