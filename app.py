"""
app.py - Al-Mizan Protocol Interactive Dashboard
============================================================================
Streamlit-based dashboard for visualizing constitutional justice in action.
Tracks gradient redistribution, audit trails, and real-time fairness metrics.

Author: Amine Rekab (@rekabamine91-lgtm)
Version: 1.0.0 (June 2026)
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from almizan import MizanValue, ZakatOrchestrator

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Al-Mizan Protocol - Constitutional Justice Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS for better styling
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #c5a059;
        text-align: center;
        font-family: 'Traditional Arabic', serif;
    }
    .constitutional-badge {
        background-color: #c5a059;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        display: inline-block;
        font-size: 0.8rem;
    }
    .justice-quote {
        background-color: #1e1e2e;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #c5a059;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Initialize Session State
# ============================================================================
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = ZakatOrchestrator(
        poverty_threshold=0.4,
        base_rate=0.025,
        tau_step=1000,
        use_entropy=True
    )
if 'iteration' not in st.session_state:
    st.session_state.iteration = 0
if 'history' not in st.session_state:
    st.session_state.history = []
if 'auto_run' not in st.session_state:
    st.session_state.auto_run = False

# ============================================================================
# Header
# ============================================================================
st.markdown("<h1 class='main-header'>⚖️ Al-Mizan Protocol: Digital Justice Dashboard</h1>", 
            unsafe_allow_html=True)

st.markdown("""
<div class='justice-quote'>
    <i>"Al-Mizan is not a policy; it is a mathematical guarantee of Digital Sovereignty."</i><br>
    This dashboard monitors how "learning capital" (gradients) is redistributed between 
    rich and poor neurons to prevent algorithmic tyranny.
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Sidebar - Constitutional Settings
# ============================================================================
st.sidebar.header("⚙️ Constitutional Settings")

poverty_thresh = st.sidebar.slider(
    "Poverty Threshold (θ)", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.4, 
    step=0.05,
    help="Neurons with entropy below this threshold receive zakat."
)

zakat_rate = st.sidebar.slider(
    "Base Zakat Rate (ζ)", 
    min_value=0.01, 
    max_value=0.10, 
    value=0.025, 
    step=0.005,
    help="Percentage of gradient redistributed from rich to poor (2.5% = Islamic zakat)."
)

use_entropy = st.sidebar.checkbox(
    "Use Entropy for Classification", 
    value=True,
    help="Use information entropy to classify rich/poor neurons (vs gradient magnitude)."
)

st.session_state.orchestrator.poverty_threshold = poverty_thresh
st.session_state.orchestrator.base_rate = zakat_rate
st.session_state.orchestrator.use_entropy = use_entropy

st.sidebar.markdown("---")

# Reset button
if st.sidebar.button("♻️ Reset Simulation", type="secondary", use_container_width=True):
    st.session_state.orchestrator.reset()
    st.session_state.iteration = 0
    st.session_state.history = []
    st.session_state.auto_run = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.write("👤 **Developer:** Amine Rekab")
st.sidebar.write("🆔 **Version:** 1.0.0")
st.sidebar.write("📄 **License:** Apache 2.0")

# ============================================================================
# Justice Cycle Function
# ============================================================================
def run_justice_cycle(num_neurons: int = 20) -> tuple:
    """
    Simulate a single justice cycle:
    1. Create neurons with random data
    2. Assign unfair raw gradients (some very high, some very low)
    3. Apply Zakat redistribution
    4. Return neurons and collected amount
    """
    st.session_state.iteration += 1
    
    neurons = []
    for i in range(num_neurons):
        n = MizanValue(data=np.random.randn(), label=f"N{i}")
        
        # Simulate unfair raw gradients
        if i < 5:  # First 5 neurons are "rich" (high gradients)
            n.grad = np.random.exponential(3.0)
        else:      # Remaining neurons are "poor" (low gradients)
            n.grad = np.random.uniform(0, 0.3)
        
        # Simulate gradient history for entropy calculation
        if i < 5:
            n.grad_history = [np.random.uniform(5, 15) for _ in range(10)]
        else:
            n.grad_history = [np.random.uniform(0, 0.5) for _ in range(10)]
        
        neurons.append(n)
    
    # Apply Digital Zakat
    collected = st.session_state.orchestrator.collect_and_distribute(neurons)
    
    return neurons, collected

# ============================================================================
# Run Justice Cycle Button
# ============================================================================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("⚖️ Apply Constitutional Justice", type="primary", use_container_width=True):
        neurons, last_collected = run_justice_cycle()
        st.session_state.last_neurons = neurons
        st.session_state.last_collected = last_collected
        st.rerun()

# ============================================================================
# Display Results
# ============================================================================
if 'last_neurons' in st.session_state:
    stats = st.session_state.orchestrator.get_statistics()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total Zakat Collected", f"{stats.get('total_collected', 0):.4f}")
    with col2:
        st.metric("📊 Current Iteration", st.session_state.iteration)
    with col3:
        if 'last_event' in stats and isinstance(stats['last_event'], dict):
            rich = stats['last_event'].get('rich_count', 0)
            poor = stats['last_event'].get('poor_count', 0)
            st.metric("⚖️ Rich : Poor Ratio", f"{rich} : {poor}")
    with col4:
        avg_rate = stats.get('average_zakat_rate', 0)
        st.metric("🕌 Avg Zakat Rate", f"{avg_rate:.3%}")
    
    st.divider()
    
    # Charts row
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📊 Gradient Distribution (After Justice)")
        grad_data = pd.DataFrame({
            "Neuron": [n.label for n in st.session_state.last_neurons],
            "Gradient Magnitude": [abs(n.grad) for n in st.session_state.last_neurons]
        })
        st.bar_chart(grad_data.set_index("Neuron"), color="#c5a059", height=400)
        
        # Add explanation
        st.caption("Blue bars show gradient magnitude after Zakat redistribution. "
                   "Notice the balance between rich and poor neurons.")
    
    with col_right:
        st.subheader("📜 Audit Trail (Live)")
        logs = st.session_state.orchestrator.get_audit_trail()
        if logs:
            df_logs = pd.DataFrame(logs).tail(10)
            display_cols = ['iteration', 'collected', 'rich_count', 'poor_count']
            if all(c in df_logs.columns for c in display_cols):
                st.dataframe(df_logs[display_cols], hide_index=True, use_container_width=True)
            
            # Last event details
            last_event = logs[-1]
            st.info(f"""
            **Last Justice Event:**
            - Iteration: {last_event.get('iteration', '-')}
            - Zakat Rate: {last_event.get('zakat_rate', 0):.3%}
            - Collected: {last_event.get('collected', 0):.4f}
            - Rich: {last_event.get('rich_count', 0)} → Poor: {last_event.get('poor_count', 0)}
            """)
    
    st.divider()
    
    # ========================================================================
    # Mathematical Analysis Section
    # ========================================================================
    with st.expander("📐 View Mathematical Integrity Analysis", expanded=False):
        st.subheader("Constitutional Formulas")
        
        st.latex(r"I(v) = \frac{1}{1 + \alpha \cdot TC + \beta \cdot \text{Var}(G)}")
        st.caption("**Integrity Score:** As Tyranny Count (TC) increases, the neuron's influence is dampened.")
        
        st.latex(r"\zeta_{\text{adaptive}}(t) = \zeta_{\text{base}} \left(1 + \exp\left(-\frac{|S_{\text{min}}|}{|S_{\text{total}}|} \cdot \frac{t}{\tau}\right)\right)")
        st.caption("**Adaptive Zakat Rate:** Dynamic adjustment for minority protection (τ = 1000).")
        
        st.latex(r"H(i) = -\sum_k p_k(a_i) \log p_k(a_i)")
        st.caption("**Information Entropy:** Neurons with H(i) < θ are 'poor' and receive zakat.")
        
        st.divider()
        
        # Current statistics
        st.subheader("Current Statistics")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Zakat Statistics:**")
            st.write(f"- Total Collected: {stats.get('total_collected', 0):.4f}")
            st.write(f"- Total Distributed: {stats.get('total_distributed', 0):.4f}")
            st.write(f"- Number of Events: {stats.get('num_events', 0)}")
        with col_b:
            if 'last_event' in stats and isinstance(stats['last_event'], dict):
                st.write("**Last Event Details:**")
                st.write(f"- Zakat Rate: {stats['last_event'].get('zakat_rate', 0):.4f}")
                st.write(f"- Rich Neurons: {stats['last_event'].get('rich_count', 0)}")
                st.write(f"- Poor Neurons: {stats['last_event'].get('poor_count', 0)}")
        
        # Sample neuron audit
        if 'last_neurons' in st.session_state:
            st.subheader("Sample Neuron Audit")
            sample_neuron = st.session_state.last_neurons[0]
            st.json(sample_neuron.get_audit_trail())
    
else:
    st.info("👆 Click the **'Apply Constitutional Justice'** button above to start the simulation and witness Al-Qist and Digital Zakat in action.")
    
    # Preview section
    st.markdown("""
    ### 🧪 What to Expect
    
    When you run the simulation, the dashboard will:
    
    1. **Generate 20 neurons** with unfair raw gradient distribution (rich vs poor)
    2. **Classify neurons** using information entropy (H(i) ≥ θ = rich, H(i) < θ = poor)
    3. **Collect Digital Zakat** (ζ = 2.5%) from rich neurons
    4. **Redistribute equally** to poor neurons
    5. **Display visualizations** showing the balanced outcome
    6. **Provide audit trail** of every justice event
    
    The result is a **decentralized learning process** where no single neuron dominates.
    """)

# ============================================================================
# Auto-Simulation (optional)
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Automation")

auto_run = st.sidebar.checkbox("▶️ Auto-Simulation Mode", value=st.session_state.auto_run)
if auto_run != st.session_state.auto_run:
    st.session_state.auto_run = auto_run
    st.rerun()

if st.session_state.auto_run:
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    for i in range(5):
        status_text.text(f"Running cycle {i+1}/5...")
        neurons, collected = run_justice_cycle()
        st.session_state.last_neurons = neurons
        st.session_state.last_collected = collected
        progress_bar.progress((i + 1) / 5)
        time.sleep(1)
    
    status_text.text("✅ Auto-simulation complete!")
    st.sidebar.session_state.auto_run = False
    st.rerun()

# ============================================================================
# Footer
# ============================================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <b>Al-Mizan Protocol v1.0.0</b> — Constitutional Autograd for Digital Sovereignty<br>
    <i>"Al-Mizan is not a policy; it is a mathematical guarantee."</i><br>
    <a href="https://github.com/rekabamine91-lgtm/al-mizan-protocol" target="_blank">GitHub Repository</a> | 
    <a href="#" target="_blank">Whitepaper (arXiv)</a>
</div>
""", unsafe_allow_html=True)
