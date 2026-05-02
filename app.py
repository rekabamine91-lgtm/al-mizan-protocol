"""
app.py - Al-Mizan Protocol Interactive Dashboard
Streamlit-based UI for Constitutional Autograd visualization

Author: Amine Rekab (@rekabamine91-lgtm)
Version: 1.0.0 (June 2026)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
from almizan.engine import MizanValue, adaptive_zakat_rate
from almizan.zakat_manager import ZakatOrchestrator

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Al-Mizan Protocol - Constitutional AI Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS for better visual appearance
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        font-family: 'Traditional Arabic', serif;
    }
    .constitutional-badge {
        background-color: #2E86AB;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        display: inline-block;
        font-size: 0.8rem;
    }
    .integrity-high {
        background-color: #27AE60;
        color: white;
        padding: 3px 8px;
        border-radius: 10px;
    }
    .integrity-low {
        background-color: #E74C3C;
        color: white;
        padding: 3px 8px;
        border-radius: 10px;
    }
    .justice-box {
        border: 2px solid #2E86AB;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Header
# ============================================================================
st.markdown("<h1 class='main-header'>⚖️ Al-Mizan Protocol: Constitutional AI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("*Where algorithms learn justice, not just error minimization*")

st.markdown("""
<div class='justice-box'>
    <b>📜 Constitutional Foundations</b><br>
    <i>"Al-Mizan is not a policy; it is a mathematical guarantee of Digital Sovereignty."</i><br><br>
    This interactive dashboard demonstrates how Al-Mizan enforces <b>Al-Qist</b> (anti-tyranny constraint) 
    and <b>Digital Zakat</b> (knowledge redistribution) inside the neural network's backward pass.
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Sidebar - Constitutional Settings
# ============================================================================
st.sidebar.header("🛠️ Constitutional Settings")
st.sidebar.markdown("---")

tau_tyr = st.sidebar.slider(
    "⚔️ Anti-Tyranny Threshold (τ_tyr)",
    min_value=1.0, max_value=20.0, value=10.0, step=0.5,
    help="Gradients exceeding this threshold are considered 'tyrannical' and dampened."
)

zakat_rate = st.sidebar.slider(
    "🕌 Digital Zakat Rate (ζ)",
    min_value=0.01, max_value=0.10, value=0.025, step=0.005,
    help="Percentage of gradient wealth redistributed from rich to poor neurons (default 2.5%)."
)

poverty_threshold = st.sidebar.slider(
    "📉 Poverty (Entropy) Threshold (θ)",
    min_value=0.1, max_value=1.0, value=0.5, step=0.05,
    help="Neurons with entropy below this threshold are considered 'poor' and receive zakat."
)

alpha = st.sidebar.number_input(
    "α (Tyranny Count Decay)",
    min_value=0.01, max_value=0.5, value=0.1, step=0.01,
    help="Weight for tyranny count in integrity score formula."
)

beta = st.sidebar.number_input(
    "β (Gradient Variance Decay)",
    min_value=0.01, max_value=0.2, value=0.05, step=0.01,
    help="Weight for gradient variance in integrity score formula."
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**📐 Mathematical Foundations**

Integrity Score:
$$I(v) = \\frac{1}{1 + \\alpha \\cdot TC + \\beta \\cdot Var(G)}$$

Adaptive Zakat:
$$\\zeta_{adaptive}(t) = \\zeta_{base} \\left(1 + e^{-r_{min} \\cdot t/\\tau}\\right)$$
""")

# ============================================================================
# Main Simulation Area
# ============================================================================
st.header("🔄 Real-time Constitutional Learning Cycle")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    This simulation trains a small neural network (10 neurons) with:
    - **Random raw gradients** (some intentionally exceed τ_tyr to trigger Al-Qist)
    - **Constitutional backward pass** with Al-Qist dampening
    - **Digital Zakat** redistribution from rich to poor neurons
    - **Full audit trail** for every justice event
    """)

with col2:
    n_neurons = st.number_input("Number of Neurons", min_value=5, max_value=20, value=10, step=1)
    n_steps = st.number_input("Training Steps", min_value=3, max_value=15, value=8, step=1)

# ============================================================================
# Run Simulation Button
# ============================================================================
if st.button("🚀 Start Constitutional Learning Cycle", type="primary", use_container_width=True):
    
    # Initialize orchestrator
    orchestrator = ZakatOrchestrator(
        poverty_threshold=poverty_threshold,
        base_rate=zakat_rate,
        tau_step=1000
    )
    
    # Create neurons (MizanValue instances)
    neurons = []
    for i in range(n_neurons):
        # Initialize with small random data
        data = np.random.uniform(-1, 1)
        neuron = MizanValue(
            data, 
            label=f"N{i}", 
            tau_tyr=tau_tyr,
            alpha=alpha,
            beta=beta,
            zakat_rate_base=zakat_rate
        )
        neurons.append(neuron)
    
    # Simulation history
    history = []
    gradients_before = []
    gradients_after = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for step in range(n_steps):
        status_text.markdown(f"**Step {step+1}/{n_steps}** – Computing gradients and applying constitutional justice...")
        
        # Simulate raw gradients (some intentionally large to test tyranny detection)
        step_grads_before = []
        step_grads_after = []
        
        for n in neurons:
            # Generate raw gradient (some large, some small)
            if np.random.random() < 0.3:  # 30% chance of "tyrannical" gradient
                raw_grad = np.random.uniform(tau_tyr + 5, tau_tyr + 20)
            else:
                raw_grad = np.random.uniform(-5, 5)
            
            n.grad = raw_grad
            step_grads_before.append(abs(raw_grad))
            
            # Apply Al-Qist (anti-tyranny)
            n.apply_al_qist()
            step_grads_after.append(abs(n.grad))
        
        gradients_before.append(step_grads_before)
        gradients_after.append(step_grads_after)
        
        # Identify minority mask (for adaptive zakat)
        # Simulate some neurons as "minority" based on entropy
        entropies = [n.compute_entropy() for n in neurons]
        minority_mask = [e < poverty_threshold for e in entropies]
        minority_ratio = sum(minority_mask) / len(neurons)
        
        # Calculate adaptive zakat rate
        current_zakat_rate = adaptive_zakat_rate(minority_ratio, step, tau=1000, base_rate=zakat_rate)
        
        # Perform collective redistribution (Zakat)
        orchestrator.collect_and_distribute(neurons, minority_mask, current_zakat_rate)
        
        # Record history
        for n in neurons:
            history.append({
                "Step": step,
                "Label": n.label,
                "Integrity Score": n.integrity_score,
                "Tyranny Count": n.tyranny_count,
                "Gradient After Justice": n.grad,
                "Zakat Received": n.zakat_received,
                "Zakat Given": n.zakat_given,
                "Entropy": n.compute_entropy(),
                "Is Poor": n.is_poor(poverty_threshold)
            })
        
        progress_bar.progress((step + 1) / n_steps)
        time.sleep(0.2)
    
    status_text.markdown("✅ **Constitutional learning cycle complete!**")
    
    # ========================================================================
    # Display Results
    # ========================================================================
    df = pd.DataFrame(history)
    
    st.markdown("---")
    st.header("📊 Simulation Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_tyranny = df["Tyranny Count"].sum()
        st.metric("⚔️ Total Tyranny Events", int(total_tyranny))
    with col2:
        avg_integrity = df["Integrity Score"].mean()
        st.metric("🌙 Avg Integrity Score", f"{avg_integrity:.3f}")
    with col3:
        total_zakat_given = df["Zakat Given"].sum()
        st.metric("🕌 Total Zakat Given", f"{total_zakat_given:.4f}")
    with col4:
        total_zakat_received = df["Zakat Received"].sum()
        st.metric("💝 Total Zakat Received", f"{total_zakat_received:.4f}")
    
    # Integrity score over time
    st.subheader("🛡️ Integrity Score Evolution")
    st.markdown("*Integrity scores decrease when neurons exhibit tyrannical behavior (|grad| > τ_tyr)*")
    
    pivot_integrity = df.pivot(index="Step", columns="Label", values="Integrity Score")
    st.line_chart(pivot_integrity)
    
    # Final zakat distribution
    st.subheader("🕌 Digital Zakat Distribution (Final Step)")
    st.markdown("*Poor neurons (low entropy) receive redistributed gradient capital from rich neurons*")
    
    final_step = df[df["Step"] == n_steps - 1]
    zakat_df = final_step[["Label", "Zakat Received", "Zakat Given", "Is Poor"]].set_index("Label")
    st.bar_chart(zakat_df[["Zakat Received", "Zakat Given"]])
    
    # Tyranny count by neuron
    st.subheader("⚔️ Tyranny Count by Neuron")
    st.markdown("*Neurons that repeatedly exceed the tyranny threshold accumulate counts and lose integrity*")
    
    tyranny_df = df.groupby("Label")["Tyranny Count"].max().reset_index()
    tyranny_df.columns = ["Neuron", "Tyranny Count"]
    st.bar_chart(tyranny_df.set_index("Neuron"))
    
    # Gradient dampening visualization
    st.subheader("📉 Gradient Dampening Effect (Al-Qist)")
    st.markdown("*How Al-Qist reduces tyrannical gradients*")
    
    # Aggregate gradient before/after
    avg_before = [np.mean(g) for g in gradients_before]
    avg_after = [np.mean(g) for g in gradients_after]
    
    dampen_df = pd.DataFrame({
        "Step": range(n_steps),
        "Before Al-Qist": avg_before,
        "After Al-Qist": avg_after
    })
    st.line_chart(dampen_df.set_index("Step"))
    
    # Entropy distribution (poor vs rich)
    st.subheader("📊 Information Entropy Distribution")
    st.markdown("*Neurons with entropy < θ = {} are classified as 'poor' and receive zakat*".format(poverty_threshold))
    
    entropy_df = df[df["Step"] == n_steps - 1][["Label", "Entropy", "Is Poor"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#E74C3C" if poor else "#27AE60" for poor in entropy_df["Is Poor"]]
    ax.bar(entropy_df["Label"], entropy_df["Entropy"], color=colors)
    ax.axhline(y=poverty_threshold, color="#2E86AB", linestyle="--", label=f"Poverty Threshold (θ={poverty_threshold})")
    ax.set_ylabel("Entropy H(i)")
    ax.set_title("Neuron Information Entropy")
    ax.legend()
    st.pyplot(fig)
    
    # ========================================================================
    # Audit Trail
    # ========================================================================
    st.divider()
    with st.expander("📜 View Complete Constitutional Audit Trail", expanded=False):
        st.dataframe(df)
        
        # Detailed audit per neuron
        st.subheader("🔍 Per-Neuron Audit Details")
        for neuron_label in df["Label"].unique():
            neuron_df = df[df["Label"] == neuron_label]
            with st.expander(f"Neuron: {neuron_label}"):
                st.write(f"**Final Integrity Score:** {neuron_df['Integrity Score'].iloc[-1]:.4f}")
                st.write(f"**Total Tyranny Count:** {neuron_df['Tyranny Count'].iloc[-1]}")
                st.write(f"**Zakat Given:** {neuron_df['Zakat Given'].iloc[-1]:.4f}")
                st.write(f"**Zakat Received:** {neuron_df['Zakat Received'].iloc[-1]:.4f}")
                st.write(f"**Final Entropy:** {neuron_df['Entropy'].iloc[-1]:.4f}")
                st.write(f"**Is Poor:** {neuron_df['Is Poor'].iloc[-1]}")
    
    # ========================================================================
    # Conclusion
    # ========================================================================
    st.divider()
    st.success(f"""
    ✅ **Constitutional Learning Cycle Completed Successfully**
    
    - **Al-Qist** applied: Tyrannical gradients were automatically dampened ({int(total_tyranny)} events detected)
    - **Digital Zakat** distributed: {total_zakat_given:.4f} gradient units redistributed from rich to poor neurons
    - **Integrity Scores** adjusted dynamically based on behavioral history
    - **Full Audit Trail** recorded for transparency and legal compliance
    """)
    
else:
    st.info("👆 Press the button above to start the constitutional learning simulation and witness Al-Qist and Digital Zakat in action.")
    
    # Show preview of what will happen
    st.markdown("""
    ### 🧪 What to Expect
    
    When you run the simulation, you will see:
    
    1. **Random raw gradients** are generated for each neuron
    2. **Al-Qist** detects any gradient exceeding `τ_tyr` and dampens it using the neuron's integrity score
    3. **Digital Zakat** redistributes gradient wealth from "rich" (high-entropy) to "poor" (low-entropy) neurons
    4. **Integrity scores** evolve over time based on each neuron's tyranny history
    5. **Full audit trail** is recorded for every justice event
    
    The dashboard visualizes:
    - Integrity score decay for tyrannical neurons
    - Zakat redistribution amounts
    - Entropy-based poverty classification
    - Gradient dampening effectiveness
    """)

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
