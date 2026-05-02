"""
app.py - Al-Mizan Protocol Interactive Dashboard
Streamlit-based UI for visualizing constitutional autograd in action
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from mizan_engine import MizanValue, JusticeLayer, AlMizanMLP, adaptive_zakat_rate
import time

# Page configuration
st.set_page_config(
    page_title="Al-Mizan Protocol - Constitutional Autograd",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Arabic flair
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        font-family: 'Traditional Arabic', serif;
    }
    .justice-badge {
        background-color: #2E86AB;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        display: inline-block;
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
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/your-repo/main/assets/logo.png", 
             use_container_width=True)
    st.markdown("## 🏛️ Al-Mizan Protocol")
    st.markdown("*Constitutional Autograd for Digital Sovereignty*")
    st.markdown("---")
    
    st.markdown("### ⚖️ Core Principles")
    st.markdown("""
    - **Al-Qist**: Anti-tyranny constraint  
    - **Digital Zakat**: Knowledge redistribution  
    - **Integrity Score**: Self-regulation  
    - **Adaptive Zakat**: Minority protection
    """)
    
    st.markdown("---")
    st.markdown("### 📜 References")
    st.markdown("""
    - [Whitepaper v1.2.2](https://arxiv.org/abs/xxxx.xxxxx)
    - [GitHub Repository](https://github.com/rekabamine91-lgtm/al-mizan-protocol)
    """)

# Main header
st.markdown("<h1 class='main-header'>🌙 Al-Mizan Constitutional Autograd</h1>", unsafe_allow_html=True)
st.markdown("*Where algorithms learn justice, not just error minimization*")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎮 Live Demo", 
    "📊 Justice Visualization", 
    "🕌 Zakat Simulation",
    "📜 Audit Trail",
    "🏛️ About"
])

# ============================================================================
# TAB 1: Live Demo
# ============================================================================
with tab1:
    st.header("🎮 Interactive Constitutional Computation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Define Variables")
        a_val = st.number_input("Value of a", value=2.0, step=0.5)
        b_val = st.number_input("Value of b", value=3.0, step=0.5)
        
        st.subheader("Expression")
        expression = st.selectbox(
            "Choose operation",
            ["a * b + a * a", "a * b", "a + b", "a ** b"]
        )
    
    with col2:
        st.subheader("Constitutional Parameters")
        tyranny_threshold = st.slider("Tyranny Threshold", 1.0, 20.0, 10.0, 0.5)
        zakat_rate = st.slider("Zakat Rate (ζ)", 0.0, 0.1, 0.025, 0.005, format="%.3f")
        apply_justice = st.checkbox("✅ Apply Al-Qist (Justice)", value=True)
    
    # Create variables
    a = MizanValue(a_val, label="a")
    b = MizanValue(b_val, label="b")
    a.tyranny_threshold = tyranny_threshold
    b.tyranny_threshold = tyranny_threshold
    a.zakat_rate = zakat_rate
    b.zakat_rate = zakat_rate
    
    # Compute
    if expression == "a * b + a * a":
        c = a * b + a * a
        formula = "a·b + a²"
    elif expression == "a * b":
        c = a * b
        formula = "a·b"
    elif expression == "a + b":
        c = a + b
        formula = "a + b"
    else:
        c = a ** b
        formula = "a^b"
    
    # Forward pass
    result = c.data
    
    # Backward pass
    if apply_justice:
        c.backward()
    else:
        # Standard backward without justice (for comparison)
        c.grad = 1.0
        c._backward()
    
    # Display results
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric("📦 Variable a", f"{a_val:.3f}")
        st.metric("📈 Gradient ∂c/∂a", f"{a.grad:.4f}")
        st.metric("⚔️ Tyranny Count", a.tyranny_count)
        st.metric("🌙 Integrity Score", f"{a.integrity_score:.3f}")
    
    with col_b:
        st.metric("📦 Variable b", f"{b_val:.3f}")
        st.metric("📈 Gradient ∂c/∂b", f"{b.grad:.4f}")
        st.metric("⚔️ Tyranny Count", b.tyranny_count)
        st.metric("🌙 Integrity Score", f"{b.integrity_score:.3f}")
    
    with col_c:
        st.metric(f"🎯 Result: {formula}", f"{result:.4f}")
        
        # Justice indicator
        if apply_justice:
            if a.tyranny_count > 0 or b.tyranny_count > 0:
                st.markdown("<span class='justice-badge'>⚖️ Justice Applied (Al-Qist)</span>", 
                           unsafe_allow_html=True)
            else:
                st.markdown("<span class='justice-badge'>✅ No Tyranny Detected</span>", 
                           unsafe_allow_html=True)
        else:
            st.markdown("<span class='justice-badge'>⚠️ Justice Disabled</span>", 
                       unsafe_allow_html=True)
    
    # Tyranny warning
    if a.tyranny_count > 2 or b.tyranny_count > 2:
        st.warning("⚔️ Tyranny detected! The network is self-correcting via Al-Qist.")
    
    st.markdown("---")
    st.caption("Al-Qist automatically dampens gradients that exceed the tyranny threshold")

# ============================================================================
# TAB 2: Justice Visualization
# ============================================================================
with tab2:
    st.header("📊 Gradient Justice Visualization")
    
    st.markdown("""
    This visualization compares **standard gradient descent** (used in PyTorch/micrograd) 
    with **Al-Mizan's Constitutional Justice**.
    """)
    
    # Simulation parameters
    n_iterations = st.slider("Number of iterations", 10, 100, 50)
    tyranny_threshold_viz = st.slider("Tyranny Threshold", 1.0, 10.0, 5.0, 0.5, key="viz_threshold")
    
    # Run simulation
    standard_grads = []
    al_mizan_grads = []
    integrity_scores = []
    
    # Create a variable
    var = MizanValue(1.0, label="test")
    var.tyranny_threshold = tyranny_threshold_viz
    
    for i in range(n_iterations):
        # Simulate gradient with occasional spikes
        if i % 10 == 0 and i > 0:
            raw_grad = 15.0  # Tyranny spike
        else:
            raw_grad = np.random.randn() * 2
        
        # Standard (no justice)
        standard_grads.append(raw_grad)
        
        # Al-Mizan with justice
        var.grad = raw_grad
        adjusted = var.apply_al_qist()
        al_mizan_grads.append(adjusted)
        integrity_scores.append(var.integrity_score)
    
    # Create plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=standard_grads,
        mode='lines+markers',
        name='Standard Gradient (PyTorch/micrograd)',
        line=dict(color='#E74C3C', width=2),
        marker=dict(size=4)
    ))
    
    fig.add_trace(go.Scatter(
        y=al_mizan_grads,
        mode='lines+markers',
        name='Al-Mizan (with Al-Qist)',
        line=dict(color='#27AE60', width=2),
        marker=dict(size=4)
    ))
    
    fig.add_trace(go.Scatter(
        y=[tyranny_threshold_viz] * n_iterations,
        mode='lines',
        name='Tyranny Threshold',
        line=dict(color='#F39C12', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Gradient Behavior: Standard vs. Constitutional",
        xaxis_title="Iteration",
        yaxis_title="Gradient Magnitude",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Integrity score over time
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=integrity_scores,
        mode='lines',
        name='Integrity Score',
        line=dict(color='#2E86AB', width=2),
        fill='tozeroy'
    ))
    
    fig2.update_layout(
        title="Variable Integrity Score Over Time",
        xaxis_title="Iteration",
        yaxis_title="Integrity Score (1.0 = fully trusted)",
        height=300
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    st.info("💡 **Insight**: When the standard gradient spikes (tyranny), Al-Mizan automatically dampens it and reduces the variable's integrity score, preventing algorithmic tyranny.")

# ============================================================================
# TAB 3: Zakat Simulation
# ============================================================================
with tab3:
    st.header("🕌 Digital Zakat Simulation")
    
    st.markdown("""
    Digital Zakat redistributes learning capital from **information-rich** neurons 
    to **information-poor** neurons, preventing knowledge monopolies.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_rich = st.number_input("Number of Rich Neurons", 1, 10, 3)
        n_poor = st.number_input("Number of Poor Neurons", 1, 10, 5)
        zakat_rate_sim = st.slider("Zakat Rate (ζ)", 0.0, 0.1, 0.025, 0.005, key="zakat_sim")
    
    # Simulate wealth distribution
    rich_grads = np.random.uniform(5, 15, n_rich)
    poor_grads = np.random.uniform(0.1, 1, n_poor)
    
    # Apply zakat
    zakat_amount = sum(rich_grads) * zakat_rate_sim
    per_poor_share = zakat_amount / n_poor if n_poor > 0 else 0
    
    rich_grads_after = rich_grads * (1 - zakat_rate_sim)
    poor_grads_after = poor_grads + per_poor_share
    
    # Create DataFrames for display
    before_df = pd.DataFrame({
        'Neuron': [f'Rich_{i}' for i in range(n_rich)] + [f'Poor_{i}' for i in range(n_poor)],
        'Gradient Before': list(rich_grads) + list(poor_grads),
        'Type': ['Rich'] * n_rich + ['Poor'] * n_poor
    })
    
    after_df = pd.DataFrame({
        'Neuron': [f'Rich_{i}' for i in range(n_rich)] + [f'Poor_{i}' for i in range(n_poor)],
        'Gradient After': list(rich_grads_after) + list(poor_grads_after),
        'Type': ['Rich'] * n_rich + ['Poor'] * n_poor
    })
    
    # Merge for display
    display_df = before_df.merge(after_df, on=['Neuron', 'Type'])
    display_df['Change'] = display_df['Gradient After'] - display_df['Gradient Before']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Zakat Collected", f"{zakat_amount:.4f}")
    with col2:
        st.metric("🕌 Per Poor Neuron Share", f"{per_poor_share:.4f}")
    with col3:
        st.metric("⚖️ Rich Reduction", f"{-zakat_rate_sim*100:.1f}%")
    
    st.dataframe(display_df.style.format({
        'Gradient Before': '{:.4f}',
        'Gradient After': '{:.4f}',
        'Change': '{:.4f}'
    }).applymap(lambda x: 'color: green' if isinstance(x, (int, float)) and x > 0 else 'color: red', subset=['Change']))
    
    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=display_df['Neuron'],
        y=display_df['Gradient Before'],
        name='Before Zakat',
        marker_color='#E74C3C'
    ))
    fig.add_trace(go.Bar(
        x=display_df['Neuron'],
        y=display_df['Gradient After'],
        name='After Zakat',
        marker_color='#27AE60'
    ))
    
    fig.update_layout(
        title="Gradient Redistribution via Digital Zakat",
        xaxis_title="Neuron",
        yaxis_title="Gradient Magnitude",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    🌟 **The Wisdom of Zakat**: Information-rich neurons share their learning capital, 
    preventing the formation of "super-neurons" that dominate decision-making. 
    This leads to more robust, decentralized intelligence.
    """)

# ============================================================================
# TAB 4: Audit Trail
# ============================================================================
with tab4:
    st.header("📜 Transparency & Audit Trail")
    
    st.markdown("""
    Unlike traditional black-box AI systems, Al-Mizan provides a **complete, verifiable audit trail** 
    of every decision, tyranny event, and zakat transfer.
    """)
    
    # Run a demo network
    if st.button("🔄 Run Demo Network & Generate Audit"):
        with st.spinner("Training constitutional network..."):
            # Create a small network
            model = AlMizanMLP(2, [4, 1])
            
            # Generate synthetic data
            X = [[MizanValue(1.0), MizanValue(0.0)], 
                 [MizanValue(0.0), MizanValue(1.0)],
                 [MizanValue(1.0), MizanValue(1.0)],
                 [MizanValue(0.0), MizanValue(0.0)]]
            y = [1.0, 1.0, 0.0, 0.0]
            
            # Train for a few iterations
            for epoch in range(5):
                predictions = []
                for x in X:
                    pred = model.forward(x)
                    predictions.append(pred)
                
                # Calculate loss
                loss = sum((pred - MizanValue(yi))**2 for pred, yi in zip(predictions, y))
                
                # Backward with justice
                model.backward(loss)
                
                # Update
                for p in model.parameters():
                    p.data -= 0.01 * p.grad
                    p.grad = 0.0
            
            # Get audit
            audit = model.get_full_audit()
            
            st.json(audit)
            
            st.success("Audit trail generated. Notice tyranny_count and zakat tracking.")
    
    st.info("""
    🔍 **What the Audit Shows**:
    - **tyranny_count**: How many times a neuron tried to dominate
    - **integrity_score**: Current trust level of each variable
    - **zakat_given/received**: Learning capital redistribution records
    - **history**: Timeline of all justice events
    """)

# ============================================================================
# TAB 5: About
# ============================================================================
with tab5:
    st.header("🏛️ About Al-Mizan Protocol")
    
    st.markdown("""
    ## Constitutional Foundations
    
    > "Al-Mizan is not a policy; it is a mathematical guarantee of Digital Sovereignty."
    
    ### The Problem
    
    Modern deep learning optimization, as popularized by frameworks like PyTorch and micrograd (Karpathy), 
    treats neurons as **passive containers** focused solely on loss minimization. This leads to:
    
    - **Algorithmic Tyranny**: Dominant patterns suppress minority features
    - **Black Box Decisions**: No transparency or accountability  
    - **Knowledge Monopolies**: Few neurons control all weights
    
    ### The Al-Mizan Solution
    
    We elevate the neuron to a **Responsible Entity** with:
    
    1. **Historical Integrity** (`tyranny_count`): Variables remember their past dominance
    2. **Al-Qist Constraint**: Automatic dampening of tyrannical gradients
    3. **Digital Zakat**: Redistribution from info-rich to info-poor neurons
    4. **Adaptive Zakat**: τ=1000 protection window for statistical minorities
    
    ### Key Innovations
    
    | Feature | Standard (PyTorch/micrograd) | Al-Mizan |
    |---------|------------------------------|----------|
    | Variable Memory | None | TyrannyCount |
    | Fairness | Post-hoc wrapper | **In-kernel constitutional** |
    | Transparency | Black box | **Audit trail** |
    | Minority Protection | None | **Adaptive Zakat** |
    
    ### Comparison with Karpathy's micrograd
    
    Andrej Karpathy's micrograd brilliantly demonstrates **how** autograd works. 
    Al-Mizan answers **why** and **for what purpose**:
    
    - **Karpathy**: "Here's how to compute gradients"
    - **Al-Mizan**: "Here's how to compute **just** gradients"
    
    ### Get Involved
    
    - 📄 [Read the Whitepaper](https://arxiv.org/abs/xxxx.xxxxx)
    - 💻 [GitHub Repository](https://github.com/rekabamine91-lgtm/al-mizan-protocol)
    - 📧 Contact: [your-email]
    
    ### Citation
    
    ```bibtex
    @misc{almizan2026constitutional,
      title={Al-Mizan Protocol: Constitutional Autograd for Digital Sovereignty},
      author={[Your Name]},
      year={2026},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }
    ```
    """)
    
    st.markdown("---")
    st.markdown("*Built with 🌙 and ⚖️ for a just digital future*")

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Al-Mizan Protocol v1.2.2 Constitutional | Digital Sovereignty is a Mathematical Guarantee</small></center>",
    unsafe_allow_html=True
)
