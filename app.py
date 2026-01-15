import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import signal

# ==========================================
# 1. 模拟数据加载器 (Data Loader)
# 在实际项目中，这里替换为 MDSplus 的连接代码
# ==========================================
def load_east_data(shot_no):
    # 模拟 5秒的数据，采样率 1ms
    time = np.linspace(0, 5, 5000)
    
    # 模拟电流 (IPRogo): 启动 -> 平顶 -> 关机
    ip = 500 * (1 - np.exp(-time/0.5)) * (1 - 1/(1+np.exp(-(time-4.5)*10)))
    
    # 模拟 H-mode 转换信号 (D_alpha): 在 2.5s 处突然下降
    d_alpha = np.random.normal(10, 1, 5000)
    d_alpha[2500:] = d_alpha[2500:] * 0.3 + 2  # 模拟 H-mode 掉落
    
    # 模拟温度剖面 (User Embedding): 32个通道
    # 形状: [Time, Channel]
    te_profile = np.zeros((5000, 32))
    for i in range(32):
        # 中心高，边缘低
        te_profile[:, i] = (1 - (i/32)**2) * ip / 500 * np.random.normal(1, 0.05, 5000)
        
    return time, ip, d_alpha, te_profile

# ==========================================
# 2. 软件界面布局 (UI Layout)
# ==========================================
st.set_page_config(layout="wide", page_title="EAST Feature Analysis Dashboard")

st.title("🔋 EAST 等离子体行为分析与特征提取平台")
st.markdown("### 基于序列推荐算法的数据预处理看板")

# --- 侧边栏 ---
with st.sidebar:
    st.header("🎮 控制台")
    shot_input = st.number_input("输入炮号 (Shot Number)", value=10086, step=1)
    
    if st.button("加载数据"):
        st.session_state['data_loaded'] = True
        # 真正加载数据
        t, ip, da, te = load_east_data(shot_input)
        st.session_state['data'] = (t, ip, da, te)
    
    st.info("当前模式: 离线分析 (Offline Analysis)")

# --- 主显示区 ---
if st.session_state.get('data_loaded'):
    t, ip, da, te = st.session_state['data']
    
    # 分栏布局
    col1, col2 = st.columns([1, 1])
    
    # === Panel 1: 基础画像 (User Profile) ===
    with col1:
        st.subheader("📊 基础画像 (Macro Signals)")
        fig_macro = go.Figure()
        fig_macro.add_trace(go.Scatter(x=t, y=ip, name='IPRogo (Current)', line=dict(color='blue')))
        fig_macro.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), title="User ID / Current")
        st.plotly_chart(fig_macro, use_container_width=True)
        
        st.markdown("**物理含义:** 对应用户的生命周期。曲线平稳代表用户活跃，归零代表流失。")

    # === Panel 2: 转化目标 (Conversion Label) ===
    with col2:
        st.subheader("🎯 转化目标 (H-mode Detection)")
        fig_label = go.Figure()
        fig_label.add_trace(go.Scatter(x=t, y=da, name='D_alpha (Radiation)', line=dict(color='orange')))
        
        # 模拟自动打标：找到数值突降的点
        h_mode_start = 2.5 
        fig_label.add_vline(x=h_mode_start, line_dash="dash", line_color="red", annotation_text="H-mode Trigger")
        
        fig_label.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), title="Conversion Event / D_alpha")
        st.plotly_chart(fig_label, use_container_width=True)
        
        st.markdown("**物理含义:** 红线处检测到 `D_alpha` 骤降，标记为 **Label=1 (转化成功)**。")

    # === Panel 3: 深度时空特征 (Spatiotemporal Embedding) ===
    st.subheader("🧠 深度时空特征 (ECE Temperature Profile)")
    
    # 绘制热力图
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=te.T, # 转置一下，y轴是通道，x轴是时间
        x=t,
        y=list(range(32)),
        colorscale='Viridis'
    ))
    fig_heatmap.update_layout(
        height=400, 
        title="User Interest Embedding (Te Profile Evolution)",
        xaxis_title="Time (s)",
        yaxis_title="Channel (Space)"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown("**搜广推视角:** 这不是普通的温度图，这是**32维用户特征向量随时间的演化流**。可以看到在 H-mode (2.5s) 后，边缘温度梯度明显变陡（颜色分界变清晰）。")

else:
    st.warning("👈 请在左侧输入炮号并点击'加载数据'")