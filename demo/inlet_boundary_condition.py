import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.integrate import trapezoid

from coronary_inlet import coronary_inlet_flow  # 傅里叶级数生理入口流量


def coronary_flow_with_womersley_profile(Q_mean, radius, r_positions, 
                                         heart_rate, viscosity=0.0035, 
                                         density=1050):
    """
    基于Womersley理论的冠脉流速剖面计算
    
    参考文献：
    Womersley JR. J Physiol. 1955;127(3):553-563.
    
    参数：
    Q_mean : float
        平均流量(mL/s)
    radius : float
        血管半径(cm)
    r_positions : array
        径向位置(cm)，0为中心，radius为壁面
    heart_rate : float
        心率(次/分钟)
    viscosity : float
        血液动力粘度(Pa·s)，默认0.0035
    density : float
        血液密度(kg/m³)，默认1050
    
    返回：
    velocity_profile : array
        速度剖面(cm/s)
    """
    
    omega = 2 * np.pi * heart_rate / 60  # 角频率(rad/s)
    alpha = radius * np.sqrt(omega * density / viscosity)  # Womersley数
    
    # 计算Womersley速度剖面
    r_norm = r_positions / radius
    j0_term = jv(0, alpha * 1j**1.5)
    j0_r_term = jv(0, alpha * r_norm * 1j**1.5)
    
    # 速度剖面(实部)
    velocity_profile = np.real(1 - j0_r_term / j0_term)
    
    # 归一化并缩放至目标流量
    velocity_profile = velocity_profile / trapezoid(velocity_profile * 2 * np.pi * r_positions, 
                                                   r_positions) * Q_mean
    
    return velocity_profile


# 示例使用和可视化
def plot_coronary_flow():
    """
    绘制冠状动脉血流波形
    """
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )
    # 时间设置
    t = np.linspace(0, 3, 1000)  # 3秒
    heart_rate = 120
    
    # 计算流量
    Q_phys = coronary_inlet_flow(t, heart_rate=heart_rate)
    Q_simp = coronary_inlet_flow(t, heart_rate=heart_rate, coronary_fraction=0.04)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 子图1：生理波形
    ax1 = axes[0, 0]
    ax1.plot(t, Q_phys, 'b-', linewidth=2)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('血流量 (mL/s)')
    ax1.set_title('冠状动脉入口生理脉动血流')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=np.mean(Q_phys), color='r', linestyle='--', 
                label=f'平均流量: {np.mean(Q_phys):.2f} mL/s')
    ax1.legend()
    
    # 子图2：简化波形
    ax2 = axes[0, 1]
    ax2.plot(t, Q_simp, 'g-', linewidth=2)
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('血流量 (mL/s)')
    ax2.set_title('简化分段函数波形')
    ax2.grid(True, alpha=0.3)
    
    # 子图3：Womersley速度剖面
    ax3 = axes[1, 0]
    radius = 0.15  # cm (1.5mm，典型冠脉半径)
    r = np.linspace(0, radius, 50)
    Q_mean = np.mean(Q_phys)
    
    # 不同时刻的速度剖面
    times_for_profile = [0, 0.2, 0.4, 0.6]  # 心动周期内不同时刻
    colors = ['blue', 'red', 'green', 'orange']
    
    for ti, color in zip(times_for_profile, colors):
        Q_t = coronary_inlet_flow(np.array([ti]), heart_rate=heart_rate)
        v_profile = coronary_flow_with_womersley_profile(
            Q_t[0], radius, r, heart_rate)
        ax3.plot(r, v_profile, color=color, linewidth=2, 
                label=f't={ti:.1f}s')
    
    ax3.set_xlabel('径向位置 (cm)')
    ax3.set_ylabel('流速 (cm/s)')
    ax3.set_title('不同时刻Womersley速度剖面')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4：流量-时间曲线的频谱分析
    ax4 = axes[1, 1]
    Q_single_cycle = Q_phys[(t >= 0) & (t < 60/heart_rate)]
    n_samples = len(Q_single_cycle)
    fft_result = np.abs(np.fft.fft(Q_single_cycle))[:n_samples//2]
    freq = np.fft.fftfreq(n_samples, d=(60/heart_rate)/n_samples)[:n_samples//2]
    
    ax4.stem(freq, fft_result/max(fft_result), 'b-', markerfmt='bo')
    ax4.set_xlabel('频率 (Hz)')
    ax4.set_ylabel('归一化幅度')
    ax4.set_title('血流波形频谱分析')
    ax4.set_xlim(0, 10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 输出关键参数
    print(f"冠状动脉血流参数：")
    print(f"平均血流量: {np.mean(Q_phys):.2f} mL/s")
    print(f"峰值血流量: {np.max(Q_phys):.2f} mL/s")
    print(f"最小血流量: {np.min(Q_phys):.2f} mL/s")
    print(f"脉动指数 (PI): {(np.max(Q_phys)-np.min(Q_phys))/np.mean(Q_phys):.2f}")
    print(f"Womersley数 α: {0.15 * np.sqrt(2*np.pi*75/60 * 1050/0.0035):.2f}")


if __name__ == "__main__":
    plot_coronary_flow()