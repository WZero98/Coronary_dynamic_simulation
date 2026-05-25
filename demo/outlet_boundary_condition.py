import numpy as np
from scipy.interpolate import interp1d

from demo.units import P_INLET_REF_MMHG, P_VENOUS_DEFAULT_MMHG


def area_from_pressure_mmhg(p_mmhg, A0, beta, p_ref=P_INLET_REF_MMHG):
    """由 tube law 反解截面积 A；压力单位 mmHg。"""
    A0 = np.asarray(A0, dtype=float)
    beta = np.asarray(beta, dtype=float)
    p_mmhg = np.asarray(p_mmhg, dtype=float)
    sqrt_A = np.sqrt(A0) + (p_mmhg - p_ref) / beta
    return np.maximum(sqrt_A ** 2, 0.1 * A0)


class RCWindkesselMurrayOutletBC:
    """
    单支血管出口：R-C Windkessel 微循环 + Murray 定律 + 血压匹配。

    1) R-C Windkessel（状态量 P_wk, mmHg）::

        C · dP_wk/dt = Q_drive - (P_wk - P_venous) / R_wk

    2) Murray 定律（出口流量与几何标度）::

        Q_murray = coronary_outlet_flow_distribution(Q_in, [r_out], method='murray')
        R_wk = R_ref · (r_inlet / r_outlet)^4        # Poiseuille 半径标度

    3) 血压匹配（目标远端压）::

        P_match = P_venous + Q_murray · R_wk
        若给定入口管腔压 p_in，则 R_wk ← (p_in - P_venous) / Q_murray · (r_in/r_out)^4

    每时间步末推进 P_wk；出口施加 Q_out（Windkessel 一致流量）与 A（由 P_wk 经 tube law 反解）。
    """

    def __init__(
        self,
        outlet_radius_cm,
        inlet_radius_cm=None,
        R_mmhg_s_per_ml=None,
        C_ml_per_mmhg=0.8,
        p_inlet_ref_mmhg=P_INLET_REF_MMHG,
        p_venous_mmhg=P_VENOUS_DEFAULT_MMHG,
        q_mean_ml_s=3.33,
        resistance_exponent=4,
        P_init_mmhg=None,
    ):
        self.r_out = float(outlet_radius_cm)
        self.r_in = float(
            inlet_radius_cm if inlet_radius_cm is not None else outlet_radius_cm
        )
        self.p_inlet_ref = float(p_inlet_ref_mmhg)
        self.p_venous = float(p_venous_mmhg)
        self.q_mean = max(float(q_mean_ml_s), 1e-6)
        self.n = int(resistance_exponent)
        self.C = float(C_ml_per_mmhg)
        if R_mmhg_s_per_ml is None:
            self.R_ref = (self.p_inlet_ref - self.p_venous) / self.q_mean
        else:
            self.R_ref = float(R_mmhg_s_per_ml)
        self.P_wk = float(
            p_inlet_ref_mmhg if P_init_mmhg is None else P_init_mmhg
        )

    def reset(self, P_init_mmhg=None):
        self.P_wk = (
            self.p_inlet_ref if P_init_mmhg is None else float(P_init_mmhg)
        )

    def set_radii(self, inlet_radius_cm, outlet_radius_cm):
        self.r_in = float(inlet_radius_cm)
        self.r_out = float(outlet_radius_cm)

    def murray_flow(self, Q_in):
        Q_m = coronary_outlet_flow_distribution(
            Q_in, [self.r_out], method="murray"
        )
        return float(np.asarray(Q_m).reshape(-1)[0])

    def murray_scaled_resistance(self, p_inlet_lumen_mmhg=None, Q_murray=None):
        """外周阻力 (mmHg·s/mL)，含 Murray–Poiseuille 半径标度。"""
        r_scale = (self.r_in / self.r_out) ** self.n
        if p_inlet_lumen_mmhg is not None and Q_murray is not None:
            dp = max(float(p_inlet_lumen_mmhg) - self.p_venous, 0.0)
            return dp / max(abs(Q_murray), 1e-6) * r_scale
        return self.R_ref * r_scale

    def pressure_match_target(self, Q_murray, R_wk):
        """血压匹配目标微循环压 (mmHg)。"""
        return self.p_venous + Q_murray * R_wk

    def evaluate(self, Q_in, p_inlet_lumen_mmhg, Q_lumen):
        """
        RK 子步：不推进 Windkessel，用当前 P_wk 与 Murray/匹配计算出口量。
        """
        Q_murray = self.murray_flow(Q_in)
        R_wk = self.murray_scaled_resistance(p_inlet_lumen_mmhg, Q_murray)
        Q_out = (self.P_wk - self.p_venous) / R_wk
        return Q_out, self.P_wk

    def advance(self, dt, Q_lumen, Q_in, p_inlet_lumen_mmhg):
        """
        时间步末：推进 Windkessel，并结合 Murray 流量与血压匹配。
        """
        if dt <= 0.0:
            return self.evaluate(Q_in, p_inlet_lumen_mmhg, Q_lumen)

        Q_murray = self.murray_flow(Q_in)
        R_wk = self.murray_scaled_resistance(p_inlet_lumen_mmhg, Q_murray)
        P_match = self.pressure_match_target(Q_murray, R_wk)

        P_old = self.P_wk
        Q_drive = 0.5 * (float(Q_lumen) + Q_murray)
        self.P_wk += (dt / self.C) * (
            Q_drive - (self.P_wk - self.p_venous) / R_wk
        )
        Q_cap = self.C * (self.P_wk - P_old) / dt
        Q_res = (self.P_wk - self.p_venous) / R_wk
        Q_wk = Q_res + Q_cap

        match_relax = min(0.35, dt / max(self.C * R_wk, 1e-6))
        self.P_wk += match_relax * (P_match - self.P_wk)

        Q_out = 0.5 * Q_wk + 0.5 * Q_murray
        return Q_out, self.P_wk


# 向后兼容别名
MicrocirculationOutletBC = RCWindkesselMurrayOutletBC


def coronary_outlet_flow_distribution(Q_total, outlet_radii, method='murray',
                                     flow_fractions=None, cardiac_phase=None):
    """
    冠状动脉出口流量分配函数。
    
    基于解剖学缩放法则和生理学原理，将总冠状动脉流量分配到各个出口。
    
    参考文献：
    1. Murray CD. "The physiological principle of minimum work applied to 
       the angle of branching of arteries." J Gen Physiol, 1926;9(6):835-841.
    2. van der Giessen AG, et al. "The influence of boundary conditions on 
       wall shear stress distribution in patient specific coronary trees."
       J Biomech, 2011;44(6):1089-1095.
    3. Taylor CA, et al. "Patient-specific modeling of cardiovascular 
       mechanics." Annu Rev Biomed Eng, 2009;11:109-134.
    
    参数：
    Q_total : float or array
        总冠状动脉入口流量(mL/s)，可以是标量或时间序列
    outlet_radii : array
        各出口血管半径(cm)
    method : str
        分配方法：
        - 'murray': Murray's Law, Q ∝ r³
        - 'area': 面积比例, Q ∝ r²
        - 'custom': 自定义流量分数
    flow_fractions : array, optional
        自定义流量分数(method='custom'时使用)，应为总和为1的数组
    cardiac_phase : str, optional
        心脏相位('systole'或'diastole')，考虑不同相位下流量分布的变化
    
    返回：
    Q_outs : array
        各出口分配的流量(mL/s)，形状为(len(outlet_radii), len(t))
    """
    
    # 处理输入
    Q_total_arr = np.atleast_1d(Q_total)
    n_outlets = len(outlet_radii)
    n_time = len(Q_total_arr)
    
    # 计算出口半径
    radii = np.array(outlet_radii)
    
    # 根据不同方法计算分配权重
    if method == 'murray':
        # Murray's Law: 父血管半径的立方等于子血管半径立方之和
        # Q ∝ r³
        weights = radii ** 3
        
    elif method == 'area':
        # 面积比例分配: Q ∝ r²
        weights = radii ** 2
        
    elif method == 'murray_modified':
        # 改进的Murray's Law (考虑冠脉特殊性)
        # 指数通常在2.27-3.0之间
        exponent = 2.7  # 冠状动脉常用指数
        weights = radii ** exponent
        
    elif method == 'custom':
        if flow_fractions is None:
            raise ValueError("使用'custom'方法时必须提供flow_fractions参数")
        weights = np.array(flow_fractions)
        
    else:
        raise ValueError(f"不支持的分配方法: {method}")
    
    # 归一化权重
    weights = weights / np.sum(weights)
    
    # 考虑心动周期相位的影响
    if cardiac_phase is not None:
        if cardiac_phase == 'systole':
            # 收缩期：心内膜下血流减少，调整权重
            # 较小血管(代表心内膜下)的权重降低
            phase_factor = np.ones_like(weights)
            small_vessels = radii < np.median(radii)
            phase_factor[small_vessels] *= 0.7  # 收缩期减少30%
            weights = weights * phase_factor
            weights = weights / np.sum(weights)
        elif cardiac_phase == 'diastole':
            # 舒张期：恢复正常灌注
            pass
    
    # 分配流量
    Q_outs = np.outer(weights, Q_total_arr).T
    
    return Q_outs.squeeze()


def coronary_outlet_flow_with_impedance(Q_inlet, outlet_radii, LAD_fraction=0.5,
                                       LCx_fraction=0.3, RCA_fraction=0.2):
    """
    基于解剖学分区的冠状动脉出口流量分配。
    
    考虑三大冠脉分支的生理性血流分配比例。
    
    参考文献：
    1. Choy JS, Kassab GS. "Scaling of myocardial mass to flow and 
       morphometry of coronary arteries." J Appl Physiol, 2008;104(5):1281-1286.
    2. Seiler C, et al. "Quantitative assessment of retrograde coronary 
       collateral flow." J Am Coll Cardiol, 1998;32(5):1414-1421.
    
    参数：
    Q_inlet : float or array
        冠状动脉总入口流量(mL/s)
    outlet_radii : list of arrays
        各主要分支的出口半径列表，格式：[LAD出口半径数组, LCx出口半径数组, RCA出口半径数组]
    LAD_fraction : float
        左前降支血流分数(默认0.5)
    LCx_fraction : float
        左旋支血流分数(默认0.3)
    RCA_fraction : float
        右冠状动脉血流分数(默认0.2)
    
    返回：
    Q_outs : list of arrays
        各分支出口的流量分配
    """
    
    # 确保分数和为1
    total_fraction = LAD_fraction + LCx_fraction + RCA_fraction
    LAD_fraction /= total_fraction
    LCx_fraction /= total_fraction
    RCA_fraction /= total_fraction
    
    Q_inlet_arr = np.atleast_1d(Q_inlet)
    
    # 分配到主要分支
    Q_LAD = LAD_fraction * Q_inlet_arr
    Q_LCx = LCx_fraction * Q_inlet_arr
    Q_RCA = RCA_fraction * Q_inlet_arr
    
    # 在各分支内按Murray's Law再分配
    Q_outs = []
    for Q_branch, radii in zip([Q_LAD, Q_LCx, Q_RCA], outlet_radii):
        if len(radii) > 0:
            Q_branch_outs = coronary_outlet_flow_distribution(
                Q_branch, radii, method='murray')
            Q_outs.append(Q_branch_outs)
        else:
            Q_outs.append(np.array([]))
    
    return Q_outs


def coronary_outlet_flow_waveform(t, outlet_ids, heart_rate=75, 
                                  cardiac_output=5.0, coronary_fraction=0.04):
    """
    生成具有生理波形的冠状动脉出口流量。
    
    考虑冠状动脉特有的舒张期优势血流模式。
    
    参考文献：
    1. Nichols WW, O'Rourke MF. "McDonald's Blood Flow in Arteries."
       6th Edition, CRC Press, 2011.
    2. Kim HJ, et al. "Patient-specific modeling of blood flow and pressure 
       in human coronary arteries." Ann Biomed Eng, 2010;38(10):3195-3209.
    
    参数：
    t : array
        时间序列(s)
    outlet_ids : list
        出口标识符列表，如['LAD1', 'LAD2', 'LCx1', 'RCA1']
    heart_rate : float
        心率(次/分钟)
    cardiac_output : float
        心输出量(L/min)
    coronary_fraction : float
        冠状动脉血流占心输出量比例
    
    返回：
    Q_outs : dict
        字典，键为出口ID，值为对应的时间序列流量(mL/s)
    """
    
    # 计算平均冠脉流量
    mean_coronary_flow = cardiac_output * 1000 / 60 * coronary_fraction
    cardiac_period = 60.0 / heart_rate
    
    # 生成生理性入口流量波形
    t_norm = (t % cardiac_period) / cardiac_period
    
    # 构建舒张期优势波形
    Q_total = mean_coronary_flow * (1.0 + 
                                    0.5 * np.sin(2 * np.pi * t_norm) +
                                    0.3 * np.sin(4 * np.pi * t_norm - 0.5) +
                                    0.15 * np.sin(6 * np.pi * t_norm - 1.0))
    
    # 确保非负
    Q_total = np.maximum(Q_total, 0.05 * mean_coronary_flow)
    
    # 示例：简单平均分配（实际应用中应根据出口尺寸分配）
    n_outlets = len(outlet_ids)
    Q_per_outlet = Q_total / n_outlets
    
    Q_outs = {outlet_id: Q_per_outlet.copy() for outlet_id in outlet_ids}
    
    return Q_outs


def calculate_outlet_flow_resistance(Q_out, P_aortic, P_venous=5.0):
    """
    根据流量和压力计算出口阻力，用于设置流量边界条件时的验证。
    
    这确保了施加的流量条件与生理压力范围一致。
    
    参数：
    Q_out : array
        出口流量(mL/s)
    P_aortic : float or array
        主动脉压力(mmHg)
    P_venous : float
        静脉压力(mmHg)，默认5 mmHg
    
    返回：
    R_eff : array
        有效出口阻力(mmHg·s/mL)
    """
    
    Q_out_arr = np.atleast_1d(Q_out)
    P_aortic_arr = np.atleast_1d(P_aortic)
    
    # 避免除零
    Q_safe = np.where(np.abs(Q_out_arr) < 1e-10, 1e-10, Q_out_arr)
    
    # 计算有效阻力
    R_eff = (P_aortic_arr - P_venous) / Q_safe
    
    return R_eff


def set_coronary_outlet_flow_bc(mesh_data, outlet_faces, Q_total, 
                                method='murray'):
    """
    为CFD求解器设置具体的流量边界条件。
    
    这是一个接口函数，展示如何将流量分配到具体网格面。
    
    参数：
    mesh_data : dict
        包含网格信息的字典，应包含'face_areas'和'outlet_ids'
    outlet_faces : dict
        出口面对应关系，键为出口ID，值为面索引列表
    Q_total : float or array
        总冠状动脉入口流量
    method : str
        流量分配方法
    
    返回：
    bc_data : dict
        CFD求解器所需的边界条件数据
    """
    
    # 提取出口面积
    outlet_areas = []
    outlet_ids = []
    
    for outlet_id, face_indices in outlet_faces.items():
        outlet_areas.append(np.sum(mesh_data['face_areas'][face_indices]))
        outlet_ids.append(outlet_id)
    
    # 计算等效半径
    outlet_radii = np.sqrt(np.array(outlet_areas) / np.pi)
    
    # 分配流量
    Q_outs = coronary_outlet_flow_distribution(
        Q_total, outlet_radii, method=method)
    
    # 构建边界条件数据
    bc_data = {
        'type': 'flow_rate',
        'outlet_ids': outlet_ids,
        'flow_rates': {oid: Q_out for oid, Q_out in zip(outlet_ids, Q_outs.T)},
        'velocity_profile': 'parabolic',  # 或 'womersley', 'flat'
        'method': method
    }
    
    return bc_data


# 示例使用和可视化
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 示例1：基本流量分配
    print("=== 示例1: 基于Murray's Law的流量分配 ===")
    outlet_radii = [0.15, 0.12, 0.10, 0.08]  # cm
    Q_total = 4.0  # mL/s
    
    Q_outs_murray = coronary_outlet_flow_distribution(
        Q_total, outlet_radii, method='murray')
    Q_outs_area = coronary_outlet_flow_distribution(
        Q_total, outlet_radii, method='area')
    
    print("出口半径(cm):", outlet_radii)
    print(f"Murray's Law分配 (mL/s): {Q_outs_murray}")
    print(f"面积比例分配 (mL/s): {Q_outs_area}")
    
    # 示例2：时间序列流量分配
    print("\n=== 示例2: 时间序列流量分配 ===")
    t = np.linspace(0, 2, 200)
    heart_rate = 75
    cardiac_period = 60 / heart_rate
    t_norm = (t % cardiac_period) / cardiac_period
    
    # 生成生理流量波形
    Q_total_t = 4.0 * (1.0 + 0.5 * np.sin(2 * np.pi * t_norm) +
                       0.3 * np.sin(4 * np.pi * t_norm - 0.5))
    
    Q_outs_t = coronary_outlet_flow_distribution(
        Q_total_t, outlet_radii, method='murray')
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1：总流量波形
    ax1 = axes[0, 0]
    ax1.plot(t, Q_total_t, 'b-', linewidth=2, label='Total flow')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Flow (mL/s)')
    ax1.set_title('Total Coronary Inlet Flow')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 子图2：各出口流量
    ax2 = axes[0, 1]
    for i, Q_out in enumerate(Q_outs_t.T):
        ax2.plot(t, Q_out, linewidth=2, 
                label=f'Outlet {i+1} (r={outlet_radii[i]:.2f} cm)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Flow (mL/s)')
    ax2.set_title('Outlet Flow Distribution (Murray\'s Law)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 子图3：不同分配方法对比
    ax3 = axes[1, 0]
    methods = ['murray', 'murray_modified', 'area']
    labels = ['Murray (r³)', 'Modified (r²·⁷)', 'Area (r²)']
    
    x = np.arange(len(outlet_radii))
    width = 0.25
    
    for i, (method, label) in enumerate(zip(methods, labels)):
        Q_method = coronary_outlet_flow_distribution(
            Q_total, outlet_radii, method=method)
        ax3.bar(x + i*width, Q_method, width, label=label, alpha=0.8)
    
    ax3.set_xlabel('Outlet Number')
    ax3.set_ylabel('Flow (mL/s)')
    ax3.set_title('Comparison of Flow Distribution Methods')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels([f'Outlet {i+1}' for i in range(len(outlet_radii))])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 子图4：流量分配比例
    ax4 = axes[1, 1]
    flow_fractions_murray = Q_outs_murray / np.sum(Q_outs_murray)
    ax4.pie(flow_fractions_murray, 
            labels=[f'Out {i+1}\n({outlet_radii[i]:.2f} cm)' 
                   for i in range(len(outlet_radii))],
            autopct='%1.1f%%', startangle=90)
    ax4.set_title('Flow Fraction Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # 示例3：考虑收缩期/舒张期的流量分配差异
    print("\n=== 示例3: 心脏相位对流量分配的影响 ===")
    Q_systole = coronary_outlet_flow_distribution(
        Q_total, outlet_radii, method='murray', cardiac_phase='systole')
    Q_diastole = coronary_outlet_flow_distribution(
        Q_total, outlet_radii, method='murray', cardiac_phase='diastole')
    
    print("收缩期分配:", Q_systole)
    print("舒张期分配:", Q_diastole)
    print("收缩期/舒张期比值:", Q_systole / Q_diastole)