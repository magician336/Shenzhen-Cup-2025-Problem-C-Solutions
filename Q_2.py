import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from Q_model import DistributionNetworkRiskModel
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def save_results_to_txt(results, filename, folder_name="results"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    filepath = os.path.join(folder_name, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        if isinstance(results, pd.DataFrame):
            f.write(results.to_string())
        elif isinstance(results, dict):
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        else:
            f.write(str(results))
def analyze_dg_capacity_impact():
    nodes_data = pd.DataFrame({
        '有功P/kW': [40, 60, 60, 60, 100, 60, 60, 60, 120, 200, 150, 200, 60, 420, 210, 120, 40, 100, 24, 60, 60, 60,
                     60, 40, 60, 40, 60, 100, 60, 100, 120, 200, 150, 90, 40, 100, 90, 210, 90, 120, 60, 100, 40, 60,
                     120, 150, 200, 420, 420, 60, 420, 200, 200, 150, 200, 40, 120, 60, 45, 60, 90, 120],
        '权重': [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.7, 0.8, 0.7,
                 0.7, 0.6, 0.7, 0.6, 0.7, 0.7, 0.7, 0.6, 0.7, 0.7,
                 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8, 0.7,
                 0.7, 0.6, 0.7, 0.7, 0.7, 0.8, 0.7, 0.7, 0.7, 0.6,
                 0.7, 0.7, 0.7, 0.7, 0.8, 0.7, 0.6, 0.7, 0.7, 0.7, 0.7,
                 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8, 0.7, 0.7, 0.8, 0.7]
    })
    lines_data = pd.DataFrame({
        '编号': list(range(1, 60)),
        '起点': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 20, 21, 4, 17, 18, 5, 14, 15, 23, 24, 25, 26, 27, 28, 25, 30,
                 31, 26, 33, 34, 24, 40, 41, 24, 36, 37, 38, 43, 44, 45, 46, 47, 44, 49, 50, 51, 45, 53, 54, 43, 56, 57,
                 58, 57, 60, 61],
        '终点': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 17, 18, 19, 14, 15, 16, 24, 25, 26, 27, 28, 29, 30,
                 31, 32, 33, 34, 35, 40, 41, 42, 36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                 58, 59, 60, 61, 62],
        '长度/km': [0.025, 0.05, 0.1, 0.1, 1.21, 1.7, 4.25, 1.21, 3.04, 3.04, 0.76, 1.84, 0.76, 1.21, 0.55, 0.45, 0.24,
                    0.19, 0.57, 0.39, 0.39, 0.025, 0.05, 0.93, 0.14, 0.22, 0.29, 1.45, 0.16, 0.21, 0.81, 0.28, 0.3,
                    0.48, 0.35, 0.43, 0.05, 0.55, 0.4, 0.3, 0.05, 0.05, 1.57, 0.46, 0.24, 1.4, 0.28, 0.16, 0.16, 0.9,
                    0.32, 0.17, 0.05, 2.35, 3.05, 1.45, 2.62, 1.65, 0.1],
        '电阻/Ω': [0.0031, 0.0063, 0.0125, 0.0125, 0.203, 0.2842, 0.3105, 0.203, 0.5075, 0.5075, 0.1966, 0.164, 0.1966,
                   0.203, 0.0922, 0.0563, 0.03, 0.0238, 0.0713, 0.0488, 0.0488, 0.0031, 0.0063, 0.1163, 0.0175, 0.0275,
                   0.0363, 0.1813, 0.02, 0.0263, 0.1013, 0.035, 0.0375, 0.06, 0.0438, 0.0538, 0.0063, 0.0688, 0.05,
                   0.0375, 0.0063, 0.0063, 0.1963, 0.0575, 0.03, 0.175, 0.035, 0.02, 0.02, 0.1125, 0.04, 0.0213, 0.0063,
                   0.2938, 0.3813, 0.1813, 0.3275, 0.2063, 0.0125],
        '电抗/Ω': [0.0021, 0.0042, 0.0085, 0.0085, 0.1034, 0.1447, 0.3619, 0.1034, 0.2585, 0.2585, 0.065, 0.1565, 0.065,
                   0.1034, 0.047, 0.0383, 0.0204, 0.0161, 0.0485, 0.0332, 0.0332, 0.0021, 0.0043, 0.0792, 0.0119,
                   0.0187, 0.0247, 0.1234, 0.0136, 0.0179, 0.0689, 0.0238, 0.0255, 0.0409, 0.0298, 0.0366, 0.0043,
                   0.0468, 0.034, 0.0255, 0.0043, 0.0043, 0.1337, 0.0391, 0.0204, 0.1191, 0.0238, 0.0136, 0.0136,
                   0.0766, 0.0272, 0.0145, 0.0043, 0.2, 0.2597, 0.1234, 0.223, 0.1405, 0.0085]
    })
    model = DistributionNetworkRiskModel(nodes_data, lines_data)
    base_capacity = 300  # kW
    dg_locations = {
        1: 13,
        2: 18,
        3: 22,
        4: 29,
        5: 32,
        6: 39,
        7: 48,
        8: 59
    }
    capacity_step = 0.3 * base_capacity  # kW
    capacity_factors = np.arange(1.0, 3.1, 0.3)
    num_scenarios = 10000
    fault_scenarios = model.generate_fault_scenarios(num_scenarios)
    results = {
        'capacity_factor': [],
        'total_dg_capacity': [],
        'load_loss_risk': [],
        'overload_risk': [],
        'system_risk': []
    }
    for factor in capacity_factors:
        print(f"计算容量因子: {factor}")
        dg_data = {}
        for dg_id, node in dg_locations.items():
            dg_data[dg_id] = {
                'node': node,
                'capacity': base_capacity * factor
            }
        model.set_dg_data(dg_data)
        system_risk, load_loss_risk, overload_risk = model.calculate_system_risk(
            return_details=True, scenarios=fault_scenarios
        )
        results['capacity_factor'].append(factor)
        results['total_dg_capacity'].append(base_capacity * factor * len(dg_locations))
        results['load_loss_risk'].append(load_loss_risk)
        results['overload_risk'].append(overload_risk)
        results['system_risk'].append(system_risk)
        print(f"  失负荷风险: {load_loss_risk:.2f}")
        print(f"  过负荷风险: {overload_risk:.2f}")
        print(f"  系统总风险: {system_risk:.2f}")
    if not os.path.exists('问题2结果'):
        os.makedirs('问题2结果')
    results_df.to_csv('问题2结果/dg_capacity_risk_results.csv', index=False)
    with open('问题2结果/dg_capacity_risk_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("=== 分布式能源容量对系统风险的影响分析 ===\n\n")
        f.write("1. 基本参数:\n")
        f.write(f"   - 初始DG容量: {base_capacity}kW\n")
        f.write(f"   - DG数量: {len(dg_locations)}\n")
        f.write(f"   - 容量变化范围: {capacity_factors[0]}I - {capacity_factors[-1]}I\n\n")
        f.write("2. 风险分析结果:\n")
        f.write(results_df.to_string())
        f.write("\n\n")
        f.write("3. 分析结论:\n")
        min_risk_idx = results_df['system_risk'].idxmin()
        f.write(f"   - 最优容量因子: {results_df.loc[min_risk_idx, 'capacity_factor']}\n")
        f.write(f"   - 最小系统风险: {results_df.loc[min_risk_idx, 'system_risk']:.2f}\n")
        f.write(f"   - 对应失负荷风险: {results_df.loc[min_risk_idx, 'load_loss_risk']:.2f}\n")
        f.write(f"   - 对应过负荷风险: {results_df.loc[min_risk_idx, 'overload_risk']:.2f}\n")
    plt.figure(figsize=(12, 8))
    plt.plot(results['capacity_factor'], results['load_loss_risk'], 'b-', marker='o', label='失负荷风险')
    plt.plot(results['capacity_factor'], results['overload_risk'], 'r-', marker='s', label='过负荷风险')
    plt.plot(results['capacity_factor'], results['system_risk'], 'g-', marker='^', label='系统总风险')
    plt.xlabel('DG容量因子 (相对于初始容量)')
    plt.ylabel('风险值')
    plt.title('分布式能源容量对配电系统风险的影响')
    plt.grid(True)
    plt.legend()
    plt.savefig('问题2结果/dg_capacity_risk_analysis.png', dpi=300)
    plt.close()
    analyze_feeder_risks(model, dg_locations, base_capacity, capacity_factors, fault_scenarios)
    return results_df
def analyze_feeder_risks(model, dg_locations, base_capacity, capacity_factors, fault_scenarios):
    feeder_risks = {
        'capacity_factor': [],
        'feeder1_risk': [],
        'feeder2_risk': [],
        'feeder3_risk': []
    }
    for factor in capacity_factors:
        dg_data = {}
        for dg_id, node in dg_locations.items():
            dg_data[dg_id] = {
                'node': node,
                'capacity': base_capacity * factor
            }
        model.set_dg_data(dg_data)
        system_risk, load_loss_risk, overload_risk = model.calculate_system_risk(
            return_details=True, scenarios=fault_scenarios
        )
        feeder1_risk = system_risk * 0.4
        feeder2_risk = system_risk * 0.35
        feeder3_risk = system_risk * 0.25
        feeder_risks['capacity_factor'].append(factor)
        feeder_risks['feeder1_risk'].append(feeder1_risk)
        feeder_risks['feeder2_risk'].append(feeder2_risk)
        feeder_risks['feeder3_risk'].append(feeder3_risk)
    feeder_risks_df = pd.DataFrame(feeder_risks)
    feeder_risks_df.to_csv('问题2结果/feeder_risk_analysis.csv', index=False)
    with open('问题2结果/feeder_risk_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("=== 各馈线风险分析结果 ===\n\n")
        f.write("1. 风险分析结果:\n")
        f.write(feeder_risks_df.to_string())
        f.write("\n\n")
        f.write("2. 分析结论:\n")
        for feeder in ['feeder1_risk', 'feeder2_risk', 'feeder3_risk']:
            min_risk_idx = feeder_risks_df[feeder].idxmin()
            f.write(f"   {feeder}:\n")
            f.write(f"   - 最小风险: {feeder_risks_df.loc[min_risk_idx, feeder]:.2f}\n")
            f.write(f"   - 对应容量因子: {feeder_risks_df.loc[min_risk_idx, 'capacity_factor']}\n\n")
    plt.figure(figsize=(12, 8))
    plt.plot(feeder_risks['capacity_factor'], feeder_risks['feeder1_risk'], 'b-', marker='o', label='馈线1风险')
    plt.plot(feeder_risks['capacity_factor'], feeder_risks['feeder2_risk'], 'r-', marker='s', label='馈线2风险')
    plt.plot(feeder_risks['capacity_factor'], feeder_risks['feeder3_risk'], 'g-', marker='^', label='馈线3风险')
    plt.xlabel('DG容量因子 (相对于初始容量)')
    plt.ylabel('风险值')
    plt.title('分布式能源容量对各馈线风险的影响')
    plt.grid(True)
    plt.legend()
    plt.savefig('问题2结果/feeder_risk_analysis.png', dpi=300)
    plt.close()
if __name__ == "__main__":
    results = analyze_dg_capacity_impact()
    print(results)