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
def get_typical_pv_curve():
    hours = np.arange(24)
    output_ratio = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.1, 0.3, 0.5, 0.7, 0.9, 1.0,
        1.0, 0.9, 0.8, 0.7, 0.6, 0.5,
        0.4, 0.3, 0.2, 0.1, 0.0, 0.0
    ])
    pv_curve = pd.DataFrame({
        'hour': hours,
        'output_ratio': output_ratio
    })
    return pv_curve
def calculate_max_pv_capacity():
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
    pv_curve = get_typical_pv_curve()
    print("典型光伏出力曲线:")
    print(pv_curve)
    plt.figure(figsize=(10, 6))
    plt.plot(pv_curve['hour'], pv_curve['output_ratio'], 'b-', marker='o')
    plt.xlabel('小时')
    plt.ylabel('归一化输出功率')
    plt.title('典型光伏日发电曲线')
    plt.grid(True)
    if not os.path.exists('问题3结果'):
        os.makedirs('问题3结果')
    plt.savefig('问题3结果/pv_curve.png', dpi=300)
    plt.close()
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
    peak_hour = pv_curve['output_ratio'].idxmax()
    morning_hour = 7
    evening_hour = 19
    analysis_hours = [peak_hour, morning_hour, evening_hour]
    hour_labels = ['峰值时段', '早晨时段', '傍晚时段']
    max_capacity_results = {}
    detailed_results = {}
    for dg_id, node in dg_locations.items():
        print(f"\n分析DG {dg_id} (节点 {node})的最大接入容量")
        detailed_results[dg_id] = {}
        base_dg_data = {}
        for id_, n in dg_locations.items():
            base_dg_data[id_] = {'node': n, 'capacity': 300}
        model.set_dg_data(base_dg_data)
        for hour_idx, hour in enumerate(analysis_hours):
            hour_label = hour_labels[hour_idx]
            output_ratio = pv_curve.loc[hour, 'output_ratio']
            print(f"  分析时间点: {hour}时 ({hour_label}), 光伏输出比例: {output_ratio:.2f}")
            num_scenarios = 10000
            fault_scenarios = model.generate_fault_scenarios(num_scenarios)
            capacities = np.arange(300, 3001, 300)
            risks = []
            detailed_risks = []
            for capacity in capacities:
                test_dg_data = base_dg_data.copy()
                test_dg_data[dg_id]['capacity'] = capacity * output_ratio
                model.set_dg_data(test_dg_data)
                system_risk, load_loss_risk, overload_risk = model.calculate_system_risk(
                    return_details=True, scenarios=fault_scenarios
                )
                detailed_risks.append({
                    'capacity': capacity,
                    'load_loss_risk': load_loss_risk,
                    'overload_risk': overload_risk,
                    'system_risk': system_risk
                })
                risks.append(system_risk)
            detailed_results[dg_id][hour_label] = pd.DataFrame(detailed_risks)
            plt.figure(figsize=(10, 6))
            plt.plot(capacities, risks, 'b-', marker='o')
            plt.xlabel('光伏容量 (kW)')
            plt.ylabel('系统风险')
            plt.title(f'DG {dg_id} (节点 {node}) 在{hour_label}的最大接入容量分析')
            plt.grid(True)
            plt.savefig(f'问题3结果/DG{dg_id}_hour{hour}_capacity_analysis.png', dpi=300)
            plt.close()
            risk_changes = np.diff(risks) / 300
            threshold = 0.01
            max_capacity_idx = 0
            for i, change in enumerate(risk_changes):
                if change > threshold:
                    max_capacity_idx = i
                    break
            if max_capacity_idx == 0 and len(risk_changes) > 0 and np.all(risk_changes <= threshold):
                max_capacity_idx = len(capacities) - 2
            max_capacity = capacities[max_capacity_idx]
            print(f"  DG {dg_id} 在{hour_label}的最大接入容量: {max_capacity}kW")
            if dg_id not in max_capacity_results:
                max_capacity_results[dg_id] = {}
            max_capacity_results[dg_id][hour_label] = max_capacity
    for dg_id, hour_data in detailed_results.items():
        for hour_label, df in hour_data.items():
            filename = f'DG{dg_id}_{hour_label}_detailed_analysis.csv'
            df.to_csv(os.path.join('问题3结果', filename), index=False)
    print("\n=== 光伏最大接入容量分析结果 ===")
    print("\n1. 各光伏在不同时段的最大接入容量 (kW):")
    results_df = pd.DataFrame(columns=hour_labels)
    for dg_id, hours_data in max_capacity_results.items():
        results_df.loc[f'DG{dg_id}'] = [hours_data.get(label, 'N/A') for label in hour_labels]
    print(results_df)
    results_df.to_csv('问题3结果/pv_max_capacity_results.csv')
    with open('问题3结果/pv_detailed_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("=== 光伏接入容量详细分析结果 ===\n\n")
        for dg_id, hour_data in detailed_results.items():
            f.write(f"\nDG {dg_id} 详细分析:\n")
            for hour_label, df in hour_data.items():
                f.write(f"\n{hour_label}:\n")
                f.write(df.to_string())
    plt.figure(figsize=(12, 8))
    bar_width = 0.25
    x = np.arange(len(max_capacity_results))
    for i, hour_label in enumerate(hour_labels):
        capacities = [max_capacity_results[dg_id].get(hour_label, 0) for dg_id in sorted(max_capacity_results.keys())]
        plt.bar(x + i * bar_width, capacities, width=bar_width, label=hour_label)
    plt.xlabel('分布式光伏')
    plt.ylabel('最大接入容量 (kW)')
    plt.title('各光伏在不同时段的最大接入容量比较')
    plt.xticks(x + bar_width, [f'DG{dg_id}' for dg_id in sorted(max_capacity_results.keys())])
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig('问题3结果/pv_max_capacity_comparison.png', dpi=300)
    plt.close()
    return results_df
if __name__ == "__main__":
    results = calculate_max_pv_capacity()
    print(results)