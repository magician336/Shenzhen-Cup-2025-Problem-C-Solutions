import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from Q_model import DistributionNetworkRiskModel
from Q_3 import get_typical_pv_curve
def save_results_to_txt(results, filename, folder_name="问题4结果"):
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
class DistributionNetworkRiskAnalysis:
    def __init__(self, load_data, topology_data, dg_locations, dg_capacity=300):
        self.load_data = load_data
        self.topology_data = topology_data
        self.dg_locations = dg_locations
        self.dg_capacity = dg_capacity
        self.rated_power = 2200
        self.rated_current = 220
        self.voltage = 10
        self.dg_failure_rate = 0.005
        self.load_failure_rate = 0.005
        self.switch_failure_rate = 0.002
        self.line_failure_rate = 0.002
        self.tie_lines = [(13, 1), (29, 2), (62, 3)]
        self.calculate_line_failure_rates()
    def calculate_line_failure_rates(self):
        self.line_failure_rates = {}
        for _, row in self.topology_data.iterrows():
            line_id = (row['起点'], row['终点'])
            length = row['长度/km']
            self.line_failure_rates[line_id] = length * self.line_failure_rate
    def calculate_load_loss_risk(self, dg_outputs):
        total_risk = 0
        for line_id, failure_rate in self.line_failure_rates.items():
            lost_load = self.calculate_lost_load(line_id, dg_outputs)
            harm = self.calculate_harm(lost_load)
            total_risk += failure_rate * harm
        return total_risk
    def calculate_overload_risk(self, dg_outputs):
        total_risk = 0
        for line_id, _ in self.line_failure_rates.items():
            current = self.calculate_line_current(line_id, dg_outputs)
            if current > self.rated_current * 1.1:
                overload_prob = self.calculate_overload_probability(current)
                harm = self.calculate_overload_harm(current)
                total_risk += overload_prob * harm
        return total_risk
    def calculate_lost_load(self, fault_line, dg_outputs):
        affected_nodes = self.get_affected_nodes(fault_line)
        total_load = sum(self.load_data[self.load_data['No.'].isin(affected_nodes)]['有功P/kW'])
        dg_power = sum(dg_outputs)
        tie_line_power = self.calculate_tie_line_power(fault_line)
        lost_load = max(0, total_load - dg_power - tie_line_power)
        return lost_load
    def calculate_line_current(self, line_id, dg_outputs):
        power = self.calculate_line_power(line_id, dg_outputs)
        current = power / (self.voltage * np.sqrt(3))
        return current
    def calculate_harm(self, lost_load):
        return lost_load * 100
    def calculate_overload_harm(self, current):
        overload_ratio = (current - self.rated_current) / self.rated_current
        return overload_ratio * 1000
    def calculate_overload_probability(self, current):
        overload_ratio = (current - self.rated_current) / self.rated_current
        return min(1.0, overload_ratio * 0.5)
    def get_affected_nodes(self, fault_line):
        start_node, end_node = fault_line
        affected_nodes = [end_node]
        return affected_nodes
    def calculate_tie_line_power(self, fault_line):
        return self.rated_power * 0.3
    def calculate_line_power(self, line_id, dg_outputs):
        start_node, end_node = line_id
        if end_node not in self.load_data['No.'].values:
            return 0
        load_power = self.load_data[self.load_data['No.'] == end_node]['有功P/kW'].values[0]
        dg_power = 0
        for i, loc in enumerate(self.dg_locations):
            if loc == end_node:
                dg_power += dg_outputs[i]
        return load_power - dg_power
    def analyze_risk_evolution(self, dg_capacity_range):
        results = []
        for capacity in dg_capacity_range:
            dg_outputs = [capacity] * len(self.dg_locations)
            load_loss_risk = self.calculate_load_loss_risk(dg_outputs)
            overload_risk = self.calculate_overload_risk(dg_outputs)
            total_risk = load_loss_risk + overload_risk
            results.append({
                'capacity': capacity,
                'load_loss_risk': load_loss_risk,
                'overload_risk': overload_risk,
                'total_risk': total_risk
            })
        results_df = pd.DataFrame(results)
        save_results_to_txt(results_df, 'risk_evolution_results.txt', '问题4结果')
        return results_df
    def plot_risk_evolution(self, results):
        plt.figure(figsize=(10, 6))
        plt.plot(results['capacity'], results['load_loss_risk'], label='失负荷风险')
        plt.plot(results['capacity'], results['overload_risk'], label='过负荷风险')
        plt.plot(results['capacity'], results['total_risk'], label='总风险')
        plt.xlabel('DG容量 (kW)')
        plt.ylabel('风险值')
        plt.title('DG容量变化对系统风险的影响')
        plt.legend()
        plt.grid(True)
        plt.savefig('问题4结果/risk_evolution.png', dpi=300)
        plt.close()
class PVStorageModel:
    def __init__(self, pv_capacity, storage_ratio=0.15):
        self.pv_capacity = pv_capacity
        self.storage_capacity = pv_capacity * storage_ratio
        self.storage_power = self.storage_capacity
        self.soc = 0.8
        self.pv_curve = get_typical_pv_curve()
        self.calculate_output_curve()
    def calculate_output_curve(self):
        pv_output = self.pv_curve['output_ratio'] * self.pv_capacity
        self.storage_energy = np.zeros(24)
        self.final_output = np.zeros(24)
        for t in range(24):
            if pv_output.iloc[t] > 0:
                if self.soc < 1.0:
                    charge_power = min(pv_output.iloc[t], self.storage_power)
                    self.soc += charge_power / self.storage_capacity
                    self.storage_energy[t] = charge_power
                    self.final_output[t] = pv_output.iloc[t] - charge_power
                else:
                    self.final_output[t] = pv_output.iloc[t]
            else:
                if self.soc > 0:
                    discharge_power = min(self.storage_power, -pv_output.iloc[t])
                    self.soc -= discharge_power / self.storage_capacity
                    self.storage_energy[t] = -discharge_power
                    self.final_output[t] = pv_output.iloc[t] + discharge_power
                else:
                    self.final_output[t] = 0
    def plot_results(self):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.pv_curve['hour'], self.pv_curve['output_ratio'] * self.pv_capacity, label='光伏出力')
        plt.plot(self.pv_curve['hour'], self.final_output, label='系统输出')
        plt.title('光伏出力与系统输出曲线')
        plt.xlabel('时间 (h)')
        plt.ylabel('功率 (kW)')
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(self.pv_curve['hour'], self.storage_energy, label='储能能量变化')
        plt.title('储能能量变化曲线')
        plt.xlabel('时间 (h)')
        plt.ylabel('能量 (kWh)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('问题4结果/pv_storage_operation.png', dpi=300)
        plt.close()
    def get_metrics(self):
        metrics = {
            '光伏利用率': np.sum(self.final_output) / np.sum(self.pv_curve['output_ratio'] * self.pv_capacity),
            '储能利用率': np.sum(np.abs(self.storage_energy)) / (self.storage_capacity * 24),
            '最大输出功率': np.max(self.final_output),
            '最小输出功率': np.min(self.final_output),
            '平均输出功率': np.mean(self.final_output)
        }
        save_results_to_txt(metrics, 'pv_storage_metrics.txt', '问题4结果')
        return metrics
class PVStorageRiskAnalysis:
    def __init__(self, risk_model, dg_locations, storage_ratio=0.15):
        self.risk_model = risk_model
        self.dg_locations = dg_locations
        self.storage_ratio = storage_ratio
        self.pv_curve = get_typical_pv_curve()
        self.detailed_results = {}
    def analyze_single_pv_storage_impact(self, dg_id, pv_capacity_range):
        results = []
        base_dg_data = {}
        for id_, node in self.dg_locations.items():
            base_dg_data[id_] = {
                'node': node,
                'capacity': 300
            }
        self.detailed_results[dg_id] = {
            'capacities': [],
            'load_loss_risks': [],
            'overload_risks': [],
            'system_risks': [],
            'storage_capacities': [],
            'max_risk_hours': []
        }
        for pv_capacity in pv_capacity_range:
            storage_capacity = pv_capacity * self.storage_ratio
            max_hour_risk = 0
            max_hour = 0
            num_scenarios = 10000
            fault_scenarios = self.risk_model.generate_fault_scenarios(num_scenarios)
            for hour in range(24):
                pv_ratio = self.pv_curve.loc[hour, 'output_ratio']
                actual_output = self.calculate_actual_output(pv_capacity, storage_capacity, hour)
                dg_data = base_dg_data.copy()
                dg_data[dg_id]['capacity'] = actual_output
                self.risk_model.set_dg_data(dg_data)
                system_risk = self.risk_model.calculate_system_risk(scenarios=fault_scenarios)
                if system_risk > max_hour_risk:
                    max_hour_risk = system_risk
                    max_hour = hour
            self.detailed_results[dg_id]['capacities'].append(pv_capacity)
            self.detailed_results[dg_id]['load_loss_risks'].append(max_hour_risk)
            self.detailed_results[dg_id]['overload_risks'].append(max_hour_risk)
            self.detailed_results[dg_id]['system_risks'].append(max_hour_risk)
            self.detailed_results[dg_id]['storage_capacities'].append(storage_capacity)
            self.detailed_results[dg_id]['max_risk_hours'].append(max_hour)
        self._save_detailed_results(dg_id)
        results_df = pd.DataFrame({
            'pv_capacity': self.detailed_results[dg_id]['capacities'],
            'storage_capacity': self.detailed_results[dg_id]['storage_capacities'],
            'max_risk': self.detailed_results[dg_id]['system_risks'],
            'max_risk_hour': self.detailed_results[dg_id]['max_risk_hours']
        })
        save_results_to_txt(results_df, f'DG{dg_id}_pv_storage_analysis.txt', '问题4结果')
        return results_df
    def calculate_actual_output(self, pv_capacity, storage_capacity, hour):
        pv_output = pv_capacity * self.pv_curve.loc[hour, 'output_ratio']
        if pv_output > 0:
            actual_output = pv_output
            charge_amount = min(pv_output, storage_capacity)
            actual_output -= charge_amount
        else:
            actual_output = min(-pv_output, storage_capacity)
        return actual_output
    def _save_detailed_results(self, dg_id):
        results_df = pd.DataFrame({
            '光伏容量(kW)': self.detailed_results[dg_id]['capacities'],
            '储能容量(kW)': self.detailed_results[dg_id]['storage_capacities'],
            '失负荷风险': self.detailed_results[dg_id]['load_loss_risks'],
            '过负荷风险': self.detailed_results[dg_id]['overload_risks'],
            '系统总风险': self.detailed_results[dg_id]['system_risks']
        })
        results_df.to_csv(f'问题4结果/DG{dg_id}_pv_storage_detailed_analysis.csv', index=False)
        with open(f'问题4结果/DG{dg_id}_pv_storage_detailed_analysis.txt', 'w', encoding='utf-8') as f:
            f.write(f"=== DG {dg_id} 光伏储能系统详细分析结果 ===\n\n")
            f.write(results_df.to_string())
            f.write("\n\n分析结论:\n")
            f.write(f"1. 最优光伏容量: {results_df.loc[results_df['系统总风险'].idxmin(), '光伏容量(kW)']}kW\n")
            f.write(f"2. 对应储能容量: {results_df.loc[results_df['系统总风险'].idxmin(), '储能容量(kW)']}kW\n")
            f.write(f"3. 最小系统风险: {results_df['系统总风险'].min()}\n")
    def plot_single_results(self, results, dg_id):
        plt.figure(figsize=(12, 8))
        plt.plot(results['pv_capacity'], results['max_risk'], 'b-', marker='o', label='系统风险')
        for _, row in results.iterrows():
            plt.annotate(f'储能:{row["storage_capacity"]:.0f}kW',
                         (row['pv_capacity'], row['max_risk']),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')
        plt.xlabel('光伏装机容量 (kW)')
        plt.ylabel('系统风险')
        plt.title(f'DG{dg_id}光伏+储能容量对系统风险的影响')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'问题4结果/DG{dg_id}_pv_storage_analysis.png', dpi=300)
        plt.close()
        self._save_detailed_results(dg_id)
def main():
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
    risk_model = DistributionNetworkRiskModel(nodes_data, lines_data)
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
    pv_storage_analysis = PVStorageRiskAnalysis(risk_model, dg_locations)
    pv_capacity_range = np.arange(300, 3001, 300)
    if not os.path.exists('问题4结果'):
        os.makedirs('问题4结果')
    for dg_id in dg_locations.keys():
        print(f"\n分析DG {dg_id}的光伏储能系统影响")
        results = pv_storage_analysis.analyze_single_pv_storage_impact(dg_id, pv_capacity_range)
        pv_storage_analysis.plot_single_results(results, dg_id)
    with open('问题4结果/pv_storage_risk_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("=== 光伏储能系统风险分析总体结果 ===\n\n")
        for dg_id in dg_locations.keys():
            f.write(f"\nDG {dg_id} 分析结果:\n")
            try:
                with open(f'问题4结果/DG{dg_id}_pv_storage_detailed_analysis.txt', 'r', encoding='utf-8') as dg_file:
                    f.write(dg_file.read())
            except FileNotFoundError:
                f.write("结果文件未找到\n")
if __name__ == "__main__":
    main()