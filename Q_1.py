import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
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
class DistributionNetworkRiskModel:
    def __init__(self, nodes_data, lines_data):
        self.nodes_data = nodes_data
        self.lines_data = lines_data
        self.node_weights = {}
        for i, row in nodes_data.iterrows():
            self.node_weights[i + 1] = row.get('权重', 1)
        self.node_loads = {}
        for i, row in nodes_data.iterrows():
            self.node_loads[i + 1] = row['有功P/kW']
        self.dg_data = {}
        self.feeder_capacity = 2200
        self.feeder_current_limit = 220
        self.voltage = 10
        self.dg_failure_rate = 0.005
        self.load_failure_rate = 0.005
        self.switch_failure_rate = 0.002
        self.line_failure_rate_per_km = 0.002
        self.tie_switches = {
            "S13-1": (13, 43),
            "S29-2": (19, 29),
            "S62-3": (23, 62)
        }
        self.substation_switches = {
            "CB1": 1,
            "CB2": 43,
            "CB3": 23
        }
        self.feeder_regions = {
            "Feeder1": list(range(1, 23)),
            "Feeder2": list(range(23, 43)),
            "Feeder3": list(range(43, 63))
        }
        self.G = self._build_network()
    def _build_network(self):
        G = nx.Graph()
        for i in range(1, len(self.nodes_data) + 1):
            G.add_node(i, load=self.nodes_data.loc[i - 1, '有功P/kW'])
        for _, row in self.lines_data.iterrows():
            start = int(row['起点'])
            end = int(row['终点'])
            length = float(row['长度/km'])
            resistance = float(row['电阻/Ω'])
            reactance = float(row['电抗/Ω'])
            G.add_edge(start, end, length=length, resistance=resistance,
                       reactance=reactance, failure_rate=length * self.line_failure_rate_per_km)

        return G
    def set_dg_data(self, dg_data):
        self.dg_data = dg_data
    def calculate_load_loss_risk(self, num_simulations=10000):
        count = 0
        total_load_loss = 0
        load_total_harm = 0
        harm_list = []
        for _ in range(num_simulations):
            node_failure = np.random.rand(len(self.nodes_data)) < self.load_failure_rate
            line_failure = np.random.rand(len(self.lines_data)) < self.line_failure_rate_per_km * self.lines_data[
                '长度/km'].values
            switch_failure = np.random.rand(len(self.tie_switches)) < self.switch_failure_rate
            harm = 0
            load_loss = 0
            duration_types = ['瞬时', '短时', '中时', '长时']
            duration_probs = [0.75, 0.15, 0.09, 0.01]
            duration_type = np.random.choice(duration_types, p=duration_probs)
            if duration_type == '瞬时':
                duration = np.random.exponential(scale=0.033)
            elif duration_type == '短时':
                duration = np.random.weibull(a=2) * 0.5
            elif duration_type == '中时':
                duration = np.random.lognormal(mean=np.log(1), sigma=0.5)
            else:
                duration = np.random.lognormal(mean=np.log(8), sigma=0.8)
            duration = min(duration, 48)
            for i, is_failed in enumerate(node_failure, start=1):
                if is_failed:
                    loss = self._calculate_node_failure_load_loss(i)
                    weight = self.node_weights.get(i, 1)
                    harm += weight * loss * duration
                    load_loss += loss
            for i, is_failed in enumerate(line_failure):
                if is_failed:
                    u = int(self.lines_data.iloc[i]['起点'])
                    v = int(self.lines_data.iloc[i]['终点'])
                    loss = self._calculate_line_failure_load_loss(u, v)
                    affected_nodes = set()
                    G_temp = self.G.copy()
                    G_temp.remove_edge(u, v)
                    for f_name, nodes in self.feeder_regions.items():
                        substation_node = [node for cb, node in self.substation_switches.items() if node in nodes][0]
                        for node in nodes:
                            if node != substation_node and not nx.has_path(G_temp, substation_node, node):
                                affected_nodes.add(node)
                    if affected_nodes:
                        avg_weight = np.mean([self.node_weights.get(n, 1) for n in affected_nodes])
                    else:
                        avg_weight = 1
                    harm += avg_weight * loss * duration
                    load_loss += loss
            tie_switch_keys = list(self.tie_switches.keys())
            for i, is_failed in enumerate(switch_failure):
                if is_failed:
                    nodes = self.tie_switches[tie_switch_keys[i]]
                    loss = self._calculate_switch_failure_load_loss(nodes)
                    weight = (self.node_weights.get(nodes[0], 1) + self.node_weights.get(nodes[1], 1)) / 2
                    harm += weight * loss * duration
                    load_loss += loss
            harm_list.append(harm)
            total_load_loss += load_loss
            load_total_harm += harm
            if load_loss > 0:
                count += 1
        load_loss_pos = count / num_simulations
        load_loss_risk = load_loss_pos * load_total_harm
        load_loss_risk /= 700
        return load_loss_risk
    def _calculate_node_failure_load_loss(self, node):
        feeder = None
        for f_name, nodes in self.feeder_regions.items():
            if node in nodes:
                feeder = f_name
                break
        if feeder is None:
            return 0
        if node in [dg_info['node'] for dg_info in self.dg_data.values()]:
            for dg_id, dg_info in self.dg_data.items():
                if dg_info['node'] == node:
                    return dg_info['capacity']
        loss = self.node_loads.get(node, 0)
        transferable_load = 0
        for tie_switch, (node1, node2) in self.tie_switches.items():
            if node in self.feeder_regions[feeder]:
                other_node = node2 if node1 == node else node1 if node2 == node else None
                if other_node is not None:
                    other_feeder = None
                    for f_name, nodes in self.feeder_regions.items():
                        if other_node in nodes and f_name != feeder:
                            other_feeder = f_name
                            break
                    if other_feeder:
                        other_remaining_capacity = self._calculate_remaining_capacity(other_feeder)
                        transferable_load += min(loss, other_remaining_capacity)
        transferable_load = min(transferable_load, loss)
        return loss - transferable_load
    def _calculate_line_failure_load_loss(self, u, v):
        G_temp = self.G.copy()
        G_temp.remove_edge(u, v)
        affected_nodes = set()
        for f_name, nodes in self.feeder_regions.items():
            substation_node = [node for cb, node in self.substation_switches.items() if node in nodes][0]
            for node in nodes:
                if node != substation_node and not nx.has_path(G_temp, substation_node, node):
                    affected_nodes.add(node)
        total_loss = sum(self.node_loads.get(node, 0) for node in affected_nodes)
        for dg_id, dg_info in self.dg_data.items():
            if dg_info['node'] in affected_nodes:
                total_loss -= min(dg_info['capacity'], total_loss)
        for tie_switch, (node1, node2) in self.tie_switches.items():
            if node1 in affected_nodes and node2 not in affected_nodes:
                feeder = None
                for f_name, nodes in self.feeder_regions.items():
                    if node2 in nodes:
                        feeder = f_name
                        break
                remaining_capacity = self._calculate_remaining_capacity(feeder)
                transferable_load = min(total_loss, remaining_capacity)
                total_loss -= transferable_load
            elif node2 in affected_nodes and node1 not in affected_nodes:
                feeder = None
                for f_name, nodes in self.feeder_regions.items():
                    if node1 in nodes:
                        feeder = f_name
                        break
                remaining_capacity = self._calculate_remaining_capacity(feeder)
                transferable_load = min(total_loss, remaining_capacity)
                total_loss -= transferable_load
        return total_loss
    def _calculate_switch_failure_load_loss(self, nodes):
        u, v = nodes
        G_temp = self.G.copy()
        if G_temp.has_edge(u, v):
            G_temp.remove_edge(u, v)
        feeder_u = feeder_v = None
        for f_name, node_list in self.feeder_regions.items():
            if u in node_list:
                feeder_u = f_name
            if v in node_list:
                feeder_v = f_name
        sub_u = [node for cb, node in self.substation_switches.items() if node in self.feeder_regions[feeder_u]][0]
        sub_v = [node for cb, node in self.substation_switches.items() if node in self.feeder_regions[feeder_v]][0]
        reachable_u = set(nx.node_connected_component(G_temp, sub_u))
        reachable_v = set(nx.node_connected_component(G_temp, sub_v))
        loss = 0
        for node in self.feeder_regions[feeder_u]:
            if node not in reachable_v and node != sub_u:
                loss += self.node_loads.get(node, 0)
        for node in self.feeder_regions[feeder_v]:
            if node not in reachable_u and node != sub_v:
                loss += self.node_loads.get(node, 0)
        return loss

    def _calculate_remaining_capacity(self, feeder):
        nodes = self.feeder_regions[feeder]
        total_load = sum(self.node_loads.get(node, 0) for node in nodes)
        dg_contribution = 0
        for dg_id, dg_info in self.dg_data.items():
            if dg_info['node'] in nodes:
                dg_contribution += dg_info['capacity']
        net_load = max(0, total_load - dg_contribution)
        return self.feeder_capacity - net_load
    def calculate_overload_risk(self):
        total_risk = 0
        base_risk = 0.1
        for feeder_name, nodes in self.feeder_regions.items():
            total_load = sum(self.node_loads.get(node, 0) for node in nodes)
            dg_capacity = 0
            for dg_id, dg_info in self.dg_data.items():
                if dg_info['node'] in nodes:
                    dg_capacity += dg_info['capacity']
            net_load = max(0, total_load - dg_capacity)
            excess_power = 0
            if net_load > 0:
                current = net_load / (self.voltage * 1000 * np.sqrt(3))
            else:
                current = 0
                excess_power = dg_capacity - total_load
                for tie_switch, (node1, node2) in self.tie_switches.items():
                    if node1 in nodes or node2 in nodes:
                        adjacent_feeder = None
                        for f_name, f_nodes in self.feeder_regions.items():
                            if f_name != feeder_name and (node1 in f_nodes or node2 in f_nodes):
                                adjacent_feeder = f_name
                                break
                        if adjacent_feeder:
                            adj_total_load = sum(
                                self.node_loads.get(node, 0) for node in self.feeder_regions[adjacent_feeder])
                            adj_dg_capacity = 0
                            for dg_id, dg_info in self.dg_data.items():
                                if dg_info['node'] in self.feeder_regions[adjacent_feeder]:
                                    adj_dg_capacity += dg_info['capacity']
                            adj_net_load = max(0, adj_total_load - adj_dg_capacity)
                            adj_remaining_capacity = self.feeder_capacity - adj_net_load
                            transferable_power = min(excess_power, adj_remaining_capacity)
                            excess_power -= transferable_power
            if excess_power > 0:
                overload_probability = min(1.0, excess_power / self.feeder_capacity)
                overload_consequence = excess_power
                total_risk += overload_probability * overload_consequence
            if current > self.feeder_current_limit * 1.1:
                overload_ratio = current / (self.feeder_current_limit * 1.1)
                overload_probability = min(1.0, (overload_ratio - 1) * 2)
                overload_consequence = (current - self.feeder_current_limit * 1.1) * self.feeder_capacity / (
                        self.feeder_current_limit * 1.1)
                total_risk += overload_probability * overload_consequence
            else:
                load_ratio = current / self.feeder_current_limit
                potential_risk = base_risk * (1 + load_ratio)
                total_risk += potential_risk
        total_risk /= 700
        return total_risk
    def calculate_system_risk(self,return_details=False):
        load_loss_risk = self.calculate_load_loss_risk()
        overload_risk = self.calculate_overload_risk()
        system_risk = load_loss_risk + overload_risk
        self.detailed_results = {
            '失负荷风险': load_loss_risk,
            '过负荷风险': overload_risk,
            '系统总风险': system_risk
        }
        self._save_detailed_results()
        if return_details:
            return system_risk, load_loss_risk, overload_risk
        return system_risk
    def _save_detailed_results(self):
        if not os.path.exists('问题1结果'):
            os.makedirs('问题1结果')
        with open('问题1结果/system_risk_analysis.txt', 'w', encoding='utf-8') as f:
            f.write("=== 系统风险分析结果 ===\n\n")
            for key, value in self.detailed_results.items():
                f.write(f"{key}: {value}\n")
            f.write("\n分析结论:\n")
            f.write(
                f"1. 失负荷风险占比: {(self.detailed_results['失负荷风险'] / self.detailed_results['系统总风险'] * 100):.2f}%\n")
            f.write(
                f"2. 过负荷风险占比: {(self.detailed_results['过负荷风险'] / self.detailed_results['系统总风险'] * 100):.2f}%\n")
    def draw_network_topology(self, loss_nodes=None, line_current=None):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, with_labels=True, node_color='lightblue',
                node_size=500, font_size=8, font_weight='bold')
        node_labels = {n: f"{n}\n{self.G.nodes[n]['load']}kW" for n in self.G.nodes()}
        nx.draw_networkx_labels(self.G, pos, node_labels, font_size=8)
        edge_labels = {(u, v): f"{self.G[u][v]['length']}km" for u, v in self.G.edges()}
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels, font_size=6)
        plt.title('配电网拓扑结构')
        plt.axis('off')
        plt.savefig('问题1结果/network_topology.png', dpi=300)
        plt.close()
        pos = nx.spring_layout(self.G, seed=42)
        plt.figure(figsize=(12, 10))
        CB_nodes = list(self.substation_switches.values())
        DG_nodes = [dg_info['node'] for dg_info in self.dg_data.values()]
        node_colors = ['red' if n in CB_nodes else 'green' if n in DG_nodes else 'skyblue' for n in self.G.nodes]
        nx.draw(self.G, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=10)
        plt.title("配电网拓扑结构图（红=变电站，绿=DG）", fontsize=16)
        plt.axis("off")
        plt.savefig('问题1结果/NEW1.png', dpi=300)
        plt.close()
        plt.figure(figsize=(12, 10))
        if loss_nodes is None:
            loss_nodes = []
        node_colors = [
            'orange' if n in loss_nodes else 'red' if n in CB_nodes else 'green' if n in DG_nodes else 'lightgray' for n
            in self.G.nodes]
        nx.draw(self.G, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=10)
        plt.title("失电节点分布图（橙色=失电节点）", fontsize=16)
        plt.axis("off")
        plt.savefig('问题1结果/NEW2.png', dpi=300)
        plt.close()
        rated_current = 220
        overload_limit = rated_current * 1.1
        edge_colors = []
        edge_widths = []
        if line_current is None:
            line_current = {}
        for (a, b) in self.G.edges:
            current = line_current.get((a, b), line_current.get((b, a), 0))
            abs_I = abs(current)
            if abs_I > overload_limit:
                edge_colors.append('red')
                edge_widths.append(3)
            else:
                edge_colors.append('gray')
                edge_widths.append(1 + abs_I / 100)
        plt.figure(figsize=(12, 10))
        nx.draw(self.G, pos, with_labels=True, node_size=400, edge_color=edge_colors, width=edge_widths,
                node_color='lightblue', font_size=9)
        plt.title("线路电流热度图（红色=过载）", fontsize=16)
        plt.axis("off")
        plt.savefig('问题1结果/NEW3.png', dpi=300)
        plt.close()
if __name__ == "__main__":
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
    dg_data = {
        1: {'node': 13, 'capacity': 300},
        2: {'node': 18, 'capacity': 300},
        3: {'node': 22, 'capacity': 300},
        4: {'node': 29, 'capacity': 300},
        5: {'node': 32, 'capacity': 300},
        6: {'node': 39, 'capacity': 300},
        7: {'node': 48, 'capacity': 300},
        8: {'node': 59, 'capacity': 300}
    }
    model.set_dg_data(dg_data)
    system_risk = model.calculate_system_risk()
    print(f"系统总风险: {system_risk}")
    model.draw_network_topology()
    print("网络拓扑图已保存为 'network_topology.png'")