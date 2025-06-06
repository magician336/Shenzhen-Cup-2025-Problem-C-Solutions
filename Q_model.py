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
class DistributionNetworkRiskModel:  # 定义类
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
    def generate_fault_scenarios(self, num_simulations=10000):
        scenarios = []
        duration_types = ['瞬时', '短时', '中时', '长时']
        duration_probs = [0.75, 0.15, 0.09, 0.01]
        for _ in range(num_simulations):
            node_failure = np.random.rand(len(self.nodes_data)) < self.load_failure_rate
            line_failure = np.random.rand(len(self.lines_data)) < self.line_failure_rate_per_km * self.lines_data[
                '长度/km'].values
            switch_failure = np.random.rand(len(self.tie_switches)) < self.switch_failure_rate
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
            scenarios.append({
                'node_failure': node_failure,
                'line_failure': line_failure,
                'switch_failure': switch_failure,
                'duration': duration,
                'duration_type': duration_type
            })
        return scenarios
    def calculate_load_loss_risk(self, num_simulations=10000, scenarios=None):
        count = 0
        total_load_loss = 0
        load_total_harm = 0
        harm_list = []
        for scenario in scenarios:
            if scenarios is not None:
                node_failure = scenario.get('node_failure', np.zeros(len(self.nodes_data), dtype=bool))
                line_failure = scenario.get('line_failure', np.zeros(len(self.lines_data), dtype=bool))
                switch_failure = scenario.get('switch_failure', np.zeros(len(self.tie_switches), dtype=bool))
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
                    # 取受影响节点的平均权重
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
    def calculate_system_risk(self, return_details=False, scenarios=None):
        load_loss_risk = self.calculate_load_loss_risk(scenarios=scenarios)
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