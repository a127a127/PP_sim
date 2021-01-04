import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from tqdm import tqdm

CARE_LAYERS = []
STEP_CYCLES = 1000

class MappingGraph:
    def __init__(self, hw_config, model_config):
        global CARE_LAYERS, STEP_CYCLES
        if model_config.Model_type == 'Lenet':
            #CARE_LAYERS = [0, 2, 4]
            CARE_LAYERS = [0, 1, 2, 3, 4]
            #CARE_LAYERS = [2, 4]
            STEP_CYCLES = 10000
        elif model_config.Model_type == "Caffenet":
            CARE_LAYERS = [2, 4, 5, 6]
            #CARE_LAYERS = [4, 5, 6]
            STEP_CYCLES = 2000
        elif model_config.Model_type == "Test":
            CARE_LAYERS = [0, 1]
            STEP_CYCLES = 20
        self.Router_num_x = hw_config.Router_num_x
        self.Router_num_y = hw_config.Router_num_y
        self.router_width = self.Router_num_x
        self.router_height = self.Router_num_y
        self.router_size = self.router_width * self.router_height
        self.router_active_array = [0] * self.router_size

        self.PE_num_x = hw_config.PE_num_x
        self.PE_num_y = hw_config.PE_num_y
        assert hw_config.PE_num_x == 2
        assert hw_config.PE_num_y == 2
        self.pe_width = self.Router_num_x * self.PE_num_x
        self.pe_height = self.Router_num_y * self.PE_num_y

        assert hw_config.CU_num == 12
        self.CU_num = hw_config.CU_num
        self.CU_num_x = 4
        self.CU_num_y = 3

        self.cu_width = self.Router_num_x * self.PE_num_x * self.CU_num_x
        self.cu_height = self.Router_num_y * self.PE_num_y * self.CU_num_y
        self.cu_size = self.cu_width * self.cu_height
        self.cu_active_array = [0] * self.cu_size

        self.model_config = model_config

        # Init matplotlib figure
        self.H = 9.00
        self.W = 14.40
        self.fig = None
        self.init_fig()

        self.node_size = int((self.W * 25 / self.cu_width) ** 2)


        # Init networkx graph for Router
        self.router_G = nx.grid_2d_graph(self.router_width, self.router_height)
        self.router_pos = {}

        p = []
        for i in range(0, self.router_width):
            for j in range(0, self.router_height):
                x = i*(self.CU_num_x+1)*self.PE_num_x + self.CU_num_x
                y = 1 + j*(self.CU_num_y+1)*self.PE_num_y + self.CU_num_y
                p.append([x,y])

        nodes = list(self.router_G.nodes)
        for i in range(0, len(nodes)):
            self.router_pos[nodes[i]] = p[i]


        # Init networkx graph for CU
        self.cu_G = nx.grid_2d_graph(self.cu_width, self.cu_height)
        self.cu_pos = {}

        nodes = list(self.cu_G.nodes)
        self.cu_G.remove_edges_from(self.cu_G.edges)
        #edges = list(self.G.edges)
        #G.add_edge((1,1), (2,2))

        p = []
        for i in range(0, self.cu_width):
            for j in range(0, self.cu_height):
                p.append([i + (i//self.CU_num_x),1 + j + (j//self.CU_num_y)])
        #for i in range(0, len(nodes)):
        #    G.nodes[nodes[i]]['pos'] = p[i]

        for i in range(0, len(nodes)):
            self.cu_pos[nodes[i]] = p[i]
        #cmap = plt.cm.get_cmap('rainbow')


        # Empty graph for text
        self.empty_G = nx.Graph()
        self.empty_node = (-1,-1)
        self.empty_G.add_node(self.empty_node)
        self.empty_pos = {self.empty_node: [0, 0]}

        # Weight mapping
        self.weight_mapping = self.allocate_cu_active_array()
        self.cu_color_default = None
        self.max_nlayer = 0
        self.nlayer_mapping = {}

    def init_fig(self):
        if self.fig != None:
            plt.close(self.fig)
        self.fig = plt.figure(None, figsize=(self.W, self.H))
        plt.subplot2grid((1, 4), (0, 0), colspan=3)
        plt.tight_layout()

    def allocate_cu_active_array(self):
        return self.cu_active_array.copy()

    def allocate_router_active_array(self):
        return self.router_active_array.copy()

    def position_idx_to_x_y(self, position_idx):
        x = (position_idx[1] * self.PE_num_x * self.CU_num_x) + (position_idx[3] * self.CU_num_x) + (position_idx[4] // self.CU_num_y)
        y = (position_idx[0] * self.PE_num_y * self.CU_num_y) + (position_idx[2] * self.CU_num_y) + (position_idx[4] % self.CU_num_y)
        return (x, y)

    def position_idx_to_idx(self, position_idx):
        (x, y) = self.position_idx_to_x_y(position_idx)
        return x*self.cu_height + y

    def position_idx_to_router_idx(self, position_idx):
        x = position_idx[1]
        y = position_idx[0]
        return x * self.router_height + y

    def set_mapping(self, idx, nlayer):
        if not nlayer in self.nlayer_mapping:
            self.nlayer_mapping[nlayer] = len(self.nlayer_mapping)
        nlayer = self.nlayer_mapping[nlayer] + 1
        assert (self.weight_mapping[idx] == 0 or self.weight_mapping[idx] == nlayer)
        self.weight_mapping[idx] = nlayer
        if nlayer > self.max_nlayer:
            self.max_nlayer = nlayer

    def get_mapping(self, nlayer):
        if nlayer in self.nlayer_mapping:
            return self.nlayer_mapping[nlayer]
        else:
            return -1

    def get_layer_color(self, nlayer_plus_one):
        if nlayer_plus_one == 0:
            return (0.5, 0.5, 0.5, 0.1)
        else:
            v = (nlayer_plus_one - 1) % 6
            if v == 0:
                return (1, 0.3, 0.3, 0.1)
            elif v == 1:
                return (1, 0.3, 1, 0.1)
            elif v == 2:
                return (0.3, 1, 0.3, 0.1)
            elif v == 3:
                return (0.3, 1, 1, 0.1)
            elif v == 4:
                return (0.3, 0.3, 1, 0.1)
            elif v == 5:
                return (1, 1, 0.3, 0.1)

    def ensure_cu_color_default(self):
        if self.cu_color_default == None:
            self.cu_color_default = self.allocate_cu_active_array()
            for k, v in enumerate(self.weight_mapping):
                self.cu_color_default[k] = self.get_layer_color(v)

    def draw(self, text, filename, active_cu, active_pe=None, active_router=None, active_router_edge=None, active_layers=[], active_windows=None):
        #if np.count_nonzero(active_cu) == 0 and np.count_nonzero(active_router) == 0:
        #    return
        assert len(active_cu) == len(self.weight_mapping)

        # Draw cu mapping
        self.ensure_cu_color_default()

        cu_color = self.cu_color_default.copy()
        for k, v in enumerate(active_cu):
            if v != 0:
                t = list(cu_color[k])
                t[3] = 0.8
                cu_color[k] = tuple(t)

        nx.draw(self.cu_G, self.cu_pos, node_color = cu_color, node_shape='s', node_size=self.node_size)

        # Draw PE
        current_axis = plt.gca()
        padding = 0.1
        for i in range(0, self.pe_width):
            for j in range(0, self.pe_height):
                x = (i*(self.CU_num_x+1)) - 0.5 - padding
                y = 1 + (j*(self.CU_num_y+1)) - 0.5 - padding
                current_axis.add_patch(
                    Rectangle((x, y),
                        width = self.CU_num_x + padding*2,
                        height = self.CU_num_y + padding*2,fill=False
                    )
                )

        # Draw Router
        if active_router == None:
            active_router = [0] * self.router_size
        router_color = [(0.2, 0, 0)] * self.router_size
        for k, v in enumerate(active_router):
            if v != 0:
                t = list(router_color[k])
                t[0] = 1
                router_color[k] = tuple(t)

        nx.draw(self.router_G, self.router_pos, node_color = router_color, node_shape='o', node_size=self.node_size)

        # Draw text
        nx.draw(self.empty_G, pos = self.empty_pos, node_color='none', node_shape='s', node_size=self.node_size)
        self.fig.text(.5, 0.03, text, ha='center', fontsize=15)

        # Draw model graph
        plt.subplot2grid((1, 4), (0, 3))

        from Model import Model
        model = Model(self.model_config)

        show_layers = []
        layer_height = []
        layer_height_offset = []
        layer_width = []
        model_height = 0
        model_width = 0
        height_padding = 3
        for i in range(0, len(self.model_config.layer_list)):
            if not i in CARE_LAYERS:
                continue
            if self.model_config.layer_list[i].layer_type == 'convolution' or self.model_config.layer_list[i].layer_type == 'fully':
                h = model.input_h[i]
                w = model.input_w[i]
            else:
                h = 1
                w = 1
            show_layers.append(i)
            layer_height.append(h)
            layer_width.append(w)
            model_height += h
            model_width = max(model_width, w)
        total_width = model_width * 2 + 3 + 1
        total_height = model_height + height_padding * (len(show_layers) + 1)

        nlayers = len(show_layers)
        model_color = [0] * nlayers
        v=-1
        for i in range(0, nlayers):
            model_color[i] = self.get_layer_color(self.get_mapping(show_layers[i]) + 1)
            if show_layers[i] in active_layers:
                t = list(model_color[i])
                t[3] = 0.8
                model_color[i] = tuple(t)
            # alpha to color:
            t = list(model_color[i])
            t[0] = 1 - (1 - t[0]) * t[3]
            t[1] = 1 - (1 - t[1]) * t[3]
            t[2] = 1 - (1 - t[2]) * t[3]
            t[3] = 1
            model_color[i] = tuple(t)

        model_G = nx.grid_2d_graph(1, nlayers)

        p = []
        labels = {}
        height = total_height - 1
        for i in range(0, nlayers):
            layer_height_offset.append(height - height_padding)
            p.append([1+(model_width//2), height - height_padding - (layer_height[i]//2)])
            labels[(0, i)] = f'{self.model_config.layer_list[show_layers[i]].layer_type}\nInput Tensor:'
            height -= layer_height[i] + height_padding

        nodes = list(model_G.nodes)
        model_pos = {}
        for i in range(0, len(nodes)):
            model_pos[nodes[i]] = p[i]

        nx.draw(model_G, model_pos, node_color = model_color, node_shape='o', node_size=int((self.W * 10 * model_width / total_width) ** 2))
        nx.draw_networkx_labels(model_G, model_pos, labels)

        current_axis = plt.gca()
        padding = 0.1
        for k in range(0, nlayers):
            if self.model_config.layer_list[k].layer_type == 'convolution' or self.model_config.layer_list[k].layer_type == 'fully':
                model_G = nx.grid_2d_graph(layer_width[k], layer_height[k])
                model_G.remove_edges_from(model_G.edges)

                p = []
                for i in range(0, layer_width[k]):
                    for j in range(0, layer_height[k]):
                        p.append([1+model_width+1+i, layer_height_offset[k]-j])

                nodes = list(model_G.nodes)
                model_pos = {}
                for i in range(0, len(nodes)):
                    model_pos[nodes[i]] = p[i]
                nx.draw(model_G, model_pos, node_shape='s', node_color='gray', node_size=int((self.W * 10 / total_width) ** 2))

                x = 1 + model_width + 1 - 0.5 - padding
                y = layer_height_offset[k] - layer_height[k] + 0.5 - padding
                current_axis.add_patch(
                    Rectangle((x, y),
                        width = layer_width[k] + padding*2,
                        height = layer_height[k] + padding*2,
                        fill=False
                    )
                )

                if active_windows != None:
                    for window_id in active_windows:
                        if show_layers[k] == window_id[0]:
                            x = 1 + model_width + 1 - 0.5 + window_id[2]
                            y = layer_height_offset[k] - window_id[3] + 0.5
                            current_axis.add_patch(
                                Rectangle((x, y),
                                    width = window_id[4]-window_id[2],
                                    height = window_id[3]-window_id[1],
                                    fill=False, color='red'
                                )
                            )


        # Draw bottom-left and up-right invisible node (for border)
        nx.draw(self.empty_G, pos = self.empty_pos, node_color='none', node_shape='o', node_size=int((self.H * 15 / nlayers) ** 2))
        nx.draw(self.empty_G, pos = {self.empty_node: [total_width, total_height]}, node_color='none', node_shape='o', node_size=int((self.H * 15 / nlayers) ** 2))

        plt.tight_layout()


        plt.savefig(f"{filename}.png", format="PNG")

        self.init_fig()

class Visualizer:
    def weightMappingByCO(hw_config, model_config, Computation_order, filename):
        CU_num = hw_config.CU_num

        graph = MappingGraph(hw_config, model_config)

        node_color_rd = graph.allocate_cu_active_array()
        node_color_cu = graph.allocate_cu_active_array()
        node_color_agg = graph.allocate_router_active_array()
        node_color_pe = graph.allocate_cu_active_array()
        node_color_act = graph.allocate_cu_active_array()
        node_color_pool = graph.allocate_cu_active_array()
        node_color_trans = graph.allocate_router_active_array()

        for event in tqdm(Computation_order):
            if event.event_type == 'cu_operation':
                idx = graph.position_idx_to_idx(event.position_idx)
                graph.set_mapping(idx, event.nlayer)

        # 以 CU 為單位
        # 要有 Router
        # 要把 PE 跟 CU 的結構畫出來

        nlayer = 0
        def draw_all():
            if np.count_nonzero(node_color_cu) == 0:
                return
            #graph.draw("Read from edram to input register", f"{filename}-{nlayer}-0-edram_rd_ir", node_color_rd)
            graph.draw("CU operation", f"{filename}-{nlayer}-1-cu_operation", node_color_cu, active_layers=[nlayer])
            #graph.draw("PE shift and add", f"{filename}-{nlayer}-2-pe_saa", node_color_pe)
            if np.count_nonzero(node_color_agg) > 0:
                graph.draw("Transfer data to aggregator", f"{filename}-{nlayer}-3-trans_to_aggregator", node_color_cu, node_color_agg, active_layers=[nlayer])
            #graph.draw("Activation", f"{filename}-{nlayer}-4-activation", node_color_act)
            #graph.draw("Pooling", f"{filename}-{nlayer}-5-pooling", node_color_pool)
            if np.count_nonzero(node_color_trans) > 0:
                graph.draw("Transfer data to next layer", f"{filename}-{nlayer}-6-trans", node_color_cu, node_color_trans, active_layers=[nlayer])

        print(f'Visualize Computation_order ({len(Computation_order)}) by layer')
        for event in tqdm(Computation_order):
            if event.nlayer != nlayer:
                draw_all()
                nlayer = event.nlayer
                node_color_rd = graph.allocate_cu_active_array()
                node_color_cu = graph.allocate_cu_active_array()
                node_color_agg = graph.allocate_router_active_array()
                node_color_pe = graph.allocate_cu_active_array()
                node_color_act = graph.allocate_cu_active_array()
                node_color_pool = graph.allocate_cu_active_array()
                node_color_trans = graph.allocate_router_active_array()
            if event.event_type == 'edram_rd_ir':
                idx = graph.position_idx_to_idx(event.position_idx)
                assert (node_color_rd[idx] == 1 or node_color_rd[idx] == 0)
                node_color_rd[idx] = 1
            elif event.event_type == 'cu_operation':
                idx = graph.position_idx_to_idx(event.position_idx)
                assert (node_color_cu[idx] == 1 or node_color_cu[idx] == 0)
                node_color_cu[idx] = 1
            elif event.event_type == 'pe_saa':
                position_idx = list(event.position_idx)
                for i in range(CU_num):
                    idx = graph.position_idx_to_idx(position_idx + [i])
                    assert (node_color_pe[idx] == 1 or node_color_pe[idx] == 0)
                    node_color_pe[idx] = 1
            elif event.event_type == 'activation':
                position_idx = list(event.position_idx)
                for i in range(CU_num):
                    idx = graph.position_idx_to_idx(position_idx + [i])
                    assert (node_color_act[idx] == 1 or node_color_act[idx] == 0)
                    node_color_act[idx] = 1
            elif event.event_type == 'pooling':
                position_idx = list(event.position_idx)
                for i in range(CU_num):
                    idx = graph.position_idx_to_idx(position_idx + [i])
                    assert (node_color_pool[idx] == 1 or node_color_pool[idx] == 0)
                    node_color_pool[idx] = 1
            elif event.event_type == 'data_transfer':
                position_idx = event.position_idx
                position_idx_from = list(position_idx[0])
                position_idx_to = list(position_idx[1])

                idx_from = graph.position_idx_to_router_idx(position_idx_from)
                idx_to = graph.position_idx_to_router_idx(position_idx_to)
                is_agg = Computation_order[event.proceeding_event[0]].nlayer == nlayer
                if is_agg:
                    node_color_agg[idx_from] = 1
                    node_color_agg[idx_to] = 1
                else:
                    node_color_trans[idx_from] = 1
                    node_color_trans[idx_to] = 1

                #if position_idx_from == position_idx_to:
                #    continue
                #is_agg = Computation_order[event.proceeding_event[0]].nlayer == nlayer
                #if is_agg:
                #    for i in range(CU_num):
                #        idx_from = graph.position_idx_to_idx(position_idx_from + [i])
                #        idx_to = graph.position_idx_to_idx(position_idx_to + [i])
                #        #if nlayer == 4:
                #        #print(nlayer, idx_from, idx_to)
                #        #print(event)
                #        assert node_color_agg[idx_from] != 2
                #        #assert node_color_agg[idx_to] != 1
                #        node_color_agg[idx_from] = 1
                #        node_color_agg[idx_to] = 2
                #else:
                #    for i in range(CU_num):
                #        idx_from = graph.position_idx_to_idx(position_idx_from + [i])
                #        idx_to = graph.position_idx_to_idx(position_idx_to + [i])
                #        #print(nlayer, idx_from, idx_to)
                #        #print(event)
                #        assert node_color_trans[idx_from] != 2
                #        #assert node_color_trans[idx_to] != 1
                #        node_color_trans[idx_from] = 1
                #        node_color_trans[idx_to] = 2
        # draw last layer
        draw_all()

        graph.draw('Weight mapping', filename, graph.weight_mapping, active_layers=[i for i in range(nlayer+1)])

    def visualizeSimulation(hw_config, model_config, Computation_order, simulation_log, filename):
        print(f'len(simulation_log) = {len(simulation_log)}')
        print(f'len(Computation_order) = {len(Computation_order)}')
        CU_num = hw_config.CU_num

        graph = MappingGraph(hw_config, model_config)

        print('Prepare weight mapping...')
        for event in tqdm(Computation_order):
            if event.event_type == 'cu_operation':
                idx = graph.position_idx_to_idx(event.position_idx)
                graph.set_mapping(idx, event.nlayer)

        time_queue = []
        for event_idx in simulation_log:
            time_queue.append((simulation_log[event_idx][0], event_idx, 'start'))
            time_queue.append((simulation_log[event_idx][1], event_idx, 'end'))
        time_queue.sort()

        active_cu = graph.allocate_cu_active_array()
        active_router = [0] * graph.router_size

        last_cycle = None
        last_count = 0
        for (cycle, event_idx, status) in tqdm(time_queue):
            current_active_count = (np.count_nonzero(active_cu) + np.count_nonzero(active_router))
            if (last_cycle == None or cycle != last_cycle) and (last_count != current_active_count) and abs(current_active_count-last_count) > 2:
                if last_cycle == None:
                    last_cycle = 0
                print(last_cycle, cycle, last_count, (np.count_nonzero(active_cu) + np.count_nonzero(active_router)))
                graph.draw(f"Cycle [{last_cycle} ~ {cycle})", f"{filename}-{cycle}", active_cu, active_router)
                last_cycle = cycle
                last_count = current_active_count

            if status == 'start':
                active = 1
            else:
                active = -1

            event = Computation_order[event_idx]

            if event.event_type == 'edram_rd_ir':
                idx = graph.position_idx_to_idx(event.position_idx)
                active_cu[idx] += active
            elif event.event_type == 'cu_operation':
                idx = graph.position_idx_to_idx(event.position_idx)
                active_cu[idx] += active
            elif event.event_type == 'pe_saa':
                position_idx = list(event.position_idx)
                for i in range(CU_num):
                    idx = graph.position_idx_to_idx(position_idx + [i])
                    #active_cu[idx] += active
            elif event.event_type == 'activation':
                position_idx = list(event.position_idx)
                for i in range(CU_num):
                    idx = graph.position_idx_to_idx(position_idx + [i])
                    #active_cu[idx] += active
            elif event.event_type == 'pooling':
                position_idx = list(event.position_idx)
                for i in range(CU_num):
                    idx = graph.position_idx_to_idx(position_idx + [i])
                    #active_cu[idx] += active
            elif event.event_type == 'data_transfer':
                position_idx = event.position_idx
                position_idx_from = list(position_idx[0])
                position_idx_to = list(position_idx[1])
                if position_idx_from == position_idx_to:
                    continue

                print(position_idx)
                temp_idx = position_idx_from.copy()
                print(temp_idx)
                active_router[temp_idx[1]*graph.router_height + temp_idx[0]] += active
                while temp_idx[1] != position_idx_to[1]:
                    temp_idx[1] += (position_idx_to[1] - temp_idx[1]) // abs(position_idx_to[1] - temp_idx[1])
                    print(temp_idx)
                    active_router[temp_idx[1]*graph.router_height + temp_idx[0]] += active
                while temp_idx[0] != position_idx_to[0]:
                    temp_idx[0] += (position_idx_to[0] - temp_idx[0]) // abs(position_idx_to[0] - temp_idx[0])
                    print(temp_idx)
                    active_router[temp_idx[1]*graph.router_height + temp_idx[0]] += active

    def visualizeSimulation2(hw_config, model_config, Computation_order, simulation_log, filename):
        print(f'len(simulation_log) = {len(simulation_log)}')
        print(f'len(Computation_order) = {len(Computation_order)}')
        CU_num = hw_config.CU_num

        graph = MappingGraph(hw_config, model_config)

        print('Prepare weight mapping...')
        for event in tqdm(Computation_order):
            if event.event_type == 'cu_operation':
                idx = graph.position_idx_to_idx(event.position_idx)
                graph.set_mapping(idx, event.nlayer)

        time_queue = []
        print('Preprocess window events...')
        for (start_cycle, end_cycle, event) in tqdm(simulation_log['window_event']):
            if not event.nlayer in CARE_LAYERS:
                continue
            time_queue.append((start_cycle, event, 'start'))
            time_queue.append((end_cycle-1, event, 'end'))
        for (start_cycle, transfer_data, src, des) in tqdm(simulation_log['data_transfer']):
            #if not transfer_data[0][0] in CARE_LAYERS:
            #    continue
            time_queue.append((start_cycle, src, 'data_transfer'))
            time_queue.append((start_cycle, des, 'data_transfer'))
        for (start_cycle, event) in tqdm(simulation_log['pooling']):
            #if not transfer_data[0][0] in CARE_LAYERS:
            #    continue
            time_queue.append((start_cycle, event, 'pooling'))
        time_queue.sort(key=lambda k: k[0])

        index = 0
        def draw(start_cycle, end_cycle, active_cu, active_pe, active_router, window_ids=None):
            nonlocal index
            active_layers = set()
            active_windows = set()
            for window_id, count in window_ids.items():
                if count > 0:
                    active_layers.add(window_id[0])
                    active_windows.add(window_id)
            print(start_cycle, end_cycle, active_layers, active_windows, active_router)
            graph.draw(f"Cycle [{start_cycle} ~ {end_cycle})", f"{filename}-{index:003}", active_cu, active_pe, active_router, active_layers=active_layers, active_windows=active_windows)
            index += 1

        last_cycle = (time_queue[0][0] // STEP_CYCLES) * STEP_CYCLES
        active_cu = graph.allocate_cu_active_array()
        deactive_cu = graph.allocate_cu_active_array()
        active_router = graph.allocate_router_active_array()
        active_pe = None

        window_id_count = {}
        window_id_deactive_count = {}
        for (cycle, event, status) in tqdm(time_queue):
            while cycle - last_cycle >= STEP_CYCLES:
                draw(last_cycle, last_cycle+STEP_CYCLES, active_cu, active_pe, active_router, window_id_count)
                # Clean up this range events
                for k, v in enumerate(active_cu):
                    active_cu[k] -= deactive_cu[k]
                for window_id in window_id_deactive_count:
                    window_id_count[window_id] -= window_id_deactive_count[window_id]
                deactive_cu = graph.allocate_cu_active_array()
                active_router = graph.allocate_router_active_array()
                window_id_deactive_count = {}
                last_cycle += STEP_CYCLES

            if status == 'data_transfer':
                active_router[graph.position_idx_to_router_idx(event)] = 1
            elif status == 'pooling':
                #active_router[graph.position_idx_to_router_idx(event)] = 1
                pass
            else: # cu_operation
                window_id = event.window_id

                idx = graph.position_idx_to_idx(event.position_idx)

                if status == 'start':
                    active_cu[idx] += 1
                    if not window_id in window_id_count:
                        window_id_count[window_id] = 0
                    window_id_count[window_id] += 1
                else: # status == 'end'
                    deactive_cu[idx] += 1
                    if not window_id in window_id_deactive_count:
                        window_id_deactive_count[window_id] = 0
                    window_id_deactive_count[window_id] += 1

        draw(last_cycle, last_cycle+STEP_CYCLES, active_cu, active_pe, active_router, window_id_count)


    def visualizeGif(hw_config, model_config, Computation_order, filename):
        print(f'len(Computation_order) = {len(Computation_order)}')

        graph = MappingGraph(hw_config, model_config)

        print('Prepare weight mapping...')
        for event in tqdm(Computation_order):
            if event.event_type == 'cu_operation':
                idx = graph.position_idx_to_idx(event.position_idx)
                graph.set_mapping(idx, event.nlayer)

        active_cu = graph.allocate_cu_active_array()

        index = 0
        def draw_all(window_id):
            nonlocal index
            graph.draw(f"Active CU", f"{filename}-{index:03}", active_cu, active_layers=[window_id[0]], window_id=window_id)
            index += 1

        last_window_id = None
        for event in tqdm(Computation_order):
            if not event.nlayer in CARE_LAYERS:
                continue

            if hasattr(event, 'window_id'):
                if event.window_id != last_window_id:
                    # New window
                    if last_window_id != None:
                        draw_all(last_window_id)
                        active_cu = graph.allocate_cu_active_array()
                    last_window_id = event.window_id
                idx = graph.position_idx_to_idx(event.position_idx)
                active_cu[idx] = 1
        draw_all(last_window_id)
