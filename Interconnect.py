from Router import Router
import numpy as np
import time

class Interconnect(object):
    def __init__(self, hw_config):
        self.router_array = []
        self.rt_h = hw_config.Router_num_y
        self.rt_w = hw_config.Router_num_x
        for h in range(self.rt_h):
            for w in range(self.rt_w):
                self.router_array.append(Router(h, w))
        self.router_array = np.array(self.router_array)
        self.router_array = self.router_array.reshape(self.rt_h, self.rt_w)
        self.busy_router = set()

    def input_packet(self, packet):
        source = packet.source
        rt_y, rt_x = source[0], source[1]
        # pe_y, pe_x = source[2], source[3]
        router = self.router_array[rt_y][rt_x]
        router.input_queue.append(packet)
        self.busy_router.add(router)

    def step(self):
        arrived_packet = []
        br = set()
        for router in self.busy_router:
            packet, direction = router.step()
            h = router.pos[0]
            w = router.pos[1]
            if self.router_array[h][w].input_queue:
                br.add(self.router_array[h][w])
            
            if direction == "arrived":
                arrived_packet.append(packet)
            else:
                if direction == "north":
                    h -= 1
                elif direction == "south":
                    h += 1
                elif direction == "east":
                    w += 1
                elif direction == "west":
                    w -= 1
                else:
                    print("interconnect error")
                    exit()
                self.router_array[h][w].input_queue.append(packet)
                br.add(self.router_array[h][w])
        self.busy_router = br
        return arrived_packet

    def __str__(self):
        return str(self.__dict__)
    