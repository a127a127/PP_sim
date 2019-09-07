import os
import csv
from PE import PE
import numpy as np
from math import ceil, floor
from EventMetaData import EventMetaData
from FetchEvent import FetchEvent
from HardwareMetaData import HardwareMetaData

from Interconnect import Interconnect
from Packet import Packet

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Controller(object):
    def __init__(self, ordergenerator, isPipeLine, trace, mapping_str):
        self.ordergenerator = ordergenerator
        self.isPipeLine = isPipeLine
        self.trace = trace
        self.mapping_str = mapping_str
        self.Computation_order = self.ordergenerator.Computation_order
        self.hd_info = HardwareMetaData()
        self.input_bit = self.ordergenerator.model_info.input_bit
        self.cycle_ctr = 0
        self.edram_read_cycles = \
            ceil(self.hd_info.Xbar_num * self.hd_info.Xbar_h * \
            self.input_bit * self.hd_info.eDRAM_read_latency / \
            self.hd_info.cycle_time)
        # Energy
        self.Total_energy_cycle = 0
        self.Total_energy_edram_buffer = 0
        self.Total_energy_bus = 0
        self.Total_energy_router = 0
        self.Total_energy_activation = 0
        self.Total_energy_pe_shift_and_add = 0
        self.Total_energy_cu_shift_and_add = 0
        self.Total_energy_pooling = 0
        self.Total_energy_or = 0
        self.Total_energy_adc = 0
        self.Total_energy_dac = 0
        self.Total_energy_crossbar = 0
        self.Total_energy_ir_in_cu = 0
        self.Total_energy_or_in_cu = 0
        self.Total_energy_interconnect = 0
        # PE object
        self.PE_array = []
        for rty_idx in range(self.hd_info.Router_num_y):
            for rtx_idx in range(self.hd_info.Router_num_x):
                for pey_idx in range(self.hd_info.PE_num_y):
                    for pex_idx in range(self.hd_info.PE_num_x):
                        pe_pos = (rty_idx, rtx_idx, pey_idx, pex_idx)
                        pe = PE(pe_pos, self.input_bit)
                        self.PE_array.append(pe)
        # Statistics
        self.color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        self.mem_acc_ctr = 0
        self.data_transfer_ctr = 0 
        self.act_xb_ctr = 0
        self.pe_saa_stall_cycle = 0
        ## Utilization
        self.energy_utilization = []
        self.xbar_utilization = []
        self.pe_state_for_plot = [[], []]
        self.cu_state_for_plot = [[], []]
        self.xb_state_for_plot = [[], []]
        self.buffer_size = []
        for i in range(len(self.PE_array)):
            self.buffer_size.append([])
        # Interconnect
        self.fetch_array = []
        self.interconnect = Interconnect(self.hd_info.Router_num_y, self.hd_info.Router_num_x, self.input_bit)
        self.interconnect_step = self.hd_info.Router_flit_size / self.input_bit * self.hd_info.cycle_time * self.hd_info.Frequency # scaling from ISAAC
        self.interconnect_step = floor(self.interconnect_step)
        self.interconnect_step = 4000
        self.data_transfer_trigger = []
        self.data_transfer_erp = []
        # Pipeline control
        if not self.isPipeLine:
            self.pipeline_layer_stage = 0
            self.pipeline_stage_record = []
            self.events_each_layer = []
            for layer in range(self.ordergenerator.model_info.layer_length):
                self.events_each_layer.append(0)
            for e in self.Computation_order:
                self.events_each_layer[e.nlayer] += 1
            self.this_layer_event_ctr = 0
            self.this_layer_cycle_ctr = 0
            self.cycles_each_layer = []

    def run(self):
        for e in self.Computation_order:
            if e.preceding_event_count == e.current_number_of_preceding_event:
                if e.event_type == 'edram_rd_ir':
                    pos = e.position_idx
                    rty, rtx, pey, pex, cuy, cux = pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]
                    idx = pex + pey * self.hd_info.PE_num_x + rtx * self.hd_info.PE_num + rty * self.hd_info.Router_num_x * self.hd_info.PE_num
                    cu_idx = cux + cuy * self.hd_info.CU_num_x
                    self.PE_array[idx].CU_array[cu_idx].edram_rd_ir_erp.append(e)
                else:
                    print("Computation order error: event\"", e.event_type, "\".")
                    print("exit...")
                    exit()

        isDone = False
        while not isDone:
            self.Total_energy_cycle = 0
            self.cycle_ctr += 1
            self.this_layer_cycle_ctr += 1
            self.act_xb_ctr = 0

            if self.trace:
                print("cycle:", self.cycle_ctr)

            ### Interconnect 
            for s in range(self.interconnect_step):
                self.interconnect.step()
                self.Total_energy_interconnect += self.interconnect.step_energy_consumption

            # Packets arrive: Store data, trigger event
            arrived_packet = self.interconnect.get_arrived_packet()
            for pk in arrived_packet:
                if self.trace:
                    pass
                    #print("\tArrived packet:", pk)
                if not self.isPipeLine:
                    self.this_layer_event_ctr += 1
                rty, rtx = pk.destination[0], pk.destination[1]
                pey, pex = pk.destination[2], pk.destination[3]
                pe_idx = pex + pey * self.hd_info.PE_num_x + rtx * self.hd_info.PE_num + rty * self.hd_info.PE_num * self.hd_info.Router_num_x
                pe = self.PE_array[pe_idx]
                
                self.Total_energy_edram_buffer += self.hd_info.Energy_edram_buffer * self.input_bit # write
                self.Total_energy_cycle += self.hd_info.Energy_edram_buffer * self.input_bit

                pro_event = self.Computation_order[pk.pro_event_idx]
                if pro_event.event_type == "edram_rd_ir":
                    # store data
                    if self.trace:
                        pass
                        #print("\t\twrite data into buffer:", pk.data)
                    pe.edram_buffer.put(pk.data)
                    # trigger event
                    cuy, cux = pro_event.position_idx[4], pro_event.position_idx[5]

                    cu_idx = cux + cuy * self.hd_info.CU_num_x
                    cu = pe.CU_array[cu_idx]
                    pro_event.current_number_of_preceding_event += 1
                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                        if self.trace:
                            print("\t\tProceeding event is triggered.", \
                               pro_event.event_type, pro_event.position_idx, "index:", self.Computation_order.index(pro_event))
                        pe.edram_rd_ir_trigger.append([pro_event, [cu_idx]])
                elif pro_event.event_type == "edram_rd_pool":
                    # store data
                    if self.trace:
                        print("\t\twrite data into buffer:", pk.data)
                    pe.edram_buffer.put(pk.data)
                    # trigger event
                    pro_event.current_number_of_preceding_event += 1
                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                        if self.trace:
                            pass
                            #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                        pe.edram_rd_pool_trigger.append([pro_event, []])
                elif pro_event.event_type == "pe_saa":
                    # trigger event
                    pro_event.current_number_of_preceding_event += 1
                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                        if self.trace:
                            pass
                            #print("\t\tProceeding event is triggered.", pro_event.event_type)
                        
                        pe.pe_saa_trigger.append([pro_event, []])

            # Event: data_transfer
            for event in self.data_transfer_erp.copy():
                if self.trace:
                    print("\tdo data_transfer, layer:", event.nlayer, ",order index:", self.Computation_order.index(event), \
                            "pos:", event.position_idx, "data:", event.outputs)
                self.data_transfer_erp.remove(event)

                src = event.position_idx[0]
                des = event.position_idx[1]
                
                pro_event_idx = event.proceeding_event[0]
                if self.Computation_order[pro_event_idx].event_type == "edram_rd_ir":
                    packet = Packet(src, des, [event.nlayer+1, event.outputs[0]], pro_event_idx)
                elif self.Computation_order[pro_event_idx].event_type == "edram_rd_pool":
                    packet = Packet(src, des, [event.nlayer+1, event.outputs[0]], pro_event_idx)
                else:
                    packet = Packet(src, des, [], pro_event_idx)
                self.interconnect.input_packet(packet)

                self.Total_energy_edram_buffer += self.hd_info.Energy_edram_buffer * self.input_bit # read
                self.Total_energy_cycle += self.hd_info.Energy_edram_buffer * self.input_bit

            ### Fetch data from off-chip memory
            for FE in self.fetch_array.copy():
                FE.cycles_counter += 1
                if FE.cycles_counter == FE.fetch_cycle:
                    src  = (0, FE.event.position_idx[1], -1, -1)
                    des  = FE.event.position_idx[0:4]
                    if FE.event.event_type == "edram_rd_ir":
                        if not self.isPipeLine:
                            self.this_layer_event_ctr -= len(FE.event.inputs)
                        FE.event.preceding_event_count += len(FE.event.inputs)
                        for inp in FE.event.inputs:
                            data = inp[1:]
                            pro_event_idx = self.Computation_order.index(FE.event)
                            packet = Packet(src, des, [FE.event.nlayer, data], pro_event_idx)
                            self.interconnect.input_packet(packet)

                    elif FE.event.event_type == "edram_rd_pool":
                        if not self.isPipeLine:
                            self.this_layer_event_ctr -= len(FE.event.inputs)
                        FE.event.preceding_event_count += len(FE.event.inputs)

                        for data in FE.event.inputs:
                            pro_event_idx = self.Computation_order.index(FE.event)
                            packet = Packet(src, des, [FE.event.nlayer, data], pro_event_idx)
                            self.interconnect.input_packet(packet)
                    self.fetch_array.remove(FE)

            ### Event: edram_rd_ir
            for pe in self.PE_array:
                for cu in pe.CU_array:
                    if not cu.state and not cu.state_edram_rd_ir: # 這個條件才能消耗一個edram read event
                        if cu.edram_rd_ir_erp: # 也要有read event才能做事
                            event = cu.edram_rd_ir_erp[0]

                            ## Is Data in eDRAM buffer
                            isData_ready = True
                            # inputs: [[num_input, fm_h, fm_w, fm_c]]
                            for inp in event.inputs:
                                data = inp[1:]
                                #print(event.nlayer, data)
                                if not pe.edram_buffer.check([event.nlayer, data]):
                                    # Data not in buffer
                                    if self.trace:
                                        print("\tData not ready for edram_rd_ir. Data: layer", event.nlayer, \
                                            event.event_type, "index:", self.Computation_order.index(event), data,
                                            "position:", cu.position)
                                        #print("\tBuffer:", pe.edram_buffer.buffer)
                                    isData_ready = False
                                    break
                            
                            if not isData_ready:
                                self.mem_acc_ctr += 1
                                cu.edram_rd_ir_erp.remove(event)
                                pe_idx = self.PE_array.index(pe)
                                cu_idx = pe.CU_array.index(cu)
                                self.fetch_array.append(FetchEvent(event, [pe_idx, cu_idx]))
                            else:
                                # 開始執行edram read event
                                if self.trace:
                                    print("\tdo edram_rd_ir, nlayer:", event.nlayer,", cu_pos:", cu.position, ",order index:", self.Computation_order.index(event))
                                    print("\t\tread data:", event.inputs)
                                if not self.isPipeLine:
                                    self.this_layer_event_ctr += 1

                                num_data = len(event.outputs)
                                self.Total_energy_edram_buffer += self.hd_info.Energy_edram_buffer * self.input_bit * num_data
                                self.Total_energy_bus += self.hd_info.Energy_bus * self.input_bit * num_data
                                self.Total_energy_ir_in_cu += self.hd_info.Energy_ir_in_cu * self.input_bit * num_data
                                self.Total_energy_cycle += (self.hd_info.Energy_edram_buffer + self.hd_info.Energy_bus + self.hd_info.Energy_ir_in_cu) * num_data

                                cu.state = True
                                cu.state_edram_rd_ir = True
                                cu.edram_rd_ir_erp.remove(event)
                                cu.edram_rd_event = event
                                cu.edram_rd_cycle_ctr += 1

                                if cu.edram_rd_cycle_ctr == self.edram_read_cycles: # finish edram read
                                    cu.edram_rd_cycle_ctr = 0
                                    ### add next event counter: ou
                                    for proceeding_index in event.proceeding_event:
                                        pro_event = self.Computation_order[proceeding_index]
                                        pro_event.current_number_of_preceding_event += 1

                                        if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                            if self.trace:
                                                pass
                                                #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                            pos = pro_event.position_idx
                                            cu_y, cu_x, xb_y, xb_x = pos[4], pos[5], pos[6], pos[7]
                                            cu_idx = cu_x + cu_y * self.hd_info.CU_num_x
                                            xb_idx = xb_x + xb_y * self.hd_info.Xbar_num_x
                                            cu.ou_trigger.append([pro_event, [cu_idx, xb_idx]])

                    elif cu.state and cu.state_edram_rd_ir:
                        cu.edram_rd_cycle_ctr += 1
                        if cu.edram_rd_cycle_ctr == self.edram_read_cycles: # finish edram read
                            cu.edram_rd_cycle_ctr = 0
                            cu.state_edram_rd_ir = False
                            event = cu.edram_rd_event
                            ### add next event counter: ou
                            for proceeding_index in event.proceeding_event:
                                pro_event = self.Computation_order[proceeding_index]
                                pro_event.current_number_of_preceding_event += 1

                                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                    if self.trace:
                                        pass
                                        #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                    pos = pro_event.position_idx
                                    cu_y, cu_x, xb_y, xb_x = pos[4], pos[5], pos[6], pos[7]
                                    cu_idx = cu_x + cu_y * self.hd_info.CU_num_x
                                    xb_idx = xb_x + xb_y * self.hd_info.Xbar_num_x
                                    cu.ou_trigger.append([pro_event, [cu_idx, xb_idx]])

                                ### add next event counter: ou
                                for proceeding_index in event.proceeding_event:
                                    pro_event = self.Computation_order[proceeding_index]
                                    pro_event.current_number_of_preceding_event += 1

                                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                        if self.trace:
                                            pass
                                            #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                        pos = pro_event.position_idx
                                        cu_y, cu_x, xb_y, xb_x = pos[4], pos[5], pos[6], pos[7]
                                        cu_idx = cu_x + cu_y * self.hd_info.CU_num_x
                                        xb_idx = xb_x + xb_y * self.hd_info.Xbar_num_x
                                        cu.ou_trigger.append([pro_event, [cu_idx, xb_idx]])
                        else:
                            break
            
            ### Event: ou
            for pe in self.PE_array:
                for cu in pe.CU_array:
                    for xb in cu.XB_array:
                        for event in xb.ou_erp.copy():
                            for idx in range(len(xb.state_ou)):
                                if not xb.state_ou[idx]:
                                    if self.trace:
                                        pass
                                        print("\tdo ou, xb_pos:", xb.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                                    
                                    if not self.isPipeLine:
                                        self.this_layer_event_ctr += 1
                                    
                                    self.Total_energy_ir_in_cu += self.hd_info.Energy_ir_in_cu * self.hd_info.OU_h
                                    self.Total_energy_dac += self.hd_info.Energy_dac
                                    self.Total_energy_crossbar += self.hd_info.Energy_crossbar
                                    self.Total_energy_cycle += self.hd_info.Energy_ir_in_cu * self.hd_info.OU_h + \
                                                                self.hd_info.Energy_dac + \
                                                                self.hd_info.Energy_crossbar
                                    self.act_xb_ctr += 1

                                    xb.state_ou[idx] = True
                                    xb.ou_erp.remove(event)

                                    ### add next event counter: adc
                                    for proceeding_index in event.proceeding_event:
                                        pro_event = self.Computation_order[proceeding_index]
                                        pro_event.current_number_of_preceding_event += 1
                                        
                                        if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                            if self.trace:
                                                pass
                                                print("\t\tProceeding event is triggered.", pro_event.event_type)

                                            pos = pro_event.position_idx
                                            cu_y, cu_x = pos[4], pos[5]
                                            cu_idx = cu_x + cu_y * self.hd_info.CU_num_x
                                            xb.adc_trigger.append([pro_event, [cu_idx]])
                                    break
            ### Event: adc
            for pe in self.PE_array:
                for cu in pe.CU_array:
                    for event in cu.adc_erp.copy():
                        for idx in range(len(cu.state_adc)):
                            if not cu.state_adc[idx]:
                                if self.trace:
                                    pass
                                    print("\tdo adc, cu_pos:", cu.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                                if not self.isPipeLine:
                                    self.this_layer_event_ctr += 1

                                self.Total_energy_adc += self.hd_info.Energy_adc
                                self.Total_energy_cycle += self.hd_info.Energy_adc

                                cu.adc_erp.remove(event)
                                cu.state_adc[idx] = True

                                ### add next event counter: cu_saa
                                for proceeding_index in event.proceeding_event:
                                    pro_event = self.Computation_order[proceeding_index]
                                    pro_event.current_number_of_preceding_event += 1

                                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                        if self.trace:
                                            pass
                                            print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx, self.Computation_order.index(pro_event))
                                        cu.cu_saa_trigger.append([pro_event, []])
                                break

            ### Event: cu_saa 
            for pe in self.PE_array:
                for cu in pe.CU_array:
                    for event in cu.cu_saa_erp.copy():
                        if self.trace:
                            pass
                            print("\tdo cu_saa, cu_pos:", cu.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                        if not self.isPipeLine:
                            self.this_layer_event_ctr += 1

                        cu.cu_saa_erp.remove(event)

                        need_saa = 0
                        for saa_amount in range(len(event.inputs)): # amounts of saa 
                            need_saa += 1
                            for idx in range(len(cu.state_cu_saa)):
                                if not cu.state_cu_saa[idx]:
                                    self.Total_energy_or_in_cu += self.hd_info.Energy_or_in_cu * self.input_bit # read
                                    self.Total_energy_cu_shift_and_add += self.hd_info.Energy_shift_and_add
                                    self.Total_energy_or_in_cu += self.hd_info.Energy_or_in_cu * self.input_bit # write
                                    self.Total_energy_cycle += self.hd_info.Energy_or_in_cu * self.input_bit + \
                                                                self.hd_info.Energy_shift_and_add + \
                                                                self.hd_info.Energy_or_in_cu * self.input_bit

                                    cu.state_cu_saa[idx] = True
                                    break
                        if need_saa > len(cu.state_cu_saa):
                            print("no enough cu saa per cycle..")
                            exit()
                        
                        ### add next event counter: pe_saa, data_transfer
                        for proceeding_index in event.proceeding_event:
                            pro_event = self.Computation_order[proceeding_index]
                            pro_event.current_number_of_preceding_event += 1

                            if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                if self.trace:
                                    pass
                                    #print("\t\tProceeding event is triggered.", pro_event.event_type)
                                if pro_event.event_type == "pe_saa":
                                    cu.pe_saa_trigger.append([pro_event, []])

                                elif pro_event.event_type == "edram_wr":
                                    pe.edram_wr_trigger.append([pro_event, []])

            ### Event: pe_saa 
            for pe in self.PE_array:
                for event in pe.pe_saa_erp.copy():
                    if self.trace:
                        #pass
                        print("\tdo pe_saa, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                    if not self.isPipeLine:
                        self.this_layer_event_ctr += 1
                    
                    pe.pe_saa_erp.remove(event)
                    

                    saa_amount = len(event.inputs)
                    if saa_amount > len(pe.state_pe_saa):
                        print("Not enough pe_saa per cycle")
                        exit()

                    for s in range(saa_amount):
                        for idx in range(len(pe.state_pe_saa)):
                            if not pe.state_pe_saa[idx]:
                                self.Total_energy_or += self.hd_info.Energy_or * self.input_bit
                                self.Total_energy_bus += self.hd_info.Energy_bus * self.input_bit
                                self.Total_energy_pe_shift_and_add += self.hd_info.Energy_shift_and_add
                                self.Total_energy_bus += self.hd_info.Energy_bus * self.input_bit
                                self.Total_energy_or += self.hd_info.Energy_or * self.input_bit
                                self.Total_energy_cycle += self.hd_info.Energy_or * self.input_bit + \
                                                            self.hd_info.Energy_bus * self.input_bit + \
                                                            self.hd_info.Energy_shift_and_add + \
                                                            self.hd_info.Energy_bus * self.input_bit + \
                                                            self.hd_info.Energy_or * self.input_bit
                                pe.state_pe_saa[idx] = True
                                break

                        ### add next event counter: activation
                        for proceeding_index in event.proceeding_event:
                            pro_event = self.Computation_order[proceeding_index]
                            pro_event.current_number_of_preceding_event += 1
                            
                            if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                if self.trace:
                                    pass
                                    #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                pe.activation_trigger.append([pro_event, []])

            ### Event: activation 
            for pe in self.PE_array:
                for event in pe.activation_erp.copy():
                    for idx in range(len(pe.state_activation)):
                        if not pe.state_activation[idx]:
                            if self.trace:
                                print("\tdo activation, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                            if not self.isPipeLine:
                                self.this_layer_event_ctr += 1

                            self.Total_energy_or += self.hd_info.Energy_or * self.input_bit
                            self.Total_energy_bus += self.hd_info.Energy_bus * self.input_bit
                            self.Total_energy_activation += self.hd_info.Energy_activation
                            self.Total_energy_cycle += self.hd_info.Energy_bus * self.input_bit + self.hd_info.Energy_activation
                            
                            pe.state_activation[idx] = True
                            pe.activation_erp.remove(event)

                            ### add next event counter: edram_wr
                            for proceeding_index in event.proceeding_event:
                                pro_event = self.Computation_order[proceeding_index]
                                pro_event.current_number_of_preceding_event += 1
                                
                                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                    if self.trace:
                                        pass
                                        #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx, self.Computation_order.index(pro_event))
                                    pe.edram_wr_trigger.append([pro_event, []])
                            break

            ### Event: edram write 
            for pe in self.PE_array:
                for event in pe.edram_wr_erp.copy():
                    for idx in range(len(pe.state_edram_wr)):
                        if not pe.state_edram_wr[idx]:
                            if self.trace:
                                print("\tdo edram_wr, pe_pos:", pe.position, "layer:", event.nlayer, \
                                        ",order index:", self.Computation_order.index(event), "data:", event.outputs)
                            if not self.isPipeLine:    
                                self.this_layer_event_ctr += 1

                            #self.Total_energy_or += self.hd_info.Energy_or * self.input_bit
                            
                            self.Total_energy_bus += self.hd_info.Energy_bus * self.input_bit
                            self.Total_energy_edram_buffer += self.hd_info.Energy_edram_buffer * self.input_bit
                            self.Total_energy_cycle += self.hd_info.Energy_bus * self.input_bit + \
                                                        self.hd_info.Energy_edram_buffer * self.input_bit

                            pe.state_edram_wr[idx] = True
                            pe.edram_wr_erp.remove(event)

                            pe.edram_buffer.put([event.nlayer+1, event.outputs[0]])

                            ### add next event counter: edram_rd_ir, edram_rd_pool, data_transfer
                            for proceeding_index in event.proceeding_event:
                                pro_event = self.Computation_order[proceeding_index]
                                pro_event.current_number_of_preceding_event += 1
                            
                                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                    if self.trace:
                                        pass
                                        #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                    pos = pro_event.position_idx
                                    if pro_event.event_type == "edram_rd_ir":
                                        cu_y, cu_x = pos[4], pos[5]
                                        cu_idx = cu_x + cu_y * self.hd_info.CU_num_x
                                        pe.edram_rd_ir_trigger.append([pro_event, [cu_idx]])
                                    elif pro_event.event_type == "edram_rd_pool":
                                        pe.edram_rd_pool_trigger.append([pro_event, []])
                                    elif pro_event.event_type == "data_transfer":
                                        self.data_transfer_trigger.append([pro_event, []])
                            break
            
            ### Event: edram_rd_pool
            for pe in self.PE_array:
                if pe.edram_rd_pool_erp:
                    event = pe.edram_rd_pool_erp[0]
                else:
                    continue
                if not pe.state_edram_rd_pool:
                    ## Data in eDRAM buffer
                    isData_ready = True
                    for data in event.inputs:
                        #print(event.nlayer, data)
                        if not pe.edram_buffer.check([event.nlayer, data]):
                            # Data not in buffer
                            if self.trace:
                                print("\tData not ready for edram_rd_pool. Data: layer", event.nlayer, event.event_type, data)
                            isData_ready = False
                            pe.edram_rd_pool_erp.remove(event)
                            break
                    
                    if not isData_ready:
                        pe_idx = self.PE_array.index(pe)
                        self.fetch_array.append(FetchEvent(event, [pe_idx]))

                    else:
                        if self.trace:
                            print("\tdo edram_rd_pool, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                        if not self.isPipeLine:
                            self.this_layer_event_ctr += 1

                        num_data = len(event.outputs)
                        self.Total_energy_edram_buffer += self.hd_info.Energy_edram_buffer * self.input_bit * num_data
                        self.Total_energy_bus += self.hd_info.Energy_bus * self.input_bit * num_data
                        self.Total_energy_cycle += self.hd_info.Energy_edram_buffer * self.input_bit * num_data + \
                                                    self.hd_info.Energy_bus * self.input_bit * num_data
                        
                        pe.state_edram_rd_pool = True
                        pe.edram_rd_pool_erp.remove(event)
                        
                        ### add next event counter: pooling
                        for proceeding_index in event.proceeding_event:
                            pro_event = self.Computation_order[proceeding_index]
                            pro_event.current_number_of_preceding_event += 1
                            
                            if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                if self.trace:
                                    pass
                                    #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                pos = pro_event.position_idx
                                pe.pooling_trigger.append([pro_event, []])                                
                    
            ### Event: pooling 
            for pe in self.PE_array:
                for event in pe.pooling_erp.copy():
                    for idx in range(len(pe.state_pooling)):
                        if not pe.state_pooling[idx]:
                            if self.trace:
                                print("\tdo pooling, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                            if not self.isPipeLine:
                                self.this_layer_event_ctr += 1

                            self.Total_energy_pooling += self.hd_info.Energy_pooling
                            self.Total_energy_cycle += self.hd_info.Energy_pooling
                            
                            pe.state_pooling[idx] = True
                            pe.pooling_erp.remove(event)

                            ### add next event counter: edram_wr
                            for proceeding_index in event.proceeding_event:
                                pro_event = self.Computation_order[proceeding_index]
                                pro_event.current_number_of_preceding_event += 1

                                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                    if self.trace:
                                        pass
                                        #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                    pe.edram_wr_trigger.append([pro_event, []])
                            break



            ### Pipeline stage control ###
            if not self.isPipeLine:
                self.pipeline_stage_record.append(self.pipeline_layer_stage)
                if self.this_layer_event_ctr == self.events_each_layer[self.pipeline_layer_stage]:
                    self.pipeline_layer_stage += 1
                    self.cycles_each_layer.append(self.this_layer_cycle_ctr)
                    self.this_layer_event_ctr = 0
                    self.this_layer_cycle_ctr = 0



            ### Trigger events ###
            ## Trigger interconnect
            for trigger in self.data_transfer_trigger.copy():
                pro_event = trigger[0]
                if not self.isPipeLine:
                    if pro_event.nlayer == self.pipeline_layer_stage:
                        self.data_transfer_erp.append(pro_event)
                        self.data_transfer_trigger.remove(trigger)
                else:
                    self.data_transfer_erp.append(pro_event)
                    self.data_transfer_trigger.remove(trigger)

            for pe in self.PE_array:
                ## Trigger activation 
                for trigger in pe.activation_trigger.copy():
                    pro_event = trigger[0]
                    if not self.isPipeLine:
                        if pro_event.nlayer == self.pipeline_layer_stage:
                            pe.activation_erp.append(pro_event)
                            pe.activation_trigger.remove(trigger)
                    else:
                        pe.activation_erp.append(pro_event)
                        pe.activation_trigger.remove(trigger)

                ## Trigger edram_wr 
                for trigger in pe.edram_wr_trigger.copy():
                    pro_event = trigger[0]
                    if not self.isPipeLine:
                        if pro_event.nlayer == self.pipeline_layer_stage:
                            pe.edram_wr_erp.append(pro_event)
                            pe.edram_wr_trigger.remove(trigger)
                    else:
                        pe.edram_wr_erp.append(pro_event)
                        pe.edram_wr_trigger.remove(trigger)

                ## Trigger edram_rd_ir
                for trigger in pe.edram_rd_ir_trigger.copy():
                    pro_event = trigger[0]
                    cu_idx = trigger[1][0]
                    if not self.isPipeLine:
                        if pro_event.nlayer == self.pipeline_layer_stage:
                            pe.CU_array[cu_idx].edram_rd_ir_erp.append(pro_event)
                            pe.edram_rd_ir_trigger.remove(trigger)
                    else:
                        pe.CU_array[cu_idx].edram_rd_ir_erp.append(pro_event)
                        pe.edram_rd_ir_trigger.remove(trigger)
                
                ## Trigger pooling 
                for trigger in pe.pooling_trigger.copy():
                    pro_event = trigger[0]
                    if not self.isPipeLine:
                        if pro_event.nlayer == self.pipeline_layer_stage:
                            pe.pooling_erp.append(pro_event)
                            pe.pooling_trigger.remove(trigger)
                    else:
                        pe.pooling_erp.append(pro_event)
                        pe.pooling_trigger.remove(trigger)

                ## Trigger edram_rd_ir_pool 
                for trigger in pe.edram_rd_pool_trigger.copy():
                    pro_event = trigger[0]
                    if not self.isPipeLine:
                        if pro_event.nlayer == self.pipeline_layer_stage:
                            pe.edram_rd_pool_erp.append(pro_event)
                            pe.edram_rd_pool_trigger.remove(trigger)
                    else:
                        pe.edram_rd_pool_erp.append(pro_event)
                        pe.edram_rd_pool_trigger.remove(trigger)
                for cu in pe.CU_array:
                    ## Trigger ou operation 
                    for trigger in cu.ou_trigger.copy():
                        pro_event = trigger[0]
                        xb_idx = trigger[1][1]
                        if not self.isPipeLine:
                            if pro_event.nlayer == self.pipeline_layer_stage:
                                cu.XB_array[xb_idx].ou_erp.append(pro_event)
                                cu.ou_trigger.remove(trigger)
                        else:
                            cu.XB_array[xb_idx].ou_erp.append(pro_event)
                            cu.ou_trigger.remove(trigger)
                    ## Trigger pe saa 
                    for trigger in cu.pe_saa_trigger.copy():
                        pro_event = trigger[0]
                        if not self.isPipeLine:
                            if pro_event.nlayer == self.pipeline_layer_stage:
                                pe.pe_saa_erp.append(pro_event)
                                cu.pe_saa_trigger.remove(trigger)
                        else:
                            pe.pe_saa_erp.append(pro_event) 
                            cu.pe_saa_trigger.remove(trigger)
                    ### Trigger adc
                    for xb in cu.XB_array:
                        for trigger in xb.adc_trigger.copy():
                            pro_event = trigger[0]
                            cu_idx = trigger[1][0]
                            if not self.isPipeLine:
                                if pro_event.nlayer == self.pipeline_layer_stage:
                                    pe.CU_array[cu_idx].adc_erp.append(pro_event)
                                    xb.adc_trigger.remove(trigger)
                            else:
                                pe.CU_array[cu_idx].adc_erp.append(pro_event)
                                xb.adc_trigger.remove(trigger)
                    ### Trigger cu_saa
                    for trigger in cu.cu_saa_trigger.copy():
                        pro_event = trigger[0]
                        if not self.isPipeLine:
                            if pro_event.nlayer == self.pipeline_layer_stage:
                                cu.cu_saa_erp.append(pro_event)
                                cu.cu_saa_trigger.remove(trigger)
                        else:
                            cu.cu_saa_erp.append(pro_event)
                            cu.cu_saa_trigger.remove(trigger)

                ### Trigger pe saa (for data transfer) 
                for trigger in pe.pe_saa_trigger.copy():
                    pro_event = trigger[0]
                    if not self.isPipeLine:
                        if pro_event.nlayer == self.pipeline_layer_stage:
                            pe.pe_saa_erp.append(pro_event)
                            pe.pe_saa_trigger.remove(trigger)
                    else:
                        pe.pe_saa_erp.append(pro_event)
                        pe.pe_saa_trigger.remove(trigger)
                        
            ### Record State ###
            for pe in self.PE_array:
                if pe.check_state():
                    self.pe_state_for_plot[0].append(self.cycle_ctr)
                    self.pe_state_for_plot[1].append(self.PE_array.index(pe))
                for cu in pe.CU_array:
                    if cu.check_state():
                        self.cu_state_for_plot[0].append(self.cycle_ctr)
                        self.cu_state_for_plot[1].append(self.PE_array.index(pe) * self.hd_info.CU_num + \
                                                        pe.CU_array.index(cu))
                    
                    for xb in cu.XB_array:
                        if xb.check_state():
                            self.xb_state_for_plot[0].append(self.cycle_ctr)
                            self.xb_state_for_plot[1].append( \
                                self.PE_array.index(pe) * self.hd_info.CU_num + \
                                pe.CU_array.index(cu) * self.hd_info.Xbar_num + cu.XB_array.index(xb)
                                )
            ### Reset ###
            for pe in self.PE_array:
                pe.reset()
                for cu in pe.CU_array:
                    cu.reset()
                    for xb in cu.XB_array:
                        xb.reset()

            for pe in self.PE_array:
                for cu in pe.CU_array:
                    if cu.state:
                        if cu.cu_saa_erp or cu.adc_erp:
                            continue
                        if cu.state_edram_rd_ir:
                            continue

                        isCUBusy = False
                        for xb in cu.XB_array:
                            if xb.ou_erp:
                                isCUBusy = True
                                break
                        if isCUBusy:
                            continue
                        cu.state = False

            self.energy_utilization.append(self.Total_energy_cycle*1e09)
            self.xbar_utilization.append(self.act_xb_ctr)
            
            ### Finish?
            isDone = True
            if self.fetch_array or self.data_transfer_erp or self.data_transfer_trigger:
                isDone = False
            if self.interconnect.busy():
                isDone = False

            for pe in self.PE_array:
                if pe.pe_saa_erp or pe.activation_erp or pe.pooling_erp or pe.edram_wr_erp or pe.edram_rd_pool_erp:
                    isDone = False
                    break
                elif pe.activation_trigger or pe.edram_wr_trigger or pe.edram_rd_pool_trigger \
                    or pe.edram_rd_ir_trigger or pe.pooling_trigger:
                    isDone = False
                    break
                for cu in pe.CU_array:
                    if cu.state or cu.edram_rd_ir_erp or cu.cu_saa_erp or cu.adc_erp:
                        isDone = False
                        break
                    elif cu.ou_trigger or cu.pe_saa_trigger or cu.cu_saa_trigger:
                        isDone = False
                        break
                    for xb in cu.XB_array:
                        if xb.adc_trigger:
                            isDone = False
                            break
                    if not isDone:
                        break
                if not isDone:
                    break


            # Buffer size utilization
            for pe_idx in range(len(self.PE_array)):
                self.buffer_size[pe_idx].append(self.PE_array[pe_idx].edram_buffer.count())
        
        ### Buffer size ###
        self.max_buffer_size = 0 # num of data
        self.total_buffer_size = 0 
        for pe in self.PE_array:
            self.total_buffer_size += len(pe.edram_buffer.buffer)
            self.max_buffer_size = max(len(pe.edram_buffer.buffer), self.max_buffer_size)
        self.avg_buffer_size = self.total_buffer_size / len(self.PE_array)

    def print_statistics_result(self):
        print("Total Cycles:", self.cycle_ctr)
        print("Cycles each layer:", self.cycles_each_layer)
        print("Cycles time:", self.hd_info.cycle_time, "ns\n")
        
        print('memory accesss times:', self.mem_acc_ctr)
        print('max_buffer_size', self.max_buffer_size, "(", self.max_buffer_size*2, "B)")
        print("Avg buffer size:", self.avg_buffer_size)
        print()
        

        self.Total_energy_cu = self.Total_energy_cu_shift_and_add + \
                                self.Total_energy_adc + \
                                self.Total_energy_dac + \
                                self.Total_energy_crossbar + \
                                self.Total_energy_ir_in_cu + \
                                self.Total_energy_or_in_cu
        self.Total_energy_pe = self.Total_energy_cu + \
                                self.Total_energy_edram_buffer + \
                                self.Total_energy_bus + \
                                self.Total_energy_activation + \
                                self.Total_energy_pe_shift_and_add + \
                                self.Total_energy_pooling + \
                                self.Total_energy_or
        self.Total_energy = self.Total_energy_pe + self.Total_energy_interconnect
        
        print("Power breakdown:")

        print("\tTotal:", self.Total_energy, "nJ")
        print("\tChip level")
        print("\t\tPE: %.4e (%.2f%%)" %(self.Total_energy_pe, self.Total_energy_pe/self.Total_energy*100))
        print("\t\tInterconnect: %.4e (%.2f%%)" %(self.Total_energy_interconnect, self.Total_energy_interconnect/self.Total_energy*100))
        print()
        print("\tPE level")
        print("\t\tCU: %.4e (%.2f%%)" %(self.Total_energy_cu, self.Total_energy_cu/self.Total_energy_pe*100))
        print("\t\tEdram Buffer: %.4e (%.2f%%)" %(self.Total_energy_edram_buffer, self.Total_energy_edram_buffer/self.Total_energy_pe*100))
        print("\t\tBus: %.4e (%.2f%%)" %(self.Total_energy_bus, self.Total_energy_bus/self.Total_energy_pe*100))
        print("\t\tActivation: %.4e (%.2f%%)" %(self.Total_energy_activation, self.Total_energy_activation/self.Total_energy_pe*100))
        print("\t\tShift and Add: %.4e (%.2f%%)" %(self.Total_energy_pe_shift_and_add, self.Total_energy_pe_shift_and_add/self.Total_energy_pe*100))
        print("\t\tPooling: %.4e (%.2f%%)" %(self.Total_energy_pooling, self.Total_energy_pooling/self.Total_energy_pe*100))
        print("\t\tOR: %.4e (%.2f%%)" %(self.Total_energy_or, self.Total_energy_or/self.Total_energy_pe*100))
        print()
        print("\tCU level")
        print("\t\tShift and Add: %.4e (%.2f%%)" %(self.Total_energy_cu_shift_and_add, self.Total_energy_cu_shift_and_add/self.Total_energy_cu*100))
        print("\t\tADC: %.4e (%.2f%%)" %(self.Total_energy_adc, self.Total_energy_adc/self.Total_energy_cu*100))
        print("\t\tDAC: %.4e (%.2f%%)" %(self.Total_energy_dac, self.Total_energy_dac/self.Total_energy_cu*100))
        print("\t\tCrossbar Array: %.4e (%.2f%%)" %(self.Total_energy_crossbar, self.Total_energy_crossbar/self.Total_energy_cu*100))
        print("\t\tIR: %.4e (%.2f%%)" %(self.Total_energy_ir_in_cu, self.Total_energy_ir_in_cu/self.Total_energy_cu*100))
        print("\t\tOR: %.4e (%.2f%%)" %(self.Total_energy_or_in_cu, self.Total_energy_or_in_cu/self.Total_energy_cu*100))
        print()

        if self.isPipeLine:
            pipe_str = "pipeline"
        else:
            pipe_str = "non_pipeline"
        
        ### non-pipeline stage
        if not self.isPipeLine:
            with open('./statistics/non_pipeline/'+self.mapping_str+'/stage.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in range(self.cycle_ctr):
                    writer.writerow([row+1, self.pipeline_stage_record[row]])

        fre = 1
        ### Energy per 100 cycle
        with open('./statistics/'+pipe_str+'/'+self.mapping_str+'/Energy.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(0, self.cycle_ctr, fre):
                writer.writerow([row+1, self.energy_utilization[row]])
        
        plt.bar(range(1, self.cycle_ctr+1), self.energy_utilization)
        #plt.show()
        plt.title(self.mapping_str+", "+pipe_str)
        plt.ylabel('Energy (nJ)')
        plt.xlabel('Cycle')
        plt.ylim([0, 20])
        #plt.xlim([0,])
        plt.savefig('./statistics/'+pipe_str+'/'+self.mapping_str+'/energy_utilization.png')
        plt.clf()
        
        ### Power breakdown
        # chip
        labels = 'PE', 'Interconnect'
        value = [self.Total_energy_pe, self.Total_energy_interconnect]
        plt.pie(value , labels = labels, autopct='%1.1f%%')
        plt.axis('equal')
        plt.title(self.mapping_str+", "+pipe_str)
        plt.savefig('./statistics/'+pipe_str+'/'+self.mapping_str+'/Power_breakdown_chip.png')
        plt.clf()


        ### PE usage
        with open('./statistics/'+pipe_str+'/'+self.mapping_str+'/PE_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(0, len(self.pe_state_for_plot[0]), fre):
                writer.writerow([self.pe_state_for_plot[0][row], self.pe_state_for_plot[1][row]])

        plt.scatter(self.pe_state_for_plot[0], self.pe_state_for_plot[1], s=3, c='blue')
        plt.title(self.mapping_str+", "+pipe_str)
        plt.xlabel('Cycle')
        plt.ylabel('PE number')
        plt.ylim([-1, 16])
        #plt.xlim([1, 250])  ### @@
        plt.xticks(np.arange(0,250, 20), fontsize=8)
        plt.savefig('./statistics/'+pipe_str+'/'+self.mapping_str+'/PE_utilization.png')
        plt.clf()
        
        ### CU usage
        with open('./statistics/'+pipe_str+'/'+self.mapping_str+'/CU_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(0, len(self.cu_state_for_plot[0]), fre):
                writer.writerow([self.cu_state_for_plot[0][row], self.cu_state_for_plot[1][row]])

        plt.scatter(self.cu_state_for_plot[0], self.cu_state_for_plot[1], s=2, c='blue')
        plt.title(self.mapping_str+", "+pipe_str)
        plt.xlabel('Cycle')
        plt.ylabel('CU number')
        #plt.ylim([-1, 64])
        #plt.xlim([1, 250])  ### @@
        plt.yticks(np.arange(0,64, 2), fontsize=6)
        plt.xticks(np.arange(0,250, 20), fontsize=8)
        plt.savefig('./statistics/'+pipe_str+'/'+self.mapping_str+'/CU_utilization.png')
        plt.clf()

        ### XB usage
        with open('./statistics/'+pipe_str+'/'+self.mapping_str+'/XB_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(0, len(self.xb_state_for_plot[0]), fre):
                writer.writerow([self.xb_state_for_plot[0][row], self.xb_state_for_plot[1][row]])

        plt.scatter(self.xb_state_for_plot[0], self.xb_state_for_plot[1], s=1, c='blue')
        plt.title(self.mapping_str+", "+pipe_str)
        plt.xlabel('Cycle')
        plt.ylabel('XB number')
        plt.yticks(np.arange(0,64, 2), fontsize=6)
        plt.xticks(np.arange(0,250, 20), fontsize=8)
        plt.savefig('./statistics/'+pipe_str+'/'+self.mapping_str+'/XB_utilization.png')
        plt.clf()
  
        ### On chip Buffer
        with open('./statistics/'+pipe_str+'/'+self.mapping_str+'/OnchipBuffer.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(self.cycle_ctr):
                c = [row+1]
                for i in range(len(self.PE_array)):
                    c.append(self.buffer_size[i][row])
                writer.writerow(c)

        for i in range(len(self.PE_array)):
            plt.plot(range(1, self.cycle_ctr+1), self.buffer_size[i]) #, c=self.color[i])
        plt.title(self.mapping_str+", "+pipe_str)
        plt.xlabel('Cycle')
        plt.ylabel('Buffer size (number of data)')
        plt.ylim([0, self.max_buffer_size+5])
        plt.savefig('./statistics/'+pipe_str+'/'+self.mapping_str+'/OnChipBuffer_size_utilization.png')
        plt.clf()
