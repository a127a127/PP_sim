import os
import csv
from PE import PE
import numpy as np
from math import ceil
from EventMetaData import EventMetaData
from FetchEvent import FetchEvent
from HardwareMetaData import HardwareMetaData

from NetworkTransfer import NetworkTransfer
from TransferEvent import TransferEvent

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Controller(object):
    def __init__(self, ordergenerator, isPipeLine, trace):
        self.ordergenerator = ordergenerator
        self.isPipeLine = isPipeLine
        self.trace = trace
        self.Computation_order = self.ordergenerator.Computation_order
        self.trace = trace

        self.cycle_ctr = 0
        self.cycle_power = 0
        self.total_energy = 0


        self.hardware_information = HardwareMetaData()
        # Leakage
        self.eDRAM_buffer_leakage = self.hardware_information.eDRAM_buffer_leakage
        self.Router_leakage = self.hardware_information.Router_leakage
        self.SA_leakage = self.hardware_information.SA_leakage
        self.Act_leakage = self.hardware_information.Act_leakage
        self.PE_SAA_leakage = self.hardware_information.PE_SAA_leakage
        self.Pool_leakage = self.hardware_information.Pool_leakage
        self.DAC_leakage = self.hardware_information.DAC_leakage
        self.MUX_leakage = self.hardware_information.MUX_leakage
        self.SA_leakage = self.hardware_information.SA_leakage
        self.Crossbar_leakage = self.hardware_information.Crossbar_leakage
        self.CU_SAA_leakage = self.hardware_information.CU_SAA_leakage

        # Dynamic Energy
        self.eDRAM_rd_ir_energy = self.hardware_information.eDRAM_rd_ir_energy
        self.edram_rd_pool_energy = self.hardware_information.edram_rd_pool_energy
        self.ou_operation_energy = self.hardware_information.ou_operation_energy
        self.pe_saa_energy = self.hardware_information.pe_saa_energy
        self.cu_saa_energy = self.hardware_information.cu_saa_energy
        self.activation_energy = self.hardware_information.activation_energy
        self.pooling_energy = self.hardware_information.pooling_energy
        self.edram_wr_energy = self.hardware_information.edram_wr_energy

        self.RT_num_y = self.hardware_information.Router_num_y
        self.RT_num_x = self.hardware_information.Router_num_x
        self.PE_num_y = self.hardware_information.PE_num_y
        self.PE_num_x = self.hardware_information.PE_num_x
        self.PE_num = self.hardware_information.PE_num
        self.CU_num_y = self.hardware_information.CU_num_y
        self.CU_num_x = self.hardware_information.CU_num_x
        self.XB_num_y = self.hardware_information.Xbar_num_y
        self.XB_num_x = self.hardware_information.Xbar_num_x
        
        
        ### for statistics
        self.mem_acc_ctr = 0
        ### utilization
        self.power_utilization = []
        self.PE_utilization = []
        self.CU_utilization = []
        self.xbar_utilization = []
        self.pooling_utilization = []
        self.cu_saa_utilization = []
        self.pe_saa_utilization = []
        self.activation_utilization = []
        self.OU_usage_utilization = []

        
        self.input_bit = self.ordergenerator.model_information.input_bit
        self.PE_array = []
        for rty_idx in range(self.RT_num_y):
            for rtx_idx in range(self.RT_num_x):
                for pey_idx in range(self.PE_num_y):
                    for pex_idx in range(self.PE_num_x):
                        pe_pos = (rty_idx, rtx_idx, pey_idx, pex_idx)
                        pe = PE(pe_pos, self.input_bit)
                        self.PE_array.append(pe)
        #print(self.PE_array[0].CU_array[0].XB_array[0])

        self.fetch_array = []
        self.network_transfer = NetworkTransfer()
        self.transfer_trigger = []

        #for i in range(len(self.pe_traverse_idx)):
        #    pass
            # self.PE_utilization.append([])
            # self.CU_utilization.append([])
            # self.pooling_utilization.append([])
            # self.OU_usage_utilization.append([])
            # self.buffer_size.append([])
            # self.shift_and_add_utilization.append([])
            # self.activation_utilization.append([])
            # self.xbar_utilization.append([])
        
        ### Pipeline control ###
        if not self.isPipeLine:
            print("non-pipeline")
            self.pipeline_layer_stage = 0
            self.num_layer = len(self.ordergenerator.layer_list)
            print("num_layer:", self.num_layer)
            
            self.events_each_layer = []
            for layer in range(self.num_layer):
                self.events_each_layer.append(0)
            for e in self.Computation_order:
                self.events_each_layer[e.nlayer] += 1
            print(self.events_each_layer)

            self.this_layer_event_ctr = 0

        print("total event:", len(self.Computation_order))

    def run(self): 
        #fetch_array = list()
        #color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        #cu_state_for_plot = [[],[]]
        #cu_transfer_ctr = 0

        for e in self.Computation_order:
            # traverse computation order
            # if current_number_of_preceding_event==preceding_event_count的event
            # append the event to event ready pool
            if e.preceding_event_count == e.current_number_of_preceding_event:
                if e.event_type == 'edram_rd_ir':
                    pos = e.position_idx
                    rty, rtx, pey, pex, cuy, cux = pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]
                    idx = pex + pey * self.PE_num_x + rtx * self.PE_num + rty * rtx * self.PE_num
                    cu_idx = cux + cuy * self.CU_num_x
                    self.PE_array[idx].CU_array[cu_idx].edram_rd_ir_erp.append(e)
                else:
                    # error
                    print("Computation order error: event\"", e.event_type, "\".")
                    print("exit")
                    exit()
        # for pe in self.PE_array:
        #     print(pe.position, pe.edram_rd_ir_erp)

    
        isDone = False
        while not isDone:
            
            self.cycle_power = 0
            self.cycle_ctr += 1
            if self.trace:
                print('cycle:', self.cycle_ctr)
            
            # pipeline_stage = 10000           
            # if not self.ordergenerator.isPipeline:
            #     for index in range(len(self.Computation_order)):
            #         if not self.Computation_order[index].finished:
            #             if self.Computation_order[index].nlayer < pipeline_stage:
            #                 pipeline_stage = self.Computation_order[index].nlayer
            #             self.CU_unfinished_event_index = index
            #             break
            
            ### Data transfer in Chip ###
            arrived = self.network_transfer.step()
            for TF_event in arrived:
                if self.trace:
                    print("\tData arrived ", TF_event.event.event_type, ",order index:", self.Computation_order.index(TF_event.event), "destination:", TF_event.event.position_idx[1])
                self.this_layer_event_ctr += 1
                ### add next event counter: pe_saa, edram_rd_ir, edram_rd_pool
                for proceeding_index in TF_event.event.proceeding_event:
                    pro_event = self.Computation_order[proceeding_index]
                    pro_event.current_number_of_preceding_event += 1
                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                        if self.trace:
                            print("\t\tProceeding event is triggered.", pro_event.event_type)
                        self.transfer_trigger.append([TF_event, pro_event])
                
            ### Fetch data from off-chip memory ###
            for FE in self.fetch_array.copy():
                FE.cycles_counter += 1
                #print(FE.cycles_counter, end=' ')
                if FE.cycles_counter == FE.fetch_cycle:
                    #print('@', end='/')
                    
                    if FE.event.event_type == "edram_rd_ir":
                        pe_idx = FE.index[0]
                        cu_idx = FE.index[1]
                        for inp in FE.event.inputs:
                            data = inp[1:]
                            self.PE_array[pe_idx].edram_buffer.put([FE.event.nlayer, data])

                        self.PE_array[pe_idx].CU_array[cu_idx].edram_rd_ir_erp.insert(0, FE.event)
                    
                    elif FE.event.event_type == "edram_rd_pool":
                        for data in FE.event.inputs:
                            self.PE_array[pe_idx].edram_buffer.put([FE.event.nlayer, data])
                        self.PE_array[pe_idx].edram_rd_pool_erp.insert(0, FE.event)

                    self.fetch_array.remove(FE)
                    self.mem_acc_ctr += 1
                    #print(XB_array[0].OnchipBuffer.arr) 

            ### Event: edram_rd_ir ###
            for pe in self.PE_array:
                for cu in pe.CU_array:
                    if cu.edram_rd_ir_erp:
                        event = cu.edram_rd_ir_erp[0]
                    else:
                        continue
                    #print("\tevent:", event.event_type)
                    if not cu.state and not cu.state_edram_rd_ir:

                        ## Data in eDRAM buffer?
                        isData_ready = True
                        # inputs: [[num_input, fm_h, fm_w, fm_c]]
                        for inp in event.inputs:
                            data = inp[1:]
                            #print(event.nlayer, data)
                            if not pe.edram_buffer.check([event.nlayer, data]):
                                # Data not in buffer
                                if self.trace:
                                    print("\tData not ready for edram_rd_ir. Data: layer", event.nlayer, event.event_type, data)
                                isData_ready = False
                                cu.edram_rd_ir_erp.remove(event)
                                break
                        
                        if not isData_ready:
                            pe_idx = self.PE_array.index(pe)
                            cu_idx = pe.CU_array.index(cu)
                            #print("PE_array index", pe_idx, "data not ready")
                            cu.state_edram_rd_ir = True
                            self.fetch_array.append(FetchEvent(event, [pe_idx, cu_idx]))

                        else:
                            ## Check how many event can be done in a cycle
                            if self.trace:  
                                print("\tdo edram_rd_ir, cu_pos:", cu.position, ",order index:", self.Computation_order.index(event))
                            self.this_layer_event_ctr += 1
                            cu.state = True
                            cu.state_edram_rd_ir = True
                            cu.edram_rd_ir_erp.remove(event)
                            
                            ### add next event counter: ou_operation
                            for proceeding_index in event.proceeding_event:
                                pro_event = self.Computation_order[proceeding_index]
                                pro_event.current_number_of_preceding_event += 1
                                
                                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                    if self.trace:
                                        print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                    pos = pro_event.position_idx
                                    cu_y, cu_x, xb_y, xb_x = pos[4], pos[5], pos[6], pos[7]
                                    cu_idx = cu_x + cu_y * self.CU_num_x
                                    xb_idx = xb_x + xb_y * self.XB_num_x
                                    cu.ou_operation_trigger.append([pro_event, [cu_idx, xb_idx]])                                

            ### Event: ou_operation ###
            for pe in self.PE_array:
                for cu in pe.CU_array:
                    for xb in cu.XB_array:
                        for event in xb.ou_operation_erp.copy():
                            for idx in range(len(xb.state_ou_operation)):
                                #print(xb.state_ou_operation[idx], idx)
                                if not xb.state_ou_operation[idx]:
                                    if self.trace:
                                        print("\tdo ou_operation, xb_pos:", xb.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                                    self.this_layer_event_ctr += 1
                                    xb.state_ou_operation[idx] = True
                                    xb.ou_operation_erp.remove(event)

                                    ### add next event counter: cu_saa
                                    for proceeding_index in event.proceeding_event:
                                        pro_event = self.Computation_order[proceeding_index]
                                        pro_event.current_number_of_preceding_event += 1
                                        
                                        if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                            if self.trace:
                                                print("\t\tProceeding event is triggered.", pro_event.event_type)
                                            pos = pro_event.position_idx
                                            cu_y, cu_x = pos[4], pos[5]
                                            cu_idx = cu_x + cu_y * self.CU_num_x
                                            xb.cu_saa_trigger.append([pro_event, [cu_idx]])
                                    break

            ### Event: cu_saa ###
            for pe in self.PE_array:
                for cu in pe.CU_array:
                    for event in cu.cu_saa_erp.copy():
                        for idx in range(len(cu.state_cu_saa)):
                            if not cu.state_cu_saa[idx]:
                                if self.trace:
                                    print("\tdo cu_saa, cu_pos:", cu.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                                self.this_layer_event_ctr += 1
                                cu.state_cu_saa[idx] = True
                                cu.cu_saa_erp.remove(event)

                                ### add next event counter: pe_saa, data_transfer
                                for proceeding_index in event.proceeding_event:
                                    pro_event = self.Computation_order[proceeding_index]
                                    pro_event.current_number_of_preceding_event += 1

                                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                        if self.trace:
                                            print("\t\tProceeding event is triggered.", pro_event.event_type)
                                        if pro_event.event_type == "pe_saa":
                                            cu.pe_saa_trigger.append([pro_event, []])
                                        else:
                                            src = pro_event.position_idx[0]
                                            des = pro_event.position_idx[1][0] # des:只會有一個PE的SAA
                                            self.network_transfer.transfer_list.append(TransferEvent(pro_event, src, des, 0))
                                break

            ### Event: pe_saa ###
            for pe in self.PE_array:
                for event in pe.pe_saa_erp.copy():
                    for idx in range(len(pe.state_pe_saa)):
                        if not pe.state_pe_saa[idx]:
                            if self.trace:
                                print("\tdo pe_saa, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                            self.this_layer_event_ctr += 1
                            pe.state_pe_saa[idx] = True
                            pe.pe_saa_erp.remove(event)

                            ### add next event counter: activation
                            for proceeding_index in event.proceeding_event:
                                pro_event = self.Computation_order[proceeding_index]
                                pro_event.current_number_of_preceding_event += 1
                                
                                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                    if self.trace:
                                        print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                    pe.activation_trigger.append([pro_event, []])
                            break    

            ### Event: activation ###
            for pe in self.PE_array:
                for event in pe.activation_erp.copy():
                    for idx in range(len(pe.state_activation)):
                        if not pe.state_activation[idx]:
                            if self.trace:
                                print("\tdo activation, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                            self.this_layer_event_ctr += 1
                            pe.state_activation[idx] = True
                            pe.activation_erp.remove(event)

                            ### add next event counter: edram_wr
                            for proceeding_index in event.proceeding_event:
                                pro_event = self.Computation_order[proceeding_index]
                                pro_event.current_number_of_preceding_event += 1
                                
                                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                    if self.trace:
                                        print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx, self.Computation_order.index(pro_event))
                                    pe.edram_wr_trigger.append([pro_event, []])
                            break

            ### Event: edram write ###
            for pe in self.PE_array:
                for event in pe.edram_wr_erp.copy():
                    for idx in range(len(pe.state_edram_wr)):
                        if not pe.state_edram_wr[idx]:
                            if self.trace:
                                print("\tdo edram_wr, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                            self.this_layer_event_ctr += 1
                            pe.state_edram_wr[idx] = True
                            pe.edram_wr_erp.remove(event)

                            ### add next event counter: edram_rd_ir, edram_rd_pool, data_transfer
                            for proceeding_index in event.proceeding_event:
                                pro_event = self.Computation_order[proceeding_index]
                                pro_event.current_number_of_preceding_event += 1
                            
                                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                    if self.trace:
                                        print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                    pos = pro_event.position_idx
                                    if pro_event.event_type == "edram_rd_ir":
                                        cu_y, cu_x = pos[4], pos[5]
                                        cu_idx = cu_x + cu_y * self.CU_num_x
                                        pe.edram_rd_ir_trigger.append([pro_event, [cu_idx]])
                                    elif pro_event.event_type == "edram_rd_pool":
                                        pe.edram_rd_pool_trigger.append([pro_event, []])
                                    elif pro_event.event_type == "data_transfer":
                                        src = pro_event.position_idx[0]
                                        des_list = pro_event.position_idx[1]
                                        for des in des_list:
                                            self.network_transfer.transfer_list.append(TransferEvent(pro_event, src, des[:-2], des))
                            break
            
            ### Event: edram_rd_pool ###
            for pe in self.PE_array:
                if pe.edram_rd_pool_erp:
                    event = pe.edram_rd_pool_erp[0]
                else:
                    continue
                if not pe.state_edram_rd_pool:
                    
                    ## Data in eDRAM buffer?
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
                        pe.state_edram_rd_pool = True
                        self.fetch_array.append(FetchEvent(event, [pe_idx]))

                    else:
                        ## Check how many event can be done in a cycle
                        if self.trace:
                            print("\tdo edram_rd_pool, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                        self.this_layer_event_ctr += 1
                        pe.state_edram_rd_pool = True
                        pe.edram_rd_pool_erp.remove(event)
                        
                        ### add next event counter: pooling
                        for proceeding_index in event.proceeding_event:
                            pro_event = self.Computation_order[proceeding_index]
                            pro_event.current_number_of_preceding_event += 1
                            
                            if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                if self.trace:
                                    print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                pos = pro_event.position_idx
                                pe.pooling_trigger.append([pro_event, []])                                
                    
            ### Event: pooling ###
            for pe in self.PE_array:
                for event in pe.pooling_erp.copy():
                    for idx in range(len(pe.state_pooling)):
                        if not pe.state_pooling[idx]:
                            if self.trace:
                                print("\tdo pooling, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                            self.this_layer_event_ctr += 1
                            pe.state_pooling[idx] = True
                            pe.pooling_erp.remove(event)

                            ### add next event counter: edram_wr
                            for proceeding_index in event.proceeding_event:
                                pro_event = self.Computation_order[proceeding_index]
                                pro_event.current_number_of_preceding_event += 1

                                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                    if self.trace:
                                        print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                    pe.edram_wr_trigger.append([pro_event, []])
                            break


            ### Trigger events ###
            for trigger in self.transfer_trigger.copy():
                TF_event = trigger[0]
                pro_event = trigger[1]
                if not self.isPipeLine:
                        if pro_event.nlayer == self.pipeline_layer_stage:
                            if pro_event.event_type == "pe_saa":
                                rty, rtx = TF_event.event.position_idx[1][0][0], TF_event.event.position_idx[1][0][1]
                                pey, pex = TF_event.event.position_idx[1][0][2], TF_event.event.position_idx[1][0][3]
                                pe_idx = pex + pey * self.PE_num_x + rtx * self.PE_num + rty * self.PE_num * self.RT_num_x
                                self.PE_array[pe_idx].pe_saa_erp.append(pro_event)
                            elif pro_event.event_type == "edram_rd_ir":
                                rty, rtx = TF_event.destination_cu[0], TF_event.destination_cu[1]
                                pey, pex = TF_event.destination_cu[2], TF_event.destination_cu[3]
                                cuy, cyx = TF_event.destination_cu[4], TF_event.destination_cu[5]
                                pe_idx = pex + pey * self.PE_num_x + rtx * self.PE_num + rty * self.PE_num * self.RT_num_x
                                cu_idx = cux + cuy * self.CU_num_x
                                self.PE_array[pe_idx].CU_array[cu_idx].edram_rd_ir_erp.append(pro_event)
                            elif pro_event.event_type == "edram_rd_pool":
                                rty, rtx = TF_event.event.position_idx[1][0][0], TF_event.event.position_idx[1][0][1]
                                pey, pex = TF_event.event.position_idx[1][0][2], TF_event.event.position_idx[1][0][3]
                                pe_idx = pex + pey * self.PE_num_x + rtx * self.PE_num + rty * self.PE_num * self.RT_num_x
                                self.PE_array[pe_idx].edram_rd_pool_erp.append(pro_event)
                            self.transfer_trigger.remove(trigger)
                else:
                    if pro_event.event_type == "pe_saa":
                        rty, rtx = TF_event.event.position_idx[1][0][0], TF_event.event.position_idx[1][0][1]
                        pey, pex = TF_event.event.position_idx[1][0][2], TF_event.event.position_idx[1][0][3]
                        pe_idx = pex + pey * self.PE_num_x + rtx * self.PE_num + rty * self.PE_num * self.RT_num_x
                        self.PE_array[pe_idx].pe_saa_erp.append(pro_event)
                    elif pro_event.event_type == "edram_rd_ir":
                        rty, rtx = TF_event.destination_cu[0], TF_event.destination_cu[1]
                        pey, pex = TF_event.destination_cu[2], TF_event.destination_cu[3]
                        cuy, cyx = TF_event.destination_cu[4], TF_event.destination_cu[5]
                        pe_idx = pex + pey * self.PE_num_x + rtx * self.PE_num + rty * self.PE_num * self.RT_num_x
                        cu_idx = cux + cuy * self.CU_num_x
                        self.PE_array[pe_idx].CU_array[cu_idx].edram_rd_ir_erp.append(pro_event)
                    elif pro_event.event_type == "edram_rd_pool":
                        rty, rtx = TF_event.event.position_idx[1][0][0], TF_event.event.position_idx[1][0][1]
                        pey, pex = TF_event.event.position_idx[1][0][2], TF_event.event.position_idx[1][0][3]
                        pe_idx = pex + pey * self.PE_num_x + rtx * self.PE_num + rty * self.PE_num * self.RT_num_x
                        self.PE_array[pe_idx].edram_rd_pool_erp.append(pro_event)
                    self.transfer_trigger.remove(trigger)

            for pe in self.PE_array:
                ## Trigger activation ###
                for trigger in pe.activation_trigger.copy():
                    pro_event = trigger[0]
                    if not self.isPipeLine:
                        if pro_event.nlayer == self.pipeline_layer_stage:
                            pe.activation_erp.append(pro_event)
                            pe.activation_trigger.remove(trigger)
                    else:
                        pe.activation_erp.append(pro_event)
                        pe.activation_trigger.remove(trigger)

                ## Trigger edram_wr ###
                for trigger in pe.edram_wr_trigger.copy():
                    pro_event = trigger[0]
                    if not self.isPipeLine:
                        if pro_event.nlayer == self.pipeline_layer_stage:
                            pe.edram_wr_erp.append(pro_event)
                            pe.edram_wr_trigger.remove(trigger)
                    else:
                        pe.edram_wr_erp.append(pro_event)
                        pe.edram_wr_trigger.remove(trigger)

                ## Trigger edram_rd_ir ###
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
                
                ## Trigger pooling ###
                for trigger in pe.pooling_trigger.copy():
                    pro_event = trigger[0]
                    if not self.isPipeLine:
                        if pro_event.nlayer == self.pipeline_layer_stage:
                            pe.pooling_erp.append(pro_event)
                            pe.pooling_trigger.remove(trigger)
                    else:
                        pe.pooling_erp.append(pro_event)
                        pe.pooling_trigger.remove(trigger)

                ## Trigger edram_rd_ir_pool ###
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
                    ## Trigger ou operation ###
                    for trigger in cu.ou_operation_trigger.copy():
                        pro_event = trigger[0]
                        xb_idx = trigger[1][1]
                        if not self.isPipeLine:
                            if pro_event.nlayer == self.pipeline_layer_stage:
                                cu.XB_array[xb_idx].ou_operation_erp.append(pro_event)
                                cu.ou_operation_trigger.remove(trigger)
                        else:
                            cu.XB_array[xb_idx].ou_operation_erp.append(pro_event)
                            cu.ou_operation_trigger.remove(trigger)
                    ## Trigger pe saa ###
                    for trigger in cu.pe_saa_trigger.copy():
                        pro_event = trigger[0]
                        if not self.isPipeLine:
                            if pro_event.nlayer == self.pipeline_layer_stage:
                                pe.pe_saa_erp.append(pro_event) 
                                #cu.pe_saa_trigger = []
                                cu.pe_saa_trigger.remove(trigger)
                        else:
                            pe.pe_saa_erp.append(pro_event) 
                            #cu.pe_saa_trigger = []
                            cu.pe_saa_trigger.remove(trigger)

                    for xb in cu.XB_array:
                        ### Trigger cu_saa ###
                        for trigger in xb.cu_saa_trigger.copy():
                            pro_event = trigger[0]
                            cu_idx = trigger[1][0]
                            if not self.isPipeLine:
                                if pro_event.nlayer == self.pipeline_layer_stage:
                                    pe.CU_array[cu_idx].cu_saa_erp.append(pro_event)
                                    xb.cu_saa_trigger.remove(trigger)
                            else:
                                pe.CU_array[cu_idx].cu_saa_erp.append(pro_event)
                                xb.cu_saa_trigger.remove(trigger)


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
                        if cu.cu_saa_erp:
                            break

                        isCUBusy = False
                        for xb in cu.XB_array:
                            if xb.ou_operation_erp:
                                isCUBusy = True
                                break
                        if isCUBusy:
                            break
                        cu.state = False
                        
            """
            # self.power_utilization.append(self.cycle_power)

            if not self.ordergenerator.isPipeline and pipeline_stage != 10000:
                if self.pipeline_lock < pipeline_stage:
                    self.pipeline_lock = pipeline_stage
                self.pipeline_stage_record.append(pipeline_stage)
            """
            
            ### Finish?
            isDone = True
            
            if self.fetch_array or self.network_transfer.transfer_list or self.transfer_trigger:
                isDone = False
            else:
                for pe in self.PE_array:
                    if pe.pe_saa_erp or pe.activation_erp or pe.pooling_erp or pe.edram_wr_erp or pe.edram_rd_pool_erp:
                        isDone = False
                        break
                    elif pe.activation_trigger or pe.edram_wr_trigger or pe.edram_rd_pool_trigger \
                        or pe.edram_rd_ir_trigger or pe.pooling_trigger:
                        isDone = False
                        break
                    for cu in pe.CU_array:
                        if cu.state or cu.edram_rd_ir_erp:
                            isDone = False
                            break
                        elif cu.ou_operation_trigger or cu.pe_saa_trigger:
                            isDone = False
                            break
                        for xb in cu.XB_array:
                            if xb.cu_saa_trigger:
                                isDone = False
                                break
                        if not isDone:
                            break
                    if not isDone:
                        break

            if self.cycle_ctr == 50:
                isDone = True

            if self.this_layer_event_ctr == self.events_each_layer[self.pipeline_layer_stage]:
                print("pipeline_layer_stage finished:", self.pipeline_layer_stage)
                self.pipeline_layer_stage += 1
                self.this_layer_event_ctr = 0
            
        print('total cycles:', self.cycle_ctr)
        print("this_layer_event_ctr:", self.this_layer_event_ctr)
        print('Power:')
        """
        print("\t total:", self.total_power)
        print("\t crossbar and sensing:", self.power_ou)
        print("\t Shift and add (CU):", self.power_cu_saa)
        print("\t Shift and add (PE):", self.power_pe_saa)
        print("\t Activation:", self.power_act)
        print("\t Pooling:", self.power_pool)
        print("\t eDRAM:", self.power_buffer)
        """
        #print('memory accesss times:', self.mem_acc_ctr)

        # if self.ordergenerator.isPipeline:
        #     pipe_str = "pipeline"
        # else:
        #     pipe_str = "non_pipeline"

        # ### Pipeline stage
        # if not self.ordergenerator.isPipeline:
        #     with open('./statistics/Pipeline_stage_'+pipe_str+'.csv', 'w', newline='') as csvfile:
        #         # 建立 CSV 檔寫入器
        #         writer = csv.writer(csvfile)
        #         for row in range(self.cycle_ctr):
        #             writer.writerow([row+1, self.pipeline_stage_record[row]])

        # ### Power
        # with open('./statistics/Power_'+pipe_str+'.csv', 'w', newline='') as csvfile:
        #     # 建立 CSV 檔寫入器
        #     writer = csv.writer(csvfile)
        #     for row in range(self.cycle_ctr):
        #         writer.writerow([row+1, self.power_utilization[row]])

        
        # plt.plot(range(1, self.cycle_ctr+1), self.power_utilization)
        # #plt.show()
        # plt.ylabel('Power (mW)')
        # plt.xlabel('Cycle')
        # plt.ylim([0, 2.5])
        # plt.savefig('./statistics/power_utilization_'+pipe_str+'.png')
        # plt.clf()
        
        # ### CU usage
        # with open('./statistics/CU_utilization_'+pipe_str+'.csv', 'w', newline='') as csvfile:
        #     # 建立 CSV 檔寫入器
        #     writer = csv.writer(csvfile)
        #     for row in range(len(cu_state_for_plot[0])):
        #         writer.writerow([cu_state_for_plot[0][row], cu_state_for_plot[1][row]])
        #     """
        #     for row in range(self.cycle_ctr):
        #         c = [row+1]
        #         for i in range(self.ordergenerator.CU_num):
        #             c.append(self.CU_utilization[i][row])
        #         writer.writerow(c)
        #     """

        # plt.scatter(cu_state_for_plot[0], cu_state_for_plot[1])
        # #for i in range(self.ordergenerator.CU_num):
        # #    plt.scatter(range(1, self.cycle_ctr), self.CU_utilization[i], c=color[i])
        # #plt.show()
        # plt.xlabel('Cycle')
        # plt.ylabel('CU number')
        # plt.ylim([-1, self.ordergenerator.CU_num])
        # plt.savefig('./statistics/CU_utilization_'+pipe_str+'.png')
        # plt.clf()

        # ### Xbar
        # for i in range(self.ordergenerator.CU_num):
        #     plt.plot(range(1, self.cycle_ctr+1), self.xbar_utilization[i], c=color[i])
        # #plt.show()
        # plt.xlabel('Cycle')
        # plt.ylabel('Crossbar utilization')
        # plt.savefig('./statistics/xbar_utilization_'+pipe_str+'.png')
        # plt.clf()

        # ### Pooling
        # for i in range(self.ordergenerator.CU_num):
        #     plt.plot(range(1, self.cycle_ctr+1), self.pooling_utilization[i], c=color[i])
        # #plt.show()
        # plt.xlabel('Cycle')
        # plt.ylabel('Pooling utilization')
        # plt.savefig('./statistics/pooling_utilization_'+pipe_str+'.png')
        # plt.clf()

        # ### Shift and Add
        # for i in range(self.ordergenerator.CU_num):
        #     plt.plot(range(1, self.cycle_ctr+1), self.shift_and_add_utilization[i], c=color[i])
        # #plt.show()
        # plt.xlabel('Cycle')
        # plt.ylabel('Shift and add unit utilization')
        # plt.savefig('./statistics/saa_utilization_'+pipe_str+'.png')
        # plt.clf()
        # """
        # for i in range(self.ordergenerator.CU_num):
        #     plt.plot(range(1, self.cycle_ctr+1), self.saa_rate_utilization[i], c=color[i])
        # #plt.show()
        # plt.savefig('./statistics/saa_rate_utilization_'+pipe_str+'.png')
        # plt.clf()
        # """

        # ### Activation Unit
        # for i in range(self.ordergenerator.CU_num):
        #     plt.plot(range(1, self.cycle_ctr+1), self.activation_utilization[i], c=color[i])
        # #plt.show()
        # plt.xlabel('Cycle')
        # plt.ylabel('Activation unit utilization')
        # plt.savefig('./statistics/activation_utilization_'+pipe_str+'.png')
        # plt.clf()

        # ### OU usage
        # OU_usage = []
        # for i in range(self.ordergenerator.CU_num):
        #     if len(self.OU_usage_utilization[i]) != 0:
        #         OU_usage.append(sum(self.OU_usage_utilization[i])/len(self.OU_usage_utilization[i]))
        #     else:
        #         OU_usage.append(0)

        # plt.xlabel('Cycle')
        # plt.ylabel('Average OU usage')
        # plt.bar(range(self.ordergenerator.CU_num), OU_usage)
        # #plt.plot(range(self.ordergenerator.CU_num), OU_usage)
        # plt.savefig("./statistics/OU_usage_utilization_"+pipe_str+".png")
        # plt.clf()

        # print('cu_transfer_ctr:', cu_transfer_ctr)


        # ### On chip Buffer
        # with open('./statistics/OnchipBuffer_'+pipe_str+'.csv', 'w', newline='') as csvfile:
        #     建立 CSV 檔寫入器
        #     writer = csv.writer(csvfile)
        #     for row in range(self.cycle_ctr):
        #         c = [row+1]
        #         for i in range(self.ordergenerator.CU_num):
        #             c.append(self.buffer_size[i][row])
        #         writer.writerow(c)

        # for i in range(self.ordergenerator.CU_num):
        #     plt.plot(range(1, self.cycle_ctr+1), self.buffer_size[i], c=color[i])
        # plt.xlabel('Cycle')
        # plt.ylabel('Buffer size')
        # plt.ylim([0, self.ordergenerator.buffer_size+1])
        # plt.savefig("./statistics/OnChipBuffer_size_utilization_"+pipe_str+".png")
        # plt.clf()
        
        # plt.plot(range(1, self.cycle_ctr+1), self.buffer_size)
        # plt.xlabel("Cycle")
        # plt.ylabel("Number of data")
        # plt.ylim([0, self.ordergenerator.buffer_size+100])
        # plt.savefig("./statistics/BufferSize_"+pipe_str+".png")
        # plt.clf()
        