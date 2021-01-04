from Model import Model
from Mapping import LIDR, HIDR
from OrderGenerator import OrderGenerator
from Controller import Controller
from ModelConfig import ModelConfig
from HardwareConfig import HardwareConfig
from Visualizer import Visualizer

import time, sys, os, pickle, math
import jsons, jsbeautifier
from tqdm import tqdm

# 1. 本來mappy.py, Ordergenerator 吃 model_config, 現在吃model_info model_info = Model(model_config)
# 2. Model.py: 多一個Model_type參數
# 3. Mapping 多紀錄CU index
# 4. Controller: 支援前面Ordergenerator的修改


def main():
    start_time = time.time()
    model      = sys.argv[1]
    mapping    = sys.argv[2]
    scheduling = sys.argv[3]
    partition_h = int(sys.argv[4])
    partition_w = int(sys.argv[5])
    mapping_str = mapping+sys.argv[4]+"_"+sys.argv[5]
    buffer_size_str = sys.argv[6]
    buffer_size = int(sys.argv[6])
    
    model_config = ModelConfig(model)
    model_info = Model(model_config)
    hw_config = HardwareConfig(buffer_size)
    hw_config.eDRAM_buffer_rd_wr_data_per_cycle = int(hw_config.eDRAM_buffer_bandwidth * 8 // model_info.input_bit * hw_config.cycle_time)
    hw_config.eDRAM_buffer_read_to_IR_cycles = math.ceil(hw_config.Xbar_h * hw_config.Xbar_num / hw_config.eDRAM_buffer_rd_wr_data_per_cycle)

    LoadOrder = True
    filename = './order_file/'+model_config.Model_type+'_'+mapping_str+'_'+scheduling+'_'+buffer_size_str+'.pkl'
    try:
        with open(filename, 'rb') as input:
            order_generator = pickle.load(input)
    except FileNotFoundError:
        print("Order file not found.")
        LoadOrder = False

    ### output path ###
    path = './statistics/'+model_config.Model_type+'/'+mapping_str+'/'+scheduling+'/'+buffer_size_str
    if not os.path.exists(path):
        os.makedirs(path)

    ### Mapping ##
    if not LoadOrder:
        cant_use_pe = (13, 12, 1, 1) # 讓不同的實驗設定下，使用相同數量的PE
        # Used PE: Lenet:6, Cifar10: 5, DeepID: 6, Caffenet: 321, Overfeat: 568, VGG16: 708
        if model == "Lenet":
            cant_use_pe = (0, 1, 1, 0)
        elif model == "Cifar10":
            cant_use_pe = (0, 1, 0, 1)
        elif model == "DeepID":
            cant_use_pe = (0, 1, 1, 0)
        elif model == "Caffenet":
            cant_use_pe = (6, 2, 0, 1)
        elif model == "Overfeat":
            cant_use_pe = (10, 12, 0, 0)
        elif model == "VGG16":
            cant_use_pe = (13, 4, 0, 0)

        cant_use_pe = (10000, 0, 1, 1, 3, 2)
        
        start_mapping_time = time.time()
        print("--- Mapping ---")
        print("Mapping policy:  ", end="")
        if mapping == "LIDR":
            print("Low input data reuse mapping")
            mapping_information = LIDR(model_info, hw_config, partition_h, partition_w, cant_use_pe)
        elif mapping == "HIDR":
            print("High input data reuse mapping")
            mapping_information = HIDR(model_info, hw_config, partition_h, partition_w, cant_use_pe)

        end_mapping_time = time.time()
        print("--- Mapping is finished in %s seconds ---\n" % (end_mapping_time - start_mapping_time))

    ### Print layer info ###
    if True:
        for nlayer in range(model_info.layer_length):
            if model_info.layer_list[nlayer].layer_type == "convolution" or model_info.layer_list[nlayer].layer_type == "fully":
                strides = model_info.strides[nlayer]
                pad = model_info.pad[nlayer]
                o_height = model_info.input_h[nlayer+1]
                o_width = model_info.input_w[nlayer+1]

                print(f'  - {nlayer} {model_info.layer_list[nlayer].layer_type}: [{model_info.input_c[nlayer]}, {model_info.input_h[nlayer]}, {model_info.input_w[nlayer]}] x [{model_info.filter_n[nlayer]}, {model_info.filter_c[nlayer]}, {model_info.filter_h[nlayer]}, {model_info.filter_w[nlayer]}] s: {strides}, p: {pad} -> [{model_info.input_c[nlayer+1]}, {o_height}, {o_width}]')
            else:
                o_height = model_info.input_h[nlayer+1]
                o_width = model_info.input_w[nlayer+1]
                print(f'  - {nlayer} {model_info.layer_list[nlayer].layer_type}: [{model_info.input_c[nlayer]}, {model_info.input_h[nlayer]}, {model_info.input_w[nlayer]}] x [{model_info.pooling_h[nlayer]}, {model_info.pooling_w[nlayer]}] s: {strides}, p: {pad} -> [{model_info.input_c[nlayer+1]}, {o_height}, {o_width}]')


    ### Buffer Replacement ###
    # print("Buffer replacement policy: ", end="")
    # replacement = "LRU"
    # if replacement == "Ideal":
    #     print("Ideal")
    # elif replacement == "LRU":
    #     print("LRU")

    ### Trace ###
    isTrace_order      = False
    isTrace_controller = False

    ### Generate computation order graph ### 
    if not LoadOrder:
        start_order_time = time.time()
        print("--- Generate computation order ---")
        order_generator = OrderGenerator(model_info, hw_config, mapping_information, isTrace_order)
        end_order_time = time.time()
        print("--- Computation order graph is generated in %s seconds ---\n" % (end_order_time - start_order_time))
        
        # Save Order
        if not os.path.exists('./order_file/'):
            os.makedirs('./order_file/')
        with open(filename, 'wb') as output:
            pickle.dump(order_generator, output, pickle.HIGHEST_PROTOCOL)
    
    else:
        with open(filename, 'rb') as input:
            order_generator = pickle.load(input)

    ### Dump JSON ###
    if False:
        json_name = f"{model}-{mapping}-{scheduling}-{partition_h}-{partition_w}-{buffer_size}"
        print(f"Dumping JSON to {json_name}.json...")
        with open(f"{json_name}.json", "w") as outfile:
            opts = jsbeautifier.default_options()
            opts.indent_with_tabs = True
            opts.indent_level = 1
            model_config_json = jsons.dumps(model_config)
            hw_config_json = jsons.dumps(hw_config)

            #json = jsons.dumps({
            #    "order_generator.Computation_order": order_generator.Computation_order,
            #})

            model_config_json = jsbeautifier.beautify(model_config_json, opts)
            hw_config_json = jsbeautifier.beautify(hw_config_json, opts)
            outfile.write(f'{{\n\t"model_config": {model_config_json},\n\t"hw_config": {hw_config_json},\n\t"order_generator.Computation_order": [\n')
            for index, event in enumerate(tqdm(order_generator.Computation_order)):
                outfile.write(f'\t\t// {index}:\n')
                outfile.write(f'\t\t{jsons.dumps(event)},\n')
            outfile.write(f'\t]\n}}\n')
        print(f"Done")

    #Visualizer.weightMappingByCO(hw_config, model_config, order_generator.Computation_order, f"{model}")
    #return
    #Visualizer.visualizeGif(hw_config, model_config, order_generator.Computation_order, f"{model}")
    #return
    
    log = {}

    ## Power and performance simulation ###
    start_simulation_time = time.time()
    print("--- Power and performance simulation---")
    controller = Controller(model_config, hw_config, order_generator, isTrace_controller, mapping_str, scheduling, path, log)
    end_simulation_time = time.time()
    print("--- Simulate in %s seconds ---\n" % (end_simulation_time - start_simulation_time))
    end_time = time.time()
    print("--- Run in %s seconds ---\n" % (end_time - start_time))

    Visualizer.visualizeSimulation2(hw_config, model_config, order_generator.Computation_order, log, f"{model}")

if __name__ == '__main__':
    main()

