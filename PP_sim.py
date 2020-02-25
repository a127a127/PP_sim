#from Mapping import DefaultMapping
#from Mapping import HighParallelismMapping
from Mapping import SameColumnFirstMapping
from Mapping import SameRowFirstMapping
from Mapping import ParallelsimMapping

from OrderGenerator import OrderGenerator
from Controller import Controller

import time, sys, os
import pickle

def main():
    start_time = time.time()
    mapping = int(sys.argv[1])
    scheduling = int(sys.argv[2])
    paral = int(sys.argv[3])

    ### Mapping ##
    start_mapping_time = time.time()
    print("--- Mapping ---")
    print("Mapping policy:  ", end="")
    if mapping == 0: # DefaultMapping
        print("Default Mapping")
        exit()
        #mapping_information = DefaultMapping()
        #mapping_str = "Default_Mapping"
    elif mapping == 1: # HighParallelismMapping
        print("High Parallelism Mapping")
        exit()
        #mapping_information = HighParallelismMapping()
        #mapping_str = "High_Parallelism_Mapping"
    elif mapping == 2: # SameColumnFirstMapping
        print("Same Column First Mapping")
        mapping_information = SameColumnFirstMapping()
        mapping_str = "Same_Column_First_Mapping"
    elif mapping == 3: # SameRowFirstMapping
        print("Same Row First Mapping")
        mapping_information = SameRowFirstMapping()
        mapping_str = "Same_Row_First_Mapping"
    elif mapping == 4: # ParallelsimMapping
        print("Parallelsim Mapping"+sys.argv[3])
        mapping_information = ParallelsimMapping(paral)
        mapping_str = "Parallelsim_Mapping"+sys.argv[3]

    end_mapping_time = time.time()
    print("--- Mapping is finished in %s seconds ---\n" % (end_mapping_time - start_mapping_time))

    # lp = mapping_information.layer_mapping_to_xbar[0][0][0][0][0][0][0][0][0]
    # lp = mapping_information.layer_mapping_to_pe[0][0][0][0][1]
    # print(lp[0])
    # exit()

    ### Scheduling ###
    print("Scheduling policy: ", end="")
    if scheduling == 0: # Non-pipeline
        print("Non-pipeline")
        scheduling_str = "Non_pipeline"
    elif scheduling == 1: # Pipeline
        print("Pipeline")
        scheduling_str = "Pipeline"
    print()
    ### dir ###
    if not os.path.exists('./statistics/'+mapping_str+'/'+scheduling_str):
            os.makedirs('./statistics/'+mapping_str+'/'+scheduling_str)
    
    ### Generate Order ###
    isGenerateOrder = True

    ### Trace ###
    isTrace_order      = True
    isTrace_controller = False

    ### Generate computation order graph ### 
    if isGenerateOrder:
        start_order_time = time.time()
        print("--- Generate computation order graph ---")

        order_generator = OrderGenerator(mapping_information, isTrace_order)

        end_order_time = time.time()
        print("--- Computation order graph is generated in %s seconds ---\n" % (end_order_time - start_order_time))

    else:
        print("Load order_generator from file \"ComputationOrder\"")
        start_load = time.time()
        with open ('ComputationOrder', 'rb') as fp:
            order_generator = pickle.load(fp)
        end_load = time.time()
        print("--- Load computation order in %s seconds ---\n" %(end_load - start_load))

    #exit()
    ## delete mapping data
    # del order_generator.mp_info
    # del mapping_information

    ## Power and performance simulation ###
    start_simulation_time = time.time()
    print("--- Power and performance simulation---")

    controller = Controller(order_generator, isTrace_controller, mapping_str, scheduling_str)
    controller.run()

    end_simulation_time = time.time()
    print("--- Simulate in %s seconds ---\n" % (end_simulation_time - start_simulation_time))
    end_time = time.time()
    print("--- Run in %s seconds ---\n" % (end_time - start_time))
    controller.print_statistics_result()

if __name__ == '__main__':
    main()

