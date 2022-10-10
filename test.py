import os
import argparse
import random
import json
import numpy as np
import torch
import logging
import data_loader

from args import Configs
from models import TReS, Net


# Local path
# modelPath = "/home/gohjiayi/Desktop/TReS/output_diy/diy_1_2021/sv/model_1_2021_4"
# relative path
modelPath = "output_diy/diy_1_2021/sv/model_1_2021_4"


def main(config,device): 
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpunum
    
    folder_path = {
        'live':     config.datapath,
        'csiq':     config.datapath,
        'tid2013':  config.datapath,
        'kadid10k': config.datapath,
        'clive':    config.datapath,
        'koniq':    config.datapath,
        'fblive':   config.datapath,
        'diy':    config.datapath,
        'va':   config.datapath,
        }

    img_num = {
        'live':     list(range(0, 29)), # 982 in total actually
        'csiq':     list(range(0, 30)),
        'kadid10k': list(range(0, 80)),
        'tid2013':  list(range(0, 25)),
        'clive':    list(range(0, 1162)),
        'koniq':    list(range(0, 10073)),
        'fblive':   list(range(0, 39810)),
        'diy':    list(range(0, 1582)),
        'va':    list(range(0, 391)),
        }
    

    print('Testing on {} dataset...'.format(config.dataset))
    


    
    SavePath = config.svpath
    svPath = SavePath+ config.dataset + '_' + str(config.vesion)+'_'+str(config.seed)+'/'+'sv'
    os.makedirs(svPath, exist_ok=True)
        
    
    
     # fix the seed if needed for reproducibility
    if config.seed == 0:
        pass
    else:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)


    
    pretrained_path = config.svpath + config.dataset + '_' + str(config.vesion)+'_'+str(config.seed)+'/'+'sv/'
    print('path: {}'.format(pretrained_path))
    # The following original lines are for training something
    path = pretrained_path + 'test_index_'+str(config.vesion)+'_'+str(config.seed)+'.json'
    path2 = pretrained_path + 'train_index_'+str(config.vesion)+'_'+str(config.seed)+'.json'
    
    # # The following update is for training on something and testing on another thing
    # # In order to test with a new dataset, we replace the paths with json files with the indexes

    # # updated this for VA
    # path = "output_va/va_1_2021/sv/test_index_1_2021.json"
    # path2 = "output_va/va_1_2021/sv/train_index_1_2021.json"
    
    # # updated this for CLIVE
    # path = "output_liveitw/clive_1_2021/sv/test_index_1_2021.json"
    # path2 = "output_liveitw/clive_1_2021/sv/train_index_1_2021.json"

    # updated this for LIVE
    path = "output/live_4_2021/sv/test_index_4_2021.json"
    path2 = "output/live_4_2021/sv/train_index_4_2021.json"



    with open(path) as json_file:
	    test_index = json.load(json_file)
    with open(path2) as json_file:
	    train_index =json.load(json_file)


    test_loader = data_loader.DataLoader(config.dataset, folder_path[config.dataset],
                                             test_index, config.patch_size,
                                             config.test_patch_num, istrain=False)
    test_data = test_loader.get_data()


    solver = TReS(config,device, svPath, folder_path[config.dataset], train_index, test_index,Net)
    # solver.load_state_dict(torch.load(modelPath))
    # solver.eval()

    version_test_save = 1000
    srcc_computed, plcc_computed = solver.test(test_data,version_test_save,svPath,config.seed,pretrained=1)
    print('srcc_computed {}, plcc_computed {}'.format(srcc_computed, plcc_computed))



    # pretrained_path = config.svpath + config.dataset + '_' + str(config.vesion)+'_'+str(config.seed)+'/'+'sv/'
    # print('path: {}'.format(pretrained_path))
    # path = pretrained_path + 'test_index_'+str(config.vesion)+'_'+str(config.seed)+'.json'
    # path2 = pretrained_path + 'train_index_'+str(config.vesion)+'_'+str(config.seed)+'.json'
    # # JY TODO: in order to test with a new dataset, we replace the paths with json files with the indexes

    # with open(path) as json_file:
    #     test_index = json.load(json_file)
    # with open(path2) as json_file:
    #     train_index = json.load(json_file)

    # test_loader = data_loader.DataLoader(config.dataset, folder_path[config.dataset],
    #                                             test_index, config.patch_size,
    #                                             config.test_patch_num, istrain=False)
    # test_data = test_loader.get_data()



if __name__ == '__main__':
    
    config = Configs()
    print(config)

    if torch.cuda.is_available():
            if len(config.gpunum)==1:
                device = torch.device("cuda", index=int(config.gpunum))
            else:
                device = torch.device("cpu")
        
    main(config,device)