from ultralytics import YOLO
import os
import csv
import torch
import pandas as pd
from numpy import mean


# 获取文件夹下的cfg
def get_cfg(cfg_folder):
    cfg_paths = []
    nums = []
    cfg_files = [f for f in os.listdir(cfg_folder) if f.lower().endswith('.yaml')]
    for cfg_file in cfg_files:
        cfg_path = os.path.join(cfg_folder, cfg_file)
        cfg_paths.append(cfg_path)
        base_name = os.path.basename(cfg_path)
        num = os.path.splitext(base_name)[0]
        nums.append(num)
    nums = [int(x) for x in nums]
    paired_lists = list(zip(nums, cfg_paths))
    sorted_lists = sorted(paired_lists, key=lambda x: x[0])
    nums, cfg_paths= zip(*sorted_lists)
    return cfg_paths


def train(cfg_folder, dataset_yaml = None, mode = 'cfg', resume=None):

    folder = cfg_folder
    if (mode == 'dataset'):
        folder = dataset_yaml
    train_cfg(folder,mode,resume)


def train_cfg (cfg_folder,mode='cfg',resume=None):
    cfgs = get_cfg(cfg_folder)

    # all results in a file
    exp_id = []
    Nauplius_Live_test_precision = []
    Nauplius_Live_test_Recall = []
    Nauplius_Live_test_ap50 = []
    Nauplius_Live_test_ap95 = []
    Nauplius_Live_test_F1 = []

    Nauplius_Shell_test_precision = []
    Nauplius_Shell_test_Recall = []
    Nauplius_Shell_test_ap50 = []
    Nauplius_Shell_test_ap95 = []
    Nauplius_Shell_test_F1 = []

    All_test_precision = []
    All_test_Recall = []
    All_test_ap50 = []
    All_test_ap95 = []
    All_test_F1 = []
    

    if resume==None:
        with open('Jayc_Fast_start/cfgs/results_on_test.csv','a', newline = '')as file:
            writer = csv.writer(file)
            writer.writerow(['exp_id' , 
                            'All_precision',
                            'All_Recall',
                            'All_ap50',
                            'All_ap95',
                            'All_F1',
                            'N_Live_precision',
                            'N_Live_Recall',
                            'N_Live_ap50',
                            'N_Live_ap95',
                            'N_Live_F1',
                            'N_Shell_precision',
                            'N_Shell_Recall',
                            'N_Shell_ap50',
                            'N_Shell_ap95',
                            'N_Shell_F1'
                            ])

    for cfg in cfgs:

        print("doing ", cfg)
        base_name = os.path.basename(cfg)
        name_without_extension = os.path.splitext(base_name)[0]
        if int(name_without_extension)<resume:
            continue
        if name_without_extension:
            project = 'Jayc_Fast_start/cfgs/' + name_without_extension
        else:
            project=None
        model = YOLO("yolov8n.pt")
        if mode == 'cfg': 
            model.train(cfg=cfg,
                        project = project )  # predict
        elif mode == 'dataset':
            model.train(data = cfg, 
                        epochs = 300,
                        patience=0,
                        batch=64,
                        workers=8,
                        project = project)

        metrics = model.val( split='test',
                    device = 0)  # no arguments needed, dataset and settings remembered
        metrics.box.ap  # 各类
        metrics.box.ap50 # 各类
        metrics.box.f1
        metrics.box.p
        metrics.box.r
        metrics.names

        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # a list contains map50-95 of each category
        # metrics.
        names=[]
        for i in range(len(metrics.names)):
            names.append(metrics.names[i])
        names.append('all')

        precison = metrics.box.p.copy().tolist()
        precison.append(mean(precison))
        recall = metrics.box.r.copy().tolist()
        recall.append(mean(recall))
        f1 = metrics.box.f1.copy().tolist()
        f1.append(mean(f1))
        ap50 = metrics.box.ap50.copy().tolist()
        ap50.append(metrics.box.map50)
        ap95 = metrics.box.ap.copy().tolist()
        ap95.append(metrics.box.map)

        val_path = 'Jayc_Fast_start/cfgs/' + name_without_extension + '/' + name_without_extension +'.csv'

        csvframe = pd.DataFrame({'classes':names,
                                'precision':precison,
                                'recall':recall,
                                'ap50': ap50,
                                'ap95': ap95,
                                'f1': f1,})
        csvframe.to_csv(val_path,sep=',')

        Nauplius_Live_test_precision.append(precison[0])
        Nauplius_Live_test_Recall.append(recall[0])
        Nauplius_Live_test_ap50.append(ap50[0])
        Nauplius_Live_test_ap95.append(ap95[0])
        Nauplius_Live_test_F1.append(f1[0])

        Nauplius_Shell_test_precision.append(precison[1])
        Nauplius_Shell_test_Recall.append(recall[1])
        Nauplius_Shell_test_ap50.append(ap50[1])
        Nauplius_Shell_test_ap95.append(ap95[1])
        Nauplius_Shell_test_F1.append(f1[1])

        All_test_precision.append(precison[2])
        All_test_Recall.append(recall[2])
        All_test_ap50.append(ap50[2])
        All_test_ap95.append(ap95[2])
        All_test_F1.append(f1[2])
        exp_id.append(name_without_extension)

        with open('Jayc_Fast_start/cfgs/results_on_test.csv','a', newline = '')as file:
            writer = csv.writer(file)
            writer.writerow([name_without_extension, 
                            precison[2],
                            recall[2],
                            ap50[2],
                            ap95[2],
                            f1[2],
                            precison[0],
                            recall[0],
                            ap50[0],
                            ap95[0],
                            f1[0],
                            precison[1],
                            recall[1],
                            ap50[1],
                            ap95[1],
                            f1[1]
                            ])



        del model.trainer.train_loader
        del model
        torch.cuda.empty_cache()





    print("finish")
    result_path = 'Jayc_Fast_start/cfgs/final.csv'
    final_csvframe = pd.DataFrame({'exp_id' : exp_id,
                                'Nauplius_Live_test_precision':Nauplius_Live_test_precision,
                                'Nauplius_Live_test_Recall':Nauplius_Live_test_Recall,
                                'Nauplius_Live_test_ap50':Nauplius_Live_test_ap50,
                                'Nauplius_Live_test_ap95': Nauplius_Live_test_ap95,
                                'Nauplius_Live_test_F1': Nauplius_Live_test_F1,
                                'Nauplius_Shell_test_precision': Nauplius_Shell_test_precision,
                                'Nauplius_Shell_test_Recall': Nauplius_Shell_test_Recall,
                                'Nauplius_Shell_test_ap50': Nauplius_Shell_test_ap50,
                                'Nauplius_Shell_test_ap95': Nauplius_Shell_test_ap95,
                                'Nauplius_Shell_test_F1': Nauplius_Shell_test_F1,
                                'All_test_precision': All_test_precision,
                                'All_test_Recall': All_test_Recall,
                                'All_test_ap50': All_test_ap50,
                                'All_test_ap95': All_test_ap95,
                                'All_test_F1': All_test_F1
                                })
    final_csvframe.to_csv(result_path, mode='a', sep=',', header=False)





if __name__ == "__main__":
    mode =  'dataset'   # 'dataset' or 'cfg' 可以是
    cfg = 'Jayc_Fast_start/cfgs'
    resume = 0
    dataset_yaml = '/home/jayc/jayc/CV/Dataset/fine-tune_Jan_24'
    train(cfg_folder = cfg, dataset_yaml = dataset_yaml, mode = mode, resume=resume)