import argparse

import torch
def get_free_gpu():
    if not torch.cuda.is_available():
        return None

    # 获取GPU数量和每个GPU的属性
    gpu_count = torch.cuda.device_count()
    free_memory = []
    for i in range(gpu_count):
        # 获取每个GPU的内存信息
        mem_info = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        free_memory.append(mem_info)

    # 找到内存最小的GPU，这可能是空闲的GPU
    min_mem = min(free_memory)
    free_gpu_id = free_memory.index(min_mem)

    return free_gpu_id

def _process_arg():
    r"""
    Function creates a namespace to read terminal-based arguments for running the experiment

    Args
        - None 

    Return:
        - args : argparse.Namespace

    """

    parser = argparse.ArgumentParser(description='Configurations for SurvPath Survival Prediction Training')

    #---> study related
    parser.add_argument('--study', type=str, help='study name',default='tcga_luad')
    parser.add_argument('--task', type=str, choices=['survival'],default='survival')
    #这里这个 n_classes 是什么意思? n_bins=args.n_classes, 就是分箱的作用
    parser.add_argument('--n_classes', type=int, default=4, help='number of classes (4 bins for survival)')
    parser.add_argument('--results_dir', default='./survpath_version3_topk_0.8_heads_16_enhanceWSI', help='results directory (default: ./results)')
    #这里可以选择不同的 基因组学的方法
    parser.add_argument("--type_of_path", type=str, default="combine", choices=["xena", "hallmarks", "combine"])

    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')

    #----> data related
    parser.add_argument('--data_root_dir', type=str, default='/devdata/xumingjie_data/dataset/wsiDataset/LUADprocess/step02/FEATURES_DIRECTORY/tumor_vs_normal_resnet_features/pt_files', help='data directory')
    # parser.add_argument('--data_root_dir', type=str, default='/devdata/xumingjie_data/dataset/wsiDataset/LUADprocess/step02/FEATURES_DIRECTORY/tumor_vs_normal_resnet_features/h5_files', help='data directory')
    parser.add_argument('--label_file', type=str, default='/devdata/xumingjie_data/dataset/SurvpATH-MAIN/SurvPath-main/datasets_csv/metadata/tcga_luad.csv', help='Path to csv with labels')
    parser.add_argument('--omics_dir', type=str, default='/devdata/xumingjie_data/dataset/SurvpATH-MAIN/SurvPath-main/datasets_csv/raw_rna_data/xena/luad', help='Path to dir with omics csv for all modalities')
    parser.add_argument('--num_patches', type=int, default=4000, help='number of patches')
    #根据不同的标签 训练
    parser.add_argument('--label_col', type=str, default="survival_months", help='type of survival (OS, DSS, PFI)')


    parser.add_argument("--wsi_projection_dim", type=int, default=256)
    parser.add_argument("--encoding_layer_1_dim", type=int, default=8)
    parser.add_argument("--encoding_layer_2_dim", type=int, default=16)
    parser.add_argument("--"
                        "encoder_dropout", type=float, default=0.25)

    #----> split related 
    parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--split_dir', type=str, default='splits', help='manually specify the set of splits to use, '
                    +'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--which_splits', type=str, default="5foldcv", help='where are splits')
        
    #----> training related  每一折训练多少epochs
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--opt', type=str, default="adam", help="Optimizer")
    parser.add_argument('--reg_type', type=str, default="None", help="regularization type [None, L1, L2]")
    parser.add_argument('--weighted_sample', action='store_true', default=True, help='enable weighted sampling')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--bag_loss', type=str, choices=['ce_surv', "nll_surv", "nll_rank_surv", "rank_surv", "cox_surv"], default='nll_surv',
                        help='survival loss function (default: ce)')
    parser.add_argument('--alpha_surv', type=float, default=0.5, help='weight given to uncensored patients')
    parser.add_argument('--reg', type=float, default=1e-5, help='weight decay / L2 (default: 1e-5)')
    parser.add_argument('--lr_scheduler', type=str, default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=1)

    #---> model related
    parser.add_argument('--fusion', type=str, default=None)
    #！！！！！！！！！！！！！！！！！！！！这里换模型
    parser.add_argument('--modality', type=str, default="survpath_version3")
    parser.add_argument('--encoding_dim', type=int, default=768, help='WSI encoding dim')
    parser.add_argument('--use_nystrom', action='store_true', default=False, help='Use Nystrom attentin in SurvPath.')
    parser.add_argument('--device', type=str, default='cuda:' + str(get_free_gpu()))

    args = parser.parse_args()

    if not (args.task == "survival"):
        print("Task and folder does not match")
        exit()

    return args