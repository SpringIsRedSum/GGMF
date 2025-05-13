from ast import Lambda
import numpy as np
import pdb
import os
from custom_optims.radam import RAdam
from models.model_ABMIL import ABMIL
from models.model_DeepMISL import DeepMISL
from models.model_MLPOmics import MLPOmics
from models.model_MLPWSI import MLPWSI
from models.model_SNNOmics import SNNOmics
from models.model_MaskedOmics import MaskedOmics
from models.model_MCATPathways import MCATPathways
from models.model_SurvPath import SurvPath
from models.model_SurvPath_with_nystrom import SurvPath_with_nystrom
from models.model_TMIL import TMIL
# from models.model_motcat import MCATPathwaysMotCat 缺失这个文件
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv

from transformers import (
    get_constant_schedule_with_warmup, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup
)


#----> pytorch imports
import torch
from torch.nn.utils.rnn import pad_sequence

from utils.general_utils import _get_split_loader, _print_network, _save_splits
from utils.loss_func import NLLSurvLoss

import torch.optim as optim



def _get_splits(datasets, cur, args):
    r"""
    Summarize the train and val splits and return them individually
    
    Args:
        - datasets : tuple
        - cur : Int 
        - args: argspace.Namespace
    
    Return:
        - train_split : SurvivalDataset
        - val_split : SurvivalDataset
    
    """

    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val splits...', end=' ')
    train_split, val_split = datasets
    _save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    return train_split,val_split


def _init_loss_function(args):
    r"""
    Init the survival loss function
    
    Args:
        - args : argspace.Namespace 
    
    Returns:
        - loss_fn : NLLSurvLoss or NLLRankSurvLoss
    
    """
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    else:
        raise NotImplementedError
    print('Done!')
    return loss_fn

def _init_optim(args, model):
    r"""
    Init the optimizer 
    
    Args: 
        - args : argspace.Namespace 
        - model : torch model 
    
    Returns:
        - optimizer : torch optim 
    """
    print('\nInit optimizer ...', end=' ')

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "radam":
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "lamb":
        optimizer = Lambda(model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError

    return optimizer







def _init_model(args):
    
    print('\nInit Model...', end=' ')
    if args.type_of_path == "xena":
        omics_input_dim = 1577
    elif args.type_of_path == "hallmarks":
        omics_input_dim = 4241
    elif args.type_of_path == "combine":
        omics_input_dim = 4999
    elif args.type_of_path == "multi":
        if args.study == "tcga_brca":
            omics_input_dim = 9947
        else:
            omics_input_dim = 14933
    else:
        omics_input_dim = 0
    
    # omics baselines
    if args.modality == "mlp_per_path":

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "dropout" : args.encoder_dropout, "num_classes" : args.n_classes,
        }
        model = MaskedOmics(**model_dict)

    elif args.modality == "omics":

        model_dict = {
            "input_dim": omics_input_dim, "projection_dim": 64, "dropout": args.encoder_dropout
        }
        model = MLPOmics(**model_dict)

    elif args.modality == "snn":

        model_dict = {
             "omic_input_dim" : omics_input_dim, 
        }
        model = SNNOmics(**model_dict)

    elif args.modality in ["abmil_wsi", "abmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = ABMIL(**model_dict)

    # unimodal and multimodal baselines
    elif args.modality in ["deepmisl_wsi", "deepmisl_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = DeepMISL(**model_dict)

    elif args.modality == "mlp_wsi":
        
        model_dict = {
            "wsi_embedding_dim":1024, "wsi_projection_dim":256,
            'n_classes':args.n_classes,
            "device": args.device

        }
        model = MLPWSI(**model_dict)

    elif args.modality in ["transmil_wsi", "transmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = TMIL(**model_dict)

    elif args.modality == "coattn":

        model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MCATPathways(**model_dict)

    elif args.modality == "coattn_motcat":

        model_dict = {
            'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes,
            "ot_reg":0.1, "ot_tau":0.5, "ot_impl":"pot-uot-l2"
        }
        model = MCATPathwaysMotCat(**model_dict)

    # survpath 
    elif args.modality in ["survpath","survpath_Manba","survpath_version1","survpath_version2","survpath_version3"]:
        if args.modality == "survpath":
            #omic_sizes 就是选中基因的个数 n_classes 是分箱的个数
            model_dict = {'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes}

            #是否使用 Nystrom attentin
            if args.use_nystrom:
                model = SurvPath_with_nystrom(**model_dict)
            else:
                model = SurvPath(**model_dict)
        elif args.modality == "survpath_version1":
            from models import model_SurvPath_version1
            # omic_sizes 就是选中基因的个数 n_classes 是分箱的个数
            model_dict = {'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes,"topk_ratio" : 0.8}
            model = model_SurvPath_version1.SurvPath(**model_dict)


        elif args.modality == "survpath_version2":
            from models import model_SurvPre_version2
            # omic_sizes 就是选中基因的个数 n_classes 是分箱的个数
            model_dict = {'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes,"topk_ratio" : 0.5}
            model = model_SurvPre_version2.SurvPath(**model_dict)

        elif args.modality == "survpath_version3":
            from models import model_SurvPre_version3
            model_dict = {'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes, "topk_ratio": 0.8,"num_heads":16}
            model = model_SurvPre_version3.SurvPath(**model_dict)

        elif args.modality == "survpath_Manba":
            # 得知道 **model_dict 里面的 键 和 值 是什么?
            # 还得知道 NdMamba2_2d 里面初始化需要什么? 搞清楚NdMamba2_2d输入的维度形状 ，然后去改 __init__
            from models.myModel import SurvPathMamba
            torch.autograd.set_detect_anomaly(True)
            # omic_sizes 就是选中基因的个数 n_classes 是分箱的个数
            model_dict = {'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes,"num_patches" : args.num_patches}
            model = SurvPathMamba(**model_dict)

            # 需求是 需要让基因组的长度和svs堆叠的时候能满足 64的倍数




    else:
        raise NotImplementedError

    # if torch.cuda.is_available():
    #     gpu = str(get_free_gpu())
    #     print("使用gpu" + gpu)
    model = model.to(torch.device(args.device))

    print('Done!')
    _print_network(args.results_dir, model)

    return model

def _init_loaders(args, train_split, val_split):
    r"""
    Init dataloaders for the train and val datasets 

    Args:
        - args : argspace.Namespace 
        - train_split : SurvivalDataset 
        - val_split : SurvivalDataset 
    
    Returns:
        - train_loader : Pytorch Dataloader 
        - val_loader : Pytorch Dataloader

    """

    print('\nInit Loaders...', end=' ')
    if train_split:
        train_loader = _get_split_loader(args, train_split, training=True, testing=False, weighted=args.weighted_sample, batch_size=args.batch_size)
    else:
        train_loader = None

    if val_split:
        val_loader = _get_split_loader(args, val_split,  testing=False, batch_size=1)
    else:
        val_loader = None
    print('Done!')

    return train_loader,val_loader

def _extract_survival_metadata(train_loader, val_loader):
    r"""
    Extract censorship and survival times from the train and val loader and combine to get numbers for the fold
    We need to do this for train and val combined because when evaulating survival metrics, the function needs to know the 
    distirbution of censorhsip and survival times for the trainig data
    
    Args:
        - train_loader : Pytorch Dataloader
        - val_loader : Pytorch Dataloader
    
    Returns:
        - all_survival : np.array
    
    """

    all_censorships = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.censorship_var].to_numpy()],
        axis=0)# 将沿着行的方向（垂直方向）合并成一个大的NumPy数组 只有censorship_var这一列

    all_event_times = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.label_col].to_numpy()],
        axis=0)# 将沿着行的方向（垂直方向）合并成一个大的NumPy数组 只有生存时间这一列

    #scikit-survival 库创建生存数据对象 未删失 ->事件发生 -> True
    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    return all_survival

def _unpack_data(modality, device, data):
    r"""
    Depending on the model type, unpack the data and put it on the correct device
    
    Args:
        - modality : String 
        - device : torch.device 
        - data : tuple 
    
    Returns:
        - data_WSI : torch.Tensor
        - mask : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - data_omics : torch.Tensor
        - clinical_data_list : list
        - mask : torch.Tensor
    
    """
    
    if modality in ["mlp_per_path", "omics", "snn"]:
        data_WSI = data[0]
        mask = None
        data_omics = data[1].to(device)
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
    
    elif modality in ["mlp_per_path_wsi", "abmil_wsi", "abmil_wsi_pathways", "deepmisl_wsi", "deepmisl_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways"]:
        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality in ["coattn", "coattn_motcat"]:
        
        data_WSI = data[0].to(device)
        data_omic1 = data[1].type(torch.FloatTensor).to(device)
        data_omic2 = data[2].type(torch.FloatTensor).to(device)
        data_omic3 = data[3].type(torch.FloatTensor).to(device)
        data_omic4 = data[4].type(torch.FloatTensor).to(device)
        data_omic5 = data[5].type(torch.FloatTensor).to(device)
        data_omic6 = data[6].type(torch.FloatTensor).to(device)
        data_omics = [data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6]

        y_disc, event_time, censor, clinical_data_list, mask = data[7], data[8], data[9], data[10], data[11]
        mask = mask.to(device)
    elif modality in ["survpath","survpath_Manba","survpath_version1","survpath_version2","survpath_version2","survpath_version3"]:

        data_WSI = data[0].to(device)
        data_omics = []
        for item in data[1][0]:
            data_omics.append(item.to(device))
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
        #print("core_ut_387行",data_omics)
        # return (patch_features, omic_list, label, event_time, c, clinical_data, mask,self.gene_names)
    else:
        raise ValueError('Unsupported modality:', modality)
    y_disc, event_time, censor = y_disc.to(device), event_time.to(device), censor.to(device)
    if(modality == "survpath_version3"):


        return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask,data[7]

    return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask

def _process_data_and_forward(model, modality, device, data):
    r"""
    Depeding on the modality, process the input data and do a forward pass on the model 
    
    Args:
        - model : Pytorch model
        - modality : String
        - device : torch.device
        - data : tuple
    
    Returns:
        - out : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - clinical_data_list : List
        这里 clinical_data_list 设置成列表的原因 估计 是因为在基因组csv文件里的数据不一定每个都能用
    """
    data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask ,gene_names = _unpack_data(modality, device, data)

    if modality in ["coattn", "coattn_motcat"]:  
        
        out = model(
            x_path=data_WSI, 
            x_omic1=data_omics[0], 
            x_omic2=data_omics[1], 
            x_omic3=data_omics[2], 
            x_omic4=data_omics[3], 
            x_omic5=data_omics[4], 
            x_omic6=data_omics[5]
            )  

    elif modality in ['survpath',"survpath_Manba","survpath_version1","survpath_version2","survpath_version3"]:
        if(modality == "survpath"):

            input_args = {"x_path": data_WSI.to(device)}
            for i in range(len(data_omics)):#将每个病人选取的基因 变成tensor 然后放到 gpu 上去
                input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)

            input_args["gene_names"] = gene_names  # 判断基因重要性✅ 必须加上


            input_args["return_attn"] = False
            out = model(**input_args)
        elif(modality == "survpath_version1"):# 输入输出可能有变化
            input_args = {"x_path": data_WSI.to(device)}
            for i in range(len(data_omics)):  # 将每个病人选取的基因 变成tensor 然后放到 gpu 上去
                input_args['x_omic%s' % str(i + 1)] = data_omics[i].type(torch.FloatTensor).to(device)

            input_args["return_attn"] = False
            out = model(**input_args)
        elif (modality == "survpath_version2"):  # 输入输出可能有变化
            input_args = {"x_path": data_WSI.to(device)}
            for i in range(len(data_omics)):  # 将每个病人选取的基因 变成tensor 然后放到 gpu 上去
                input_args['x_omic%s' % str(i + 1)] = data_omics[i].type(torch.FloatTensor).to(device)

            input_args["return_attn"] = False
            out = model(**input_args)

        elif (modality == "survpath_version3"):
            input_args = {"x_path": data_WSI.to(device)}
            for i in range(len(data_omics)):  # 将每个病人选取的基因 变成tensor 然后放到 gpu 上去
                input_args['x_omic%s' % str(i + 1)] = data_omics[i].type(torch.FloatTensor).to(device)
            #检测 有没有
            input_args["return_attn"] = False
            input_args["gene_names"] = gene_names

            out = model(**input_args)

    elif modality == "omics":
        input_args = {}# 这里就是有若干个基因
        for i in range(len(data_omics)):#将每个病人选取的基因 变成tensor 然后放到 gpu 上去
            input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)

        out = model(**input_args)
    elif modality == "mlp_wsi":
        out = model(data_WSI)

    else:
        out = model(
            data_omics = data_omics, 
            data_WSI = data_WSI, 
            mask = mask
            )
        
    if len(out.shape) == 1:
            out = out.unsqueeze(0)
    return out, y_disc, event_time, censor, clinical_data_list


def _calculate_risk(h):
    r"""
    Take the logits of the model and calculate the risk for the patient 
    
    Args: 
        - h : torch.Tensor 
    
    Returns:
        - risk : torch.Tensor 
    
    """
    hazards = torch.sigmoid(h)  # 转化为 概率
    # 1 - hazards 变为各个区间的生存概率 然后累积乘积 变成每个区间存活的概率
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()#将生存概率转换为总体风险
    return risk, survival.detach().cpu().numpy()

def _update_arrays(all_risk_scores, all_censorships, all_event_times, all_clinical_data, event_time, censor, risk, clinical_data_list):
    r"""
    Update the arrays with new values 
    
    Args:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - risk : torch.Tensor
        - clinical_data_list : List
    
    Returns:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
    
    """
    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    all_clinical_data.append(clinical_data_list)
    return all_risk_scores, all_censorships, all_event_times, all_clinical_data

def _train_loop_survival(epoch, model, modality, loader, optimizer, scheduler, loss_fn):
    r"""
    Perform one epoch of training 

    Args:
        - epoch : Int
        - model : Pytorch model
        - modality : String 
        - loader : Pytorch dataloader
        - optimizer : torch.optim
        - loss_fn : custom loss function class 
    
    Returns:
        - c_index : Float
        - total_loss : Float 
    
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []

    torch.autograd.set_detect_anomaly(True)
    # one epoch
    for batch_idx, data in enumerate(loader):
        # 1. DataLoader 内部调用 __len__ 确定总迭代次数
        # 2. 每次迭代：
        #    a. 调用 __getitem__ 获取 batch_size 个样本
        #    b. 使用 collate_fn 将这些样本组织成批次
        #    c. 返回组织好的批次数据
        # 3. data 就是 collate_fn 处理后的批次数据

        optimizer.zero_grad()
        # h 是模型的预测结果(预测在各个箱子的概率)  y_disc 是标签(实际上的分箱结果)
        # print("core_570行")
        # print(len(data))
        # print("\n=== Training Loop Debug ===")
        # print(f"1. Modality: {modality}")
        # print(f"2. Device: {device}")
        # print(f"3. Model type: {type(model)}")
        h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, modality, device, data)
        # if torch.isnan(h).any():
        #     print("h:",h) #h: tensor([[nan, nan, nan, nan]], device='cuda:0', grad_fn=<AddmmBackward0>)
        #     print("y_disc:",y_disc)
        #     print("event_time:",event_time)
        #     print("censor:",censor)
        #     print("clinical_data_list:",clinical_data_list)
        #     exit(509)
        loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
        loss_value = loss.item()


        loss = loss / y_disc.shape[0]  #除以样本数量 得到平均损失
        
        risk, _ = _calculate_risk(h) # risk是总体风险

        all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

        total_loss += loss_value
        # print("522行",total_loss)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (batch_idx % 20) == 0:
            print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))

            gene_importance = model.gene_importance.cpu().numpy()
            gene_names = loader.dataset.gene_names
            # 检查是否有非零的重要性分数
            # if np.any(gene_importance != 0):
            #     # 获取前10个重要的基因
            #     top_10_indices = np.argsort(gene_importance)[-10:][::-1]
            #     top_10_genes = [gene_names[i] for i in top_10_indices]
            #     top_10_importance = gene_importance[top_10_indices]
            #
            #     # 打印结果
            #     print("\nTop 10 Important Genes:")
            #     for gene, importance in zip(top_10_genes, top_10_importance):
            #         print("%s: %.4f" % (gene, importance))
            # else:
            #     print("\nNo gene importance scores available yet. The model is still initializing or training.")



    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    # 删失 和 删失作为一个样本对 不删失 和 不删失 作为一个样本对
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))

    return c_index, total_loss

def _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores):
    r"""
    Calculate various survival metrics 
    
    Args:
        - loader : Pytorch dataloader
        - dataset_factory : SurvivalDatasetFactory
        - survival_train : np.array
        - all_risk_scores : np.array
        - all_censorships : np.array
        - all_event_times : np.array
        - all_risk_by_bin_scores : np.array
        
    Returns:
        - c_index : Float
        - c_index_ipcw : Float
        - BS : np.array
        - IBS : Float
        - iauc : Float
    
    """
    #这里是硬编码使用 survival_months_dss 但是由于缺少这部分数据 我们采用survival_months
    # data = loader.dataset.metadata["survival_months_dss"]24//11/8
    data = loader.dataset.metadata["survival_months"]
    bins_original = dataset_factory.bins
    #这里为什么只 获取了 第2 第3 个边界值???
    which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])

    #---> delete the nans and corresponding elements from other arrays 
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    #<---

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS, iauc = 0., 0., 0., 0.

    # change the datatype of survival test to calculate metrics 
    try:
        survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw, BS, IBS, iauc
   
    # cindex2 (cindex_ipcw)
    try:
        # IPCW 这个评价指标到底在计算什么?
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.
    
    # brier score 
    try:
        # brier_score用于评估模型预测的生存时间与实际观察到的生存时间之间的差异
        _, BS = brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing BS')
        BS = 0.
    
    # IBS
    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.

    # iauc
    try:
        _, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1-all_risk_by_bin_scores[:, 1:], times=which_times_to_eval_at[1:])
    except:
        print('An error occured while computing iauc')
        iauc = 0.
    
    return c_index, c_index_ipcw, BS, IBS, iauc

def _summary(dataset_factory, model, modality, loader, loss_fn, survival_train=None):
    r"""
    Run a validation loop on the trained model 
    
    Args: 
        - dataset_factory : SurvivalDatasetFactory
        - model : Pytorch model
        - modality : String
        - loader : Pytorch loader
        - loss_fn : custom loss function clas
        - survival_train : np.array
    
    Returns:
        - patient_results : dictionary
        - c_index : Float
        - c_index_ipcw : Float
        - BS : List
        - IBS : Float
        - iauc : Float
        - total_loss : Float

    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.

    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    all_logits = []
    all_slide_ids = []

    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    with torch.no_grad():
        for data in loader:
            if modality == "survpath_version3":
                data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask,_ = _unpack_data(
                    modality, device, data)
            else:
                data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)

            if modality in ["coattn", "coattn_motcat"]:  
                h = model(
                    x_path=data_WSI, 
                    x_omic1=data_omics[0], 
                    x_omic2=data_omics[1], 
                    x_omic3=data_omics[2], 
                    x_omic4=data_omics[3], 
                    x_omic5=data_omics[4], 
                    x_omic6=data_omics[5]
                )  
            elif modality in ["survpath",'survpath_Manba',"survpath_version1","survpath_version2","survpath_version3"]:

                input_args = {"x_path": data_WSI.to(device)}
                for i in range(len(data_omics)):
                    input_args['x_omic%s' % str(i + 1)] = data_omics[i].type(torch.FloatTensor).to(device)
                input_args["return_attn"] = False
                input_args['mask'] = mask

                # 先调用模型获取输出
                h = model(**input_args)

                print("core_util 750")
                # 然后检查是否有基因重要性信息
                if hasattr(model, 'gene_importance') and hasattr(loader.dataset, 'gene_names'):
                    gene_importance = model.gene_importance.cpu().numpy()
                    gene_names = loader.dataset.gene_names

                    # 获取前10个重要的基因
                    top_10_indices = np.argsort(gene_importance)[-10:][::-1]
                    top_10_genes = [gene_names[i] for i in top_10_indices]
                    top_10_importance = gene_importance[top_10_indices]

                    # 打印结果
                    print("\nTop 10 Important Genes:")
                    for gene, importance in zip(top_10_genes, top_10_importance):
                        print("%s: %.4f" % (gene, importance))

                    # # 将基因重要性信息添加到结果中
                    # if "gene_importance" not in patient_results[case_id]:
                    #     patient_results[case_id]["gene_importance"] = {
                    #         'top_genes': top_10_genes,
                    #         'importance_scores': top_10_importance.tolist()
                    #     }
            elif modality == "omics":
                input_args = {}  # 这里就是有若干个基因
                for i in range(len(data_omics)):  # 将每个病人选取的基因 变成tensor 然后放到 gpu 上去
                    input_args['x_omic%s' % str(i + 1)] = data_omics[i].type(torch.FloatTensor).to(device)

                h = model(**input_args)
            elif modality == "mlp_wsi":
                h = model(data_WSI)

            else:
                h = model(
                    data_omics = data_omics, 
                    data_WSI = data_WSI, 
                    mask = mask
                    )
                    
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]


            risk, risk_by_bin = _calculate_risk(h) # risk_by_bin记录的是值是 总体风险
            all_risk_by_bin_scores.append(risk_by_bin)
            #_update_arrays 更新 传递的几个列表
            all_risk_scores, all_censorships, all_event_times, clinical_data_list = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)
            all_logits.append(h.detach().cpu().numpy())
            total_loss += loss_value
            all_slide_ids.append(slide_ids.values[count])
            count += 1
    # 这些值 是 怎么算的
    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    
    patient_results = {}
    for i in range(len(all_slide_ids)):
        slide_id = slide_ids.values[i]  #遍历每一张 svs 由于上面按照 "行" 拼接起来了 这样容易获取数据
        case_id = slide_id[:12]
        patient_results[case_id] = {} #一个病人 一个 字典
        patient_results[case_id]["time"] = all_event_times[i]
        patient_results[case_id]["risk"] = all_risk_scores[i]
        patient_results[case_id]["censorship"] = all_censorships[i]
        patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["logits"] = all_logits[i]
    
    c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores)

    return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss


def _get_lr_scheduler(args, optimizer, dataloader):
    scheduler_name = args.lr_scheduler
    warmup_epochs = args.warmup_epochs
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs

    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * len(dataloader)
    else:
        warmup_steps = 0
    if scheduler_name=='constant':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_name=='cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    elif scheduler_name=='linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    return lr_scheduler

def _step(cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader):
    r"""
    Trains the model for the set number of epochs and validates it.
    
    Args:
        - cur
        - args
        - loss_fn
        - model
        - optimizer
        - lr scheduler 
        - train_loader
        - val_loader
        
    Returns:
        - results_dict : dictionary
        - val_cindex : Float
        - val_cindex_ipcw  : Float
        - val_BS : List
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """
    #scikit-survival库的 生存数据对象 包含了 事件是否发生的标志 和 生存的时间
    all_survival = _extract_survival_metadata(train_loader, val_loader)
    
    for epoch in range(args.max_epochs):
        _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn)
        # _, val_cindex, _, _, _, _, total_loss = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)
        # print('Val loss:', total_loss, ', val_c_index:', val_cindex)
    # save the trained model
    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    
    results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)
    
    print('Final Val c-index: {:.4f}'.format(val_cindex))
    # print('Final Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}'.format(
    #     val_cindex, 
    #     val_cindex_ipcw,
    #     val_IBS,
    #     val_iauc
    #     ))

    return results_dict, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss)

def _train_val(datasets, cur, args):
    """   
    Performs train val test for the fold over number of epochs

    Args:
        - datasets : tuple
        - cur : Int 
        - args : argspace.Namespace 
    
    Returns:
        - results_dict : dict
        - val_cindex : Float
        - val_cindex2 : Float
        - val_BS : Float
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """

    #----> gets splits and summarize 返回对应的训练集和数据集 并把对应的 svs 分布保存到对应是csv文件中
    train_split, val_split = _get_splits(datasets, cur, args)
    
    #----> init loss function  NLLSurvLoss后面也可以自己定义
    loss_fn = _init_loss_function(args)


    #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    #----> init model # 加载模型 这里面的 elif 分支就是以后自定的模型加载方式
    model = _init_model(args)
    
    #---> init optimizer 返回优化器
    optimizer = _init_optim(args, model)

    #---> init loaders 数据加载器！！！！！！！！ 这里要看一下
    train_loader, val_loader = _init_loaders(args, train_split, val_split)

    # lr scheduler  设置 学利率的变化方式 稳定训练过程
    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    #---> do train val 这里开始训练 并 加载数据集
    results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss) = _step(cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)

    return results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss)
