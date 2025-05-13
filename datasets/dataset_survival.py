from __future__ import print_function, division
from cProfile import label
import os
import pdb
from unittest import case
import pandas as pd
import dgl 
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.general_utils import _series_intersection

from utils.general_utils import _get_split_loader, _print_network, _save_splits

ALL_MODALITIES = ['rna_clean.csv']  

class SurvivalDatasetFactory:

    def __init__(self,
        study,
        label_file, 
        omics_dir,
        seed, 
        print_info, 
        n_bins, 
        label_col, 
        eps=1e-6,
        num_patches=4096,
        is_mcat=False,
        is_survpath=True,
        type_of_pathway="combine",
        ):
        r"""
        Initialize the factory to store metadata, survival label, and slide_ids for each case id. 

        Args:
            - study : String 
            - label_file : String 
            - omics_dir : String
            - seed : Int
            - print_info : Boolean
            - n_bins : Int
            - label_col: String
            - eps Float
            - num_patches : Int 
            - is_mcat : Boolean
            - is_survapth : Boolean 
            - type_of_pathway : String

        Returns:
            - None
        """

        #---> self
        self.study = study
        self.label_file = label_file
        self.omics_dir = omics_dir
        self.seed = seed
        self.print_info = print_info
        #这里 self.train_ids, self.val_ids 为什么全是None ???
        self.train_ids, self.val_ids  = (None, None)
        self.data_dir = None
        self.label_col = label_col
        self.n_bins = n_bins
        self.num_patches = num_patches
        self.is_mcat = is_mcat
        self.is_survpath = is_survpath
        self.type_of_path = type_of_pathway




        if self.label_col == "survival_months":
            self.survival_endpoint = "OS"
            self.censorship_var = "censorship"
        elif self.label_col == "survival_months_pfi":
            self.survival_endpoint = "PFI"
            self.censorship_var = "censorship_pfi"
        elif self.label_col == "survival_months_dss":
            self.survival_endpoint = "DSS"
            self.censorship_var = "censorship_dss"

        #---> process omics data
        #这个函数最开始创建的字典用来保存模态信息 self.all_modalities
        self._setup_omics_data() 
        
        #---> labels, metadata, patient_df
        self._setup_metadata_and_labels(eps)

        #---> prepare for weighted sampling
        self._cls_ids_prep() #self.patient_cls_ids  self.slide_cls_ids

        #---> load all clinical data 
        self._load_clinical_data() #self.clinical_data 加载 clinical csv文件

        #---> summarize
        self._summarize()#就简单打印一些病例信息

        #---> read the signature files for the correct model/ experiment
        if self.is_mcat:#模型构建是否添加这个模块
            self._setup_mcat()
        elif self.is_survpath:# 主要是这个 模式
            self._setup_survpath()
        else:
            self.omic_names = []
            self.omic_sizes = []
       
    def _setup_mcat(self):
        r"""
        Process the signatures for the 6 functional groups required to run MCAT baseline
        
        Args:
            - self 
        
        Returns:
            - None 
        
        """
        self.signatures = pd.read_csv("./datasets_csv/metadata/signatures.csv")
        self.omic_names = []
        for col in self.signatures.columns:
            omic = self.signatures[col].dropna().unique()
            #这里将 从x里获取的病例基因信息与基因通路文件 取交集 然后排序
            omic = sorted(_series_intersection(omic, self.all_modalities["rna"].columns))
            self.omic_names.append(omic)
        # print("dataset_survival_131行",self.omic_names)
        self.omic_sizes = [len(omic) for omic in self.omic_names]




    def _setup_survpath(self):

        r"""
        Process the signatures for the 331 pathways required to run SurvPath baseline. Also provides functinoality to run SurvPath with 
        MCAT functional families (use the commented out line of code to load signatures)
        
        Args:
            - self 
        
        Returns:
            - None 
        
        """

        # for running survpath with mcat signatures 
        # self.signatures = pd.read_csv("./datasets_csv/metadata/signatures.csv")
        
        # running with hallmarks, reactome, and combined signatures
        self.signatures = pd.read_csv("./datasets_csv/metadata/{}_signatures.csv".format(self.type_of_path))
        
        self.omic_names = []
        for col in self.signatures.columns:
            omic = self.signatures[col].dropna().unique()
            omic = sorted(_series_intersection(omic, self.all_modalities["rna"].columns))
            self.omic_names.append(omic)
        print("dataset_survival 159",self.omic_names,"长度为",len(self.omic_names))
        self.omic_sizes = [len(omic) for omic in self.omic_names]
            

    def _load_clinical_data(self):
        r"""
        Load the clinical data for the patient which has grade, stage, etc.
        
        Args:
            - self 
        
        Returns:
            - None
            
        """
        path_to_data = "./datasets_csv/clinical_data/{}_clinical.csv".format(self.study)
        self.clinical_data = pd.read_csv(path_to_data, index_col=0)
    
    def _setup_omics_data(self):
        """
        read the csv with the omics data
        """
        self.all_modalities = {}
        for modality in ALL_MODALITIES:# 这里如果有足够多的数据 是可以一起加载的
            print("\n正在加载模态: {}".format(modality))
            file_path = os.path.join(self.omics_dir, modality)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError("找不到文件: {}".format(file_path))
                
            data = pd.read_csv(file_path, engine='python', index_col=0)
            
            print("已加载 {} 数据形状: {}".format(modality, data.shape))
            if data.empty:
                raise ValueError("从 {} 加载的数据为空".format(file_path))
                
            self.all_modalities[modality.split('_')[0]] = data

    def _setup_metadata_and_labels(self, eps):
        r"""
        Process the metadata required to run the experiment. Clean the data. Set up patient dicts to store slide ids per patient.
        Get label dict.
        
        Args:
            - self
            - eps : Float 
        
        Returns:
            - None 
        
        """

        #---> read labels 
        self.label_data = pd.read_csv(self.label_file, low_memory=False)

        #---> minor clean-up of the labels 
        uncensored_df = self._clean_label_data()

        #---> create discrete labels
        # self.bins = q_bins在此初始化 self.patients_df添加了 label列 值是对应的分箱编号
        self._discretize_survival_months(eps, uncensored_df)#传递的是 发生了感兴趣(死亡)的病例
    
        #---> get patient info, labels, and metada
        self._get_patient_dict()#self.patient_dict一个字典 其中键是case_id 值是对应的slide_id(以np序列存储)

        #self.num_classes self.label_dict 初始化了这两个值 并且 更新了 label对应的值
        self._get_label_dict()
        # self.patient_data 字典
        self._get_patient_data()

    def _clean_label_data(self):
        r"""
        Clean the metadata. For breast, only consider the IDC subtype.
        
        Args:
            - self 
        
        Returns:
            - None
            
        """

        # if "IDC" in self.label_data['oncotree_code']: # must be BRCA (and if so, use only IDCs)
        #     self.label_data = self.label_data[self.label_data['oncotree_code'] == 'IDC']

        self.patients_df = self.label_data.drop_duplicates(['case_id']).copy()

        uncensored_df = self.patients_df[self.patients_df[self.censorship_var] < 1]
        #uncensored_df 将 未删失 的病人挑出来
        return uncensored_df

    def _discretize_survival_months(self, eps, uncensored_df):
        """
        将连续的生存时间离散化为分类标签，使用中位数填充缺失值
        """
        # 检查数据类型
        # print("\n开始分箱前的数据检查:")
        # print("uncensored_df[self.label_col] 的数据类型:", uncensored_df[self.label_col].dtype)
        # print("self.patients_df[self.label_col] 的数据类型:", self.patients_df[self.label_col].dtype)
        # print("self.label_data[self.label_col] 的数据类型:", self.label_data[self.label_col].dtype)
        #
        # 如果不数值型，尝试转换
        if not np.issubdtype(uncensored_df[self.label_col].dtype, np.number):
            # 处理所有数据框中的#VALUE!
            for df in [uncensored_df, self.patients_df, self.label_data]:
                # 先将非#VALUE!的值转为数值型
                valid_mask = df[self.label_col] != '#VALUE!'
                valid_data = df[valid_mask][self.label_col].str.strip().astype(float)
                
                # 计算中位数（使用uncensored_df的中位数）
                median_survival = valid_data.median()
                # print("使用中位数填充:", median_survival)
                
                # 替换#VALUE!为中位数并转换类型
                df.loc[~valid_mask, self.label_col] = median_survival
                df[self.label_col] = df[self.label_col].astype(float)
        
        # print("\n转换后的数据类型检查:")
        # print("uncensored_df[self.label_col] 的数据类型:", uncensored_df[self.label_col].dtype)
        # print("self.patients_df[self.label_col] 的数据类型:", self.patients_df[self.label_col].dtype)
        # print("self.label_data[self.label_col] 的数据类型:", self.label_data[self.label_col].dtype)
        #
        # 继续原来的分操作
        disc_labels, q_bins = pd.qcut(uncensored_df[self.label_col], q=self.n_bins, retbins=True, labels=False)
        q_bins[-1] = self.label_data[self.label_col].max() + eps
        q_bins[0] = self.label_data[self.label_col].min() - eps

        #disc_labels: 存储每个病人的生存时间落在哪个分箱（0 到 n_bins-1 的整数）
        #q_bins: 存储分箱的边界值

        disc_labels, q_bins = pd.cut(self.patients_df[self.label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        self.patients_df.insert(2, 'label', disc_labels.values.astype(int))
        self.bins = q_bins# q_bins 是分箱的边界含有 n + 1 个 值
        
    def _get_patient_data(self):
        r"""
        Final patient data is just the clinical metadata + label for the patient 
        
        Args:
            - self 
        
        Returns: 
            - None
        
        """
        patients_df = self.label_data[~self.label_data.index.duplicated(keep='first')]
        #'case_id' 记录的是所有 case_id(np形式) 'label'记录的是前面步骤记录的 分箱和 删失 等信息(np形式)
        patient_data = {'case_id': patients_df["case_id"].values, 'label': patients_df['label'].values} # only setting the final data to self
        self.patient_data = patient_data

    def _get_label_dict(self):
        r"""
        For the discretized survival times and censorship, we define labels and store their counts.
        
        Args:
            - self 
        
        Returns:
            - self 
        
        """

        label_dict = {}
        key_count = 0
        #len(bins) = 5  range(4) i-> 0 1 2 3
        # key_count 和 c 的作用是什么?   {(0,-1e-06) : key_count}
        for i in range(len(self.bins)-1):
            for c in [0, 1]:
                label_dict.update({(i, c):key_count})
                key_count+=1 #key_count -> 8 有多少种组合?

        for i in self.label_data.index:#遍历每一行
            key = self.label_data.loc[i, 'label']#分箱的时候这个在那个组
            self.label_data.at[i, 'disc_label'] = key #添加了  disc_label 列 对应的值是 分箱的时候在那个箱

            censorship = self.label_data.loc[i, self.censorship_var]
            key = (key, int(censorship)) #这个元组 保存的是 分箱的箱号 和 是否是 删失
            self.label_data.at[i, 'label'] = label_dict[key] # 此时label 的值 对应为 是哪一种组合？

        self.num_classes=len(label_dict)
        self.label_dict = label_dict

    def _get_patient_dict(self):
        r"""
        For every patient store the respective slide ids

        Args:
            - self 
        
        Returns:
            - None
        """
    
        patient_dict = {}
        temp_label_data = self.label_data.set_index('case_id')#temp_label_data以'case_id'为索引，有重复
        for patient in self.patients_df['case_id']:
            slide_ids = temp_label_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient:slide_ids})#存储为numpy数组，这时一个病人就和若干个slide对应上了
        self.patient_dict = patient_dict
        self.label_data = self.patients_df#此时也更新了 相当于去重了 并且有了 label 列
        self.label_data.reset_index(drop=True, inplace=True)

    def _cls_ids_prep(self):
        r"""
        Find which patient/slide belongs to which label and store the label-wise indices of patients/ slides

        Args:
            - self 
        
        Returns:
            - None

        """
        # patient_cls_ids 里面有 n 个列表用于储存 划分为不同时间段的病人
        self.patient_cls_ids = [[] for i in range(self.num_classes)]   
        # Find the index of patients for different labels
        for i in range(self.num_classes):
        #找出 self.patient_data 中 'label' = i 的所有 标注信息
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0] 

        # Find the index of slides for different labels
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):# self.label_data['label'] 本身是个字典将一个 case_id 与若干个slide对上了
            self.slide_cls_ids[i] = np.where(self.label_data['label'] == i)[0]

    def _summarize(self):
        r"""
        Summarize which type of survival you are using, number of cases and classes
        
        Args:
            - self 
        
        Returns:
            - None 
        
        """
        if self.print_info:
            print("label column: {}".format(self.label_col))
            print("number of cases {}".format(len(self.label_data)))
            print("number of classes: {}".format(self.num_classes))

    def _patient_data_prep(self):
        patients = np.unique(np.array(self.label_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.label_data[self.label_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.label_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        _, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def return_splits(self, args, csv_path, fold):
        """
        Create the train and val splits for the fold
        """
        assert csv_path 
        all_splits = pd.read_csv(csv_path)
        print("Defining datasets...")

        # 获取训练集
        train_split, scaler = self._get_split_from_df(args, all_splits=all_splits, split_key='train', fold=fold, scaler=None)

        
        # 获取验证集
        val_split = self._get_split_from_df(args, all_splits=all_splits, split_key='val', fold=fold, scaler=scaler)

        args.omic_sizes = args.dataset_factory.omic_sizes
        datasets = (train_split, val_split)
        
        return datasets

    def _get_scaler(self, data):
        r"""
        Define the scaler for training dataset. Use the same scaler for validation set
        
        Args:
            - self 
            - data : np.array

        Returns: 
            - scaler : MinMaxScaler
        
        """
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data)# 消除不同基因表达量级的差异 -1 ~ 1
        return scaler
    
    def _apply_scaler(self, data, scaler):
        r"""
        Given the datatype and a predefined scaler, apply it to the data 
        
        Args:
            - self
            - data : np.array 
            - scaler : MinMaxScaler 
        
        Returns:
            - data : np.array """
        
        # find out which values are missing
        zero_mask = data == 0

        # transform data
        transformed = scaler.transform(data)
        data = transformed

        # rna -> put back in the zeros 
        data[zero_mask] = 0.
        
        return data

    def _get_split_from_df(self, args, all_splits, split_key: str='train', fold = None, scaler=None, valid_cols=None):
        r"""
        Initialize SurvivalDataset object for the correct split and after normalizing the RNAseq data 
        
        Args:
            - self 
            - args: argspace.Namespace 
            - all_splits: pd.DataFrame 
            - split_key : String 
            - fold : Int 
            - scaler : MinMaxScaler
            - valid_cols : List 

        Returns:
            - SurvivalDataset 
            - Optional: scaler (MinMaxScaler)
        
        """

        if not scaler:
            scaler = {}
        #all_splits 是打开的分割集 df    such as TCGA-05-4244
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        # 处理split中的ID，只保留TCGA-XX-XXXX部分  其实不需要这一步 可以先验证 是不是 这个样子
        processed_split = split.apply(lambda x: '-'.join(x.split('-')[:3]))

        # print("\n=== 详细的数据匹配信息 ===")
        # print("1. Split中的样本ID格式:")
        # print("前5个原始split样本:")
        # print(split.head())
        # print("前5个处理后的split样本:")
        # print(processed_split.head())

        # 使用处理后的ID进行匹配 是担心这里 case_id 与 分割的数据集中的 case_id 无法对应
        mask = self.label_data['case_id'].isin(processed_split.tolist())
        #args.dataset_factory.label_data 其实就是 self.label_data
        df_metadata_slide = args.dataset_factory.label_data.loc[mask, :].reset_index(drop=True)

        # print("\n2. Label Data中的case_id格式:")
        # print("前5个case_id:")
        # print(self.label_data['case_id'].head())

        omics_data_for_split = {}
        for key in args.dataset_factory.all_modalities.keys():
            raw_data_df = args.dataset_factory.all_modalities[key]

            # print("\n3. {}模态的索引格式:".format(key))
            # print("前5个索引:")
            # print(raw_data_df.index[:5])
            # print("\n{}模态的索引长度: {}".format(key, len(raw_data_df.index[0])))
            #
            # print("\n4. 比较示例:")
            # print("原始Split样本示例:", split.iloc[0])
            # print("处理后Split样本示例:", processed_split.iloc[0])
            # print("{}模态索引示例: {}".format(key, raw_data_df.index[0]))

            # 使用处理后的ID进行匹配
            mask = raw_data_df.index.isin(processed_split.tolist())
            filtered_df = raw_data_df[mask]

            # print("\n5. 匹配结果:")
            # print("Split样本数: {}".format(len(processed_split)))
            # print("匹配到的{}模态样本数: {}".format(key, len(filtered_df)))

            # if filtered_df.empty:
            #     print("警告: filtered_df 为空!")
            #     print("处理后的分割值:", processed_split.tolist())
            #     print("原始数据索引样本:", raw_data_df.index[:5])
            #     raise ValueError("在模态 {} 的 {} 分割中未找到数据".format(key, split_key))

            filtered_df = filtered_df[~filtered_df.index.duplicated()]
            filtered_df["temp_index"] = filtered_df.index
            filtered_df.reset_index(inplace=True, drop=True)
            #temp_index 就是 case_id
            
            # if filtered_df.empty:
            #     print("警告: filtered_df 为空!")
            #     print("分割值:", split.tolist())
            #     print("原始数据索引样本:", raw_data_df.index[:5])
            #     raise ValueError("在模态 {} 的 {} 分割中未找到数据".format(key, split_key))

            clinical_data_mask = self.clinical_data.case_id.isin(split.tolist())
            clinical_data_for_split = self.clinical_data[clinical_data_mask]
            clinical_data_for_split = clinical_data_for_split.set_index("case_id")
            clinical_data_for_split = clinical_data_for_split.replace(np.nan, "N/A")

            # 检查数据对齐
            mask = [True if item in list(filtered_df["temp_index"]) else False for item in df_metadata_slide.case_id]
            df_metadata_slide = df_metadata_slide[mask]
            df_metadata_slide.reset_index(inplace=True, drop=True)

            mask = [True if item in list(filtered_df["temp_index"]) else False for item in clinical_data_for_split.index]
            clinical_data_for_split = clinical_data_for_split[mask]
            clinical_data_for_split = clinical_data_for_split[~clinical_data_for_split.index.duplicated(keep='first')]

            if split_key == "train":
                case_ids = filtered_df["temp_index"]
                df_for_norm = filtered_df.drop(labels="temp_index", axis=1)
                # print("归一化前的形状: {}".format(df_for_norm.shape))
                
                # if df_for_norm.empty:
                #     raise ValueError("归一化前的 df_for_norm 为空")

                #将所有数据打平 然后重整为一列
                flat_df = df_for_norm.values.flatten().reshape(-1, 1)
                # print("展平后的数据形状: {}".format(flat_df.shape))
                
                # if flat_df.size == 0:
                #     raise ValueError("展平后的数据为空")

                # 标准化基因数据 本身为0 的还是 0
                scaler_for_data = self._get_scaler(flat_df)
                normed_flat_df = self._apply_scaler(data = flat_df, scaler = scaler_for_data)

                filtered_normed_df = pd.DataFrame(normed_flat_df.reshape([df_for_norm.shape[0], df_for_norm.shape[1]]))
                filtered_normed_df["temp_index"] = case_ids
                #重新变为原来的数据  相当于提前 把基因组数据 预处理过了 并添加上原始 基因名称
                filtered_normed_df.rename(columns=dict(zip(filtered_normed_df.columns[:-1], df_for_norm.columns)), inplace=True)
                scaler[key] = scaler_for_data #归一化的字典保留相关数据 rna ,因为可能使用多种基因来源
            else:
                case_ids = filtered_df["temp_index"]
                df_for_norm = filtered_df.drop(labels="temp_index", axis=1)
                flat_df = df_for_norm.values.flatten().reshape(-1, 1)
                scaler_for_data = scaler[key]
                normed_flat_df = self._apply_scaler(data = flat_df, scaler = scaler_for_data)
                filtered_normed_df = pd.DataFrame(normed_flat_df.reshape([df_for_norm.shape[0], df_for_norm.shape[1]]))
                filtered_normed_df["temp_index"] = case_ids
                filtered_normed_df.rename(columns=dict(zip(filtered_normed_df.columns[:-1], df_for_norm.columns)), inplace=True)
                
            omics_data_for_split[key] = filtered_normed_df

        if split_key == "train":
            sample=True
        elif split_key == "val":
            sample=True
        # 注意这里 train 的时候设置了 sample为True!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        split_dataset = SurvivalDataset(
            split_key=split_key,
            fold=fold,
            study_name=args.study,
            modality=args.modality,
            patient_dict=args.dataset_factory.patient_dict,
            metadata=df_metadata_slide,
            omics_data_dict=omics_data_for_split,
            data_dir=args.data_root_dir,
            num_classes=self.num_classes,
            label_col = self.label_col,
            censorship_var = self.censorship_var,
            valid_cols = valid_cols,
            is_training=split_key=='train',
            clinical_data = clinical_data_for_split,
            num_patches = self.num_patches,
            omic_names = self.omic_names,
            sample=sample
            )

        if split_key == "train":
            return split_dataset, scaler
        else:
            return split_dataset
    
    def __len__(self):
        return len(self.label_data)
    

class SurvivalDataset(Dataset):

    def __init__(self,
        split_key,
        fold,
        study_name,
        modality,
        patient_dict,
        metadata, 
        omics_data_dict,
        data_dir, 
        num_classes,
        label_col="survival_months_DSS",
        censorship_var = "censorship_DSS",
        valid_cols=None,
        is_training=True,
        clinical_data=-1,
        num_patches=4000,
        omic_names=None,
        sample=True,
        ): 

        super(SurvivalDataset, self).__init__()

        #---> self
        self.split_key = split_key
        self.fold = fold
        self.study_name = study_name
        self.modality = modality
        self.patient_dict = patient_dict
        self.metadata = metadata 
        self.omics_data_dict = omics_data_dict
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.censorship_var = censorship_var
        self.valid_cols = valid_cols
        self.is_training = is_training
        self.clinical_data = clinical_data
        self.num_patches = num_patches
        self.omic_names = omic_names
        self.num_pathways = len(omic_names)
        self.sample = sample

        # for weighted sampling
        self.slide_cls_id_prep()

        # 添加一个属性来存储基因映射关系

        self.gene_names = []
        if self.omic_names is not None:
            for pathway_genes in self.omic_names:
                self.gene_names.extend(pathway_genes)
            # 去重并保持顺序
            self.gene_names = list(dict.fromkeys(self.gene_names))



    def _get_valid_cols(self):
        r"""
        Getter method for the variable self.valid_cols 
        """
        return self.valid_cols

    def slide_cls_id_prep(self):
        r"""
        For each class, find out how many slides do you have
        
        Args:
            - self 
        
        Returns: 
            - None
        
        """
        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.metadata['label'] == i)[0]

            
    def __getitem__(self, idx):
        r"""
        Given the modality, return the correctly transformed version of the data
        
        Args:
            - idx : Int 
        
        Returns:
            - variable, based on the modality 
        
        """
        
        label, event_time, c, slide_ids, clinical_data, case_id = self.get_data_to_return(idx)

        if self.modality in ['omics', 'snn', 'mlp_per_path']:

            df_small = self.omics_data_dict["rna"][self.omics_data_dict["rna"]["temp_index"] == case_id]
            df_small = df_small.drop(columns="temp_index")
            df_small = df_small.reindex(sorted(df_small.columns), axis=1)
            omics_tensor = torch.squeeze(torch.Tensor(df_small.values))
            
            return (torch.zeros((1,1)), omics_tensor, label, event_time, c, clinical_data)
        
        #@TODO what is the difference between tmil_abmil and transmil_wsi
        elif self.modality in ["mlp_per_path_wsi", "abmil_wsi", "abmil_wsi_pathways", "deepmisl_wsi", "deepmisl_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways"]:

            df_small = self.omics_data_dict["rna"][self.omics_data_dict["rna"]["temp_index"] == case_id]
            df_small = df_small.drop(columns="temp_index")
            df_small = df_small.reindex(sorted(df_small.columns), axis=1)
            omics_tensor = torch.squeeze(torch.Tensor(df_small.values))
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)
            
            #@HACK: returning case_id, remove later
            return (patch_features, omics_tensor, label, event_time, c, clinical_data, mask)

        elif self.modality in ["coattn", "coattn_motcat"]:
            
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)

            omic1 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[0]].iloc[idx])
            omic2 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[1]].iloc[idx])
            omic3 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[2]].iloc[idx])
            omic4 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[3]].iloc[idx])
            omic5 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[4]].iloc[idx])
            omic6 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[5]].iloc[idx])

            return (patch_features, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c, clinical_data, mask)
        
        elif self.modality in ["survpath","survpath_Manba","survpath_version1","survpath_version2","survpath_version3"]:
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)
            # patch_features 是对应的 wsi 图片 (一个病人的所有图片都拼接在一起了 mask 用来标志是否要填0来补充)
            omic_list = []
            selectgen = []

            for i in range(self.num_pathways):
                omic_list.append(torch.tensor(self.omics_data_dict["rna"][self.omic_names[i]].iloc[idx]))
                # 确保 selectgen 包含基因名称
                selectgen.extend(self.omic_names[i])

            return (patch_features, omic_list, label, event_time, c, clinical_data, mask, selectgen)


        
        else:
            raise NotImplementedError('Model Type [%s] not implemented.' % self.modality)

    def get_data_to_return(self, idx):
        r"""
        Collect all metadata and slide data to return for this case ID 
        
        Args:
            - idx : Int 
        
        Returns: 
            - label : torch.Tensor
            - event_time : torch.Tensor
            - c : torch.Tensor
            - slide_ids : List
            - clinical_data : tuple
            - case_id : String
        
        """
        case_id = self.metadata['case_id'][idx]
        label = torch.Tensor([self.metadata['disc_label'][idx]]) # disc 对应的是分箱 分在那个箱子
        event_time = torch.Tensor([self.metadata[self.label_col][idx]])
        c = torch.Tensor([self.metadata[self.censorship_var][idx]])
        slide_ids = self.patient_dict[case_id]
        clinical_data = self.get_clinical_data(case_id)# 病情在那个阶段

        return label, event_time, c, slide_ids, clinical_data, case_id
    
    def _load_wsi_embs_from_path(self, data_dir, slide_ids):
        """
        Load all the patch embeddings from a list a slide IDs.

        Args:
            - self
            - data_dir : String
            - slide_ids : List

        Returns:
            - patch_features : torch.Tensor
            - mask : torch.Tensor

        """
        patch_features = []
        # load all slide_ids corresponding for the patient
        for slide_id in slide_ids:
            wsi_path = os.path.join(data_dir, '{}.pt'.format(slide_id.rstrip('.svs')))
            wsi_bag = torch.load(wsi_path)
            patch_features.append(wsi_bag)
        patch_features = torch.cat(patch_features, dim=0)

        if self.sample:
            # 这里是设置的是否 填充 patch 因为有的病例缺乏图片 方法是简单的全部 填 0
            max_patches = self.num_patches
            #这里堆叠起来设置的上限是 4000
            n_samples = min(patch_features.shape[0], max_patches)
            idx = np.sort(np.random.choice(patch_features.shape[0], n_samples, replace=False))
            patch_features = patch_features[idx, :]

            # make a mask
            if n_samples == max_patches:
                # sampled the max num patches, so keep all of them
                mask = torch.zeros([max_patches])
            else:
                # sampled fewer than max, so zero pad and add mask
                original = patch_features.shape[0]
                how_many_to_add = max_patches - original
                zeros = torch.zeros([how_many_to_add, patch_features.shape[1]])
                patch_features = torch.concat([patch_features, zeros], dim=0)
                mask = torch.concat([
                torch.zeros([original]), torch.ones([how_many_to_add])])

        else:
            mask = torch.ones([1])# 如果不去 补充 全填 1

        return patch_features, mask

    # def _load_wsi_embs_from_path(self, data_dir, slide_ids):
    #     """使用密度信息增强patch特征"""
    #     patch_features_list = []
    #     patch_coords_list = []
    #
    #     # 1. 加载特征和坐标
    #     for slide_id in slide_ids:
    #         h5_path = os.path.join(data_dir, f'{slide_id.rstrip(".svs")}.h5')
    #         import h5py
    #         with h5py.File(h5_path, 'r') as hf:
    #             # 将坐标转换为浮点类型
    #             coords = torch.from_numpy(hf['coords'][:]).float()  # 确保是float类型
    #             features = torch.from_numpy(hf['features'][:])
    #
    #             # 计算局部密度
    #             density_scores = self._compute_local_density(coords)
    #             # 使用密度信息增强特征
    #             enhanced_features = self._enhance_features_by_density(features, density_scores)
    #
    #             patch_features_list.append(enhanced_features)
    #             patch_coords_list.append(coords)
    #
    #     patch_features = torch.cat(patch_features_list, dim=0)
    #     patch_coords = torch.cat(patch_coords_list, dim=0)
    #
    #     # 处理采样和padding
    #     if self.sample:
    #         max_patches = self.num_patches
    #         total_patches = len(patch_features)
    #         if total_patches > max_patches:
    #             selected_indices = self._density_based_sampling(patch_coords, max_patches)
    #             patch_features = patch_features[selected_indices]
    #             mask = torch.zeros(max_patches)
    #         else:
    #             padding_size = max_patches - total_patches
    #             feature_padding = torch.zeros(padding_size, patch_features.shape[1])
    #             patch_features = torch.cat([patch_features, feature_padding])
    #             mask = torch.cat([torch.zeros(total_patches), torch.ones(padding_size)])
    #     else:
    #         mask = torch.ones([1])
    #
    #     return patch_features, mask
    #
    # def _compute_local_density(self, coords, k=10):
    #     """计算每个patch的局部密度"""
    #     # 确保坐标是浮点类型
    #     coords = coords.float()
    #     # 计算patch间距离
    #     distances = torch.cdist(coords, coords)
    #     # 获取最近的k个邻居的平均距离
    #     topk_dists, _ = torch.topk(distances, k=min(k + 1, len(distances)),
    #                                dim=1, largest=False)
    #     # 计算密度分数 (使用距离的倒数)
    #     density_scores = 1.0 / (topk_dists[:, 1:].mean(dim=1) + 1e-6)
    #     # 归一化密度分数
    #     density_scores = (density_scores - density_scores.min()) / \
    #                      (density_scores.max() - density_scores.min() + 1e-6)
    #     return density_scores
    #
    # def _enhance_features_by_density(self, features, density_scores):
    #     """基于密度分数增强特征"""
    #     # 将密度分数转换为注意力权重
    #     attention_weights = torch.sigmoid(density_scores).unsqueeze(1)
    #
    #     # 增强特征
    #     enhanced_features = features * (1.0 + attention_weights)
    #
    #     return enhanced_features
    #
    # def _density_based_sampling(self, coords, num_samples):
    #     """基于密度的采样策略"""
    #     # 计算密度分数
    #     density_scores = self._compute_local_density(coords)
    #
    #     # 使用密度分数作为采样概率
    #     probs = F.softmax(density_scores, dim=0)
    #
    #     # 基于概率进行采样
    #     selected_indices = torch.multinomial(probs,
    #                                          num_samples=min(num_samples, len(coords)),
    #                                          replacement=False)
    #
    #     return selected_indices
    
    def get_clinical_data(self, case_id):
        """
        Load all the patch embeddings from a list a slide IDs. 

        Args:
            - data_dir : String 
            - slide_ids : List
        
        Returns:
            - patch_features : torch.Tensor 
            - mask : torch.Tensor

        """
        try:
            stage = self.clinical_data.loc[case_id, "stage"]
        except:
            stage = "N/A"
        
        try:
            grade = self.clinical_data.loc[case_id, "grade"]
        except:
            grade = "N/A"

        try:
            subtype = self.clinical_data.loc[case_id, "subtype"]
        except:
            subtype = "N/A"
        
        clinical_data = (stage, grade, subtype)
        return clinical_data
    
    def getlabel(self, idx):
        r"""
        Use the metadata for this dataset to return the survival label for the case 
        
        Args:
            - idx : Int 
        
        Returns:
            - label : Int 
        
        """
        label = self.metadata['label'][idx]
        return label




    def __len__(self):
        return len(self.metadata) 