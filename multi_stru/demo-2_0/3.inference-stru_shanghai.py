import os
import pandas as pd
#这里决定是否有flash_attn
from model.demo_model_v5_flash import *
#from model.demo_model_v4_no_flash import *
encoder_type = "flash_attn"
import time
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import shutil
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from torch.cuda.amp import GradScaler, autocast


###进行的任务
file_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path=os.path.join(file_dir,"input/0617_mini-dataset_v1.csv") #测试数据集
embed_stru_path=os.path.join(file_dir,"input/0617-embed_v1/") #测试embed

model_params=os.path.join(file_dir,"temp/demo2_0-largedata-with_similar_stru/model-epoch=0.pth") #模型测试路径

print("[dataset_path]cur path:",dataset_path)
print("[embed_stru_path]cur path:",embed_stru_path)
print("[model_path]cur path:",model_params)

cls_task=True
mlm_task=False
if  mlm_task:
    mask_ratio = {"ag": 0.1, "ab_h": 0.1, "ab_l": 0.1}
    print("[mask_ratio]:",mask_ratio)
else:
    mask_ratio = {"ag": 0., "ab_h": 0., "ab_l": 0.}
    print("no mask!!")
batch_size = 32
num_epochs = 50
print("[batch_size]:",batch_size)
print("[num_epochs]:",num_epochs)

prot_bert_path=os.path.join(file_dir,"prot_bert") # general prot_bert path
###Shanghai
# prot_bert_path="/home/data/byc/multi_stru/prot_bert" #Shanghai use me!
###Tianhe
#prot_bert_path="/fs1/home/caolab/prot_bert" #Tianhe use me!
###pp2
#prot_bert_path="/mnt/d/byc/project/10-生物物理所/task/prot_bert/" #pp2 use me!
print("[prot_bert_path]cur path:",prot_bert_path)
cuda_available = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print("[num_gpus]:",num_gpus)
if cuda_available:
    device = torch.device("cuda:2")
else:
    device = torch.device("cpu")
print("[cur device]device:",device)
print("===="*4)
task_id="demo2_0-test" #代表对应任务名
#demo2_0模型;数据集为minidata,无任何结构输入
print("===="*4)
print("===="*4)
print("[task_id]:",task_id)
print(f"[cls_task]{cls_task},[mlm_task]:{mlm_task}")
print("===="*4)
print("===="*4)
#demo2_0模型;数据集为minidata,无任何结构输入
temp_path=f"temp/{task_id}"

def ensure_new_folder(folder_path):
    """
    确保指定路径的文件夹是新创建的。如果文件夹已存在，则删除并重新创建。

    参数:
    folder_path (str): 文件夹的路径。
    """
    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 文件夹存在，使用shutil.rmtree删除文件夹
        shutil.rmtree(folder_path)
        print(f"文件夹 {folder_path} 已存在，已被删除。")

    # 创建一个新的空文件夹
    os.makedirs(folder_path)
    print(f"新文件夹 {folder_path} 已创建。")
ensure_new_folder(temp_path)

class ProteinDataset(Dataset):
    def __init__(self, heavy, light, antigen, tokenizer, max_len_list, labels=None,stru_embed_id=None,mlm_mask_ratio={"ag":0.,"ab_h":0.,"ab_l":0.}): #mlm_mask_ratio for ag,ab_h,ab_l
        self.heavy = heavy
        self.light = light
        self.antigen = antigen
        self.tokenizer = tokenizer
        self.max_len_list = max_len_list  # [max_len_heavy, max_len_light, max_len_antigen]
        self.labels = labels
        self.stru_embed_id = stru_embed_id
        self.mlm_mask_ratio=mlm_mask_ratio

    def __len__(self):
        return len(self.heavy)

    def __getitem__(self, idx):
        heavy = self.heavy[idx]
        #print("heavy:",heavy)
        light = self.light[idx]
        antigen = self.antigen[idx]
        label = self.labels[idx] if self.labels is not None else None
        stru_embed_id = self.stru_embed_id[idx] if self.labels is not None else None


        ag_input_ids, ag_attention_mask = mask_helper(antigen, self.max_len_list[2], self.tokenizer)
        ab_h_input_ids, ab_h_attention_mask = mask_helper(heavy, self.max_len_list[0], self.tokenizer)
        ab_l_input_ids, ab_l_attention_mask = mask_helper(light, self.max_len_list[1], self.tokenizer)
        D={}
        for cur_name,cur_input_ids in zip(["ag","ab_h","ab_l"],[ag_input_ids,ab_h_input_ids,ab_l_input_ids]):
            cur_input_ids=cur_input_ids.squeeze(0)
            if self.mlm_mask_ratio[cur_name]!=0: #如果需要mask
                non_zero_positions = [i[0] for i in torch.nonzero(cur_input_ids).tolist()][1:-1] #不考虑首尾
                num_masks = int(len(non_zero_positions) * self.mlm_mask_ratio[cur_name])
                mask_indices = random.sample(non_zero_positions, num_masks) #找mask
                D[f"{cur_name}_input_ids"]=torch.tensor([25 if i in mask_indices else x for i, x in enumerate(cur_input_ids.tolist())])
                D[f"{cur_name}_mask_loc"] = torch.tensor([x if i in mask_indices else 0 for i, x in enumerate(cur_input_ids.tolist())]) #0代表这个位置被mlm了,否则存放这个位置的真实值,直接用来mlm了
            else:
                D[f"{cur_name}_input_ids"]=torch.tensor([25 if i in [] else x for i, x in enumerate(cur_input_ids.tolist())])
                D[f"{cur_name}_mask_loc"] = torch.tensor([x if i in [] else 0 for i, x in enumerate(cur_input_ids.tolist())]) #0代表这个位置被mlm了,否则存放这个位置的真实值,直接用来mlm了
            #print("[D]:", D)
            #统计长度
            #SXASXSA
        #print(ab_h_input_ids)
        return {
            'ag_input_ids': D['ag_input_ids'].squeeze(0),
            'ag_attention_mask': ag_attention_mask.squeeze(0),
            'ab_h_input_ids': D['ab_h_input_ids'].squeeze(0),
            'ab_h_attention_mask': ab_h_attention_mask.squeeze(0),
            'ab_l_input_ids': D['ab_l_input_ids'].squeeze(0),
            'ab_l_attention_mask': ab_l_attention_mask.squeeze(0),
            'ag_mask_loc': D['ag_mask_loc'].squeeze(0),
            'ab_h_mask_loc': D['ab_h_mask_loc'].squeeze(0),
            'ab_l_mask_loc': D['ab_l_mask_loc'].squeeze(0),
            'stru_embed_id':stru_embed_id, #对应的id
            'label': label
        }


# # 预测函数
def cls_predict(dataloader, model, device):
    model.eval()  # 设置模型为评估模式
    predictions = []
    final_labels = []
    with torch.cuda.amp.autocast(enabled=True):
        with torch.no_grad():  # 关闭梯度计算
            for batch in dataloader:
                ag_input_ids = batch['ag_input_ids'].to(device)
                ag_attention_mask = batch['ag_attention_mask'].to(device, dtype=torch.float16)
                ab_h_input_ids = batch['ab_h_input_ids'].to(device)
                ab_h_attention_mask = batch['ab_h_attention_mask'].to(device, dtype=torch.float16)
                ab_l_input_ids = batch['ab_l_input_ids'].to(device)
                ab_l_attention_mask = batch['ab_l_attention_mask'].to(device, dtype=torch.float16)
                stru_embed_id = batch['stru_embed_id'].to(device, dtype=torch.int)
                labels = batch['label'].long().to(device)
                #print("example1:", ab_h_input_ids[0])
                #print("example_mask:", ag_attention_mask[0])

                cls_output = model(ag=ag_input_ids, attention_mask_ag=ag_attention_mask,
                                ab_h=ab_h_input_ids, attention_mask_ab_h=ab_h_attention_mask,
                                ab_l=ab_l_input_ids, attention_mask_ab_l=ab_l_attention_mask,
                                   stru_ids=stru_embed_id)["cls_prob"]
                preds = cls_output[:,1].cpu().numpy().tolist() #提取预测为1的概率
                predictions.extend(preds)
                final_labels.extend(labels.cpu().numpy().tolist())
                #break
    #print(predictions, final_labels)
    return predictions, final_labels


# 创建数据集和数据加载器
#tokenizer = BertTokenizer.from_pretrained("/fs1/home/caolab/prot_bert", do_lower_case=False)
tokenizer = BertTokenizer.from_pretrained(prot_bert_path, do_lower_case=False)
max_len_list = [150,150,400]

# 加载数据集
dataset = pd.read_csv(dataset_path)#.head(1000)
dataset=dataset.dropna(subset="bind_value")
dataset["test_seed"]=10
#print("only seq!!")

#代表训练,测试,验证
test_id=[10]
Heavy = dataset['Heavy'].tolist()
#print("AAA:",Heavy)
Light = dataset['Light'].tolist()
Antigen = dataset['variant_seq'].tolist()
stru_embed_id =dataset["stru_embed_id"].tolist()
labels = dataset['bind_value'].tolist()  # 假设使用 'bind_value' 作为标签
five_fold_seed = dataset["test_seed"].tolist()
# 拆分数据集
# 根据five-fold_seed划分数据集

test_dataset = ProteinDataset(
    [Heavy[i] for i in range(len(Heavy)) if five_fold_seed[i] in test_id],
    [Light[i] for i in range(len(Light)) if five_fold_seed[i] in test_id],
    [Antigen[i] for i in range(len(Antigen)) if five_fold_seed[i] in test_id],
    tokenizer,
    max_len_list,
    [labels[i] for i in range(len(labels)) if five_fold_seed[i] in test_id],
    [stru_embed_id[i] for i in range(len(labels)) if five_fold_seed[i] in test_id],
    mlm_mask_ratio=mask_ratio
)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model = Demo_seq_BCR(bert="/fs1/home/caolab/prot_bert", freeze_bert=False, ab_freeze_layer_count=16).to(device)
# model = Demo_seq_BCR(bert="/fs1/home/caolab/prot_bert", freeze_bert=True, ab_freeze_layer_count=-1,
#                      cls_task=True,mlm_task=True).to(device) #双任务
model = Demo_seq_BCR(bert=prot_bert_path, freeze_bert=True, ab_freeze_layer_count=-1,
                    heads=36,local_stru_embed_path=embed_stru_path,node_attr_dim=1152,
                     ag_time=5,ab_ag_dim=2300,
                     cls_task=cls_task,mlm_task=mlm_task).to(device) #双任务
#print("model!!!")
state_dict=torch.load(model_params)
model.load_state_dict(state_dict)
## 定义损失函数
#cls-loss
cls_criterion = nn.CrossEntropyLoss() #适用于二分类or多分类
#mlm-loss

def compute_masked_mlm_loss(mlm_output, mask_loc,show_sample=True,return_acc=True):
    """
    计算掩码语言模型（MLM）的交叉熵损失。

    参数:
    - mlm_output: MLM 输出张量，形状为 [batch_size, seq_length, vocab_size]
    - mask_loc: 掩码位置张量，形状为 [batch_size, seq_length]
    - target_labels: 目标标签张量，形状为 [batch_size, seq_length]

    返回:
    - loss: 交叉熵损失
    """
    # 展平张量
    mlm_output = mlm_output.reshape(-1, mlm_output.shape[-1])  # 展平为 [batch_size * seq_length, vocab_size]
    mask_loc = mask_loc.reshape(-1)  # 展平为 [batch_size * seq_length]

    # 提取 mask_loc 的非零位置
    mask_indices = torch.nonzero(mask_loc, as_tuple=True)[0]

    # 提取对应的 mlm 值和目标标签
    mlm_masked = mlm_output[mask_indices]
    mask_loc = mask_loc[mask_indices]

    # 将 mask_loc 的非零位置的值设置为 -5，这样最大值为0-19
    mask_loc -= 5

    # 计算交叉熵损失
    mlm_criterion = nn.CrossEntropyLoss()
    mlm_loss = mlm_criterion(mlm_masked, mask_loc)
    if return_acc:
        _, pred = torch.max(mlm_masked, dim=-1)  # 预测值
        correct_num = (pred == mask_loc).sum().item()
        total_num = mask_loc.shape[0]
    else:
        correct_num = 0
        total_num = 0

    if show_sample:
        mlm_masked=mlm_masked[:10,:]
        mask_loc=mask_loc[:10]
        amino_acids_custom_order = 'LAGVESIDKRTPNQFYMHCW'
        _, max_indices = torch.max(mlm_masked, dim=-1) #预测值
        pred_output = [amino_acids_custom_order[idx] for idx in max_indices.tolist()]
        true_output = [amino_acids_custom_order[idx] for idx in mask_loc.tolist()]
        print("[pred Amino]:",pred_output)
        print("[true Amino]:",true_output)

    return mlm_loss,correct_num,total_num


test_predictions, test_labels = cls_predict(test_dataloader, model, device)
original_dataset=dataset

# 假设five_fold_seed列用于区分验证集和测试集
test_indices = original_dataset.index.tolist()
original_dataset.loc[test_indices, 'Test Predictions'] = test_predictions
original_dataset.to_csv(f'{temp_path}/predictions_results.csv', index=False)
#break


