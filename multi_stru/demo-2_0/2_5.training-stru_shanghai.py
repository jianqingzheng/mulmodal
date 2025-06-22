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

file_dir = os.path.dirname(os.path.abspath(__file__))
###进行的任务
cls_task=True
mlm_task=False
if  mlm_task:
    mask_ratio = {"ag": 0.1, "ab_h": 0.1, "ab_l": 0.1}
    print("[mask_ratio]:",mask_ratio)
else:
    mask_ratio = {"ag": 0., "ab_h": 0., "ab_l": 0.}
    print("no mask!!")
batch_size = 32
num_epochs = 500
print("[batch_size]:",batch_size)
print("[num_epochs]:",num_epochs)
prot_bert_path=os.path.join(file_dir,"prot_bert") # general prot_bert path
###Shanghai
# prot_bert_path="/home/data/byc/multi_stru/prot_bert" #Shanghai use me!
###Tianhe
#prot_bert_path="/fs1/home/caolab/prot_bert" #Tianhe use me!
###pp2
#prot_bert_path="/mnt/d/byc/project/10-生物物理所/task/prot_bert/" #pp2 use me!
cuda_available = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print("[num_gpus]:",num_gpus)
if cuda_available:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("[cur device]device:",device)
dataset_path=os.path.join(file_dir,"input/0617_mini-dataset_v1.csv") #测试数据集
embed_stru_path=os.path.join(file_dir,"input/0617-embed_v1/") #测试embed
# dataset_path="input/0617_mini-dataset_v1.csv"
# embed_stru_path="input/0617-embed_v1/"
print("[prot_bert_path]cur path:",prot_bert_path)
print("[dataset_path]cur path:",dataset_path)
print("[embed_stru_path]cur path:",embed_stru_path)
print("===="*4)
task_id="demo2_0-minidata-with_stru" #代表对应任务名
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
    with torch.cuda.amp.autocast(enabled=amp_val):
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

#代表训练,测试,验证
val_id=[9]
test_id=[10]
train_id=[i for i in dataset["five-fold_seed=408"].unique() if i not in val_id+test_id] #代表train_id对应的值
train_id.sort() #排个序
print(train_id,val_id,test_id)
Heavy = dataset['Heavy'].tolist()
#print("AAA:",Heavy)
Light = dataset['Light'].tolist()
Antigen = dataset['variant_seq'].tolist()
stru_embed_id =dataset["stru_embed_id"].tolist()
labels = dataset['bind_value'].tolist()  # 假设使用 'bind_value' 作为标签
five_fold_seed = dataset['five-fold_seed=408'].tolist()
# 拆分数据集
# 根据five-fold_seed划分数据集
train_dataset = ProteinDataset(
    [Heavy[i] for i in range(len(Heavy)) if five_fold_seed[i] in train_id],
    [Light[i] for i in range(len(Light)) if five_fold_seed[i] in train_id],
    [Antigen[i] for i in range(len(Antigen)) if five_fold_seed[i] in train_id],
    tokenizer,
    max_len_list,
    [labels[i] for i in range(len(labels)) if five_fold_seed[i] in train_id],
    [stru_embed_id[i] for i in range(len(labels)) if five_fold_seed[i] in train_id],
    mlm_mask_ratio=mask_ratio
)

val_dataset = ProteinDataset(
    [Heavy[i] for i in range(len(Heavy)) if five_fold_seed[i] in val_id],
    [Light[i] for i in range(len(Light)) if five_fold_seed[i] in val_id],
    [Antigen[i] for i in range(len(Antigen)) if five_fold_seed[i] in val_id],
    tokenizer,
    max_len_list,
    [labels[i] for i in range(len(labels)) if five_fold_seed[i] in val_id],
    [stru_embed_id[i] for i in range(len(labels)) if five_fold_seed[i] in val_id],
    mlm_mask_ratio=mask_ratio
)

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

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型

#model = Demo_seq_BCR(bert="/fs1/home/caolab/prot_bert", freeze_bert=False, ab_freeze_layer_count=16).to(device)
# model = Demo_seq_BCR(bert="/fs1/home/caolab/prot_bert", freeze_bert=True, ab_freeze_layer_count=-1,
#                      cls_task=True,mlm_task=True).to(device) #双任务
model = Demo_seq_BCR(bert=prot_bert_path, freeze_bert=True, ab_freeze_layer_count=-1,
                    heads=36,local_stru_embed_path=embed_stru_path,node_attr_dim=1152,
                     ag_time=5,ab_ag_dim=2300,
                     cls_task=cls_task,mlm_task=mlm_task).to(device) #双任务
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


##优化器
lr = 1e-5 #1e-5
schedule_interval = 1
schedule_ratio = 0.9
amp_val = True
params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(params_to_optimize, lr=lr, eps=1e-8) #eps=1e-8
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, schedule_interval, gamma=schedule_ratio
)
scaler = torch.cuda.amp.GradScaler(enabled=amp_val)

##删除temp文件

# 创建一个空列表来存储每个epoch的信息
epoch_info = []
last_time_val_loss = float("Inf")
# 训练模型
for epoch in range(num_epochs):
    start_time = time.time()  # 记录当前时间
    model.train()
    total_cls_loss = 0
    total_mlm_loss = 0

    for batch in train_dataloader:
        ag_input_ids = batch['ag_input_ids'].to(device)
        ag_attention_mask = batch['ag_attention_mask'].to(device, dtype=torch.float16)
        ab_h_input_ids = batch['ab_h_input_ids'].to(device)
        ab_h_attention_mask = batch['ab_h_attention_mask'].to(device, dtype=torch.float16)
        ab_l_input_ids = batch['ab_l_input_ids'].to(device)
        ab_l_attention_mask = batch['ab_l_attention_mask'].to(device, dtype=torch.float16)
        stru_embed_id = batch['stru_embed_id'].to(device, dtype=torch.int)
        if cls_task:
            labels = batch['label'].long().to(device)
            print("example1-ab_h_input_ids:", ab_h_input_ids[0])
            print("example1-stru_ids:", stru_embed_id)

        if mlm_task:
            ag_mask_loc = batch['ag_mask_loc'].to(device)
            ab_h_mask_loc = batch['ab_h_mask_loc'].to(device)
            ab_l_mask_loc = batch['ab_l_mask_loc'].to(device)
            print("example1-ab_h_mask_loc:", ab_h_mask_loc[0])


        # 使用 autocast 上下文进行前向传播和损失计算
        with torch.cuda.amp.autocast(enabled=amp_val):
            output_dict = model(
                ag=ag_input_ids,
                attention_mask_ag=ag_attention_mask,
                ab_h=ab_h_input_ids,
                attention_mask_ab_h=ab_h_attention_mask,
                ab_l=ab_l_input_ids,
                attention_mask_ab_l=ab_l_attention_mask,
                stru_ids=stru_embed_id
            )
            loss = 0

            if cls_task:
                cls_prob=output_dict["cls_prob"]
                print("cls_prob:",cls_prob)
                # cls:
                cls_loss = cls_criterion(cls_prob, labels)  # 假设标签是二分类问题
                loss += cls_loss
                total_cls_loss += cls_loss.item()

            if mlm_task:
                ab_h_mlm=output_dict["ab_h_mlm"]
                ab_l_mlm=output_dict["ab_l_mlm"]
                ag_mlm=output_dict["ag_mlm"]
                print("ab_h_mlm:",ab_h_mlm[0][0]) #代表第一个样本的第一个位置
                print("[ab_h_mlm]:", ab_h_mlm.shape)
                #mlm:(先分别考虑ab_h,ab_l,ag)
                print("[ab_h]:")
                ab_h_mlm_loss,_,_=compute_masked_mlm_loss(ab_h_mlm,ab_h_mask_loc,show_sample=True,return_acc=False)
                print("[ab_l]:")
                ab_l_mlm_loss,_,_=compute_masked_mlm_loss(ab_l_mlm,ab_l_mask_loc,show_sample=True,return_acc=False)
                print("[ag]:")
                ag_mlm_loss,_,_=compute_masked_mlm_loss(ag_mlm,ag_mask_loc,show_sample=True,return_acc=False)
                mlm_loss = ab_h_mlm_loss + ab_l_mlm_loss + ag_mlm_loss
                loss += mlm_loss
                total_mlm_loss += mlm_loss.item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        #break

    end_time = time.time()  # 记录当前时间
    epoch_duration = end_time - start_time  # 计算 epoch 持续时间
    print("=====train=====")
    print(f'Epoch [{epoch+1}/{num_epochs}], CLS_Loss: {total_cls_loss / len(train_dataloader):.4f}, MLM_Loss: {total_mlm_loss / len(train_dataloader):.4f}, Duration: {epoch_duration:.2f}s')
    print("==========")
    train_cls_loss = total_cls_loss / len(train_dataloader)
    train_mlm_loss = total_mlm_loss / len(train_dataloader)
    # 验证模型
    model.eval()
    total_cls_loss = 0
    cls_correct = 0
    total_mlm_loss = 0
    mlm_correct = 0
    mlm_total = 0

    with torch.no_grad():
        for batch in val_dataloader:
            ag_input_ids = batch['ag_input_ids'].to(device)
            ag_attention_mask = batch['ag_attention_mask'].to(device, dtype=torch.float16)
            ab_h_input_ids = batch['ab_h_input_ids'].to(device)
            ab_h_attention_mask = batch['ab_h_attention_mask'].to(device, dtype=torch.float16)
            ab_l_input_ids = batch['ab_l_input_ids'].to(device)
            ab_l_attention_mask = batch['ab_l_attention_mask'].to(device, dtype=torch.float16)
            stru_embed_id = batch['stru_embed_id'].to(device, dtype=torch.int)
            if cls_task:
                labels = batch['label'].long().to(device)
                print("example1-ab_h_input_ids:", ab_h_input_ids[0])
            if mlm_task:
                ag_mask_loc = batch['ag_mask_loc'].to(device)
                ab_h_mask_loc = batch['ab_h_mask_loc'].to(device)
                ab_l_mask_loc = batch['ab_l_mask_loc'].to(device)
                print("example1-ab_h_mask_loc:", ab_h_mask_loc[0])

            with torch.cuda.amp.autocast(enabled=amp_val):
                output_dict = model(
                    ag=ag_input_ids,
                    attention_mask_ag=ag_attention_mask,
                    ab_h=ab_h_input_ids,
                    attention_mask_ab_h=ab_h_attention_mask,
                    ab_l=ab_l_input_ids,
                    attention_mask_ab_l=ab_l_attention_mask,
                    stru_ids=stru_embed_id
                )
                loss = 0

                if cls_task:
                    cls_prob = output_dict["cls_prob"]
                    print("cls_prob:", cls_prob)
                    # cls:
                    cls_loss = cls_criterion(cls_prob, labels)  # 假设标签是二分类问题
                    total_cls_loss += cls_loss.item()
                    loss += cls_loss
                    _, predicted = torch.max(cls_prob, 1)
                    cls_correct += (predicted == labels).sum().item()

                if mlm_task:
                    ab_h_mlm = output_dict["ab_h_mlm"]
                    ab_l_mlm = output_dict["ab_l_mlm"]
                    ag_mlm = output_dict["ag_mlm"]
                    print("ab_h_mlm:", ab_h_mlm[0][0])  # 代表第一个样本的第一个位置
                    print("[ab_h_mlm]:", ab_h_mlm.shape)
                    # mlm:(先分别考虑ab_h,ab_l,ag)
                    print("[ab_h]:")
                    ab_h_mlm_loss,ab_h_corr_num,ab_h_total_num = compute_masked_mlm_loss(ab_h_mlm, ab_h_mask_loc, show_sample=True, return_acc=True)
                    print("[ab_l]:")
                    ab_l_mlm_loss,ab_l_corr_num,ab_l_total_num = compute_masked_mlm_loss(ab_l_mlm, ab_l_mask_loc, show_sample=True, return_acc=True)
                    print("[ag]:")
                    ag_mlm_loss,ag_corr_num,ag_total_num = compute_masked_mlm_loss(ag_mlm, ag_mask_loc, show_sample=True, return_acc=True)
                    mlm_loss = ab_h_mlm_loss + ab_l_mlm_loss + ag_mlm_loss
                    total_mlm_loss += mlm_loss.item()
                    loss += mlm_loss
                    mlm_correct += ab_h_corr_num + ab_l_corr_num + ag_corr_num
                    mlm_total += ab_h_total_num + ab_l_total_num + ag_total_num
            #break

    cls_accuracy = cls_correct / len(val_dataset)
    mlm_accuracy = mlm_correct / max(mlm_total,1)
    print("=====valid=====")
    print(f'Epoch [{epoch + 1}/{num_epochs}], \
            \nCLS_state: {cls_task}, \
            CLS_Loss: {total_cls_loss / len(train_dataloader):.4f},\
            CLS_accuracy: {cls_accuracy:.4f},\
            \nMLM_state: {mlm_task}, \
            MLM_Loss: {total_mlm_loss / len(train_dataloader):.4f},\
            MLM_accuracy: {cls_accuracy:.4f},\
            \nDuration: {epoch_duration:.2f}s')
    print("==========")
    val_cls_loss = total_cls_loss / len(val_dataloader)
    val_mlm_loss = total_mlm_loss / len(val_dataloader)
    # 记录epoch信息
    epoch_info.append({
        'Epoch': epoch + 1,
        "CLS_state":cls_task,
        'CLS_Train Loss': train_cls_loss,
        'CLS_Validation Loss': val_cls_loss,
        'CLS_Validation_Accuracy': cls_accuracy,
        "MLM_state":mlm_task,
        'MLM_Train Loss': train_mlm_loss,
        'MLM_Validation Loss': val_mlm_loss,
        'MLM_Validation_Accuracy': mlm_accuracy,
        'Duration': epoch_duration,
        "encoder_type": encoder_type,
    })
    # 将epoch信息保存到DataFrame
    df_epoch_info = pd.DataFrame(epoch_info)
    # 保存DataFrame到CSV文件
    df_epoch_info.to_csv(f'{temp_path}/epoch_info.csv', index=False)
    #进行一次预测
    #continue #先不跑后半部分

    if cls_task and last_time_val_loss>val_cls_loss: #如果效果更好就预测一次
        last_time_val_loss=val_cls_loss
        val_predictions, val_labels = cls_predict(val_dataloader, model, device)
        test_predictions, test_labels = cls_predict(test_dataloader, model, device)

        # 将预测结果并回到原始数据集
        # original_dataset = pd.read_csv("input/250606-HPV_dataset.csv")
        # original_dataset = original_dataset.dropna(subset=["bind_value"])  # 删除缺失值
        original_dataset=dataset

        # 假设five_fold_seed列用于区分验证集和测试集
        val_indices = original_dataset[original_dataset['five-fold_seed=408'].isin(val_id)].index.tolist()
        test_indices = original_dataset[original_dataset['five-fold_seed=408'].isin(test_id)].index.tolist()

        # 将预测结果添加到原始数据集
        print(len(val_indices))
        print(len(val_predictions))
        original_dataset.loc[val_indices, 'Validation Predictions'] = val_predictions
        original_dataset.loc[test_indices, 'Test Predictions'] = test_predictions

        # 保存结果到新的CSV文件
        original_dataset.to_csv(f'{temp_path}/epoch={epoch}-predictions_results.csv', index=False)
        # 保存模型
        torch.save(model.state_dict(), f'{temp_path}/model-epoch={epoch}.pth')
    #break


