import re
import math
import torch
# import Bio.PDB
# import numpy as np
# import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from model.flash_attn_v1 import FlashTransformerEncoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
#print("import math!")
#PositionalEncoding-https://blog.csdn.net/qq_43645301/article/details/109279616
#这里要求输入为序列,样本,feature
#print("no flash!!!")
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ReptileModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def point_grad_to(self, target, device=torch.device('cuda')):
        '''
        Set .grad attribute of each parameter to be proportional
        to the difference between self and target
        '''
        for p, target_p in zip(self.parameters(), target.parameters()):
            if p.grad is None:
                if self.is_cuda():
                    p.grad = torch.autograd.Variable(torch.zeros(p.size())).to(device)
                else:
                    p.grad = torch.autograd.Variable(torch.zeros(p.size()))
            p.grad.data.zero_()  # not sure this is required
            p.grad.data.add_(p.data - target_p.data)

    def is_cuda(self):
        return next(self.parameters()).is_cuda



class Demo_seq_BCR(ReptileModel):
    def __init__(self,ab_ag_dim=700,node_attr_dim=1024, num_layers=8, dropout=0.,hidden_dim=2048,
                 heads=8,freeze_bert=False, ab_freeze_layer_count=-1,
                 cls_task=True,mlm_task=False, #分别对应是否进行两种task
                 bert=None):
        super(ReptileModel, self).__init__()
        #任务设置
        self.cls_task=cls_task
        self.mlm_task=mlm_task
        #bert
        self.ab_ag_dim = ab_ag_dim
        self.bert_name = bert
        if self.bert_name:
            self.ab_bert = BertModel.from_pretrained(self.bert_name)
            # freeze the ? layers
            self.freeze_bert = freeze_bert
            self.ab_freeze_layer_count = ab_freeze_layer_count
            _freeze_bert(self.ab_bert, freeze_bert=freeze_bert, freeze_layer_count=ab_freeze_layer_count)

        #transformer
        self.pos_encoder = PositionalEncoding(node_attr_dim, dropout) #代表编码位置信息
        self.node_attr_dim = node_attr_dim
        self.learnable_cls = nn.Parameter(torch.randn(node_attr_dim))  # for initial param of <cls>
        self.num_layers=num_layers
        self.heads=heads
        self.hidden_dim=hidden_dim
        flash_attn_layer = FlashTransformerEncoderLayer(d_model=self.node_attr_dim, nhead=self.heads,dim_feedforward=self.hidden_dim) #用来自己给自己做self_attn
        self.flash_encoder = TransformerEncoder(flash_attn_layer, num_layers=self.num_layers)
        print("flash!!!")
        #传统
        # encoder_layer = nn.TransformerEncoderLayer(d_model=node_attr_dim, nhead=heads, dim_feedforward=hidden_dim)
        # self.traditional_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # print("traditional!!!")
        #decoder
        if self.cls_task:
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.node_attr_dim, 2) #代表分成两类
            self.softmax = F.softmax
        if self.mlm_task:
            d_model=self.node_attr_dim #代表层数
            self.mlm_decoder = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 20),
            )


    def forward(self, ag=None,attention_mask_ag=None,
                      ab_h=None,attention_mask_ab_h=None,
                      ab_l=None, attention_mask_ab_l=None,):

        if self.bert_name:
        #if True:
            ab_h = self.ab_bert(input_ids=ab_h, attention_mask=attention_mask_ab_h).last_hidden_state
            ab_l = self.ab_bert(input_ids=ab_l, attention_mask=attention_mask_ab_l).last_hidden_state
            ag = self.ab_bert(input_ids=ag, attention_mask=attention_mask_ag).last_hidden_state

        #flash-transformer
        #learnable_cls = self.learnable_cls.unsqueeze(0).expand(ab_h.size(0), -1) #将learnable_cls扩展成(batchsize,1024)
        learnable_cls = self.learnable_cls.unsqueeze(0).unsqueeze(1).expand(ab_h.shape[0], 1, -1)
        seq_embed=torch.cat((learnable_cls,ab_h, ab_l, ag), dim=1)
        # 位置编码
        seq_embed=self.pos_encoder(seq_embed.transpose(0, 1)).transpose(0, 1) #编码后再转回来
        cls_attn = torch.ones(attention_mask_ab_h.shape[0],1, device=ab_h.device)
        seq_attn=torch.cat((cls_attn,attention_mask_ab_h,attention_mask_ab_l,attention_mask_ag), dim=1) #1 for valid
        # print("[seq_embed]shape:",seq_embed.shape)
        # print("[seq_attn]shape:",seq_attn.shape)
        # print("[seq_attn]:",seq_attn)

        #合并
        final_embed=self.flash_encoder(seq_embed,src_key_padding_mask=seq_attn)
        #cpu测试时使用:
        # final_embed=self.traditional_encoder(seq_embed,src_key_padding_mask=seq_attn.T)
        #self.traditional_encoder
        #输出结果
        ##CLS:
        output_dict={}
        if self.cls_task:
            # 池化：取 [CLS] token 输出 (batch_size, hidden_size)
            pooled = final_embed[:, 0, :]
            # Dropout + 全连接
            logits = self.classifier(self.dropout(pooled))
            probabilities = self.softmax(logits, dim=1)
            output_dict["cls_prob"]=probabilities

        ##MLM:
        if self.mlm_task:
            #计算长度
            len_list = [1] + [i.shape[1] for i in [ab_h, ab_l, ag]]
            for i in range(1, len(len_list)):
                len_list[i] += len_list[i - 1]
            mlm_embed=self.mlm_decoder(final_embed)
            output_dict["ab_h_mlm"] = mlm_embed[:,len_list[0]:len_list[1],:]
            output_dict["ab_l_mlm"] = mlm_embed[:,len_list[1]:len_list[2],:]
            output_dict["ag_mlm"] = mlm_embed[:,len_list[2]:len_list[3],:]

        return output_dict

def _freeze_bert(
        bert_model: BertModel, freeze_bert=True, freeze_layer_count=-1
):
    """Freeze parameters in BertModel (in place)
    Args:
        bert_model: HuggingFace bert model
        freeze_bert: Bool whether to freeze the bert model
        freeze_layer_count: If freeze_bert, up to what layer to freeze.
    Returns:
        bert_model
    """
    if freeze_bert:
        # freeze the entire bert model
        for param in bert_model.parameters():
            param.requires_grad = False
    else:
        # freeze the embeddings
        for param in bert_model.embeddings.parameters():
            param.requires_grad = False
        if freeze_layer_count != -1:
            if freeze_layer_count > 0 :
                # freeze layers in bert_model.encoder
                for layer in bert_model.encoder.layer[:freeze_layer_count]:
                    for param in layer.parameters():
                        param.requires_grad = False

            if freeze_layer_count < 0 :
                # freeze layers in bert_model.encoder
                for layer in bert_model.encoder.layer[freeze_layer_count:]:
                    for param in layer.parameters():
                        param.requires_grad = False
    return None


#model = Demo_seq_BCR(ab_ag_dim=bert_name)

def get_frozen_bert(key_word="prot_bert"):
    ab_bert=BertModel.from_pretrained(key_word)
    _freeze_bert(ab_bert, freeze_bert=True, freeze_layer_count=None)
    return ab_bert


def get_unfrozen_bert(key_word="prot_bert",n=28):
    ab_bert=BertModel.from_pretrained(key_word)
    _freeze_bert(ab_bert, freeze_bert=False, freeze_layer_count=n)
    return ab_bert

def mask_helper(sequence,max_seq_len,tokenizer):
    sequence=re.sub(r'[UZOB*_]', "X", sequence)
    ids_seq=tokenizer(" ".join(list(sequence)), return_tensors='pt', max_length=max_seq_len,
              padding='max_length',
              truncation=True)
    return ids_seq["input_ids"],ids_seq["attention_mask"]


if __name__ == "__main__":
    # 测试
    Heavy = "VQLQESGPGLVKPSQTLSLTCTVSGGSISDGDYYWSWFRQPPGSGLEWIGNSYYSGSTNHNPSLKSRATISIDTSKNQLSLRLTSVTVADTAVYYCARSGVFGNSYNRYFDPWGQGTLVTVSSA"
    Light = "AIQMTQSPPSLYASVGDRVTITCRASQGVRNDLGWYQQKPGKAPTLLIFGTYRLKSGVPSRFRGSGSGTDFTLTITRLQPEDFATYYCLQDHEFPFTFGGGTKVEIKR"
    Antigen = "KVVATDAYVTRTNIFYHASSSRLLAVGHPYFSIKRANKTVVPKVSGYQYRVFKVVLPDPNKFALPDSSLFDPTTQRLVWACTGLEVGRGQPLGVGVSGHPFLNKYDDVENSGSGGNPGQDNRVNVGMDYKQTQLCMVGCAPPLGEHWGKGKQCTNTPVQAGDCPPLELITSVIQDGDMVDTGFGAMNFADLQTNKSDVPIDICGTTCKYPDYLQMAADPYGDRLFFFLRKEQMFARHFFNRAGEVGEPVPDTLIIKGSGNRTSVGSSIYVNTPSGSLVSSEAQLFNKPYWLQKAQGHNNGICWGNQLFVTVVDTTRSTNMTLCASVTTSSTYTNSDYKEYMRHVEEYDLQFIFQLCSITLSAEVVAYIHTMNPSVLEDWNFPDPYKNLSFWEVNLKEKFSSELDQYPLGRKFLLQS"
    # 代表Antigen,Heavy,Light
    bert_name = "/fs1/home/caolab/prot_bert"  # 加载bert
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"cur seq model path:{bert_name};device-{device}")
    tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=False)
    temp = [mask_helper(seq, max_seq_len, tokenizer) for seq, max_seq_len in \
            zip([Heavy, Light, Antigen], [150, 150, 400])]
    ab_h_input_ids, ab_l_input_ids, ag_input_ids = [temp[i][0].to(device) for i in range(len(temp))]
    ab_h_attention_mask, ab_l_attention_mask, ag_attention_mask = [temp[i][1].to(device, dtype=float) for i in \
                                                                   range(len(temp))]
    print(ab_h_attention_mask.shape, ab_h_attention_mask) #1 for not masked
    # model = Demo_seq_BCR(bert=bert_name,freeze_bert=True)
    model = Demo_seq_BCR(bert=bert_name, freeze_bert=False, ab_freeze_layer_count=8).to(device, dtype=torch.float16)
    with torch.cuda.amp.autocast(enabled=True):
        outputs = model(ag=ag_input_ids, attention_mask_ag=ag_attention_mask,\
                        ab_h=ab_h_input_ids, attention_mask_ab_h=ab_h_attention_mask,\
                        ab_l=ab_l_input_ids, attention_mask_ab_l=ab_l_attention_mask)

    # 打印输出
    print(outputs)