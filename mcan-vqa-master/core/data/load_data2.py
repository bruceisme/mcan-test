# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.data.data_utils import img_feat_path_load, img_feat_load, ques_load, tokenize, ans_stat
from core.data.data_utils import proc_img_feat, proc_ques, proc_ans
import sys,re
import transformers
import numpy as np
import glob, json, torch, time
import torch.utils.data as Data


class DataSet(Data.Dataset):
    def __init__(self, __C, split_key):
        self.__C = __C
        self.problems=json.load(open(__C.PROBLEM_PATH, 'r'))
        self.captions = json.load(open(__C.CAPTION_PATH, 'r'))["captions"]
        self.image_list = json.load(open(__C.LIST_PATH, 'r'))
        self.pid_splits = json.load(open(__C.PID_PATH, 'r'))
        self.ids=[]
        for qid in self.problems:
            label=np.zeros(__C.len_choice)
            label[self.problems[qid]['answer']]=1
            self.problems[qid]['label']=label
            self.problems[qid]['caption'] = self.captions[qid] if qid in self.captions else ""
        
        self.ids=self.pid_splits['%s' % (split_key)]
        self.data_size=self.ids.__len__()
        print('== Dataset size:', self.data_size)
        self.image_type=(196,4096)
        self.ans_size = __C.len_choice
        self.use_caption=__C.use_caption
        
        # Tokenize
        print('Tokenizing questions and answers...')
        self.tokenizer = eval(
            f'transformers.{__C.BERT_MODEL}Tokenizer.from_pretrained(\"{__C.BERT_VERSION}\")'
        )
        print('Common data process is done.\n')


        # --------------------------
        # ---- Raw data loading ----
        # --------------------------

        # Loading all image paths
        # if self.__C.PRELOAD:
        self.image_feat_path_dict={}
        for qid in self.ids:
            if qid in self.image_list:
                image_feat_path=__C.FEAT_SAVE_PATH + f'/{qid}.npz'
                self.image_feat_path_dict[qid]=image_feat_path




    def __getitem__(self, idx):
        # For code safety
        context={}
        img_feat       = np.zeros(1)
        con_ix   = np.zeros(1)
        #ques_mask      = np.zeros(1)
        label         = np.zeros(self.ans_size)

        # Load the run data(context(w/o caption) + image + label) from list
        sub_idx=self.ids[idx]
        context['question'] = self.problems[sub_idx]["question"]
        #context['choice'] = self.problems[sub_idx]['choices']
        if self.problems[sub_idx]["hint"] != "" :
            context['hint']= self.problems[sub_idx]["hint"]
                
        if (self.use_caption==True) & (self.problems[sub_idx]['caption'] != ""):
            context['caption']=self.problems[sub_idx]['caption']
            
        for num, x in enumerate(self.problems[sub_idx]['choices']):
            context[f'choice{num}']=x
        #print(context)
        
        con_ix = self.bert_tokenize(context, self.__C.MAX_TOKEN)
        
        if sub_idx in self.image_list:
            img_feat=np.load(self.image_feat_path_dict[sub_idx])
        else:
            img_feat=np.zeros(self.image_type)
            
      
        #if self.split_key == 'train':
        #    # Process answer
        #    label= self.problems[sub_idx]["label"]
    
        label= self.problems[sub_idx]["label"]
 
        return  torch.tensor(img_feat, dtype=torch.float), \
                torch.tensor(con_ix, dtype=torch.long), \
                torch.tensor(label, dtype=torch.float)


    def __len__(self):
        return self.data_size

    def bert_tokenize(self, context, max_token):
        token_tem= ['[CLS]'] 
        for key, value in context.items():
            re.sub(r"([.,'!?\"()*#:;])",'', value.lower()).replace('-', ' ').replace('/', ' ').replace('\n',' ').split()
            token_tem += self.tokenizer.tokenize(value) + ['[SEP]']
            
        if len(token_tem) > max_token - 1:
            token_tem = token_tem[:max_token-1]
        token_tem = token_tem + ['[SEP]']
        # print(tokens)
        t_ids = self.tokenizer.convert_tokens_to_ids(token_tem)
        t_ids = t_ids + [0] * (max_token - len(t_ids))
        con_ix = np.array(t_ids, np.int64)

        return con_ix