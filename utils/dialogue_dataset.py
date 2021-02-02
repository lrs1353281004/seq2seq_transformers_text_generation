import torch
from torch.utils.data import Dataset
from tqdm import tqdm
class dialogue_dataset(Dataset):
    def __init__(self, data_root,tokenizer):
        """
        :param data_root:   数据集路径
        """
        self.data_root = data_root
        with open(data_root,'r',encoding='utf-8') as f:
            data_all=f.readlines()
        data_all=[item.strip().split('_split_') for item in data_all]
        self.data = data_all
        self.tokenizer=tokenizer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        q,a = self.data[index]
        sample = {'q_token':self.tokenizer.tokenize(list(q.strip())) ,'a_token':self.tokenizer.tokenize(['[START]']+list(a.strip())+['[END]'])}
        return sample
def collate_func(batch_dic):
    from torch.nn.utils.rnn import pad_sequence
    batch_len=len(batch_dic)
    src_ids_batch=[]
    tgt_ids_batch=[]
    src_pad_mask_batch=[]
    tgt_pad_mask_batch=[]
    tgt_mask_batch=[]
    for i in range(batch_len):
        dic=batch_dic[i]
        src_ids_batch.append(torch.tensor(dic['q_token']))
        tgt_ids_batch.append(torch.tensor(dic['a_token']))
        src_pad_mask_batch.append(torch.tensor([True]*len(dic['q_token'])))
        tgt_pad_mask_batch.append(torch.tensor([True]*len(dic['a_token'])))
    res={}
    res['src_ids']=pad_sequence(src_ids_batch,batch_first=True)
    res['tgt_ids']=pad_sequence(tgt_ids_batch,batch_first=True)
    res['src_pad_mask']=~pad_sequence(src_pad_mask_batch,batch_first=True)
    res['tgt_pad_mask']=~pad_sequence(tgt_pad_mask_batch,batch_first=True)
    tgt_length=res['tgt_pad_mask'].shape[1]
    tgt_mask_batch=[torch.tensor([True]*(i+1)) for i in range(tgt_length)]
    res['tgt_mask']=~pad_sequence(tgt_mask_batch,batch_first=True)
    return res
if __name__ == "__main__":
    import os
    CURRENT_PATH=os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.path.append(CURRENT_PATH)
    from tokenizer import basic_tokenizer
    tokenizer=basic_tokenizer(os.path.join(CURRENT_PATH,'../data/basic_vocab.txt'))
    dataset = dialogue_dataset(os.path.join(CURRENT_PATH,'../data/LCCC_base/test'),tokenizer)
    batch_size=10
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True,collate_fn=collate_func)
    for batch_idx, batch in tqdm(enumerate(train_loader),total=int(len(train_loader.dataset) / batch_size) + 1):
        src_batch,tgt_batch,src_pad_batch,tgt_pad_batch,tgt_mask_batch=batch['src_ids'],batch['tgt_ids'],batch['src_pad_mask'],batch['tgt_pad_mask'],batch['tgt_mask']
        break

