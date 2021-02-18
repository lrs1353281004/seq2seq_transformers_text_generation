import torch
import torch.nn as nn
import torch.nn.functional as F
class seq_generation_loss(nn.Module):
    ''' sequence generation loss
        model_out: (seq,batch,vocab_size)
        tgt: (batch,seq)       type(tgt[i,j])==int  and  tgt[i,j]<vocab_size
    '''
    def __init__(self,device='cpu'):
        super(seq_generation_loss,self).__init__()
        self.device=device
    def forward(self,model_out,tgt):
        seq_size,batch_size,vocab_size=model_out.shape
        alpha=0.05

        tgt_vocab=torch.zeros((batch_size,seq_size,vocab_size))
        tgt_vocab=tgt_vocab.to(self.device)
        tgt_shift=torch.roll(tgt,-1,dims=1)
        tgt_shift[:,-1]=0
        tgt_shift=tgt_shift.unsqueeze(-1)
        tgt_vocab=tgt_vocab.scatter_(2,tgt_shift,1-alpha)
        tgt_vocab += alpha/vocab_size
        tgt_vocab=tgt_vocab.permute(1,0,2)

        model_out = F.log_softmax(model_out,dim=2)

        tgt_shift = tgt_shift.permute(1,0,2)
        pad_mask  = tgt_shift!=0
        tgt_mask  = pad_mask.expand(seq_size,batch_size,vocab_size)
        out=torch.masked_select(model_out, tgt_mask)
        tgt=torch.masked_select(tgt_vocab, tgt_mask)

        loss = out*tgt
        loss = -loss.sum() / pad_mask.sum() 

        return loss
if __name__=="__main__":
    import os
    CURRENT_PATH=os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.path.append(os.path.join(CURRENT_PATH,'../'))
    from utils.tokenizer import basic_tokenizer
    from utils.dialogue_dataset import dialogue_dataset,collate_func
    from model.transformer_base import transformer_base
    tokenizer=basic_tokenizer(os.path.join(CURRENT_PATH,"../data/basic_vocab.txt"))
    dataset = dialogue_dataset(os.path.join(CURRENT_PATH,"../data/LCCC_base/test"),tokenizer)
    batch_size=10
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True,collate_fn=collate_func)
    for batch_idx, batch in tqdm(enumerate(train_loader),total=int(len(train_loader.dataset) / batch_size) + 1):
        src_batch,tgt_batch,src_pad_batch,tgt_pad_batch,tgt_mask_batch=batch['src_ids'],batch['tgt_ids'],batch['src_pad_mask'],batch['tgt_pad_mask'],batch['tgt_mask']
        break
    model=transformer_base(tokenizer.vocab_size)
    out=model(src_batch,tgt_batch,src_pad_batch,tgt_pad_batch,tgt_mask_batch)
    criterion=seq_generation_loss()
    loss = criterion(out,tgt_batch)
    print(loss.item())

        

