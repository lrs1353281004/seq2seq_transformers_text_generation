import torch
from utils.tokenizer import basic_tokenizer
from model.transformer_base import transformer_base
from utils._utils import preprocess
class dialogue_bot:
    def __init__(self,vocab_path,model_ckpt_path,model_params,chat_max_length):
        self.tokenizer=basic_tokenizer(vocab_path)
        self.model=transformer_base(self.tokenizer.vocab_size,model_params['embed_dim'],model_params['nheads'])
        self.model.load_state_dict(torch.load(model_ckpt_path))
        self.chat_max_len=chat_max_length
        self.unk='抱歉，我不明白你在说什么，请换一种说法。'
    def chat(self,query):
        query=preprocess(query)
        if not query:
            return '无效输入！'
        src_ids=self.tokenizer.tokenize(list(query))
        src = torch.tensor(src_ids).unsqueeze(0)
        tgt_ids=self.tokenizer.tokenize(['[START]'])
        res= []
        max_length=self.chat_max_len
        cnt=0
        self.model.eval()
        while cnt<max_length:
            tgt=torch.tensor(tgt_ids).unsqueeze(0)
            out=self.model(src,tgt)
            cur=out[-1,0,:].argmax().item()
            if self.tokenizer.detokenize([cur])[0]=='[END]':
                break
            tgt_ids.append(cur)
            res.append(cur)
            cnt+=1
        return ''.join(self.tokenizer.detokenize(res)) if res else self.unk
if __name__=="__main__":
    vocab_path='./data/basic_vocab.txt'
    ckpt_path='./ckpt/trans_lccc_v1_steps_200000.pkl'
    model_params={"embed_dim":300,"nheads":15}
    chat_bot=dialogue_bot(vocab_path,ckpt_path,model_params,50)
    while True:
        query = input('>>>>>>>>>>Input query:')
        if query=="exit" or query=="quit":
            print('再见！')
            break
        print(chat_bot.chat(query))

