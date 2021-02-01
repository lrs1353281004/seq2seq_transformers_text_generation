import os
CURRENT_PATH=os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(CURRENT_PATH)
from _utils import preprocess
class basic_tokenizer:
    def __init__(self,vocab_path):
        self.vocab=self.load_vocab(vocab_path)
        self.vocab_size=len(self.vocab)
        self.v_2_index={self.vocab[i]:i for i in range(self.vocab_size)}
        self.index_2_v={i:self.vocab[i] for i in range(self.vocab_size)}
    def load_vocab(self,vocab_path):
        with open(vocab_path,'r',encoding='utf-8') as f:
            res=f.readlines()
            res=[item.strip() for item in res]
        return res
    def tokenize(self,query_list):
        unk_inx=self.v_2_index['[UNK]']
        token_list=[self.v_2_index.get(c,unk_inx) for c in query_list]
        return token_list
    def detokenize(self,token_list):
        return [self.index_2_v[t] for t in token_list]
if __name__=="__main__":
    tokenizer=basic_tokenizer(os.path.join(CURRENT_PATH,'../data/basic_vocab.txt'))
    print(tokenizer.tokenize(['你','好']))
    print(tokenizer.detokenize([0,3,2,1]))

