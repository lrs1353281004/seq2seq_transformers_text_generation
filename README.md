# 基于transformer的文本生成问题训练pipeline
## main requirements
python 3.6  
pytorch 1.6.0+cu101

## 项目说明
基于transformer的文本生成问题pipeline。（基于对话数据进行闲聊模型训练和测试）
训练方式为teacher forcing（基于下三角mask实现，具体可参考loss部分代码）。

## 模型训练
python train.py

## 推理
python inference.py

## 训练细节参考

训练数据（LCCC_base,来源：https://github.com/thu-coai/CDial-GPT）

初始学习率：1e-4

batch_size:96

nheads_transformer:15

embed_dim:300 （使用了预训练字向量，来源:https://github.com/Embedding/Chinese-Word-Vectors ，使用的字向量链接为: https://pan.baidu.com/s/1hJKTAz6PwS7wmz9wQgmYeg ）

encode_layers=6

## 训练效果预览

训练到16个epoch（大约200万+steps，耗时约10天）

![image](https://github.com/lrs1353281004/seq_2_seq_transformers_text_generation/blob/master/pics/training_example.png)






