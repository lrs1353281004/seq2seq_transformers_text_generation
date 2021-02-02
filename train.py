# -*- coding: utf-8 -*-
import os
import time
import math
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from utils.dialogue_dataset import dialogue_dataset,collate_func
from model.transformer_base import transformer_base
from utils.tokenizer import basic_tokenizer
from utils.loss import seq_generation_loss
from utils._utils import reset_log
def Train(args):
    train_path = args['trainset_path']
    dev_path   = args['testset_path']
    resume = args['resume']
    checkpoint_path = args['checkpoint_path']
    history_path    = args['history_path']
    log_path = args['log_path']
    vocab_path=args['vocab_path']
    model_name = args['model_save_name']
    model_resume_name = args['model_resume_name']
    batch_size = args['batch_size']
    end_epoch  = args['end_epoch']
    lr = args['lr']
    loss_check_freq = args['loss_check']
    check_steps=args['check_steps']
    save_steps =args['save_steps']
    os.environ['CUDA_VISIBLE_DEVICES'] = args['GPU_ids']
    embed_path= args['embed_path']
    embed_dim = args['embed_dim']
    nheads = args['nheads_transformer']
    #########
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(history_path):
        os.makedirs(history_path)
    log_save_name = 'log_' + model_name + '.log'
    reset_log(log_path + log_save_name)
    logger = logging.getLogger(__name__)
    for k,v in args.items():
        logger.info(k+':'+ str(v))
    checkpoint_name = os.path.join(checkpoint_path, model_name + '_best_ckpt.pth')
    model_ckpt_name = os.path.join(checkpoint_path, model_name + '_best.pkl')

    if not model_resume_name:
        model_resume_name = model_ckpt_name
    localtime = time.asctime(time.localtime(time.time()))
    logger.info('#####start time:%s' % (localtime))
    time_stamp = int(time.time())
    logger.info('time stamp:%d' % (time_stamp))
    logger.info('######Model: %s' % (model_name))
    logger.info('trainset path ：%s' % (train_path))
    logger.info('valset path: %s' % (dev_path))
    logger.info('batch_size:%d' % (batch_size))
    logger.info('learning rate:%f' % (lr))
    logger.info('end epoch:%d' % (end_epoch))

    tokenizer=basic_tokenizer(vocab_path)
    trainset = dialogue_dataset(train_path,tokenizer)
    devset   = dialogue_dataset(dev_path,tokenizer)

    print("训练集样本数：%d"%(trainset.__len__()))
    logger.info("训练集样本数：%d"%(trainset.__len__()))
    print("验证集样本数：%d" % (devset.__len__()))
    logger.info("验证集样本数：%d" % (devset.__len__()))

    train_loader = DataLoader(trainset, batch_size=batch_size,num_workers=4, shuffle=True,collate_fn=collate_func,drop_last=True)
    dev_loader   = DataLoader(devset, batch_size=batch_size,num_workers=4, shuffle=False,collate_fn=collate_func)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = transformer_base(tokenizer.vocab_size,embed_dim,nheads,embed_path)
    model.to(device)

    if resume != 0:
        logger.info('Resuming from checkpoint...')
        model.load_state_dict(torch.load(model_resume_name))
        checkpoint = torch.load(checkpoint_name)
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']
    else:
        best_loss =  math.inf
        start_epoch = -1
        history = {'train_loss': [], 'val_loss': []}

    criterion = seq_generation_loss(device=device).to(device)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr) #weight_decay=1e-5
    scheduler = StepLR(optim, step_size=5, gamma=0.9)

    steps_cnt=0
    for epoch in range(start_epoch+1,end_epoch):
        print('-------------epoch:%d--------------'%(epoch))
        logger.info('-------------epoch:%d--------------'%(epoch))
        model.train()
        loss_tr = 0
        local_steps_cnt=0
        #########   train ###########
        print ('start training!')
        for batch_idx, batch in tqdm(enumerate(train_loader),
                                     total=int(len(train_loader.dataset) / batch_size) + 1):
            src_batch,tgt_batch,src_pad_batch,tgt_pad_batch,tgt_mask_batch=batch['src_ids'], \
                                    batch['tgt_ids'],batch['src_pad_mask'],batch['tgt_pad_mask'],batch['tgt_mask']

            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            src_pad_batch = src_pad_batch.to(device)
            tgt_pad_batch = tgt_pad_batch.to(device)
            tgt_mask_batch = tgt_mask_batch.to(device)

            model.zero_grad()
            out=model(src_batch,tgt_batch,src_pad_batch,tgt_pad_batch,tgt_mask_batch)
            loss = criterion(out,tgt_batch)
            loss.backward()  # compute gradients
            optim.step()  # update parameters
            steps_cnt+=1
            local_steps_cnt+=1
            loss_tr += loss.item()

            if batch_idx % loss_check_freq == 0:
                print('batch:%d' % (batch_idx))
                print('loss:%f' % (loss.item()))
            
            if steps_cnt%check_steps == 0:
                loss_tr  /= local_steps_cnt
                print('trainset loss:%f' % (loss_tr))
                logger.info('trainset loss:%f' % (loss_tr))
                history['train_loss'].append(loss_tr)
                loss_tr = 0
                local_steps_cnt = 0
                #########  val ############
                loss_va = 0
                model.eval()
                with torch.no_grad():
                    print('start validating!')
                    for batch_idx, batch in tqdm(enumerate(dev_loader),
                                                total=int(len(dev_loader.dataset) / batch_size) + 1):
                        src_batch,tgt_batch,src_pad_batch,tgt_pad_batch,tgt_mask_batch=batch['src_ids'], \
                                                batch['tgt_ids'],batch['src_pad_mask'],batch['tgt_pad_mask'],batch['tgt_mask']

                        src_batch = src_batch.to(device)
                        tgt_batch = tgt_batch.to(device)
                        src_pad_batch = src_pad_batch.to(device)
                        tgt_pad_batch = tgt_pad_batch.to(device)
                        tgt_mask_batch = tgt_mask_batch.to(device)

                        model.zero_grad()
                        out=model(src_batch,tgt_batch,src_pad_batch,tgt_pad_batch,tgt_mask_batch)
                        loss = criterion(out,tgt_batch)
                        loss_va += loss.item()

                    loss_va = loss_va / (batch_idx + 1)
                    print('valset loss:%f' % (loss_va))
                    logger.info('valset loss:%f' % (loss_va))
                    history['val_loss'].append(loss_va)

                    # save checkpoint and model
                    if loss_va < best_loss:
                        logger.info('Checkpoint Saving...')
                        print('best loss so far! Checkpoint Saving...')
                        state = {
                            'epoch': epoch ,
                            'val_loss':loss_va,
                            'history': history
                        }
                        torch.save(state, checkpoint_name)
                        best_loss = loss_va
                        ## save model
                        torch.save(model.state_dict(), model_ckpt_name)
                scheduler.step()
                logger.info("current lr:%f" % (scheduler.get_last_lr()[0]))
            if steps_cnt%save_steps==0:
                logger.info('match save steps,Checkpoint Saving...')
                torch.save(model.state_dict(), os.path.join(checkpoint_path, model_name + '_steps_'+str(steps_cnt)+'.pkl'))
if __name__ == "__main__":
    args={
        'trainset_path':'./data/LCCC_base/train',
        'testset_path':'./data/LCCC_base/test',
        'checkpoint_path':'./ckpt/',
        'history_path':'./history/',
        'log_path':'./log/',
        'vocab_path':'./data/basic_vocab.txt',
        'embed_path':'./data/pretrained_embed.pkl',   # default: ''
        'embed_dim':300, # default: 512
        'nheads_transformer':15, # embed_dim % nheads_transformer == 0
        'resume':0,
        'model_save_name':'trans_lccc_v333',
        'model_resume_name':'',
        'batch_size':64,
        'end_epoch':10,
        'check_steps':20000,
        'save_steps':50000,
        'lr':1e-4,
        'loss_check':300,
        'version_info':'use pretrained embed',
        'GPU_ids':'0'
    }
    Train(args)