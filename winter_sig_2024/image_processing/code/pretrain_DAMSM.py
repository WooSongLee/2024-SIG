from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

from model import RNN_ENCODER, CNN_ENCODER
 
import os
import sys
# 파일 경로 조작 및 시스템 경로 설정
import time
# 코드 실행 시간을 측정
import random
# 난수 생성
import pprint
# 설정 파일을 예쁘게 출력하는 용도
import datetime
import dateutil.tz
# 현재 시간과 타임스탬프를 관리
import argparse
import numpy as np
# 배열 연산을 위해 사용
from PIL import Image
# 이미지 저장 및 변환

import torch
import torch.nn as nn
# 기본적인 PyTorch 기능과 신경망 모듈
import torch.optim as optim
# 옵티마이저 관련 모듈
from torch.autograd import Variable
# 자동 미분 기능을 위한 래퍼 (PyTorch 최신 버전에서는 필요 X)
import torch.backends.cudnn as cudnn
# GPU 연산 최적화
import torchvision.transforms as transforms
# 이미지 전처리(transform) 관련 기능 제공

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

UPDATE_INTERVAL = 200
# 학습 중 손실을 출력하는 간격을 설정. 200번 반복마다 로그를 출력함.

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/DAMSM/coco.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

# 학습
def train(dataloader, cnn_model, rnn_model, batch_size,
          labels, optimizer, epoch, ixtoword, image_dir):
    cnn_model.train() # 모델이 학습 모드로 설정됨
    rnn_model.train()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    count = (epoch + 1) * len(dataloader)
    start_time = time.time()
    for step, data in enumerate(dataloader, 0):
        # print('step', step)
        rnn_model.zero_grad()
        cnn_model.zero_grad()

        imgs, captions, cap_lens, \
            class_ids, keys = prepare_data(data)


        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        words_features, sent_code = cnn_model(imgs[-1])
        # --> batch_size x nef x 17*17
        nef, att_sze = words_features.size(1), words_features.size(2)
        # words_features = words_features.view(batch_size, nef, -1)

        hidden = rnn_model.init_hidden(batch_size)
        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                 cap_lens, class_ids, batch_size)
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        loss = w_loss0 + w_loss1

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data
        #
        loss.backward()
        #
        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(rnn_model.parameters(),
                                      cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()

        if step % UPDATE_INTERVAL == 0:
            count = epoch * len(dataloader) + step

            s_cur_loss0 = s_total_loss0[0] / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1[0] / UPDATE_INTERVAL

            w_cur_loss0 = w_total_loss0[0] / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1[0] / UPDATE_INTERVAL

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  's_loss {:5.2f} {:5.2f} | '
                  'w_loss {:5.2f} {:5.2f}'
                  .format(epoch, step, len(dataloader),
                          elapsed * 1000. / UPDATE_INTERVAL,
                          s_cur_loss0, s_cur_loss1,
                          w_cur_loss0, w_cur_loss1))
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            start_time = time.time()
            # attention Maps
            img_set, _ = \
                build_super_images(imgs[-1].cpu(), captions,
                                   ixtoword, attn_maps, att_sze)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/attention_maps%d.png' % (image_dir, step)
                im.save(fullpath)
    return count

# 모델 검증 용도 (train 함수와 달리 optimizer.step()을 호출하지 않아 모델이 업데이트 되지 않음)
def evaluate(dataloader, cnn_model, rnn_model, batch_size):
    cnn_model.eval() # 모델이 평가 모드로 설정됨
    rnn_model.eval()
    s_total_loss = 0
    w_total_loss = 0
    for step, data in enumerate(dataloader, 0):
        real_imgs, captions, cap_lens, \
                class_ids, keys = prepare_data(data)

        words_features, sent_code = cnn_model(real_imgs[-1])
        # nef = words_features.size(1)
        # words_features = words_features.view(batch_size, nef, -1)

        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels,
                                            cap_lens, class_ids, batch_size)
        w_total_loss += (w_loss0 + w_loss1).data

        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        s_total_loss += (s_loss0 + s_loss1).data

        if step == 50:
            break

    s_cur_loss = s_total_loss[0] / step
    w_cur_loss = w_total_loss[0] / step

    return s_cur_loss, w_cur_loss

# 모델 생성하고 초기화 하는 역할
def build_models():
    # build model ############################################################
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)
    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()

    return text_encoder, image_encoder, labels, start_epoch


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, 'train',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)

    print(dataset.n_words, dataset.embeddings_num)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #
    dataset_val = TextDataset(cfg.DATA_DIR, 'test',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    # Train ##############################################################
    text_encoder, image_encoder, labels, start_epoch = build_models()
    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)
    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        lr = cfg.TRAIN.ENCODER_LR
        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            epoch_start_time = time.time()
            count = train(dataloader, image_encoder, text_encoder,
                          batch_size, labels, optimizer, epoch,
                          dataset.ixtoword, image_dir)
            print('-' * 89)
            if len(dataloader_val) > 0:
                s_loss, w_loss = evaluate(dataloader_val, image_encoder,
                                          text_encoder, batch_size)
                print('| end epoch {:3d} | valid loss '
                      '{:5.2f} {:5.2f} | lr {:.5f}|'
                      .format(epoch, s_loss, w_loss, lr))
            print('-' * 89)
            if lr > cfg.TRAIN.ENCODER_LR/10.:
                lr *= 0.98

            if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
                epoch == cfg.TRAIN.MAX_EPOCH):
                torch.save(image_encoder.state_dict(),
                           '%s/image_encoder%d.pth' % (model_dir, epoch))
                torch.save(text_encoder.state_dict(),
                           '%s/text_encoder%d.pth' % (model_dir, epoch))
                print('Save G/Ds models.')
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')