
from __future__ import print_function
from datasets import TextDataset
from datasets import prepare_data
from miscc.config import cfg, cfg_from_file
from model import RNN_ENCODER, CNN_ENCODER
from miscc.losses import sent_loss, words_loss
from miscc.utils import build_super_images, mkdir_p
import torch
import time
import os
import argparse
from PIL import Image
from torch.cuda.amp import autocast

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

'''
# CNN + RNN 모델 학습
def train(dataloader, cnn_model, rnn_model, batch_size,
          labels, optimizer, epoch, ixtoword, image_dir):
    # dataloader : 데이터셋을 불러오는 DataLoader 객체
    # labels : 배치 크기만큼의 정수 라벨 텐서 (0 ~ batch_size -1)
    # ixtoword : 인덱스를 단어로 변환하는 딕셔너리
    # image__dir : 학습 중 생성된 attentio map 이미지를 저장할 디렉토리

    cnn_model.train() # 모델이 학습 모드로 설정됨 → Dropout과 같이 학습 중에만 활성화되는 연산이 실행됨됨
    rnn_model.train()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    count = (epoch + 1) * len(dataloader) # 학습된 데이터 샘플 개수를 추적
    start_time = time.time() # 학습 속도를 측정하기 위한 시작 시간 저장
    for step, data in enumerate(dataloader, 0):
        # dataloader에서 데이터를 하나씩 꺼내 학습을 진행

        # print('step', step)
        # 이전 step의 그래디언트 값을 초기화화
        rnn_model.zero_grad()
        cnn_model.zero_grad()

        imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
        # prepare_data를 이용하여 데이터 정렬

        #### CNN(이미지 인코더) 적용용
        words_features, sent_code = cnn_model(imgs[-1]) # imgs[-1] : 가장 높은 해상도의 이미지
        # words_features: batch_size x nef x 17 x 17 (단어 수준 이미지 특징 맵)
        # sent_code: batch_size x nef (문장 수준 이미지 특징 벡터)
        # --> batch_size x nef x 17*17 
    
        nef, att_sze = words_features.size(1), words_features.size(2)
        # att_sze : attention map 크기 (17)
        # words_features = words_features.view(batch_size, nef, -1)
        # CNN 특징 맵을 1D 벡터로 변환하는 코드, CNN 모델에서 차원 변환을 수행하므로 주석 처리한 것으로 보임임

        #### RNN(텍스트 인코더) 적용
        hidden = rnn_model.init_hidden(batch_size)
        # 초기 hidden state 생성
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)
        # words_emb: batch_size x nef x seq_len (단어 수준 텍스트 임베딩)
        # sent_emb: batch_size x nef (문장 수준 텍스트 임베딩)

        # 단어 수준의 손실 계산
        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                 cap_lens, class_ids, batch_size)
        # attn_mpas : 학습 중 주의 맵 생성
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        loss = w_loss0 + w_loss1
        
        # 문장 수준의 손실 계산
        s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data
        #
        loss.backward() # 역전파 수행
        #
        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(rnn_model.parameters(),
                                      cfg.TRAIN.RNN_GRAD_CLIP)
        # RNN의 기울기 폭발 방지
        optimizer.step()
        # 가중치 업데이트

        if step % UPDATE_INTERVAL == 0:
            # 일정한 간격마다 손실값 및 학습 속도 출력
            count = epoch * len(dataloader) + step

            s_cur_loss0 = s_total_loss0.item() / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1.item() / UPDATE_INTERVAL

            w_cur_loss0 = w_total_loss0.item() / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1.item() / UPDATE_INTERVAL

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
            """
            build_super_images : 실제 이미지, 어텐션 맵, 캡션을 기반으로 시각적 주의를 시각화한 이미지를 생성하는 역할
            텍스트 설명과 함께 이미지와 attention map을 한 장의 이미지로 구성.
            img_set : attention map이 적용된 이미지가 저장됨
            _ : 두 번째 반환값이 있지만 사용되지 않음을 의미
            imgs[-1].cpu() : 마지막 단계의 이미지 텐서를 CPU로 이동시킴

            """

            # 이미지가 유효한 경우 실행
            if img_set is not None:
                im = Image.fromarray(img_set)
                # Numpy 배열을 PIL 이미지 객체로 변환
                image_dir_per_epoch = os.path.join(image_dir, str(epoch))
                mkdir_p(image_dir_per_epoch)
                fullpath = '%s/attention_maps_step%d.png' % (image_dir_per_epoch, step)
                # 저장 경로 설정, step : 현재 학습 단계(step) 번호
                im.save(fullpath)
    return count


# 모델 검증 용도 (train 함수와 달리 optimizer.step()을 호출하지 않아 모델이 업데이트 되지 않음)
def evaluate(dataloader, cnn_model, rnn_model, batch_size, labels):
    cnn_model.eval() # 모델이 평가 모드로 설정됨
    rnn_model.eval()
    s_total_loss = 0
    w_total_loss = 0
    for step, data in enumerate(dataloader, 0):
        real_imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

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

    s_cur_loss = s_total_loss.item() / step
    w_cur_loss = w_total_loss.item() / step

    return s_cur_loss, w_cur_loss
'''

def train(dataloader, cnn_model, rnn_model, batch_size,
          labels, optimizer, epoch, ixtoword, image_dir, scaler):
    # dataloader : 데이터셋을 불러오는 DataLoader 객체
    # labels : 배치 크기만큼의 정수 라벨 텐서 (0 ~ batch_size -1)
    # ixtoword : 인덱스를 단어로 변환하는 딕셔너리
    # image__dir : 학습 중 생성된 attentio map 이미지를 저장할 디렉토리

    cnn_model.train() # 모델이 학습 모드로 설정됨 → Dropout과 같이 학습 중에만 활성화되는 연산이 실행됨됨
    rnn_model.train()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    count = (epoch + 1) * len(dataloader) # 학습된 데이터 샘플 개수를 추적
    start_time = time.time() # 학습 속도를 측정하기 위한 시작 시간 저장
    for step, data in enumerate(dataloader, 0):
        # dataloader에서 데이터를 하나씩 꺼내 학습을 진행

        # print('step', step)
        # 이전 step의 그래디언트 값을 초기화화
        rnn_model.zero_grad()
        cnn_model.zero_grad()

        imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
        # prepare_data를 이용하여 데이터 정렬

        with autocast():
            words_features, sent_code = cnn_model(imgs[-1]) # CNN 인코더 적용
            nef, att_sze = words_features.size(1), words_features.size(2)
            hidden = rnn_model.init_hidden(batch_size) # RNN 인코더 적용
            words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)

            w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels,
                                                 cap_lens, class_ids, batch_size)
            w_total_loss0 += w_loss0.item()
            w_total_loss1 += w_loss1.item()
            loss = w_loss0 + w_loss1

            s_loss0, s_loss1 = \
            sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
            loss += s_loss0 + s_loss1
            s_total_loss0 += s_loss0.item()
            s_total_loss1 += s_loss1.item()

        scaler.scale(loss).backward()  # loss에 대해 그래디언트를 계산하고 스케일링
        torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), cfg.TRAIN.RNN_GRAD_CLIP)
        
            # optimizer.step() 전에 scaler.step()을 호출하여 가중치를 업데이트
        scaler.step(optimizer)  # 가중치 업데이트
        scaler.update()  # 스케일러 업데이트

        if step % UPDATE_INTERVAL == 0:
                # 일정한 간격마다 손실값 및 학습 속도 출력
            count = epoch * len(dataloader) + step

            s_cur_loss0 = s_total_loss0.item() / UPDATE_INTERVAL
            s_cur_loss1 = s_total_loss1.item() / UPDATE_INTERVAL

            w_cur_loss0 = w_total_loss0.item() / UPDATE_INTERVAL
            w_cur_loss1 = w_total_loss1.item() / UPDATE_INTERVAL

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
                    build_super_images(imgs[-1].cpu(), captions, ixtoword, attn_maps, att_sze)
            """
            build_super_images : 실제 이미지, 어텐션 맵, 캡션을 기반으로 시각적 주의를 시각화한 이미지를 생성하는 역할
            텍스트 설명과 함께 이미지와 attention map을 한 장의 이미지로 구성.
            img_set : attention map이 적용된 이미지가 저장됨
            _ : 두 번째 반환값이 있지만 사용되지 않음을 의미
            imgs[-1].cpu() : 마지막 단계의 이미지 텐서를 CPU로 이동시킴

            """

            # 이미지가 유효한 경우 실행
            if img_set is not None:
                im = Image.fromarray(img_set)
                # Numpy 배열을 PIL 이미지 객체로 변환
                image_dir_per_epoch = os.path.join(image_dir, str(epoch))
                mkdir_p(image_dir_per_epoch)
                fullpath = '%s/attention_maps_step%d.png' % (image_dir_per_epoch, step)
                # 저장 경로 설정, step : 현재 학습 단계(step) 번호
                im.save(fullpath)
    return count


def evaluate(dataloader, cnn_model, rnn_model, batch_size, labels):
    cnn_model.eval() # 모델이 평가 모드로 설정됨
    rnn_model.eval()
    s_total_loss = 0
    w_total_loss = 0

    with autocast():
        for step, data in enumerate(dataloader, 0):
            real_imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

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

    s_cur_loss = s_total_loss.item() / step
    w_cur_loss = w_total_loss.item() / step

    return s_cur_loss, w_cur_loss


# 텍스트 및 이미지 인코더 초기화 및 불러오기기
def build_models(dataset, batch_size):
    # build model ############################################################
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    # RNN_ENCODER : 텍스트 데이터를 인코딩하는 RNN 모델
    # n_words : 단어 크기 (텍스트에 포함된 전체 단어
    # cfg.TEXT.EMBEDDING_DIM = 256, RNN의 은닉 상태 크기

    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    # CNN_ENCODER : 이미지를 인코딩하는 CNN 모델
    # 텍스 임베딩 차원을 입력으로 받음 → 나중에 텍스트와 이미지 특징을 비교하기 위해해

    # labels = Variable(torch.LongTensor(range(batch_size)))
    labels = torch.arange(batch_size, dtype=torch.long)
    # 배치 크기만큼의 정수 라벨 텐서를 생성 : [0, 1, 2, ..., batch_size -1]
    # 모델 학습 시, 배치 내 각 샘플의 인덱스를 나타내는 역할

    start_epoch = 0
    if cfg.TRAIN.NET_E != '': 
        """ 만약 학습을 이어서 할 경우 저장된 모델을 불러서 온다. 
        cfg.TRAIN.NET_E가 ''이면 이어서 학습할 모델이 없는 것이므로 해당 단계는 넘어감"""
        # cfg.TRAIN.NET_E : 저장된 RNN 모델의 파일 경로
        state_dict = torch.load(cfg.TRAIN.NET_E)
        # 저장된 RNN 모델의 가중치를 불러옴
        text_encoder.load_state_dict(state_dict)
        # 불러온 가중치를 text_encoder에 적용
        print('Load ', cfg.TRAIN.NET_E)

        # CNN도 같은 방식으로 저장된 모델의 가중치를 가져옴    
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        # 이전 학습의 Epoch 값 찾기 : 저장된 모델 파일명에서 이전 학습이 종료된 epoch 번호를 추출
        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)

    if cfg.CUDA:
        # cfg.CUDA == True라면, GPU에서 모델을 실행하도록 .cuda()를 호출
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda() # 배치별 정수 라벨 텐서
        
    return text_encoder, image_encoder, labels, start_epoch