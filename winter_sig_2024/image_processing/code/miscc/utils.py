import os
import errno
import numpy as np
from torch.nn import init

import torch
import torch.nn as nn

from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import skimage.transform

from miscc.config import cfg


# For visualization ################################################
COLOR_DIC = {0:[128,64,128],  1:[244, 35,232],
             2:[70, 70, 70],  3:[102,102,156],
             4:[190,153,153], 5:[153,153,153],
             6:[250,170, 30], 7:[220, 220, 0],
             8:[107,142, 35], 9:[152,251,152],
             10:[70,130,180], 11:[220,20, 60],
             12:[255, 0, 0],  13:[0, 0, 142],
             14:[119,11, 32], 15:[0, 60,100],
             16:[0, 80, 100], 17:[0, 0, 230],
             18:[0,  0, 70],  19:[0, 0,  0]}
FONT_MAX = 50


def drawCaption(convas, captions, ixtoword, vis_size, off1=2, off2=2):
    num = captions.size(0)
    img_txt = Image.fromarray(convas)
    # get a font
    # fnt = None  # ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    fnt = ImageFont.truetype('Pillow/fonts/FreeMono.ttf', 50)
    # get a drawing context
    d = ImageDraw.Draw(img_txt)
    sentence_list = []
    for i in range(num):
        cap = captions[i].data.cpu().numpy()
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
            d.text(((j + off1) * (vis_size + off2), i * FONT_MAX), '%d:%s' % (j, word[:6]),
                   font=fnt, fill=(255, 255, 255, 255))
            sentence.append(word)
        sentence_list.append(sentence)
    return img_txt, sentence_list


def build_super_images(real_imgs, captions, ixtoword,
                       attn_maps, att_sze, lr_imgs=None,
                       batch_size=cfg.TRAIN.BATCH_SIZE,
                       max_word_num=cfg.TEXT.WORDS_NUM):
    # lr_imgs : 저해상도 이미지
    # att_size : 어텐션 맵 크기 (일반적으로 17)

    ## 1. 시각화할 이미지 개수 설정 (최대 8개의 이미지만 사용)
    nvis = 8
    real_imgs = real_imgs[:nvis]
    # real_imgs : 실제 이미지 데이터가 저장된 텐서 또는 리스트
    # 리스트에서 처음 nvis개 요소만 유지하고 나머지를 제거하는 슬라이싱 연산 (즉, nvis개의 이미지만 선택하여 이후의 연산에서 사용함)

    ## 2. 시각화할 크기 결정
    if lr_imgs is not None:
        lr_imgs = lr_imgs[:nvis]
        # 저해상도 이미지가 있는 경우, lr_imgs도 nvis개만 남긴다.
    if att_sze == 17:
        vis_size = att_sze * 16 # vis_size = 17 * 16 = 272 → 최종 시각화 크기 : 272*272
    else:
        vis_size = real_imgs.size(2) # real_imgs의 크기 중에 세 번째 차원의 값을 가져오는 연산

    ## 3. 배경을 위한 빈 캔버스 생성
    text_convas = \
        np.ones([batch_size * FONT_MAX,
                 (max_word_num + 2) * (vis_size + 2), 3],
                dtype=np.uint8)
    # text_convas : 텍스트 캡션을 위한 빈 이미지를 생성한다.
    
    ## 4. 단어별 색상 설정
    for i in range(max_word_num):
        istart = (i + 2) * (vis_size + 2)
        iend = (i + 3) * (vis_size + 2)
        text_convas[:, istart:iend, :] = COLOR_DIC[i]
        # 단어별 색상을 COLOR_DIC[1]에서 가져와 캔버스의 일부를 채색함함


    ## 5. 이미지 정규화 및 변환
    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255) # real_imgs의 범위를 [0, 255]로 변환
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))

    #print('[디버깅 : real_imgs의 크기]', real_imgs.shape)
    # real_imgs는 numpy 배열 형태

    pad_sze = real_imgs.shape # real_imgs의 차원 정보(크기)를 반환
    middle_pad = np.zeros([pad_sze[2], 2, 3]) # [width, 2, 3] : 가로 방향으로 2픽셀 두께의 패딩을 추가하고, 3은 채널 수를 의미
    post_pad = np.zeros([pad_sze[1], pad_sze[2], 3]) # [height, width, 3] : 원본 이미지와 동일한 hdight, width, channels를 가지는 빈 이미지를 생성하는 것으로 보임
    # np.zeros() 함수 : 0으로 채워진 배열을 생성 

    #print('[디버깅 : middle_pad의 크기]', middle_pad.shape)
    #print('[디버깅 : post_pad의 크기]', post_pad.shape)

    ## 6. 저해상도 이미지 처리 (위와 동일한 방식으로 변환환)
    if lr_imgs is not None:
        lr_imgs = \
            nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(lr_imgs)
        # [-1, 1] --> [0, 1]
        lr_imgs.add_(1).div_(2).mul_(255)
        lr_imgs = lr_imgs.data.numpy()
        # b x c x h x w --> b x h x w x c
        lr_imgs = np.transpose(lr_imgs, (0, 2, 3, 1))
    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    seq_len = max_word_num
    img_set = []
    num = nvis  # len(attn_maps)

    ## 7. 캡션을 캔버스에 그리기
    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size)
    # 이미지 위에 단어 캡션을 그리는 함수 
    # PIL.ImageDraw를 사용해 주석을 이미지에 표시
    text_map = np.asarray(text_map).astype(np.uint8)

    ## 8. Attention 맵 처리
    bUpdate = 1 # 업그레이드 여부를 나타내는 플래그 (실패하면 0으로 변경)
    for i in range(num):    # 이미지 개수만큼 반복하여 Attention 맵을 처리
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        #print('[디버깅 : attn_maps의 크기]', attn_maps[i].shape)
        #print('[디버깅 : attn의 크기]', attn.shape)

        # --> 1 x 1 x 17 x 17
        attn_max = attn.max(dim=1, keepdim=True) 
        # 가장 강한 attention 맵 찾기
        # dim=1 : 여러 개의 attention 채널이 있을 때, 채널 차원(1)에서 최대값을 찾음
        attn = torch.cat([attn_max[0], attn], 1) # 맥스 attention 추가
        #print('[디버깅 : 맥스 attention 추가 후 attn 크기]', attn.shape)

        attn = attn.view(-1, 1, att_sze, att_sze) # 다시 변환
        #print('[디버깅 : repeat 전 attn 크기]', attn.shape)
        attn = attn.repeat(1, 3, 1, 1).data.numpy() # 채널 3개로 확장
        #print('[디버깅 : repeat 후 attn 크기]', attn.shape)
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        #print('[디버깅 : transpose 이후 attn 크기]', attn.shape)

        
        ## 8-1. 원본 이미지 설정 및 패딩
        num_attn = attn.shape[0] # Attention 맵의 개수 저장
        img = real_imgs[i] # i번째 원본 이미지 저장

        if lr_imgs is None:
            lrI = img # 저해상도 이미지가 없으면 원본 이미지를 사용
        else:
            lrI = lr_imgs[i] # 저해상도 이미지가 있으면 i번째 저해상도 이미지를 사용

        row = [lrI, middle_pad] # 시각화를 위한 행(row) 리스트 생성
        row_merge = [img, middle_pad] # Attention 맵과 결합할 row 생성
        
        ## 8-2. Attention 맵 크기 조정 및 최소/최대값 찾기
        row_beforeNorm = [] # 정규화 전 Attention 맵을 저장해둘 리스트
        minVglobal, maxVglobal = 1, 0
        for j in range(num_attn):
            one_map = attn[j]

            # if one_map.ndim == 3:
            #    C, H, W = one_map
            #    expanded_maps = []
            #    for c in range(C) :
            ##        expanded_map = skimage.transform.pyramid_expand(one_map, sigma=10, upscale=vis_size // att_sze)
            #        expanded_maps.append(expanded_map)
            #    one_map = np.stack(expanded_maps, axis=0)
            #elif one_map.ndim == 2:
            #    one_map = skimage.transform.pyramid_expand(one_map, sigma=10, # sigma=20이 원래 코드이며, 나중에 20으로도 해보기기
            #                                         upscale=vis_size // att_sze)
                # Attention 맵을 vis_size 크기로 확대 / sigma=20 : 부드럽게 보이도록 Gaussian 필터 적용용
            if (vis_size // att_sze) > 1:
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=10, # sigma=20이 원래 코드이며, 나중에 20으로도 해보기기
                                                     upscale=vis_size // att_sze)
                # Attention 맵을 vis_size 크기로 확대 / sigma=20 : 부드럽게 보이도록 Gaussian 필터 적용용
            
            #print('[디버깅 : one_map 크기]', one_map.shape)
            row_beforeNorm.append(one_map)
            minV = one_map.min()
            maxV = one_map.max()
            # 개별 Attention 맵의 최대/최소값 계산
            if minVglobal > minV:
                minVglobal = minV
            if maxVglobal < maxV:
                maxVglobal = maxV
            # 값 갱신
        
        ## 8-3. Attention 맵을 정규화하여 최종 이미지 합성성
        for j in range(seq_len + 1):
            if j < num_attn:
                one_map = row_beforeNorm[j]
                one_map = (one_map - minVglobal) / (maxVglobal - minVglobal)
                one_map *= 255

                # 코드 수정 부분
                # 차원이 (height, width)일 경우, (height, width, 1) -> (height, width, 3)으로 확장
                # if one_map.ndim == 2:  # (height, width) 형태일 경우
                #     one_map = np.expand_dims(one_map, axis=-1)  # (height, width, 1)로 확장
                #     one_map = np.repeat(one_map, 3, axis=-1)  # (height, width, 3) 형태로 확장

                #print('[디버깅 : one_map의 차원]', one_map.shape)
                one_map = one_map[:, :, :3].astype(np.uint8)
                #print('[디버깅 : 3채널 변환 후 one_map의 차원]', one_map.shape)
                #print('[디버깅 : img의 차원]', img.shape)                

                PIL_im = Image.fromarray(np.uint8(img)) # 원본 이미지
                PIL_att = Image.fromarray(np.uint8(one_map)) # Attention 맵 변환
                merged = \
                    Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
                mask = Image.new('L', (vis_size, vis_size), (210))
                merged.paste(PIL_im, (0, 0)) # 원본 이미지 붙이기
                merged.paste(PIL_att, (0, 0), mask) # Attention 맵 오버레이이
                merged = np.array(merged)[:, :, :3]
            else:
                one_map = post_pad
                merged = post_pad
            row.append(one_map)
            row.append(middle_pad)
            #
            row_merge.append(merged)
            row_merge.append(middle_pad)
        
        ## 8-4. 최종 이미지 생성 및 반환
        row = np.concatenate(row, 1)
        row_merge = np.concatenate(row_merge, 1)
        txt = text_map[i * FONT_MAX: (i + 1) * FONT_MAX]
        if txt.shape[1] != row.shape[1]: # 크기 불일치 검출출
            print('txt', txt.shape, 'row', row.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row, row_merge], 0)
        img_set.append(row)
    if bUpdate: # 업그레이드가 되었으면 (1이면)
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None


def build_super_images2(real_imgs, captions, cap_lens, ixtoword,
                        attn_maps, att_sze, vis_size=256, topK=5):
    batch_size = real_imgs.size(0)
    max_word_num = np.max(cap_lens)
    text_convas = np.ones([batch_size * FONT_MAX,
                           max_word_num * (vis_size + 2), 3],
                           dtype=np.uint8)

    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    img_set = []
    num = len(attn_maps)

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size, off1=0)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        #
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = cap_lens[i]
        thresh = 2./float(num_attn)
        #
        img = real_imgs[i]
        row = []
        row_merge = []
        row_txt = []
        row_beforeNorm = []
        conf_score = []
        for j in range(num_attn):
            one_map = attn[j]
            mask0 = one_map > (2. * thresh)
            conf_score.append(np.sum(one_map * mask0))
            mask = one_map > thresh
            one_map = one_map * mask
            if (vis_size // att_sze) > 1:
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze)
            minV = one_map.min()
            maxV = one_map.max()
            one_map = (one_map - minV) / (maxV - minV)
            row_beforeNorm.append(one_map)
        sorted_indices = np.argsort(conf_score)[::-1]

        for j in range(num_attn):
            one_map = row_beforeNorm[j]
            one_map *= 255
            #
            PIL_im = Image.fromarray(np.uint8(img))
            PIL_att = Image.fromarray(np.uint8(one_map))
            merged = \
                Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
            mask = Image.new('L', (vis_size, vis_size), (180))  # (210)
            merged.paste(PIL_im, (0, 0))
            merged.paste(PIL_att, (0, 0), mask)
            merged = np.array(merged)[:, :, :3]

            row.append(np.concatenate([one_map, middle_pad], 1))
            #
            row_merge.append(np.concatenate([merged, middle_pad], 1))
            #
            txt = text_map[i * FONT_MAX:(i + 1) * FONT_MAX,
                           j * (vis_size + 2):(j + 1) * (vis_size + 2), :]
            row_txt.append(txt)
        # reorder
        row_new = []
        row_merge_new = []
        txt_new = []
        for j in range(num_attn):
            idx = sorted_indices[j]
            row_new.append(row[idx])
            row_merge_new.append(row_merge[idx])
            txt_new.append(row_txt[idx])
        row = np.concatenate(row_new[:topK], 1)
        row_merge = np.concatenate(row_merge_new[:topK], 1)
        txt = np.concatenate(txt_new[:topK], 1)
        if txt.shape[1] != row.shape[1]:
            print('Warnings: txt', txt.shape, 'row', row.shape,
                  'row_merge_new', row_merge_new.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None


####################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise