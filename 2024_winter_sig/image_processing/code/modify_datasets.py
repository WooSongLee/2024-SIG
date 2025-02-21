from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg
import json
import re

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import glob


def prepare_data(data):
    # 입력 데이터를 정리하는 역할
    # 각 데이터 샘플을 정렬하고, PyTorch 텐서(Variable)로 변환하는 과정이 포함되어 있다.
    imgs, captions, captions_lens, class_ids, keys = data
    # sort data by the length in a decreasing order
    """
    data : 다섯 개 요소로 이루어진 튜플
    image : 캡션(문장) 데이터 (PyTorch 텐서)
    caption_lens : 각 캡션이 길이를 저장한 리스트 (PyTorch 텐서)
    class_ids : 각 샘플에 대한 클래스 ID 정보 (PyTorch 텐서)
    keys : 샘플을 식별하는 키 리스트 (파일 이름 등)
    """
    
    # 캡션 길이를 기준으로 내림차순 정렬 → 패딩 처리를 쉽게 하기 위해 (길이가 긴 문장부터 짧은 문장 순으로 정렬하면 연산 속도를 최적화 가능능)
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)
    """
    sorted_cap_lens : 정렬된 길이 값
    sorted_cap_indices : 원래 순서에서 정렬된 위치의 인덱스스
    """

    device = torch.device("cuda" if cfg.CUDA else "cpu")  # 장치 설정
    
    real_imgs = []
    # 이미지 데이터도 캡션 순서와 동일하게 정렬해야 하므로 sorted_cap_indices를 이용해 정렬렬
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        real_imgs.append(imgs[i].to(device))
        # .cuda()는 GPU에서만 동작하므로, .to(device)를 사용하면 CPU/GPU 환경에 맞게 자동 변환됨
        
        #if cfg.CUDA:
        #    real_imgs.append(Variable(imgs[i]).cuda())
        #else:
        #    real_imgs.append(Variable(imgs[i]))

    ## 캡션, 클래스 ID, 키 정렬
    captions = captions[sorted_cap_indices].squeeze() 
    # squeeze()로 차원을 줄임임
    class_ids = class_ids[sorted_cap_indices].numpy()
    # Numpy 배열로 변환
    # sent_indices = sent_indices[sorted_cap_indices]

    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # keys는 리스트이므로 Numpy 배열 변환한 후 인덱싱
    # print('keys', type(keys), keys[-1])  # list

    captions = captions.to(device)
    sorted_cap_lens = sorted_cap_lens.to(device)
    """
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)
    """
    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys]    
    # 최종적으로 정렬된 데이터를 반환환


class TextDataset(data.Dataset):
    # __init__은 클래스를 생성할 때 실행되는 생성자자
    def __init__(self, data_dir, voca_dir, split='train', base_size=64,
                 transform=None, target_transform=None):
        self.transform = transform
        self.norm = transform.Compose([
            transform.toTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embaeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
            # imsize 리스트에 여러 크기의 이미지 크기를 저장
        # self.imsize = 299로 출력됨

        self.data = []
        self.data_dir = data_dir
        # .../image_processing/data/coco
        self.voca_dir = voca_dir
        # .../image_processing/output/coco_DAMSM_.../voca_dictionary

        #if data_dir.find('birds') != -1:
        #    self.bbox = self.load_bbox()
            # bbox 정보를 로드
        #else:
            # coco 데이터셋은 bounding_boxes.txt 파일이 없으니 load_bbox() 수행X
            # json 파일 형식으로 정보를 가지고 있음
        #    self.bbox = None
        
        self.split = split # 이 코드가 맞는 지 아닌 지 꼭 확인하기!!!!!!!!!!!!!
        self.bbox = None # COCO 데이터셋 만을 사용할 것이기 때문에 birds 데이터 셋이 있는 경우 고려 X
        split_dir = os.path.join(data_dir, split)
        # .../image_processing/data/coco/train

        self.filenames, self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(data_dir, voca_dir, split)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)
        # 이미지 데이터의 갯수

    def load_text_data(self, data_dir, voca_dir, split):
        filepath = os.path.join(voca_dir, split, 'captions.pickle')

        train_names = self.load_filenames(data_dir, 'train')
        val_names = self.load_filenames(data_dir, 'val')
        test_names = self.load_filenames(data_dir, 'test') # 이 코드 필요한 지 검토 필요

        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names, 'train')
            val_captions = self.load_captions(data_dir, val_names, 'val')
            # test 데이터셋은 annotation 파일이 존재하지 않으므로 load_captions 해주지 않아도 OK

            train_captions, val_captions, ixtoword, wordtoix, n_words = self.build_dictionary(train_captions, val_captions)

            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, val_captions, ixtoword, wordtoix], f, protocol = 4)
                # protocol = 2는 python 2.x 버전에 최적화된 것으로 3.x 버전부터는 protocol=4를 사용한다고 한다.
                print('Save pickle file to : ', filepath)
            
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, val_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load pickle file from: ', filepath)
        
        if split == 'train':
            captions = train_captions
            filenames = train_names
        elif split == 'val':
            captions = val_captions
            filenames = val_names
        else:  # split=='test'
            captions = []
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_filename(data_dir, split='train'):
        path = os.path.join(data_dir, split)
        all_filepath = glob.glob(os.path.join(path, "*.jpg"))
        data_filenames = [os.path.basename(f) for f in all_filepath]
        return data_filenames


    def load_captions(self, data_dir, filenames, split='train'):
        # if split == 'test':
        #    return[[] for _ in filenames]

        annotations_file = os.path.join(data_dir, 'annotations', f'captions_{split}2017.json')

        with open(annotations_file, 'r') as f:
            data = json.load(f)

        image_id_to_captions = {}
        for annotation in data['annotations']:
            image_id = annotation['image_id']
            caption = annotation['caption']

            if image_id not in image_id_to_captions:
                image_id_to_captions[image_id] = []
            image_id_to_captions[image_id].append(caption)

        all_captions = []
        for filename in filenames:
            image_id = int(filename.split('.')[0])
            # 이미지 ID를 파일 이름에서 직접 추출 (.jpg 확장자를 제외한 부분)

            if image_id in image_id_to_captions:
                captions =  image_id_to_captions[image_id]
                # captions에 image_id에 해당하는 캡션 리스트를 저장하고 있음
            else : # image_id_to_captions에 없다면
                captions = []
            
            cnt = 0
            for cap in captions:
                # print(cap)
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                
                tokenizer = RegexpTokenize(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                if len(tokens) == 0:
                    continue
                
                tokens_new = []
                for t in tokens:
                    t = t.encode('ascii', 'ignore').decode('ascii')
                    if len(t) > 0:
                        tokens_new.append(t)
                
                all_captions.append(tokens_new)
                cnt += 1

                if cnt == self.embeddings_num:
                    break
            
            if cnt < self.embeddings_num:
                print('ERROR: the captions for %s less than %d' % (filename, cnt))
        return all_captions
        # 반환 값 all_captions = 캡션 한 문장에 대한 토큰 리스트들의 리스트 (모든 캡션에 대한 값 가지고 있음)

    def build_dictionary(self, train_captions, val_captions):
        word_counts = defaultdict(float)
        # 단어의 출현 횟수를 기록하기 위한 defaultdict 객체

        captions = train_captions + val_captions
        
        for sent in captions:
            for word in sent:
                word_counts[word] += 1
        
        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = [] # 인덱스-단어 사전
        ixtoword[0] = '<en>'
        wordtoix = []
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            # vocab에 있는 단어들에 순차적으로 인덱스를 부여함
            ixtoword[ix] = w
            wordtoix[w] = ix
            ix += 1
        
        train_captions_new = []
        for t in train_captions:
        # train 데이터셋 캡션들 중 하나
            rev = [] # 캡션을 idx로 변환한 값을 저장하기 위한 변수
            for w in t:
            # 해당 캡션에서 단어 선택
                if w in wordtoix:
                # 단어-인덱스 사전에 해당 단어가 있는 경우
                    rev.append(wordtoix[w])
            train_captions_new.append(rev)

        val_captions_new = []
        for t in val_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            val_captions_new.append(rev)
        
        return [train_captions_new, val_captions_new, ixtoword, wordtoix, len(ixtoword)]
        # len(ixtoword) : 단어의 갯수를 의미

    def load_class_id(self, voca_dir, total_num):
    # 클래스 생성하는 함수
        if os.paht.isfile(voca_dir + '/class_info.pickle'):
            with open(voca_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arrange(total_num)
            # np.arrange : 0부터 total_num -1까지 랜덤으로 정수 배열 생성
            # class_id를 각 파일 이름에 대해서 부여
            # 순서대로 아이디 부여
        return class_id

        
    def __len__(self):
        return len(self.filenames)
        
    def get_caption(self, sent_ix):
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            # 캡션(num_words)이 WORDS_NUM보다 짧은 경우
            x[:num_words, 0] = sent_caption
            # 그대로 복사하고, 나머지는 0으로 패딩
        else:
            # 캡션이 WORDS_NUM보다 긴 경우 : 캡션이 너무 길면 일부 단어만 랜덤 샘플링 후 정렬하여 선택택
            ix = list(np.arange(num_words)) # [0, 1, 2, .... num_words -1]
            np.random.shuffle(ix) # 단어 순서 랜덤 섞기
            ix = ix[:cfg.TEXT.WORDS_NUM] # WORDS_NUM 개수만큼 선택
            ix = np.sort(ix) # 다시 정렬
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len
    
    # __getitem__은 클래스의 인덱스에 접근할 때 자동으로 호출되는 메서드이다. (예시 : a[2])
    # index 값을 받아서 이미지, 캡션, 캡션 길이, 클래스 ID, 이미지 파일명을 반환
    def __getitem__(self, index):
        key = self.filenames[index]
        # filenames 리스트 중 index번째 값 → int형으로 변환시키지 않고 그대로 사용하는가?
        cls_id = self.class_id[index]

        bbox = None
        data_dir = self.data_dir
        split = self.split

        img_name = '%s/%s/%s.jpg' % (data_dir, split, key)
        imgs = get_imgs(img_name, self.imsize, bbox, self.transform, normalize=self.norm)

        sent_ix = random.randint(0, self.embeddings_num)
        # 해당 이미지에 대한 여러 개의 캡션 중 하나를 무작위 선택 (랜덤 선택)
        # self.embeddings_num : 하나의 이미지당 저장된 캡션 개수

        new_sent_ix = index * self.embedings_num + sent_ix  # 전체 캡션 리스트에서의 인덱스
        cap, cap_len = self.get_caption(new_sent_ix)
        return imgs, cap, cap_len, cls_id, key


def get_imgs(img_path, imsize, bbox=None, transform=None, normalize=None):
        img = Image.open(img_path).conver('RGB')
        width, height = img.size

        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            img = img.crop([x1, y1, x2, y2])

        if transform is not None:
            img = transform(img)

        ret = []
        if cfg.GAN.B_DCGAN:
            ret = [normalize(img)]
        else:
            # 기본 설정은 False : DCGAN이 아닌 경우, branch 수만큼 다른 크기의 이미지를 생성성
            for i in range(cfg.TREE.BRANCH_NUM):
                # cfg.TREE.BRANCH_NUM = 3
                if i < (cfg.TREE.BRANCH.NUM -1): # 마지막 단계가 아닌 경우
                    re_img = transforms.Scale(imsize[i])
                else:
                    re_img = img
                ret.append(normalize(re_img))
        return ret