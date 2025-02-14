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


def get_imgs(img_path, imsize, bbox=None,   
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
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
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Scale(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))

    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64,
                 transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
            # imsize 리스트에 여러 크기의 이미지 크기를 저장장

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
            # bbox 정보를 로드
        else:
            # coco 데이터셋은 bounding_boxes.txt 파일이 없으니 load_bbox() 수행X
            # json 파일 형식으로 정보를 가지고 있음음
            self.bbox = None
        split_dir = os.path.join(data_dir, split)
        # split_dir 경로 : data_dir/train

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        # caption.pickle 파일은 캡션 데이터와 단어 사전이 저장된 파일
        # 파일 존재하면 로드하고, 존재하지 않으면 캡션을 처리하여 생성성
        train_names = self.load_filenames(data_dir, 'train')
        val_names = self.load_filenames(data_dir, 'val')
        test_names = self.load_filenames(data_dir, 'test')
        # 학습과 테스트 데이터의 파일명 리스트를 가져옴
        # train_names와 test_names에 각각 이미지 파일 이름 리스트가 저장됨
        # 예 : ['000001.jpg', '000002.jpg', '000003.jpg', ...]

        # captions.pickle이 없을 경우 캡션을 새로 로드하고, 단어 사전을 생성한 뒤 피클 파일로 저장
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            val_captions = self.load_captions(data_dir, val_names)
            test_captions = self.load_captions(data_dir, test_names)
            # load_captions를 이용하여 train과 val, test 캡션을 로드함

            # 캡션에서 사용된 단어들을 기반으로 단어 사전을 생성함
            train_captions, val_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, val_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, val_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                # 캡션과 단어 사전을 픽클 파일로 저장하여, 이후에는 재사용할 수 있도록 한다
                print('Save to: ', filepath)

        else:
            # 이미 있으면 저장되어 있는 픽클 파일 불러와서 사용용
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1], x[2]
                ixtoword, wordtoix = x[3], x[4]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)

        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        elif split == 'val':
            captions = val_captions
            filenames = val_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words
    
    
    def load_captions(self, data_dir, filenames, split='train'):
        if split == 'test':
            return [[] for _ in filename] 
        # test 데이터셋의 경우 캡션이 없으니 빈 캡션 리스트 변환
        
        annotations_file = os.path.join(data_dir, 'annotations', f'captions_{split}2017.json')
        # split에 맞는 캡션 파일 로드
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
            image_id = self.get_image_id_from_filename(filename)
            
            if image_id in image_id_to_captions:
                captions = image_id_to_captions[image_id]
            else:
                captions = []
            
            cnt = 0
            for cap in captions:
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                
                tokenizer = RegexpTokenizer(r'\w+')
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


    """
    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            # filenames 리스트에 있는 모든 이미지에 대해 반복 실행행
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
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
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions
    """

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            # 해당 경로에 filenames.pickle 파일이 존재하면
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
                # 해당 파일을 로드하여 filenames 리스트에 저장
                # 이미지 파일명이 해당 리스트에 저장됨
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            # 없는 경우 빈 문자열 반환
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)
        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)
        return imgs, caps, cap_len, cls_id, key


    def __len__(self):
        return len(self.filenames)
    
    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in xrange(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox