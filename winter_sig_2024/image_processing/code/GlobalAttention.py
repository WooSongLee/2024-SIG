"""
Global attention takes a matrix and a query metrix.
Based on each query vector q, it computes a parameterized convex combination of the matrix
based.
H_1 H_2 H_3 ... H_n
  q   q   q       q
    |  |   |       |
      \ |   |      /
              .....
          \   |  /
                  a
Constructs a unit mapping.
$$(H_1 + H_n, q) => (a)$$
Where H is of `batch x n x dim` and q is of `batch x dim`.

References:
https://github.com/OpenNMT/OpenNMT-py/tree/fc23dfef1ba2f258858b2765d24565266526dc76/onmt/modules
http://www.aclweb.org/anthology/D15-1166
"""

import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)


#### 단어 특징과 이미지 특징 간 어텐션 연산 수행
def func_attention(query, context, gamma1):
    # gamma1 : 스케일링 계수로 어텐션 강도를 조절하는 하이퍼파라미터
    """
    query(텍스트 임베딩) : batch x ndf(특징 차원) x queryL(텍스트 단어 개수) 
    context(이미지 특징 맵): batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

   
    context = context.view(batch_size, -1, sourceL)
    # 4D 이미지 특징 맵 (batch_size, ndf, ih, iw)를 (batch_size, ndf, sourceL)로 reshape
    contextT = torch.transpose(context, 1, 2).contiguous()
     # --> batch x sourceL x ndf (contextT : 각 공간 위치에서의 특징 벡터를 나타냄냄)

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL = atten(i,j,k)는 j번쨰 이미지 위치와 k번째 단어의 유사도
    attn = torch.bmm(contextT, query) # Eq. (7) in AttnGAN paper
    # 행렬 곱 연산을 이용해 query와 contextT 사이의 유사도를 계산

    # --> batch*sourceL x queryL
    attn = attn.view(batch_size*sourceL, queryL)
    attn = nn.Softmax()(attn)  # Eq. (8)
    # Softmax를 적용하여 sourceL 차원(공간적 위치)에 대해 확률 분포를 얻음음

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # 이를 통해 이미지의 각 위치가 텍스트 단어와 얼마나 관련이 있는지를 확률적으로 나타냄

    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)
    #  Eq. (9) : 어텐션 강화화
    attn = attn * gamma1 # gamma1을 곱해 특정 영역을 더 강하게 강조
    attn = nn.Softmax()(attn) # 다시 정규화
    attn = attn.view(batch_size, queryL, sourceL)

    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)
    # context와 attT(어텐션 가중치)를 곱해 최종적으로 텍스트와 관련이 있는 이미지 특징을 생성
    # weightedContext : 텍스트 단어별 중요도가 반영된 이미지 특징 벡터
    # attn : 각 단어가 이미지의 어느 부분에 집중하고 있는지를 나타내는 어텐션 맵맵

    return weightedContext, attn.view(batch_size, -1, ih, iw)


class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)

        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)
        # --> batch*queryL x sourceL
        attn = attn.view(batch_size*queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn