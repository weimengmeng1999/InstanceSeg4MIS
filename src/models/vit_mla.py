import torch
import torch.nn as nn
from einops import rearrange

from functools import partial
from dinov2.eval.setup import build_model_for_eval, get_autocast_dtype
from dinov2.utils.config import get_cfg_from_args
from dinov2.eval.utils import ModelWithIntermediateLayers

import torch.nn.functional as F

class MLAHead(nn.Module):
    def __init__(self, mla_channels=384, mlahead_channels=128, norm_cfg=None):
        super(MLAHead, self).__init__()
        self.head2 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), 
                                   nn.ReLU(),
                                   nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), 
                                   nn.ReLU())
        self.head3 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels),  
                                   nn.ReLU(),
                                   nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), 
                                   nn.ReLU())
        self.head4 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), 
                                   nn.ReLU(),
                                   nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), 
                                   nn.ReLU())
        self.head5 = nn.Sequential(nn.Conv2d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), 
                                   nn.ReLU(),
                                   nn.Conv2d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(mlahead_channels), 
                                   nn.ReLU())

    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5):
        # head2 = self.head2(mla_p2)
        head2 = F.interpolate(self.head2(
            mla_p2), 4*mla_p2.shape[-1], mode='bilinear', align_corners=True)
        head3 = F.interpolate(self.head3(
            mla_p3), 4*mla_p3.shape[-1], mode='bilinear', align_corners=True)
        head4 = F.interpolate(self.head4(
            mla_p4), 4*mla_p4.shape[-1], mode='bilinear', align_corners=True)
        head5 = F.interpolate(self.head5(
            mla_p5), 4*mla_p5.shape[-1], mode='bilinear', align_corners=True)
        return torch.cat([head2, head3, head4, head5], dim=1)

class Decoder2D(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=588, mla_channels=384, mlahead_channels=128,
                 norm_layer=nn.BatchNorm2d, num_classes=1, norm_cfg=None):
        super().__init__()
        self.img_size = img_size
        self.norm_cfg = norm_cfg
        self.mla_channels = mla_channels
        self.BatchNorm = norm_layer
        self.mlahead_channels = mlahead_channels
        self.num_classes = num_classes

        self.mlahead = MLAHead(mla_channels=self.mla_channels,
                               mlahead_channels=self.mlahead_channels, norm_cfg=self.norm_cfg)
        # self.cls = nn.Conv2d(4 * self.mlahead_channels,
        #                      self.num_classes, 3, padding=1)
        self.cls = nn.Sequential(
                    nn.Conv2d(4 * self.mlahead_channels, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)#56 256
                )
        self.cls_1 = nn.Sequential(
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True) #56 256
                )
        self.cls_2 = nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True) #56 256
                )
        self.cls_3 = nn.Conv2d(64, self.num_classes, 3, padding=1)

        self.cls_ins = nn.Conv2d(64, 3, 3, padding=1) #depends on sigma

    def forward(self, input, input1, input2, input3):
        x = self.mlahead(input, input1, input2, input3)
        x = self.cls(x)
        x = self.cls_1(x)
        x = self.cls_2(x)
        x1 = self.cls_ins(x)
        x1 = F.interpolate(x1, size=self.img_size, mode='bilinear')
        x = self.cls_3(x)
        x = F.interpolate(x, size=self.img_size, mode='bilinear')
        return x1, x

class Branchedvitmla(nn.Module):
    def __init__(self):
        super().__init__()
        # cfg = get_cfg_from_args(args)
        # model = build_model_for_eval(cfg, args.pretrained_weights)
        autocast_dtype = torch.float16
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        

        n_last_blocks_list = [1, 4]
        n_last_blocks = max(n_last_blocks_list)
        autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
        self.feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)

        self.seg_decoder = Decoder2D(num_classes=1)


    def forward(self, input):
        H = 588
        W = 588

        with torch.no_grad():
            x_tokens_list = self.feature_model(input)

            intermediate_output_last = x_tokens_list[-1:]
            intermediate_output_last_2 = x_tokens_list[-2:-1]
            intermediate_output_last_3 = x_tokens_list[-3:-2]
            intermediate_output_last_4 = x_tokens_list[-4:-3]
            #   print(x_tokens_list[-1:].shape)
            output_last = torch.cat([outputs for outputs, _ in intermediate_output_last], dim=-1)
            output_last_2 =  torch.cat([outputs for outputs, _ in intermediate_output_last_2], dim=-1)
            output_last_3 =  torch.cat([outputs for outputs, _ in intermediate_output_last_3], dim=-1)
            output_last_4 =  torch.cat([outputs for outputs, _ in intermediate_output_last_4], dim=-1)

            output_last = rearrange(output_last, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
                        p1 = 1, p2 = 1, 
                        h = H // 14, w = W // 14, 
                        c = 384)
            output_last_2 = rearrange(output_last_2, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
                        p1 = 1, p2 = 1, 
                        h = H // 14, w = W // 14, 
                        c = 384)
            output_last_3 = rearrange(output_last_3, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
                        p1 = 1, p2 = 1, 
                        h = H // 14, w = W // 14, 
                        c = 384)
            output_last_4 = rearrange(output_last_4, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
                        p1 = 1, p2 = 1, 
                        h = H // 14, w = W // 14, 
                        c = 384)

        output_ins, output = self.seg_decoder(output_last, output_last_2, output_last_3, output_last_4)

        return torch.cat([output_ins, output], 1)


