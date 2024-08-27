import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nets.yolo import YoloBody
import random
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import get_world_size


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


def conv1x1(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)


class MLP2d(nn.Module):
    def __init__(self, in_dim, inner_dim=4096, out_dim=256):
        super(MLP2d, self).__init__()
        
        self.linear1 = conv1x1(in_dim, inner_dim)
        self.bn1 = nn.BatchNorm2d(inner_dim)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.linear2 = conv1x1(inner_dim, out_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.linear2(x)
        
        return x



def Proj_Head(in_dim=1024, inner_dim=4096, out_dim=256):
    return MLP2d(in_dim, inner_dim, out_dim)


def Pred_Head(in_dim=256, inner_dim=4096, out_dim=256):
    return MLP2d(in_dim, inner_dim, out_dim)


def random_noise_levels():
    """Generates random shot and read noise from a log-log linear distribution."""
    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)
    log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = np.exp(log_shot_noise)
    
    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + np.random.normal(scale=0.26)
    # print('shot noise and read noise:', log_shot_noise, log_read_noise)
    read_noise = np.exp(log_read_noise)
    return shot_noise, read_noise


class TTNet(nn.Module):
    def __init__(self, num_classes, degration_cfg=None, aug_test=False, lighten=False, pretrained=False, ft=False):
        super(TTNet, self).__init__()
        self.name = "TTNet"
        self.yolo = YoloBody(num_classes)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if degration_cfg is None:
            self.degration_cfg = dict(darkness_range=(0.01, 1.0),
                                      gamma_range=(2.0, 3.5),
                                      rgb_range=(0.8, 0.1),
                                      red_range=(1.9, 2.4),
                                      blue_range=(1.5, 1.9),
                                      quantisation=[12, 14, 16])
        else:
            self.degration_cfg = degration_cfg
        self.aug_test = aug_test
        self.FT = ft  # Fine Tune
        self.lignten = lighten
        # build a 3-layer projector
        
        # parse arguments
        self.pixpro_p = 1.
        self.pixpro_momentum = 0.999
        self.pixpro_pos_ratio = 0.7
        self.pixpro_clamp_value = 0.
        self.pixpro_transform_layer = 1
        self.pixpro_ins_loss_weight = 0.
        
        # self.yolo.backbone.layers_out_filters = [64, 128, 256, 512, 1024]
        # self.layers_out_filters = [1024, 512, 256]
        
        self.layers_out_filters = self.yolo.backbone.layers_out_filters[-1:-4:-1]

        self.projector = nn.ModuleList([Proj_Head(in_dim=i) for i in self.layers_out_filters])

        if self.pixpro_transform_layer == 0:
            self.value_transform = Identity()
        elif self.pixpro_transform_layer == 1:
            self.value_transform = conv1x1(in_planes=256, out_planes=256)
        elif self.pixpro_transform_layer == 2:
            self.value_transform = MLP2d(in_dim=256, inner_dim=256, out_dim=256)
        else:
            raise NotImplementedError
        
        if self.pixpro_ins_loss_weight > 0.:
            self.projector_instance = Proj_Head(in_dim=1024)

            self.predictor = Pred_Head()
            
            self.avgpool = nn.AvgPool2d(13, stride=1)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        if pretrained:
            self.yolo.backbone.load_state_dict(torch.load("./weights/darknet53_backbone_weights.pth"))
            print("darknet53_backbone_weights is load!")
        
        
    def load_weights(self, path):
        print('Load weights {}.'.format(path))
        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = self.state_dict()
        pretrained_dict = torch.load(path, map_location="cuda:0" if torch.cuda.is_available() else "cpu")["model"]
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        self.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    

    def apply_ccm(self, image, ccm):
        '''
        The function of apply CCM matrix
        '''
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        return image.view(shape)
    

    def Low_Illumination_Degrading(self, img, safe_invert=False):
        
        '''
        (1)unprocess part(RGB2RAW) (2)low light corruption part (3)ISP part(RAW2RGB)
        Some code copy from 'https://github.com/timothybrooks/unprocessing', thx to their work ~
        input:
        img (Tensor): Input normal light images of shape (C, H, W).
        img_meta(dict): A image info dict contain some information like name ,shape ...
        return:
        img_deg (Tensor): Output degration low light images of shape (C, H, W).
        degration_info(Tensor): Output degration paramter in the whole process.
        '''
        
        '''
        parameter setting
        '''
        device = img.device
        config = self.degration_cfg
        # camera color matrix
        xyz2cams = [[[1.0234, -0.2969, -0.2266],
                     [-0.5625, 1.6328, -0.0469],
                     [-0.0703, 0.2188, 0.6406]],
                    [[0.4913, -0.0541, -0.0202],
                     [-0.613, 1.3513, 0.2906],
                     [-0.1564, 0.2151, 0.7183]],
                    [[0.838, -0.263, -0.0639],
                     [-0.2887, 1.0725, 0.2496],
                     [-0.0627, 0.1427, 0.5438]],
                    [[0.6596, -0.2079, -0.0562],
                     [-0.4782, 1.3016, 0.1933],
                     [-0.097, 0.1581, 0.5181]]]
        rgb2xyz = [[0.4124564, 0.3575761, 0.1804375],
                   [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]]
        
        # noise parameters and quantization step
        
        '''
        (1)反变换(RGB2RAW): 1.反色调映射, 2.反伽马矫正, 3.反颜色校正, 4.反白平衡
        '''
        img1 = img.permute(1, 2, 0)  # (C, H, W) -- (H, W, C)
        # print(img1.shape)
        # img_meta = img_metas[i]
        # inverse tone mapping
        img1 = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * img1) / 3.0)
        # inverse gamma
        epsilon = torch.FloatTensor([1e-8]).to(torch.device(device))
        gamma = random.uniform(config['gamma_range'][0], config['gamma_range'][1])
        img2 = torch.max(img1, epsilon) ** gamma
        # sRGB2cRGB
        xyz2cam = random.choice(xyz2cams)
        rgb2cam = np.matmul(xyz2cam, rgb2xyz)
        rgb2cam = torch.from_numpy(rgb2cam / np.sum(rgb2cam, axis=-1)).to(torch.float).to(torch.device(device))
        # print(rgb2cam)
        img3 = self.apply_ccm(img2, rgb2cam)
        # img3 = torch.clamp(img3, min=0.0, max=1.0)
        
        # inverse WB
        rgb_gain = random.normalvariate(config['rgb_range'][0], config['rgb_range'][1])
        red_gain = random.uniform(config['red_range'][0], config['red_range'][1])
        blue_gain = random.uniform(config['blue_range'][0], config['blue_range'][1])
        
        gains1 = np.stack([1.0 / red_gain, 1.0, 1.0 / blue_gain]) * rgb_gain
        # gains1 = np.stack([1.0 / red_gain, 1.0, 1.0 / blue_gain])
        gains1 = gains1[np.newaxis, np.newaxis, :]
        gains1 = torch.FloatTensor(gains1).to(torch.device(device))
        
        # color disorder !!!
        if safe_invert:
            img3_gray = torch.mean(img3, dim=-1, keepdim=True)
            inflection = 0.9
            zero = torch.zeros_like(img3_gray).to(torch.device(device))
            mask = (torch.max(img3_gray - inflection, zero) / (1.0 - inflection)) ** 2.0
            safe_gains = torch.max(mask + (1.0 - mask) * gains1, gains1)
            
            # img4 = img3 * gains1
            img4 = torch.clamp(img3 * safe_gains, min=0.0, max=1.0)
        
        else:
            img4 = img3 * gains1
        
        '''
        (2)低光扰动: 5.线性降光, 6.镜头噪声和读出噪声
        '''
        # darkness(low photon numbers)
        lower, upper = config['darkness_range'][0], config['darkness_range'][1]
        mu, sigma = 0.1, 0.08
        darkness = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        darkness = darkness.rvs()
        # print(darkness)
        img5 = img4 * darkness
        # add shot and read noise
        shot_noise, read_noise = random_noise_levels()
        var = img5 * shot_noise + read_noise  # here the read noise is independent
        var = torch.max(var, epsilon)
        # print('the var is:', var)
        noise = torch.normal(mean=0, std=var)
        img6 = img5 + noise
        
        '''
        (3)正变换 (RAW2RGB): 7.量化  8.白平衡 9.颜色校正 10.伽马矫正
        '''
        # quantisation noise: uniform distribution
        bits = random.choice(config['quantisation'])
        quan_noise = torch.FloatTensor(img6.size()).uniform_(-1 / (255 * bits), 1 / (255 * bits)).to(
            torch.device(device))
        # print(quan_noise)
        # img7 = torch.clamp(img6 + quan_noise, min=0)
        img7 = img6 + quan_noise
        # white balance
        gains2 = np.stack([red_gain, 1.0, blue_gain])
        gains2 = gains2[np.newaxis, np.newaxis, :]
        gains2 = torch.FloatTensor(gains2).to(torch.device(device))
        img8 = img7 * gains2
        # cRGB2sRGB
        cam2rgb = torch.inverse(rgb2cam)
        img9 = self.apply_ccm(img8, cam2rgb)
        # gamma correction
        img10 = torch.max(img9, epsilon) ** (1 / gamma)
        
        img_low = img10.permute(2, 0, 1)  # (H, W, C) -- (C, H, W)
        # degration infomations: darkness, gamma value, WB red, WB blue
        # dark_gt = torch.FloatTensor([darkness]).to(torch.device(device))
        para_gt = torch.FloatTensor([darkness, 1.0 / gamma, 1.0 / red_gain, 1.0 / blue_gain]).to(torch.device(device))
        # others_gt = torch.FloatTensor([1.0 / gamma, 1.0, 1.0]).to(torch.device(device))
        # print('the degration information:', degration_info)
        
        return img_low, para_gt
    
    def featprop(self, feat):
        N, C, H, W = feat.shape
        
        # Value transformation
        feat_value = self.value_transform(feat)
        feat_value = F.normalize(feat_value, dim=1)
        feat_value = feat_value.view(N, C, -1)
        
        # Similarity calculation
        feat = F.normalize(feat, dim=1)
        
        # [N, C, H * W]
        feat = feat.view(N, C, -1)
        
        # [N, H * W, H * W]
        attention = torch.bmm(feat.transpose(1, 2), feat)
        attention = torch.clamp(attention, min=self.pixpro_clamp_value)
        if self.pixpro_p < 1.:
            attention = attention + 1e-6
        attention = attention ** self.pixpro_p
        
        # [N, C, H * W]
        feat = torch.bmm(feat_value, attention.transpose(1, 2))
        
        return feat.view(N, C, H, W)
    
    
    def contrastive_layer(self, feat1, feat2, projector):
        proj_1 = projector(feat1)
        pred_1 = self.featprop(proj_1)
        
        proj_1 = F.normalize(proj_1, dim=1)
        pred_1 = F.normalize(pred_1, dim=1)
        
        proj_2 = projector(feat2)
        pred_2 = self.featprop(proj_2)
        
        proj_2 = F.normalize(proj_2, dim=1)
        pred_2 = F.normalize(pred_2, dim=1)
        
        return [proj_1, proj_2, pred_1, pred_2]
    
    def contrastive_instance(self, feat_1, feat_2, projector_instance):
        
        proj_instance_1 = projector_instance(feat_1)
        pred_instacne_1 = self.predictor(proj_instance_1)
        
        proj_instance_1 = F.normalize(self.avgpool(proj_instance_1).view(proj_instance_1.size(0), -1),
                                      dim=1)
        pred_instance_1 = F.normalize(self.avgpool(pred_instacne_1).view(pred_instacne_1.size(0), -1),
                                      dim=1)
        
        proj_instance_2 = projector_instance(feat_2)
        pred_instance_2 = self.predictor(proj_instance_2)
        
        proj_instance_2 = F.normalize(self.avgpool(proj_instance_2).view(proj_instance_2.size(0), -1),
                                      dim=1)
        pred_instance_2 = F.normalize(self.avgpool(pred_instance_2).view(pred_instance_2.size(0), -1),
                                      dim=1)
        
        return [proj_instance_1, proj_instance_2, pred_instance_1, pred_instance_2]
    
    def forward(self, img):
        # generate low light degration images part and get degration informations
        # generate low light images
        if self.training is True:
            if self.FT:
                yolo_out, backbone_out = self.yolo(img)
                return yolo_out, backbone_out
            else:
                batch_size = img.shape[0]
                device = img.device
                # img_ = torch.empty(size=(batch_size, img.shape[1], img.shape[2], img.shape[3])).to(torch.device(device))
                img_dark = torch.empty(size=(batch_size, img.shape[1], img.shape[2], img.shape[3])).to(torch.device(device))
                # others_gt = torch.empty(size=(batch_size, 3)).to(torch.device(device))
                
                # Generation of degraded data and AET groundtruth
                for i in range(batch_size):
                    img_dark[i], _ = self.Low_Illumination_Degrading(img[i])
                # for i in range(batch_size):
                #     img_[i], _ = self.Low_Illumination_Degrading(img[i])
                # # img_features:52,52,256；26,26,512；13,13,1024
                # yolo_out, img_features = self.yolo(img_)
                yolo_out, img_features = self.yolo(img)
                yolo_out_dark, img_dark_features = self.yolo(img_dark)
                
                k = 1
                con_info = []
                for i in range(k):
                    con_layer_info = self.contrastive_layer(img_features[i], img_dark_features[i], self.projector[i])
                    con_info.append(con_layer_info)
                
                if self.pixpro_ins_loss_weight > 0.:
                    con_info_ins = self.contrastive_instance(img_features[0], img_dark_features[0],
                                                             self.projector_instance)
                    con_info.append(con_info_ins)
                return [yolo_out, yolo_out_dark], con_info
        
        # inference
        else:
            if self.aug_test:
                batch_size = img.shape[0]
                device = img.device
                img_dark = torch.empty(size=(batch_size, img.shape[1], img.shape[2], img.shape[3])).to(
                    torch.device(device))
                for i in range(batch_size):
                    img_dark[i], _ = self.Low_Illumination_Degrading(img[i])
                img = img_dark

            x, p = self.yolo(img)
            return x, p

# if __name__=="__main__":
#     model = TTNet(12).cuda().train()
#     input = torch.rand([7, 3, 416, 416]).cuda()
#     output = model(input)
#     print("")
