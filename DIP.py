import torch
import torch.nn as nn
import os
import sys
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import get_world_size

degration_cfg = dict(darkness_range=(0.01, 1.0),
                          gamma_range=(2.0, 3.5),
                          rgb_range=(0.8, 0.1),
                          red_range=(1.9, 2.4),
                          blue_range=(1.5, 1.9),
                          quantisation=[12, 14, 16])


def apply_ccm(image, ccm):
    '''
    The function of apply CCM matrix
    '''
    shape = image.shape
    image = image.view(-1, 3)
    image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
    return image.view(shape)

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

def Low_Illumination_Degrading(img, safe_invert=False, save_each = False, save_last = True):
    device = img.device
    config = degration_cfg
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
    (1)unprocess part(RGB2RAW): 1.inverse tone, 2.inverse gamma, 3.sRGB2cRGB, 4.inverse WB digital gains
    '''
    img1 = img.permute(1, 2, 0)  # (C, H, W) -- (H, W, C)
    # print(img1.shape)
    # img_meta = img_metas[i]
    # inverse tone mapping
    img1 = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * img1) / 3.0)
    if save_each:
        inverse_tone = img1.clone().detach().numpy()
        cv2.imwrite('./picture/1inverse_tone.png', inverse_tone*255.0)
        cv2.imshow("inverse_tone", inverse_tone)
        cv2.waitKey(0)
    
    # inverse gamma
    epsilon = torch.FloatTensor([1e-8]).to(torch.device(device))
    gamma = random.uniform(config['gamma_range'][0], config['gamma_range'][1])
    img2 = torch.max(img1, epsilon) ** gamma
    if save_each:
        inverse_gamma = img2.clone().detach().numpy()
        cv2.imwrite('./picture/2inverse_gamma.png', inverse_gamma*255.0)
        cv2.imshow("inverse_gamma", inverse_tone)
        cv2.waitKey(0)
    # sRGB2cRGB
    xyz2cam = random.choice(xyz2cams)
    rgb2cam = np.matmul(xyz2cam, rgb2xyz)
    rgb2cam = torch.from_numpy(rgb2cam / np.sum(rgb2cam, axis=-1)).to(torch.float).to(torch.device(device))
    # print(rgb2cam)
    img3 = apply_ccm(img2, rgb2cam)
    # img3 = torch.clamp(img3, min=0.0, max=1.0)
    
    sRGB2cRGB = img3.clone().detach().numpy()
    if save_each:
        cv2.imwrite('./picture/3sRGB2cRGB.png', sRGB2cRGB*255.0)
        cv2.imshow("3sRGB2cRGB", sRGB2cRGB)
        cv2.waitKey(0)
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
    if save_each:
        inverse_WB = img4.clone().detach().numpy()
        cv2.imwrite('./picture/4inverse_WB.png', inverse_WB*255.0)
        cv2.imshow("inverse_WB", inverse_WB)
        cv2.waitKey(0)
    '''
    (2)low light corruption part: 5.darkness, 6.shot and read noise
    '''
    # darkness(low photon numbers)
    lower, upper = config['darkness_range'][0], config['darkness_range'][1]
    mu, sigma = 0.1, 0.08
    darkness = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    darkness = darkness.rvs()

    print("darkness= %f " %darkness)
    img5 = img4 * darkness
    if save_each:
        dark = img5.clone().detach().numpy()
        cv2.imwrite('./picture/5dark.png', dark*255.0)
        cv2.imshow("dark", dark)
        cv2.waitKey(0)
    # add shot and read noise
    shot_noise, read_noise = random_noise_levels()
    var = img5 * shot_noise + read_noise  # here the read noise is independent
    var = torch.max(var, epsilon)
    # print('the var is:', var)
    noise = torch.normal(mean=0, std=var)
    img6 = img5 + noise
    if save_each:
        darkness_noise = img6.clone().detach().numpy()
        cv2.imwrite('./picture/6darkness_noise.png', darkness_noise*255.0)
        cv2.imshow("darkness_noise", darkness_noise)
        cv2.waitKey(0)
    
    '''
    (3)ISP part(RAW2RGB): 7.quantisation  8.white balance 9.cRGB2sRGB 10.gamma correction
    '''
    # quantisation noise: uniform distribution
    bits = random.choice(config['quantisation'])
    quan_noise = torch.FloatTensor(img6.size()).uniform_(-1 / (255 * bits), 1 / (255 * bits)).to(
        torch.device(device))
    # print(quan_noise)
    # img7 = torch.clamp(img6 + quan_noise, min=0)
    img7 = img6 + quan_noise
    if save_each:
        quantisation = img7.clone().detach().numpy()
        cv2.imwrite('./picture/7quantisation.png', quantisation * 255.0)
        cv2.imshow("quantisation", quantisation)
        cv2.waitKey(0)
    # white balance
    gains2 = np.stack([red_gain, 1.0, blue_gain])
    gains2 = gains2[np.newaxis, np.newaxis, :]
    gains2 = torch.FloatTensor(gains2).to(torch.device(device))
    img8 = img7 * gains2
    if save_each:
        white_balance = img8.clone().detach().numpy()
        cv2.imwrite('./picture/8white_balance.png', white_balance * 255.0)
        cv2.imshow("white_balance", white_balance)
        cv2.waitKey(0)
    # cRGB2sRGB
    cam2rgb = torch.inverse(rgb2cam)
    img9 = apply_ccm(img8, cam2rgb)
    if save_each:
        cRGB2sRGB = img9.clone().detach().numpy()
        cv2.imwrite('./picture/9cRGB2sRGB.png', cRGB2sRGB * 255.0)
        cv2.imshow("cRGB2sRGB", cRGB2sRGB)
        cv2.waitKey(0)
    # gamma correction
    img10 = torch.max(img9, epsilon) ** (1 / gamma)
    img_low = img10
    if save_each:
        gamma_correction = img10.clone().detach().numpy()
        cv2.imwrite('./LLIS_picture/10gamma_correction.png', gamma_correction * 255.0)
        cv2.imshow("gamma_correction", gamma_correction)
        cv2.waitKey(0)
    if save_last:
        gamma_correction = img10.clone().detach().numpy()
        cv2.imwrite('./LLIS_picture/10gamma_correction.png', gamma_correction * 255.0)
        cv2.imshow("gamma_correction", gamma_correction)
        cv2.waitKey(0)
    # img_low = img10.permute(2, 0, 1)  # (H, W, C) -- (C, H, W)
    # degration infomations: darkness, gamma value, WB red, WB blue
    # dark_gt = torch.FloatTensor([darkness]).to(torch.device(device))
    para_gt = torch.FloatTensor([darkness, 1.0 / gamma, 1.0 / red_gain, 1.0 / blue_gain]).to(torch.device(device))
    # others_gt = torch.FloatTensor([1.0 / gamma, 1.0, 1.0]).to(torch.device(device))
    # print('the degration information:', degration_info)
    
    return img_low, para_gt


# img = cv2.imread('D:/CV/Detection_in_Dark/VOCdevkit/ExDark_coco/JPEGImages/000000537206.jpg')
# print(img.shape)
# img = torch.Tensor(img.transpose(2, 0, 1))/255.0
# img, _ = Low_Illumination_Degrading(img, save_each=False, save_last=True)
