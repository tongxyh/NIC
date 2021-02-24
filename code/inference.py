####       Tong Chen        ####
#### tong@smail.nju.edu.cn  ####

import argparse
import math
import os
import struct
import sys
import time
from decimal import *
D = Decimal
import decimal
from multiprocessing import Pool

decimal.getcontext().prec = 16

from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# import Util.AE as AE
import AE
import Model.model as model
from Model.context_model import Weighted_Gaussian
from Util import ops

import profile
# avoid memory leak during cpu inference for Pytorch < 1.5
# more details: https://github.com/pytorch/pytorch/issues/27971
os.environ['LRU_CACHE_CAPACITY'] = '1'
GPU = True
DEBUG = True

# index - [0-15]
models = ["mse200", "mse400", "mse800", "mse1600", "mse3200", "mse6400", "mse12800", "mse25600",
          "msssim4", "msssim8", "msssim16", "msssim32", "msssim64", "msssim128", "msssim320", "msssim640"]

@torch.no_grad()
def encode(im_dir, out_dir, model_dir, model_index, block_width, block_height):
    file_object = open(out_dir, 'wb')
    
    M, N2 = 192, 128
    if (model_index == 6) or (model_index == 7) or (model_index == 14) or (model_index == 15):
        M, N2 = 256, 192
    image_comp = model.Image_coding(3, M, N2, M, M//2)
    context = Weighted_Gaussian(M)
    ######################### Load Model #########################
    image_comp.load_state_dict(torch.load(
        os.path.join(model_dir, models[model_index] + r'.pkl'), map_location='cpu'))
    context.load_state_dict(torch.load(
        os.path.join(model_dir, models[model_index] + r'p.pkl'), map_location='cpu'))
    if GPU:
        image_comp = image_comp.cuda()
        context = context.cuda()
    ######################### Read Image #########################
    img = Image.open(im_dir)
    img = np.array(img)/255.0
    H, W, _ = img.shape
    num_pixels = H * W
    C = 3
    Head = struct.pack('2HB', H, W, model_index)
    file_object.write(Head)
    ######################### spliting Image #########################
    Block_Num_in_Width = int(np.ceil(W / block_width))
    Block_Num_in_Height = int(np.ceil(H / block_height))
    img_block_list = []
    for i in range(Block_Num_in_Height):
        for j in range(Block_Num_in_Width):
            img_block_list.append(img[i * block_height:np.minimum((i + 1) * block_height, H),j * block_width:np.minimum((j + 1) * block_width,W),...])

    ######################### Padding Image #########################
    Block_Idx = 0
    for img in img_block_list:
        block_H = img.shape[0]
        block_W = img.shape[1]

        tile = 64.
        block_H_PAD = int(tile * np.ceil(block_H / tile))
        block_W_PAD = int(tile * np.ceil(block_W / tile))
        im = np.zeros([block_H_PAD, block_W_PAD, 3], dtype='float32')
        im[:block_H, :block_W, :] = img[:, :, :3]
        im = torch.FloatTensor(im)
        im = im.permute(2, 0, 1).contiguous()
        im = im.view(1, C, block_H_PAD, block_W_PAD)
        if GPU:
            im = im.cuda()
        print('====> Encoding Image:', im_dir, "%dx%d" % (block_H, block_W), 'to', out_dir, " Block Idx: %d" % (Block_Idx))
        Block_Idx +=1

        with torch.no_grad():
            y_main, y_hyper = image_comp.encoder(im)
            y_main_q = torch.round(y_main)
            y_main_q = torch.Tensor(y_main_q.cpu().numpy().astype(np.int))
            if GPU:
                y_main_q = y_main_q.cuda()

            # y_hyper_q = torch.round(y_hyper)

            y_hyper_q, xp2 = image_comp.factorized_entropy_func(y_hyper, 2)
            y_hyper_q = torch.Tensor(y_hyper_q.cpu().numpy().astype(np.int))
            if GPU:
                y_hyper_q = y_hyper_q.cuda()

            hyper_dec = image_comp.p(image_comp.hyper_dec(y_hyper_q))

            xp3, params_prob = context(y_main_q, hyper_dec)

        # Main Arith Encode
        Datas = torch.reshape(y_main_q, [-1]).cpu().numpy().astype(np.int).tolist()
        Max_Main = max(Datas)
        Min_Main = min(Datas)
        sample = np.arange(Min_Main, Max_Main+1+1).tolist()  # [Min_V - 0.5 , Max_V + 0.5]
        _, c, h, w = y_main_q.shape
        print("Main Channel:", c)
        # sample = torch.FloatTensor(np.tile(sample, [1, c, h, w, 1]))
        # if GPU:
        #     sample = sample.cuda()

        # 3 mixed gaussian
        prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = [
            torch.chunk(params_prob, 9, dim=1)[i].squeeze(1) for i in range(9)]
        del params_prob
        # keep the weight summation of prob == 1
        probs = torch.stack([prob0, prob1, prob2], dim=-1)
        del prob0, prob1, prob2

        probs = F.softmax(probs.cpu(), dim=-1)
        # process the scale value to positive non-zero
        scale0 = torch.abs(scale0)
        scale1 = torch.abs(scale1)
        scale2 = torch.abs(scale2)
        scale0[scale0 < 1e-6] = 1e-6
        scale1[scale1 < 1e-6] = 1e-6
        scale2[scale2 < 1e-6] = 1e-6

        # t = time.time()
        precise = 16
        samples = np.arange(Min_Main, Max_Main+1+1)  # [Min_V - 0.5 , Max_V + 0.5]
        
        mean0_ = torch.reshape(mean0, [-1]).cpu().numpy().tolist()
        mean1_ = torch.reshape(mean1, [-1]).cpu().numpy().tolist()
        mean2_ = torch.reshape(mean2, [-1]).cpu().numpy().tolist()
        
        scale0_ = torch.reshape(scale0, [-1]).cpu().numpy().tolist()
        scale1_ = torch.reshape(scale1, [-1]).cpu().numpy().tolist()
        scale2_ = torch.reshape(scale2, [-1]).cpu().numpy().tolist()
        
        probs0_ = torch.reshape(probs[:, :, :, :, 0],[-1]).cpu().numpy().tolist()
        probs1_ = torch.reshape(probs[:, :, :, :, 1],[-1]).cpu().numpy().tolist()
        probs2_ = torch.reshape(probs[:, :, :, :, 2],[-1]).cpu().numpy().tolist()
        
        ###### rounding error check ######
        mean0_ = list(map(ops.round_check, mean0_))
        mean1_ = list(map(ops.round_check, mean1_))        
        mean2_ = list(map(ops.round_check, mean2_))

        scale0_ = list(map(ops.round_check, scale0_))
        scale1_ = list(map(ops.round_check, scale1_))
        scale2_ = list(map(ops.round_check, scale2_))

        # probs0_ = list(map(Decimal, probs0_))
        # probs1_ = list(map(Decimal, probs1_))
        # probs2_ = list(map(Decimal, probs2_)) # [N]
                
        def err_index(dat_):
            return [[i,x[1]] for i,x in enumerate(dat_) if x[1] != 0]

        err_mean = [err_index(mean0_), err_index(mean1_), err_index(mean2_)]
        err_scal = [err_index(scale0_), err_index(scale1_), err_index(scale2_)]
        # err_prob = [err_index(probs0_), err_index(probs1_), err_index(scale2_)]
        err_prob = [[], [], []]
        err_len = len(err_mean[0]) + len(err_mean[1]) + len(err_mean[2]) + len(err_scal[0]) + len(err_scal[1]) + len(err_scal[2]) + len(err_prob[0]) + len(err_prob[1]) + len(err_prob[2])
        if err_len > 0:
            print("[WARNING] Found %d possible rounding errors"%(err_len))
            print("\t Mean [0,1,2]:", len(err_mean[0]),len(err_mean[1]), len(err_mean[2]))
            print("\t Scal [0,1,2]:", len(err_scal[0]),len(err_scal[1]), len(err_scal[2]))
            # print("\t Prob [0,1,2]:", len(err_prob[0]),len(err_prob[1]), len(err_prob[2]))
        else:
            print("[ROUND CHECK] PASSED!")

        # print("Time (s):", time.time() - t)
        lower_ = []
        factor = (1 << precise) - (Max_Main - Min_Main + 1)
        # def cdf_sample(l0,l1,l2,p0,p1,p2):
        #     if p0[0] <= p2[0] and p1[0] <= p2[0]:
        #         return int((p0[0]*l0+p1[0]*l1+(1-p0[0]-p1[0])*l2) * factor)
        #     if p0[0] <= p1[0] and p2[0] <= p1[0]:
        #         return int((p0[0]*l0+(1-p0[0]-p2[0])*l1+p2[0]*l2) * factor)
        #     if p1[0] <= p0[0] and p2[0] <= p0[0]:
        #         return int(((1-p1[0]-p2[0])*l0+p1[0]*l1+p2[0]*l2) * factor)
        #     print("[WARNING] Found Equal prob weights [0 1 2]: ", p0[0],p1[0],p2[0])
        #     return int((p0[0]*l0+p1[0]*l1+(1-p0[0]-p1[0])*l2) * factor)
        def cdf_sample(l0,l1,l2,p0,p1,p2,d,s):
            # p0 = Decimal(p0).quantize(Decimal("0.0000001"))
            # p1 = Decimal(p1).quantize(Decimal("0.0000001"))
            # p2 = Decimal(p2).quantize(Decimal("0.0000001"))
            p0,p1,p2 = D(p0), D(p1), D(p2)
            prec_f = D("0.004")
            flag   = 0

            if p0 <= p2 and p1 <= p2:
                p = (p0*l0+p1*l1+(1-p0-p1)*l2) * factor / 10
                if d == s or s - d == 1:
                    # if p > factor - 1 or p < 1:
                    #     return p
                    # rounding check
                    if int(p - prec_f) != int(p) or int(p + prec_f) != int(p):
                        # print("[WARNING] possible round error in Probs", p)
                        flag = 1
                return p, flag
            
            if p0 <= p1 and p2 <= p1:
                p = (p0*l0+(1-p0-p2)*l1+p2*l2) * factor / 10
                if d == s or s - d == 1:
                    # if p > factor - 1 or p < 1:
                    #     return p
                    # rounding check
                    if int(p - prec_f) != int(p) or int(p + prec_f) != int(p):
                        # print("[WARNING] possible round error in Probs", p)
                        flag = 1
                return p, flag
            if p1 <= p0 and p2 <= p0:
                p =((1-p1-p2)*l0+p1*l1+p2*l2) * factor / 10
                if d == s or s - d == 1:
                    # if p > factor - 1 or p < 1:
                    #     return p
                    # rounding check
                    if int(p - prec_f) != int(p) or int(p + prec_f) != int(p):
                        # print("[WARNING] possible round error in Probs", p)
                        flag = 1
                return p, flag
            print("[WARNING] Found Equal prob weights [0 1 2]: ", p0,p1,p2)
            return int((p0*l0+p1*l1+(1-p0-p1)*l2) * factor), 2

        for sample in samples:
            # t_s = time.time()
            sample_ = np.repeat(sample, len(Datas)).tolist()
            lower0 = list(map(ops.Decimal_cdf, sample_, mean0_, scale0_))
            lower1 = list(map(ops.Decimal_cdf, sample_, mean1_, scale1_))
            lower2 = list(map(ops.Decimal_cdf, sample_, mean2_, scale2_))        
            lower_.append([cdf_sample(l0,l1,l2,p0,p1,p2,d,s) for l0,l1,l2,p0,p1,p2,d,s in zip(lower0, lower1, lower2, probs0_, probs1_, probs2_, Datas, sample_)])
            # print(sample, "Time (s):", time.time() - t_s)
        # DEBUG: 
        c_d, h_d, w_d = 132,38,27
        print("Error Location:", c_d, h_d, w_d)
        index_ = c_d*int(block_W_PAD/16*block_H_PAD/16) + h_d*int(block_W_PAD/16) + w_d
        print("[GET YOU]",y_main_q[0,c_d,h_d,w_d],lower_[int(y_main_q[0,c_d,h_d,w_d])-Min_Main][index_], lower_[int(y_main_q[0,c_d,h_d,w_d])-Min_Main+1][index_])
         
        lower,b = [],[]
        bappend = b.append
        for i in lower_:
            a = []
            for j,x in enumerate(i):
                a.append(int(x[0])*10)
                if x[1] > 0:
                    c_index = j // int(block_H_PAD/16*block_W_PAD/16)
                    h_index = (j % int(block_H_PAD/16*block_W_PAD/16)) // int(block_W_PAD/16)
                    w_index = h_index % int(block_W_PAD/16)
                    bappend([c_index, h_index, w_index])
                    print(b[-1],x[0])
            lower.append(a)
        print("[WARNING] Found %d possible rounding errors"%(len(b)))
        # int2np
        cdf_m = np.array(lower).transpose()
        # print("Time (s):", time.time() - t)
        
        cdf_m = cdf_m.astype(np.int32) + samples.astype(np.int32) - Min_Main
        cdf_main = np.reshape(cdf_m, [len(Datas), -1])

        # Cdf[Datas - Min_V]
        Cdf_lower = list(map(lambda x, y: int(y[x - Min_Main]), Datas, cdf_main))
        # Cdf[Datas + 1 - Min_V]
        Cdf_upper = list(map(lambda x, y: int(
            y[x - Min_Main]), Datas, cdf_main[:, 1:]))
        AE.encode_cdf(Cdf_lower, Cdf_upper, "main.bin")
        FileSizeMain = os.path.getsize("main.bin")
        print("[main.bin] %d bytes" % (FileSizeMain))

        # Hyper Arith Encode
        Min_V_HYPER = torch.min(y_hyper_q).cpu().numpy().astype(np.int).tolist()
        Max_V_HYPER = torch.max(y_hyper_q).cpu().numpy().astype(np.int).tolist()
        _, c, h, w = y_hyper_q.shape
        # print("Hyper Channel:", c)
        Datas_hyper = torch.reshape(
            y_hyper_q, [c, -1]).cpu().numpy().astype(np.int).tolist()
        # [Min_V - 0.5 , Max_V + 0.5]
        sample = np.arange(Min_V_HYPER, Max_V_HYPER+1+1)
        sample = np.tile(sample, [c, 1, 1])
        sample_tensor = torch.FloatTensor(sample)
        if GPU:
            sample_tensor = sample_tensor.cuda()
        lower = torch.sigmoid(image_comp.factorized_entropy_func._logits_cumulative(
            sample_tensor - 0.5, stop_gradient=False))
        cdf_h = lower.data.cpu().numpy()*((1 << precise) - (Max_V_HYPER -
                                                            Min_V_HYPER + 1))  # [N1, 1, Max-Min+1]
        cdf_h = cdf_h.astype(np.int) + sample.astype(np.int) - Min_V_HYPER
        cdf_hyper = np.reshape(np.tile(cdf_h, [len(Datas_hyper[0]), 1, 1, 1]), [
                               len(Datas_hyper[0]), c, -1])

        # Datas_hyper [256, N], cdf_hyper [256,1,X]
        Cdf_0, Cdf_1 = [], []
        for i in range(c):
            Cdf_0.extend(list(map(lambda x, y: int(
                y[x - Min_V_HYPER]), Datas_hyper[i], cdf_hyper[:, i, :])))   # Cdf[Datas - Min_V]
            Cdf_1.extend(list(map(lambda x, y: int(
                y[x - Min_V_HYPER]), Datas_hyper[i], cdf_hyper[:, i, 1:])))  # Cdf[Datas + 1 - Min_V]
        AE.encode_cdf(Cdf_0, Cdf_1, "hyper.bin")
        FileSizeHyper = os.path.getsize("hyper.bin")
        print("[hyper.bin] %d bytes" % (FileSizeHyper))

        Head_block = struct.pack('2H4h2I', block_H, block_W, Min_Main, Max_Main, Min_V_HYPER, Max_V_HYPER, FileSizeMain, FileSizeHyper)
        file_object.write(Head_block)
        # cat Head_Infor and 2 files together
        # Head = [FileSizeMain,FileSizeHyper,H,W,Min_Main,Max_Main,Min_V_HYPER,Max_V_HYPER,model_index]
        # print("Head Info:",Head)
        with open("main.bin", 'rb') as f:
            bits = f.read()
            file_object.write(bits)
        with open("hyper.bin", 'rb') as f:
            bits = f.read()
            file_object.write(bits)
    if DEBUG:
        return y_main_q, mean0, mean1, mean2, scale0, scale1, scale2, probs, cdf_main
    return 0



@torch.no_grad()
def decode(bin_dir, rec_dir, model_dir, block_width, block_height):
    if DEBUG:
        ey_main_q = np.load("debug/encode.npy")
        em0 = np.load("debug/mean0.npy")
        em1 = np.load("debug/mean1.npy")
        em2 = np.load("debug/mean2.npy")
        es0 = np.load("debug/scale0.npy")
        es1 = np.load("debug/scale1.npy")
        es2 = np.load("debug/scale2.npy")
        ep = np.load("debug/probs.npy")  
        ecdf = np.load("debug/cdf.npy")
    ############### retreive head info ###############
    T = time.time()
    file_object = open(bin_dir, 'rb')

    head_len = struct.calcsize('2HB')
    bits = file_object.read(head_len)
    [H, W, model_index] = struct.unpack('2HB', bits)
    # print("File Info:",Head)
    # Split Main & Hyper bins
    C = 3
    out_img = np.zeros([H, W, C])
    H_offset = 0
    W_offset = 0
    Block_Num_in_Width = int(np.ceil(W / block_width))
    Block_Num_in_Height = int(np.ceil(H / block_height))

    c_main = 192
    c_hyper = 128
    
    M, N2 = 192, 128
    if (model_index == 6) or (model_index == 7) or (model_index == 14) or (model_index == 15):
        M, N2 = 256, 192
    image_comp = model.Image_coding(3, M, N2, M, M//2)
    context = Weighted_Gaussian(M)
    ######################### Load Model #########################
    image_comp.load_state_dict(torch.load(
        os.path.join(model_dir, models[model_index] + r'.pkl'), map_location='cpu'))
    context.load_state_dict(torch.load(
        os.path.join(model_dir, models[model_index] + r'p.pkl'), map_location='cpu'))
    if GPU:
        image_comp = image_comp.cuda()
        context = context.cuda()
        
    for i in range(Block_Num_in_Height):
        for j in range(Block_Num_in_Width):

            Block_head_len = struct.calcsize('2H4h2I')
            bits = file_object.read(Block_head_len)
            [block_H, block_W, Min_Main, Max_Main, Min_V_HYPER, Max_V_HYPER, FileSizeMain, FileSizeHyper] = struct.unpack('2H4h2I', bits)

            precise, tile = 16, 64.

            block_H_PAD = int(tile * np.ceil(block_H / tile))
            block_W_PAD = int(tile * np.ceil(block_W / tile))

            with open("main.bin", 'wb') as f:
                bits = file_object.read(FileSizeMain)
                f.write(bits)
            with open("hyper.bin", 'wb') as f:
                bits = file_object.read(FileSizeHyper)
                f.write(bits)

            ############### Hyper Decoder ###############
            # [Min_V - 0.5 , Max_V + 0.5]
            sample = np.arange(Min_V_HYPER, Max_V_HYPER+1+1)
            sample = np.tile(sample, [c_hyper, 1, 1])
            sample_tensor = torch.FloatTensor(sample)
            if GPU:
                sample_tensor = sample_tensor.cuda()
            lower = torch.sigmoid(image_comp.factorized_entropy_func._logits_cumulative(
                sample_tensor - 0.5, stop_gradient=False))
            cdf_h = lower.data.cpu().numpy()*((1 << precise) - (Max_V_HYPER -
                                                                Min_V_HYPER + 1))  # [N1, 1, Max - Min]
            cdf_h = cdf_h.astype(np.int) + sample.astype(np.int) - Min_V_HYPER
            T2 = time.time()
            AE.init_decoder("hyper.bin", Min_V_HYPER, Max_V_HYPER)
            Recons = []
            for i in range(c_hyper):
                for j in range(int(block_H_PAD * block_W_PAD / 64 / 64)):
                    # print(cdf_h[i,0,:])
                    Recons.append(AE.decode_cdf(cdf_h[i, 0, :].tolist()))
            # reshape Recons to y_hyper_q   [1, c_hyper, H_PAD/64, W_PAD/64]
            y_hyper_q = torch.reshape(torch.Tensor(
                Recons), [1, c_hyper, int(block_H_PAD / 64), int(block_W_PAD / 64)])

            ############### Main Decoder ###############
            if GPU:
                y_hyper_q = y_hyper_q.cuda()
            hyper_dec = image_comp.p(image_comp.hyper_dec(y_hyper_q))
            h, w = int(block_H_PAD / 16), int(block_W_PAD / 16)
            samples = np.arange(Min_Main, Max_Main+1+1)  # [Min_V - 0.5 , Max_V + 0.5]

            p3d = (5, 5, 5, 5, 5, 5)
            y_main_q = torch.zeros(1, 1, c_main+10, h+10, w+10)  # 8000x4000 -> 500*250
            if GPU:
                y_main_q = y_main_q.cuda()
            AE.init_decoder("main.bin", Min_Main, Max_Main)
            hyper = torch.unsqueeze(context.conv3(hyper_dec), dim=1)

            #
            context.conv1.weight.data *= context.conv1.mask
            c_weight = context.conv1.weight
            c_bias = context.conv1.bias

            for i in range(c_main):
                T = time.time()
                for j in range(int(block_H_PAD / 16)):
                    # t_h = time.time()
                    # for k in range(int(block_W_PAD / 16)):

                    x1 = F.conv3d(y_main_q[:, :, i:i+11, j:j+11, :], weight=c_weight, bias=c_bias)  # [1,24,1,1,1]
                    params_prob = context.conv2(torch.cat((x1, hyper[:, :, i:i+1, j:j+1, :]), dim=1))

                    # 3 gaussian
                    prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = params_prob[
                        0, :, 0, 0, :]
                    # keep the weight  summation of prob == 1
                    probs = torch.stack([prob0, prob1, prob2], dim=-1)
                    probs = F.softmax(probs.cpu(), dim=-1)
                    
                    # process the scale value to positive non-zero
                    scale0, scale1, scale2 = torch.abs(scale0), torch.abs(scale1), torch.abs(scale2)
                    scale0[scale0 < 1e-6] = 1e-6
                    scale1[scale1 < 1e-6] = 1e-6
                    scale2[scale2 < 1e-6] = 1e-6

                    mean0_, mean1_, mean2_ = mean0.cpu().numpy().tolist(), mean1.cpu().numpy().tolist(), mean2.cpu().numpy().tolist()                    
                    scale0_, scale1_, scale2_ = scale0.cpu().numpy().tolist(), scale1.cpu().numpy().tolist(), scale2.cpu().numpy().tolist()
                    
                    probs0_ = np.tile(probs[:, 0].cpu().numpy().reshape((-1, 1)), samples.shape[0]).tolist()
                    probs1_ = np.tile(probs[:, 1].cpu().numpy().reshape((-1, 1)), samples.shape[0]).tolist()
                    probs2_ = np.tile(probs[:, 2].cpu().numpy().reshape((-1, 1)), samples.shape[0]).tolist() # [N, len(samples)]

                    # 3 gaussian distributions
                    factor = (1 << precise) - (Max_Main - Min_Main + 1)

                    sample_ = np.tile(samples.reshape((1,-1)), (len(mean0_),1)).tolist() # [len(samples)] -> [N, len(samples)]

                    ###### rounding error check ######
                    # TODO: decoder version round_check 
                    mean0_ = list(map(ops.round_check, mean0_))
                    mean1_ = list(map(ops.round_check, mean1_))        
                    mean2_ = list(map(ops.round_check, mean2_))

                    scale0_ = list(map(ops.round_check, scale0_))
                    scale1_ = list(map(ops.round_check, scale1_))
                    scale2_ = list(map(ops.round_check, scale2_))

                    # TODO: round_check of probs
                    # probs0_ = list(map(ops.round_check, probs0_))
                    # probs1_ = list(map(ops.round_check, probs1_))

                    # t = time.time()
                    lower0 = list(map(ops.Decimal_cdf_, sample_, mean0_, scale0_))
                    lower1 = list(map(ops.Decimal_cdf_, sample_, mean1_, scale1_))
                    lower2 = list(map(ops.Decimal_cdf_, sample_, mean2_, scale2_))
                    # print(j, "decimal cdf:",time.time()-t)
                    
                    # t = time.time()   
                    def cdf_sample(l0,l1,l2,p0,p1,p2):
                        # p0 = Decimal(p0).quantize(Decimal("0.0000001"))
                        # p1 = Decimal(p1).quantize(Decimal("0.0000001"))
                        # p2 = Decimal(p2).quantize(Decimal("0.0000001"))
                        p0, p1, p2 = D(p0), D(p1), D(p2)
                        if p0 <= p2 and p1 <= p2:
                            return int((p0*l0+p1*l1+(1-p0-p1)*l2) * factor/10)*10
                        if p0 <= p1 and p2 <= p1:
                            return int((p0*l0+(1-p0-p2)*l1+p2*l2) * factor/10)*10
                        if p1 <= p0 and p2 <= p0:
                            return int(((1-p1-p2)*l0+p1*l1+p2*l2) * factor/10)*10
                        print("[WARNING] Found Equal prob weights [0 1 2]: ", p0,p1,p2)
                        return int((p0*l0+p1*l1+(1-p0-p1)*l2) * factor)

                    lower  = [list(map(cdf_sample, l0,l1,l2,p0,p1,p2)) for l0,l1,l2,p0,p1,p2 in zip(lower0, lower1, lower2, probs0_, probs1_, probs2_)]
                    # print(j, "final mix cdf:",time.time()-t)

                    cdf_m = np.array(lower).astype(np.int) + samples.astype(np.int) - Min_Main
                    

                    for k in range(int(block_W_PAD / 16)):
                        pixs = AE.decode_cdf(cdf_m[k, :].tolist())
                        y_main_q[0, 0, i+5, j+5, k+5] = pixs
                    
                        if DEBUG:
                            np.set_printoptions(threshold = 10000, linewidth=200)
                            index = int(i*(block_H_PAD/16*block_W_PAD/16)+j*(block_W_PAD/16)) + k
                            r = ecdf[index,:] == cdf_m[k,:]
                            if r.all():
                                pass
                            else:
                                print("FOUND NOT EQUAL in CDF !!! =====>", i,j,k, '\n', ecdf[index,int(ey_main_q[0,i,j,k]-Min_Main):int(ey_main_q[0,i,j,k]-Min_Main+2)] - samples.astype(np.int)[int(ey_main_q[0,i,j,k]-Min_Main):int(ey_main_q[0,i,j,k]-Min_Main+2)] + Min_Main, '\n', cdf_m[k,int(ey_main_q[0,i,j,k]-Min_Main):int(ey_main_q[0,i,j,k]-Min_Main+2)] - samples.astype(np.int)[int(ey_main_q[0,i,j,k]-Min_Main):int(ey_main_q[0,i,j,k]-Min_Main+2)] + Min_Main)
                                print("=== Latent Features (E/D):",i,j,k,ey_main_q[0,i,j,k],y_main_q[0,0,i+5,j+5,k+5])
                                
                                print("--- Mean 0 (E/D):",em0[0,i,j,k], mean0.cpu().numpy()[k])
                                print("--- Mean 1 (E/D):",em1[0,i,j,k], mean1.cpu().numpy()[k])
                                print("--- Mean 2 (E/D):",em2[0,i,j,k], mean2.cpu().numpy()[k])

                                print("--- Scal 0 (E/D):",es0[0,i,j,k], scale0.cpu().numpy()[k])
                                print("--- Scal 1 (E/D):",es1[0,i,j,k], scale1.cpu().numpy()[k])
                                print("--- Scal 2 (E/D):",es2[0,i,j,k], scale2.cpu().numpy()[k])

                                print("--- Weigh  (E/D):",ep[0,i,j,k,:], probs.cpu().numpy()[k,:])
                            if y_main_q[0,0,i+5,j+5,k+5] == ey_main_q[0,i,j,k]:
                                pass
                            else:
                                print("FOUND NOT EQUAL in latent feature !!! =====>", i,j,k, '\n', y_main_q[0,0,i+5,j+5,k+5], '\n' ,ey_main_q[0,i,j,k])
                                print("==== TEST FAILED ====")
                                return 0
                    # print("Overall:", time.time()-t_h)
                print("Decoding Channel (%d/192), Time (s): %0.4f" % (i, time.time()-T))
            if DEBUG:
                print("TEST PASSED")
            del hyper, hyper_dec
            y_main_q = y_main_q[0, :, 5:-5, 5:-5, 5:-5]
            rec = image_comp.decoder(y_main_q)

            output_ = torch.clamp(rec, min=0., max=1.0)
            out = output_.data[0].cpu().numpy()
            out = out.transpose(1, 2, 0)
            out_img[H_offset : H_offset + block_H, W_offset : W_offset + block_W, :] = out[:block_H, :block_W, :]
            W_offset += block_W
            if W_offset >= W:
                W_offset = 0
                H_offset += block_H
    out_img = np.round(out_img * 255.0)
    out_img = out_img.astype('uint8')
    img = Image.fromarray(out_img[:H, :W, :])
    img.save(rec_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input Image")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output Bin(encode)/Image(decode)")
    parser.add_argument("-m_dir", "--model_dir", type=str, required=True, help="Directory containing trained models")
    parser.add_argument("-m", "--model", type=int, default=0, help="Model Index [0-5]")
    parser.add_argument('--encode', dest='coder_flag', action='store_true')
    parser.add_argument('--decode', dest='coder_flag', action='store_false')
    parser.add_argument("--block_width", type=int, default=2048, help="coding block width")
    parser.add_argument("--block_height", type=int, default=1024, help="coding block height")
    args = parser.parse_args()
  
    T = time.time()

    if args.coder_flag:
        values = encode(args.input, args.output, args.model_dir, args.model, args.block_width, args.block_height)
        if DEBUG:
            np.save("debug/encode",values[0].cpu().numpy())
            np.save("debug/mean0",values[1].cpu().numpy())
            np.save("debug/mean1",values[2].cpu().numpy())
            np.save("debug/mean2",values[3].cpu().numpy())
            np.save("debug/scale0",values[4].cpu().numpy())
            np.save("debug/scale1",values[5].cpu().numpy())
            np.save("debug/scale2",values[6].cpu().numpy())
            np.save("debug/probs",values[7].cpu().numpy())
            np.save("debug/cdf",values[8])   
    else:
        decode(args.input, args.output, args.model_dir, args.block_width, args.block_height)
                            
    print("Time (s):", time.time() - T)
