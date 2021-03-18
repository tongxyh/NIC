####       Tong Chen        ####
#### tong@smail.nju.edu.cn  ####
#### NIC-0.1-CLIC v0.1.22   ####
import argparse
import math
import os
import struct
import sys
import time
from functools import partial
from multiprocessing import Pool
from decimal import *
import decimal
D = Decimal
decimal.getcontext().prec = 16
from hashlib import sha224,sha256,md5
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
DEBUG = False

# index - [0-15]
models = ["mse200", "mse400", "mse800", "mse1600", "mse3200", "mse6400", "mse12800", "mse25600",
          "msssim4", "msssim8", "msssim16", "msssim32", "msssim64", "msssim128", "msssim320", "msssim640"]

# def partial_cdf(args):
#     return Decimal_cdf_partial (mean=args[0], scale=args[1])

@torch.no_grad()
def encode(im_dir, out_dir, model_dir, model_index, block_width, block_height):
    file_object = open(out_dir, 'wb')
    
    M, N2 = 192, 128
    if (model_index == 6) or (model_index == 7) or (model_index == 14) or (model_index == 15):
        M, N2 = 256, 192
    image_comp = model.Image_coding(3, M, N2, M, M//2)
    context = Weighted_Gaussian(M)
    ######################### Load Model #########################
    print("====> Loading Model: ", os.path.join(model_dir, models[model_index] + r'.pkl'))
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
    N = 1
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
        hasher = sha224()
        hash_result = sha224(str(Datas).encode("utf-8")).hexdigest()
        with open(out_dir[:-4]+'.hash', 'wb') as f:
            f.write(str(hash_result).encode("utf-8"))

        Max_Main = max(Datas)
        Min_Main = min(Datas)
        sample = np.arange(Min_Main, Max_Main+1+1).tolist()  # [Min_V - 0.5 , Max_V + 0.5]
        _, c, h, w = y_main_q.shape
        print("Main Channel:", c)
        # sample = torch.FloatTensor(np.tile(sample, [1, c, h, w, 1]))
        # if GPU:
        #     sample = sample.cuda()

        # N mixed gaussian
        #params = [torch.chunk(params_prob, 3*N, dim=1)[i].squeeze(1) for i in range(3*N)]
        params = [torch.chunk(params_prob, 9, dim=1)[i].squeeze(1) for i in range(3*N)]
        del params_prob
        # keep the weight summation of prob == 1
        probs = torch.stack(params[0:3*N:3], dim=-1)
        probs = F.softmax(probs, dim=-1)
        
        probs = [probs[:,:,:,:,i] for i in range(N)]  
        means = params[1:3*N:3]
        scales = [torch.abs(x) for x in params[2:3*N:3]] # clamp scale value to positive
        
        # t = time.time()
        precise = 16
        samples = np.arange(Min_Main, Max_Main+1+1)  # [Min_V - 0.5 , Max_V + 0.5]
        
        means_  = [torch.reshape(mean, [-1]).cpu().numpy().tolist() for mean in means] # [N, len(Data)]
        scales_ = [torch.reshape(scale, [-1]).cpu().numpy().tolist() for scale in scales]
        probs_  = [torch.reshape(prob,[-1]).cpu().numpy().tolist() for prob in probs]
        
        if DEBUG:
            # pass
            i_,j_,k_ = 137, 16, 67
            index = i_ * (block_H_PAD/16*block_W_PAD/16) + j_ * (block_W_PAD/16) + k_
            print("Before Round:", means[0][0,i_,j_,k_], means_[0][int(index)], scales[0][0,i_,j_,k_], scales_[0][int(index)])

        ###### rounding error check ######
        means_  = [list(map(ops.round_check, mean)) for mean in means_]
        scales_ = [list(map(ops.round_check, scale)) for scale in scales_]
        # probs_ = [list(map(Decimal, prob)) for prob in probs_]
        
        if DEBUG:
            # pass
            # # def loc_to_index(i=0,j=0,k=0):
            # #     pass
            index = i_ * (block_H_PAD/16*block_W_PAD/16) + j_ * (block_W_PAD/16) + k_
            print("After Round:", means_[0][int(index)], scales_[0][int(index)])

        def err_index(dat_):
            return [[i,x] for i,x in enumerate(dat_) if x[1] != 0]

        err_means = [err_index(x) for x in means_]
        err_scals = [err_index(x) for x in scales_]
        
        # err_probs = [err_index(x) for x in probs_]
        err_probs = [[], [], []]
        
        err_len = sum(list(map(len, err_means))) + sum(list(map(len, err_scals))) + sum(list(map(len, err_probs)))
        
        if err_len > 0:
            print("[WARNING] Found %d possible rounding errors"%(err_len))
            if DEBUG:
                for i,x in enumerate(err_means):
                    print("==== Mean", i)
                    for j in x:
                        print(ops.index_to_loc(j[0], H_PAD=block_H_PAD, W_PAD=block_W_PAD), j[0], j[1]) # [index, value]
                for i,x in enumerate(err_scals):
                    print("==== Scal", i)
                    for j in x:
                        print(ops.index_to_loc(j[0], H_PAD=block_H_PAD, W_PAD=block_W_PAD), j[0], j[1]) # [index, value]
                for i,x in enumerate(err_probs):
                    print("==== Prob", i)
                    for j in x:
                        print(ops.index_to_loc(j[0], H_PAD=block_H_PAD, W_PAD=block_W_PAD), j[0], j[1]) # [index, value]

        else:
            print("[ROUND CHECK] PASSED!")
        
        ## TODO: save error correction info
        def format_err(e):
            return e[0]*e[1][1]

        errm_format = [list(map(format_err,e)) for e in err_means]
        errs_format = [list(map(format_err,e)) for e in err_scals]
        with open(out_dir[:-4]+'_errs.npy', 'wb') as f:
            np.save(f, np.array(errm_format))
            np.save(f, np.array(errs_format))

        # print("Time (s):", time.time() - t)
        cdf_ = []
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
        def cdf_sample(l,p,d,s):
            # p0 = Decimal(p0).quantize(Decimal("0.0000001"))
            # p1 = Decimal(p1).quantize(Decimal("0.0000001"))
            # p2 = Decimal(p2).quantize(Decimal("0.0000001"))
            prec_f = D("0.001")
            flag   = 0

            # if p[0] <= p[2] and p[1] <= p[2]:
            #     v = (p[0]*l[0]+p[1]*l[1]+(1-p[0]-p[1])*l[2]) * factor / 10
            #     if d == s or s - d == 1:
            #         # if p > factor - 1 or p < 1:
            #         #     return p
            #         # rounding check
            #         if int(v - prec_f) != int(v) or int(v + prec_f) != int(v):
            #             # print("[WARNING] possible round error in Probs", p)
            #             flag = 1
            #     return p, flag
            
            # if p[0] <= p[1] and p[2] <= p[1]:
            #     p = (p[0]*l[0]+(1-p[0]-p[2])*l[1]+p[2]*l[2]) * factor / 10
            #     if d == s or s - d == 1:
            #         # if p > factor - 1 or p < 1:
            #         #     return p
            #         # rounding check
            #         if int(p - prec_f) != int(p) or int(p + prec_f) != int(p):
            #             # print("[WARNING] possible round error in Probs", p)
            #             flag = 1
            #     return p, flag
            # if p1 <= p0 and p2 <= p0:
            #     p =((1-p1-p2)*l0+p1*l1+p2*l2) * factor / 10
            #     if d == s or s - d == 1:
            #         # if p > factor - 1 or p < 1:
            #         #     return p
            #         # rounding check
            #         if int(p - prec_f) != int(p) or int(p + prec_f) != int(p):
            #             # print("[WARNING] possible round error in Probs", p)
            #             flag = 1
            #     return p, flag
            # for p_, l_ in zip(p,l):
            #     print(p_,l_)
            
            v = sum([D(p_)*l_ for p_,l_ in zip(p,l)])
            if v >= 1:
                return factor, flag
            quan_step = 1
            v = v * factor / quan_step
            if d == s or s - d == 1:
                if int(v - prec_f) != int(v) or int(v + prec_f) != int(v):
                    # print("[WARNING] possible round error in Probs", p)
                    flag = 1
            # print("[WARNING] Found Equal prob weights [0 1 2]: ", p0,p1,p2)
            return quan_step*int(v), flag

        probs_ = [p for p in zip(*probs_)]
        lower, b = [],[]
        bappend = b.append
        for sample in samples:
            # t_s = time.time()
            sample_ = np.repeat(sample, len(Datas)).tolist()
            lowers_ = [list(map(ops.Decimal_cdf, sample_, mean_, scale_)) for mean_, scale_ in zip(means_, scales_)] # [[lower0], [lower1], [lower2]]
            lowers_ = [l for l in zip(*lowers_)]  # [N, len(Data)] - > [len(Data), N]
            cdf_ = [cdf_sample(l,p,d,s) for l,p,d,s in zip(lowers_, probs_, Datas, sample_)] # [len(Data),2]
            lower.append([x[0] for x in cdf_])
        #     for j,x in enumerate(cdf_):
        #         if x[1] > 0:
        #             c_index = j // int(block_H_PAD/16*block_W_PAD/16)
        #             h_index = (j % int(block_H_PAD/16*block_W_PAD/16)) // int(block_W_PAD/16)
        #             w_index = (j % int(block_H_PAD/16*block_W_PAD/16)) % int(block_W_PAD/16)
        #             bappend([c_index, h_index, w_index])
        #             # print(b[-1],x[0])
        # print("[WARNING] Found %d possible rounding errors"%(len(b)))
            # print(sample, "Time (s):", time.time() - t_s)

        # DEBUG: 
        # c_d, h_d, w_d = 138,30,30
        # index_ = c_d*int(block_W_PAD/16*block_H_PAD/16) + h_d*int(block_W_PAD/16) + w_d
        # print("Error Location:", index_, c_d, h_d, w_d, Datas[index_], y_main_q[0,c_d,h_d,w_d])
        # for x,y in zip(means, means_):
        #     print(x[0,c_d,h_d,w_d], y[index_])
        # for x,y in zip(scales, scales_):
        #     print(x[0,c_d,h_d,w_d], y[index_])
        # print("[GET YOU]",y_main_q[0,c_d,h_d,w_d],lower_[int(y_main_q[0,c_d,h_d,w_d])-Min_Main][index_], lower_[int(y_main_q[0,c_d,h_d,w_d])-Min_Main+1][index_])

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
        return y_main_q, means, scales, probs, cdf_main
    return 0



@torch.no_grad()
def decode(bin_dir, rec_dir, model_dir, block_width, block_height):
    if DEBUG:
        ey_main_q = np.load("debug/encode.npy")
        # em = np.load("debug/means.npy")
        # es = np.load("debug/scales.npy")
        # ep = np.load("debug/probs.npy")  
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
    N = 1

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
            factor = (1 << precise) - (Max_Main - Min_Main + 1)

            def cdf_sample(l,p):
                # p0 = Decimal(p0).quantize(Decimal("0.0000001"))
                # p1 = Decimal(p1).quantize(Decimal("0.0000001"))
                # p2 = Decimal(p2).quantize(Decimal("0.0000001"))
                # if p0 <= p2 and p1 <= p2:
                #     return int((p0*l0+p1*l1+(1-p0-p1)*l2) * factor/10)*10
                # if p0 <= p1 and p2 <= p1:
                #     return int((p0*l0+(1-p0-p2)*l1+p2*l2) * factor/10)*10
                # if p1 <= p0 and p2 <= p0:
                #     return int(((1-p1-p2)*l0+p1*l1+p2*l2) * factor/10)*10
                # print("[WARNING] Found Equal prob weights [0 1 2]: ", p0,p1,p2)
                quan_step = 1
                v = sum([D(p_)*l_ for p_,l_ in zip(p,l)])
                if v >=1:
                    return factor
                return quan_step*int(v*factor/quan_step)

            block_H_PAD = int(tile * np.ceil(block_H / tile))
            block_W_PAD = int(tile * np.ceil(block_W / tile))

            ## LOAD ERROR CORRECTTION INFO
            with open(bin_dir[:-4]+'_errs.npy', 'rb') as f:
                err_means = np.load(f).tolist() # [N, Datas]
                err_scals = np.load(f).tolist()

            err_means_dict, err_scals_dict = {}, {}
            
            def decode_err(e, H_PAD=block_H_PAD, W_PAD=block_W_PAD):
                if e > 0:
                    return ops.index_to_loc(e, H_PAD=block_H_PAD, W_PAD=block_W_PAD), 1
                if e < 0:
                    return ops.index_to_loc(-e, H_PAD=block_H_PAD, W_PAD=block_W_PAD), -1
            for gmm_index in range(N):

                for e in err_means[gmm_index]:
                    loc,v = decode_err(e)
                    loc = list(loc)
                    loc.append(gmm_index)
                    print(loc, v)
                    err_means_dict[str.encode(str(loc))] = v

                for e in err_scals[gmm_index]:
                    loc,v = decode_err(e)
                    loc = list(loc)
                    loc.append(gmm_index)
                    print(loc, v)
                    err_scals_dict[str.encode(str(loc))] = v
           

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

            Decimal_cdf_partial = partial(ops.Decimal_cdf_, x=samples.tolist())
                    
            def partial_cdf(mean, scale):
                return Decimal_cdf_partial(mean=mean, scale=scale)
            def func(l,p):
                # [N, len(samples)], [N]
                partial_cdf = partial(cdf_sample, p=p)
                l = [l_ for l_ in zip(*l)] #[len(sample), N]
                return list(map(partial_cdf, l))

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

                    # N mixed gaussian
                    # params = [torch.chunk(params_prob, 3*N, dim=1)[i].squeeze(1) for i in range(3*N)]
                    params = [torch.chunk(params_prob, 9, dim=1)[i].squeeze(1) for i in range(3*N)]
                    
                    params = [x[0,0,0,:] for x in params]
                    # keep the weight summation of prob == 1
                    probs = torch.stack(params[0:3*N:3], dim=-1)
                    probs = F.softmax(probs, dim=-1)
                    
                    # process the scale value to positive non-zero
                    means = params[1:3*N:3]
                    scales = [torch.abs(x) for x in params[2:3*N:3]] # clamp scale value to positive

                    means  = [mean.cpu().numpy().tolist() for mean in means]                  
                    scales = [scale.cpu().numpy().tolist() for scale in scales]
                    probs = [probs[:,i].cpu().numpy().tolist() for i in range(N)]  #(N, len(Data))

                    # for gmm_index in range(N):
                    #     for k in range(int(block_W_PAD / 16)):
                    #         loc_ = str.encode(str([i,j,k,gmm_index]))
                    #         if loc_ in err_means_dict:
                    #             print(i,j,k,gmm_index)
                    #             means[gmm_index][k] = means[gmm_index][k] + 0.00001 * err_means_dict[loc_]
                    #         if loc_ in err_scals_dict:
                    #             print(i,j,k,gmm_index)
                    #             scales[gmm_index][k] = scales[gmm_index][k] + 0.00001 * err_scals_dict[loc_]

                    # 3 gaussian distributions

                    ###### rounding error check ######
                    # TODO: decoder version round_check 
                    # t = time.time()
                    means_  = [list(map(ops.round_check_test, mean)) for mean in means]
                    scales_ = [list(map(ops.round_check_test, scale)) for scale in scales]
                    # print("Round Check:",time.time()-t)

                    # TODO: round_check of probs
                    # probs0_ = list(map(ops.round_check, probs0_))
                    # probs1_ = list(map(ops.round_check, probs1_))

                    ## TODO: Multiprocessing
                    
                    # t = time.time()
                    # lowers_ = []
                    # p = Pool(2)
                    # for mean_, scale_ in zip(means_, scales_):
                    #     lowers_.append(p.map(partial_cdf, [mean_, scale_]))
                    #     # p.close()
                    #     # p.join()
                    # print("Mutli:",time.time()-t)    
                    # t = time.time()
                    lowers_ = [list(map(partial_cdf, mean_, scale_)) for mean_,scale_ in zip(means_, scales_)] # [N, len(Data), len(samples)]
                    # print("Decimal CDF:",time.time()-t)
                    
                    # t = time.time()   
                    # GMM
                    # probs_ = [p for p in zip(*probs)]    # [len(Data), N]
                    # lowers_ = [l for l in zip(*lowers_)] # [len(Data), N, len(samples)], [len(Data), N]
                    # lower = [func(l,p) for l,p in zip(lowers_, probs_)] # [len(Data), len(sample)]
                    
                    # CLIC single gaussain
                    lower = lowers_[0]
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
                                print("FOUND NOT EQUAL in CDF !!! =====>", i,j,k, '\n', "decoded:", ecdf[index,int(ey_main_q[0,i,j,k]-Min_Main):int(ey_main_q[0,i,j,k]-Min_Main+2)] - samples.astype(np.int)[int(ey_main_q[0,i,j,k]-Min_Main):int(ey_main_q[0,i,j,k]-Min_Main+2)] + Min_Main, '\n', "origin :", cdf_m[k,int(ey_main_q[0,i,j,k]-Min_Main):int(ey_main_q[0,i,j,k]-Min_Main+2)] - samples.astype(np.int)[int(ey_main_q[0,i,j,k]-Min_Main):int(ey_main_q[0,i,j,k]-Min_Main+2)] + Min_Main)
                                print("=== Latent Features (E/D):",i,j,k,ey_main_q[0,i,j,k],y_main_q[0,0,i+5,j+5,k+5])
                                for x in means:
                                    print(x[k])
                                for x in scales: 
                                    print(x[k])
                                # print("--- Mean 0 (E/D):",em0[0,i,j,k], mean0.cpu().numpy()[k])
                                # print("--- Scal 0 (E/D):",es0[0,i,j,k], scale0.cpu().numpy()[k])
                                # print("--- Weigh  (E/D):",ep[0,i,j,k,:], probs.cpu().numpy()[k,:])

                            if y_main_q[0,0,i+5,j+5,k+5] == ey_main_q[0,i,j,k]:
                                pass
                            else:
                                print("FOUND NOT EQUAL in latent feature !!! =====>", i,j,k, '\n', y_main_q[0,0,i+5,j+5,k+5], '\n' ,ey_main_q[0,i,j,k])
                                print("==== TEST FAILED ====")
                                return 0
                    # print("Overall:", time.time()-t_h)
                # print("Decoding Channel (%d/192), Time (s): %0.4f" % (i, time.time()-T))
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
            np.save("/output/encode",values[0].cpu().numpy())
            # np.save("debug/mean0",values[1].cpu().numpy())
            # np.save("debug/scale0",values[2].cpu().numpy())
            # np.save("debug/probs",values[3].cpu().numpy())
            np.save("/output/cdf",values[4])
    else:
        decode(args.input, args.output, args.model_dir, args.block_width, args.block_height)
                            
    print("Time (s):", time.time() - T)
