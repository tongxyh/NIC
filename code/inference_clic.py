####       Tong Chen        ####
#### tong@smail.nju.edu.cn  ####
#### NIC-0.1-CLIC v0.2.0   ####
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

@torch.no_grad()
def decode(bin_dir, rec_dir, model_dir, block_width, block_height):
    if DEBUG:
        ey_main_q = np.load("/output/encode.npy")
        # em = np.load("debug/means.npy")
        # es = np.load("debug/scales.npy")
        # ep = np.load("debug/probs.npy")  
        ecdf = np.load("/output/cdf.npy")
    ############### retreive head info ###############
    T = time.time()
    try:
        file_object = open(bin_dir, 'rb')
    except IOError:
        print("file not found:", bin_dir)
    print("Decoding:", bin_dir)
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
                if e >= 0:
                    return ops.index_to_loc(e, H_PAD=block_H_PAD, W_PAD=block_W_PAD), 1
                if e < 0:
                    return ops.index_to_loc(-e, H_PAD=block_H_PAD, W_PAD=block_W_PAD), -1
            for gmm_index in range(N):

                for e in err_means[gmm_index]:
                    loc,v = decode_err(e)
                    loc = list(loc)
                    loc.append(gmm_index)
                    # print(loc, v)
                    err_means_dict[str.encode(str(loc))] = v

                for e in err_scals[gmm_index]:
                    loc,v = decode_err(e)
                    loc = list(loc)
                    loc.append(gmm_index)
                    # print(loc, v)
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
            
            samples_decimal = samples.tolist()
            samples_decimal = [D(x) for x in samples_decimal]
            Decimal_cdf_partial = partial(ops.Decimal_cdf_, x=samples_decimal)
                    
            def partial_cdf(mean, scale):
                return Decimal_cdf_partial(mean=mean, scale=scale)

            # def func_mul_fast(x, sample, factor=factor, Min_Main=Min_Main):
            #     # return int(x * factor) - Min_Main + int(sample)
            #     return int(x * factor)
            
            def func_mul(x, sample, factor=factor, Min_Main=Min_Main):
                return int(x * factor) - Min_Main + sample

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
                # T = time.time()
                for j in range(int(block_H_PAD / 16)):
                    t_h = time.time()

                    # t = time.time()
                    x1 = F.conv3d(y_main_q[:, :, i:i+11, j:j+11, :], weight=c_weight, bias=c_bias)  # [1,24,1,1,1]
                    params_prob = context.conv2(torch.cat((x1, hyper[:, :, i:i+1, j:j+1, :]), dim=1))
                    # print("[1] Context Forward:",time.time()-t)

                    # N mixed gaussian
                    # t = time.time()
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
                    # print("[2] Process:",time.time()-t)

                    # t = time.time()
                    for k in range(int(block_W_PAD / 16)):
                        loc_ = str.encode(str([i,j,k,gmm_index]))
                        if loc_ in err_means_dict:
                            # print(i,j,k,gmm_index)
                            means[gmm_index][k] = means[gmm_index][k] + 0.00001 * err_means_dict[loc_]
                        if loc_ in err_scals_dict:
                            # print(i,j,k,gmm_index)
                            scales[gmm_index][k] = scales[gmm_index][k] + 0.00001 * err_scals_dict[loc_]
                    # print("[3] Round Save:",time.time()-t)

                    ###### rounding error check ###### 
                    # t = time.time()
                    means_  = [list(map(ops.round_check_test, mean)) for mean in means]
                    scales_ = [list(map(ops.round_check_test, scale)) for scale in scales]
                    # print("[4] Round Check:",time.time()-t)

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
                    lowers_ = [list(map(partial_cdf, means_[0], scales_[0]))] # [len(Data), len(samples)]
                    # print("[5] Decimal CDF:",time.time()-t)

                    # print(j, "final mix cdf:",time.time()-t)
                    # t = time.time()
                    # cdf_m = np.array(lowers_[0])
                    # print("[6] CDF Process 1:",time.time()-t)
                    # t = time.time()
                    # cdf_m = cdf_m.astype(np.int) + samples - Min_Main
                    # print("[6] CDF Process 2:",time.time()-t)

                    # t = time.time()
                    samples_ = samples.astype(np.int).tolist()
                    cdf_m = [list(map(func_mul, x, samples_)) for x in lowers_[0]]
                    # print("[6] CDF Process:",time.time()-t)
                    # t = time.time()
                    # cdf_m = [list(map(func_mul_fast, x, samples)) for x in lowers_[0]]
                    # print("[6] CDF Process Fast:",time.time()-t)

                    # t = time.time()
                    pixs = [AE.decode_cdf(x) for x in cdf_m]
                    y_main_q[0,0,i+5,j+5,5:-5] = torch.Tensor(pixs).cuda()
                    if DEBUG:
                        for k in range(int(block_W_PAD / 16)):
                            np.set_printoptions(threshold = 10000, linewidth=200)
                            index = int(i*(block_H_PAD/16*block_W_PAD/16)+j*(block_W_PAD/16)) + k
                            r = ecdf[index,:] == cdf_m[k][:]
                            if r.all():
                                pass
                            else:
                                print("FOUND NOT EQUAL in CDF !!! =====>", i,j,k, '\n', "origin:", ecdf[index,int(ey_main_q[0,i,j,k]-Min_Main):int(ey_main_q[0,i,j,k]-Min_Main+2)] - samples.astype(np.int)[int(ey_main_q[0,i,j,k]-Min_Main):int(ey_main_q[0,i,j,k]-Min_Main+2)] + Min_Main, '\n', "decoded :", cdf_m[k][int(ey_main_q[0,i,j,k]-Min_Main):int(ey_main_q[0,i,j,k]-Min_Main+2)] - samples.astype(np.int)[int(ey_main_q[0,i,j,k]-Min_Main):int(ey_main_q[0,i,j,k]-Min_Main+2)] + Min_Main)
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
                    # print("[7] Arith Decode Fast:",time.time()-t)
                    # t = time.time()
                    # for k in range(int(block_W_PAD / 16)):
                    #     pixs = AE.decode_cdf(cdf_m[k])
                    #     y_main_q[0, 0, i+5, j+5, k+5] = pixs
                    # print("[7] Arith Decode:",time.time()-t)
                    # print("[Overall]:",time.time()-t_h)

            # hash check
            Datas = torch.reshape(y_main_q[:,:,5:-5,5:-5,5:-5], [-1]).cpu().numpy().astype(np.int).tolist()
            hasher = sha224()
            hash_result = sha224(str(Datas).encode("utf-8")).hexdigest()
            with open(bin_dir[:-4]+'.hash', 'rb') as f:
                hash_gt = f.read()
                hash_gt = hash_gt.decode("utf-8")
            print(hash_result,hash_gt)
            if str(hash_result) != hash_gt:
                print("HASH TEST FAILED!", bin_dir)

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
    parser.add_argument("-m", "--model", type=int, default=0, help="Model Index [0-15]")
    parser.add_argument('--encode', dest='coder_flag', action='store_true')
    parser.add_argument('--decode', dest='coder_flag', action='store_false')
    parser.add_argument("--block_width", type=int, default=2048, help="coding block width")
    parser.add_argument("--block_height", type=int, default=1024, help="coding block height")
    args = parser.parse_args()
  
    T = time.time()

    decode(args.input, args.output, args.model_dir, args.block_width, args.block_height)
                            
    print("Time (s):", time.time() - T)
