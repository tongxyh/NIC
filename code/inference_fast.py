import argparse
import math
import os
import struct
import sys
import time
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# import Util.AE as AE
import AE
import Model.model as model
from Model.context_model import Weighted_Gaussian

# avoid memory leak during cpu inference for Pytorch < 1.5
# more details: https://github.com/pytorch/pytorch/issues/27971
os.environ['LRU_CACHE_CAPACITY'] = '1'
GPU = True
# index - [0-15]
models = ["dists_mse_gan_fast2", "mse400", "mse800", "mse1600", "mse3200", "mse6400", "mse12800", "mse25600",
          "msssim4", "msssim8", "msssim16", "msssim32", "msssim64", "msssim128", "msssim320", "msssim640"]


def gaussian(size, sigma=12):  # create guassian window
    gauss = torch.Tensor(
        [math.exp(-(x - size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(size)])
    return gauss


def edge_post_guassian(H, W, img_input, post_net, crop_size_w, crop_size_h):  # post processing
    img_input = img_input.cuda()
    img_input = img_input.unsqueeze(0)
    Block_Num_in_Width = int(np.ceil(W / crop_size_w))
    Block_Num_in_Height = int(np.ceil(H / crop_size_h))
    proc_size = 16
    edge_w_list = [crop_size_w * i for i in range(1, Block_Num_in_Width)]
    edge_h_list = [crop_size_h * i for i in range(1, Block_Num_in_Height)]

    for i in range(0, Block_Num_in_Height):
        for pos in edge_w_list:
            post_input = img_input[:, :, crop_size_h * i:crop_size_h * (i + 1),
                         pos - proc_size:pos + proc_size]
            post_output = post_net(post_input)
            window_size = post_input.shape[3]
            window = gaussian(window_size).cuda().view(1, 1, 1, -1)
            img_input[:, :, crop_size_h * i:crop_size_h * (i + 1), pos - proc_size:pos + proc_size] = \
                post_output * window + post_input * (1 - window)

    for i in range(0, Block_Num_in_Width):
        for pos in edge_h_list:
            post_input = img_input[:, :, pos - proc_size:pos + proc_size,
                         crop_size_w * i:crop_size_w * (i + 1)]
            post_output = post_net(post_input)
            window_size = post_input.shape[2]
            window = gaussian(window_size).cuda().view(1, 1, -1, 1)
            img_input[:, :, pos - proc_size:pos + proc_size, crop_size_w * i:crop_size_w * (i + 1)] = \
                post_output * window + post_input * (1 - window)
    img_input = img_input.squeeze(0)
    return img_input


@torch.no_grad()
def encode(im_dir, out_dir, model_dir, model_index, block_width, block_height):
    file_object = open(out_dir, 'wb')

    M, N2 = 192, 128
    if (model_index == 6) or (model_index == 7) or (model_index == 14) or (model_index == 15):
        M, N2 = 256, 192
    image_comp = model.Image_coding(3, M, N2, M, M // 2)
    # context = Weighted_Gaussian(M)
    ######################### Load Model #########################
    image_comp.load_state_dict(torch.load(
        os.path.join(model_dir, models[model_index] + r'.pkl'), map_location='cpu'))
    # context.load_state_dict(torch.load(
    #     os.path.join(model_dir, models[model_index] + r'p.pkl'), map_location='cpu'))
    if GPU:
        image_comp = image_comp.cuda()
        # context = context.cuda()
    ######################### Read Image #########################
    img = Image.open(im_dir)
    img = np.array(img) / 255.0
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
            img_block_list.append(img[i * block_height:np.minimum((i + 1) * block_height, H),
                                  j * block_width:np.minimum((j + 1) * block_width, W), ...])

    print('check')
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
        print('====> Encoding Image:', im_dir, "%dx%d" % (block_H, block_W), 'to', out_dir,
              " Block Idx: %d" % (Block_Idx))
        Block_Idx += 1

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

            # xp3, params_prob = context(y_main_q, hyper_dec)

        # Main Arith Encode
        Datas = torch.reshape(y_main_q, [-1]).cpu().numpy().astype(np.int).tolist()
        Max_Main = max(Datas)
        Min_Main = min(Datas)
        sample = np.arange(Min_Main, Max_Main + 1 + 1)  # [Min_V - 0.5 , Max_V + 0.5]
        _, c, h, w = y_main_q.shape
        print("Main Channel:", c)
        sample = torch.FloatTensor(np.tile(sample, [1, c, h, w, 1]))
        if GPU:
            sample = sample.cuda()
        # 3 gaussian
        # prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = [
        #     torch.chunk(params_prob, 9, dim=1)[i].squeeze(1) for i in range(9)]
        # del params_prob
        channels = hyper_dec.size()[1]
        mean = hyper_dec[:, :channels // 2, :, :]
        scale = hyper_dec[:, channels // 2:, :, :]
        # process the scale value to positive non-zero
        scale = torch.abs(scale)
        scale[scale < 1e-6] = 1e-6
        m0 = torch.distributions.normal.Normal(mean, scale)
        lower = torch.zeros(1, c, h, w, Max_Main - Min_Main + 2)
        for i in range(sample.shape[4]):
            # print("CDF:", i)
            lower0 = m0.cdf(sample[:, :, :, :, i] - 0.5)
            if GPU:
                lower0 = lower0.cuda()
            lower[:, :, :, :, i] = lower0
        del lower0

        precise = 16
        cdf_m = lower.data.cpu().numpy() * ((1 << precise) - (Max_Main -
                                                              Min_Main + 1))  # [1, c, h, w ,Max-Min+1]
        cdf_m = cdf_m.astype(np.int32) + sample.cpu().numpy().astype(np.int32) - Min_Main
        cdf_main = np.reshape(cdf_m, [len(Datas), -1])

        # Cdf[Datas - Min_V]
        Cdf_lower = list(map(lambda x, y: int(y[x - Min_Main]), Datas, cdf_main))
        # Cdf[Datas + 1 - Min_V]
        Cdf_upper = list(map(lambda x, y: int(
            y[x - Min_Main]), Datas, cdf_main[:, 1:]))
        AE.encode_cdf(Cdf_lower, Cdf_upper, "main.bin")
        FileSizeMain = os.path.getsize("main.bin")
        print("main.bin: %d bytes" % (FileSizeMain))

        # Hyper Arith Encode
        Min_V_HYPER = torch.min(y_hyper_q).cpu().numpy().astype(np.int).tolist()
        Max_V_HYPER = torch.max(y_hyper_q).cpu().numpy().astype(np.int).tolist()
        _, c, h, w = y_hyper_q.shape
        # print("Hyper Channel:", c)
        Datas_hyper = torch.reshape(
            y_hyper_q, [c, -1]).cpu().numpy().astype(np.int).tolist()
        # [Min_V - 0.5 , Max_V + 0.5]
        sample = np.arange(Min_V_HYPER, Max_V_HYPER + 1 + 1)
        sample = np.tile(sample, [c, 1, 1])
        sample_tensor = torch.FloatTensor(sample)
        if GPU:
            sample_tensor = sample_tensor.cuda()
        lower = torch.sigmoid(image_comp.factorized_entropy_func._logits_cumulative(
            sample_tensor - 0.5, stop_gradient=False))
        cdf_h = lower.data.cpu().numpy() * ((1 << precise) - (Max_V_HYPER -
                                                              Min_V_HYPER + 1))  # [N1, 1, Max-Min+1]
        cdf_h = cdf_h.astype(np.int) + sample.astype(np.int) - Min_V_HYPER
        cdf_hyper = np.reshape(np.tile(cdf_h, [len(Datas_hyper[0]), 1, 1, 1]), [
            len(Datas_hyper[0]), c, -1])

        # Datas_hyper [256, N], cdf_hyper [256,1,X]
        Cdf_0, Cdf_1 = [], []
        for i in range(c):
            Cdf_0.extend(list(map(lambda x, y: int(
                y[x - Min_V_HYPER]), Datas_hyper[i], cdf_hyper[:, i, :])))  # Cdf[Datas - Min_V]
            Cdf_1.extend(list(map(lambda x, y: int(
                y[x - Min_V_HYPER]), Datas_hyper[i], cdf_hyper[:, i, 1:])))  # Cdf[Datas + 1 - Min_V]
        AE.encode_cdf(Cdf_0, Cdf_1, "hyper.bin")
        FileSizeHyper = os.path.getsize("hyper.bin")
        print("hyper.bin: %d bytes" % (FileSizeHyper))

        Head_block = struct.pack('2H4h2I', block_H, block_W, Min_Main, Max_Main, Min_V_HYPER, Max_V_HYPER, FileSizeMain,
                                 FileSizeHyper)
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


@torch.no_grad()
def decode(bin_dir, rec_dir, model_dir, block_width, block_height):
    ############### retreive head info ###############
    T = time.time()
    file_object = open(bin_dir, 'rb')
    post_net = model.PostProcNet(128).cuda()
    post_net.load_state_dict(torch.load('./Weights/post_msssim4.pkl'))
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
    image_comp = model.Image_coding(3, M, N2, M, M // 2)
    # context = Weighted_Gaussian(M)
    ######################### Load Model #########################
    image_comp.load_state_dict(torch.load(
        os.path.join(model_dir, models[model_index] + r'.pkl'), map_location='cpu'))
    # context.load_state_dict(torch.load(
    #     os.path.join(model_dir, models[model_index] + r'p.pkl'), map_location='cpu'))
    if GPU:
        image_comp = image_comp.cuda()
        # context = context.cuda()

    for i in range(Block_Num_in_Height):
        for j in range(Block_Num_in_Width):

            Block_head_len = struct.calcsize('2H4h2I')
            bits = file_object.read(Block_head_len)
            [block_H, block_W, Min_Main, Max_Main, Min_V_HYPER, Max_V_HYPER, FileSizeMain,
             FileSizeHyper] = struct.unpack('2H4h2I', bits)

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
            sample = np.arange(Min_V_HYPER, Max_V_HYPER + 1 + 1)
            sample = np.tile(sample, [c_hyper, 1, 1])
            sample_tensor = torch.FloatTensor(sample)
            if GPU:
                sample_tensor = sample_tensor.cuda()
            lower = torch.sigmoid(image_comp.factorized_entropy_func._logits_cumulative(
                sample_tensor - 0.5, stop_gradient=False))
            cdf_h = lower.data.cpu().numpy() * ((1 << precise) - (Max_V_HYPER -
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
            sample = np.arange(Min_Main, Max_Main + 1 + 1)  # [Min_V - 0.5 , Max_V + 0.5]
            sample = torch.FloatTensor(sample)
            sample = torch.FloatTensor(np.tile(sample, [1, c_main, h, w, 1]))
            # print(sample.shape)
            if GPU:
                sample = sample.cuda()

            y_main_q = torch.zeros(1, 1, c_main, h, w)  # 8000x4000 -> 500*250
            if GPU:
                y_main_q = y_main_q.cuda()
            AE.init_decoder("main.bin", Min_Main, Max_Main)

            # 3 gaussian
            channels = hyper_dec.size()[1]
            mean = hyper_dec[:, :channels // 2, :, :]
            scale = hyper_dec[:, channels // 2:, :, :]

            # keep the weight  summation of prob == 1

            # process the scale value to positive non-zero
            scale = torch.abs(scale)
            scale[scale < 1e-6] = 1e-6
            # 3 gaussian distributions
            m0 = torch.distributions.normal.Normal(mean, scale)
            lower = torch.zeros(1, c_main, h, w, Max_Main - Min_Main + 2)
            for i in range(sample.shape[4]):
                lower0 = m0.cdf(sample[:, :, :, :, i] - 0.5)
                if GPU:
                    lower0 = lower0.cuda()
                lower[:, :, :, :, i] = lower0

            cdf_m = lower.data.cpu().numpy() * ((1 << precise) - (Max_Main -
                                                                  Min_Main + 1))  # [1, c, h, w ,Max-Min+1]
            cdf_m = cdf_m.astype(np.int) + \
                    sample.cpu().numpy().astype(np.int) - Min_Main
            Recons = []
            cdf_m = cdf_m.squeeze(0)
            for i in range(c_main):
                for j in range(int(block_H_PAD / 16)):
                    for k in range(int(block_W_PAD / 16)):
                        Recons.append(AE.decode_cdf(cdf_m[i, j, k, :].tolist()))
            y_main_q = torch.reshape(torch.Tensor(
                Recons), [1, c_main, int(block_H_PAD / 16), int(block_W_PAD / 16)])
            y_main_q = y_main_q.cuda()
            rec = image_comp.decoder(y_main_q)
            out = rec.data[0].cpu().numpy()
            out = out.transpose(1, 2, 0)
            out_img[H_offset: H_offset + block_H, W_offset: W_offset + block_W, :] = out[:block_H, :block_W, :]
            W_offset += block_W
            if W_offset >= W:
                W_offset = 0
                H_offset += block_H

    out_img = torch.Tensor(out_img).cuda()
    out_img = out_img.permute(2, 0, 1).contiguous()
    out_img = edge_post_guassian(H=H, W=W, img_input=out_img, post_net=post_net, crop_size_h=block_height,
                                 crop_size_w=block_width)
    output_ = torch.clamp(out_img, min=0., max=1.0)
    out_img = output_.data.cpu().numpy()
    out_img = out_img.transpose(1, 2, 0)
    out_img = np.round(out_img * 255.0)
    out_img = out_img.astype('uint8')
    out_img = out_img[:H, :W, :]
    img = Image.fromarray(out_img)
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
        encode(args.input, args.output, args.model_dir, args.model, args.block_width, args.block_height)
    else:
        decode(args.input, args.output, args.model_dir, args.block_width, args.block_height)
    print("Time (s):", time.time() - T)
