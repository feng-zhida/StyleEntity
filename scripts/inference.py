import os
import random

import clip
import yaml
import torch

import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

import archs
from psp_models.psp import pSp
from clip import tokenize
from basicsr.archs import build_network


print('load modules complete')


def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256), InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def inversion_image(model, image, T):
    image = T(image).unsqueeze(0)
    latent = model(image).squeeze(0)
    return latent


def run_alignment(image_path):
    import dlib
    from utils.alignment import align_face
    predictor = dlib.shape_predictor("pretrained_models/shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def get_delta_w(text, w_plus, clip_model, encoder):
    text = tokenize(text).expand(w_plus.size(0), -1).cuda()
    text_features = clip_model.encode_text(text)
    delta_w = encoder(text_features, w_plus)
    return delta_w


def prepare_models(args):
    # build model
    with open(args.opt) as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)
    clip_model, _ = clip.load('ViT-B/16')
    mapper = build_network(opt['encoder'])
    net_g = build_network(opt['network_g'])

    # to cuda
    clip_model.cuda().eval()
    mapper.cuda().eval()
    net_g.cuda().eval()

    # load model weight
    net_g.load_state_dict(
        torch.load(opt['path']['pretrain_network_g'], map_location='cpu')['params_ema'])
    mapper.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['params_ema'])

    # setting inversion model
    inversion_model = pSp('pretrained_models/e4e_ffhq_encode.pt', 1024)
    inversion_model.cuda().eval()
    return clip_model, mapper, net_g, inversion_model


@torch.no_grad()
def main(args):
    # prepare models
    clip_model, mapper, net_g, inversion_model = prepare_models(args)

    # getting orignal image
    T = get_transforms()
    if args.align:
        image = run_alignment(args.image_path).convert('RGB')
    else:
        image = Image.open(args.image_path).convert('RGB')
    image = T(image).unsqueeze(0).cuda()

    # inverse image to W+
    w_plus = inversion_model(image)

    # smooth_num is the N in paper
    if args.smooth_num > 0:
        # random choice N names
        with open(args.names) as f:
            names = f.read().strip().split('\n')
        random.shuffle(names)
        names = names[:args.smooth_num]

        # Manipulation Vector Normalization
        delta_w = 0
        for name in names:
            tgt_desc = args.target_text.format(name)
            src_desc = name
            delta_w += get_delta_w(tgt_desc, w_plus, clip_model, mapper) \
                       - get_delta_w(src_desc, w_plus, clip_model, mapper)
        delta_w = delta_w / len(names)
    else:
        tgt_desc = f'{args.front_text} {args.back_text}'
        delta_w = get_delta_w(tgt_desc, w_plus, clip_model, mapper)
    ori, _ = net_g([w_plus], input_is_latent=True, randomize_noise=False)
    w_plus = w_plus + args.alpha * delta_w
    out, _ = net_g([w_plus], input_is_latent=True, randomize_noise=False)

    img = to_img(torch.cat([ori, out], -1))
    if args.save_path is None:
        img.show()
    else:
        img.save(args.save_path)


def to_img(out):
    out = out.squeeze(0).permute(1, 2, 0).clip(-1, 1) / 2 + .5
    out = out.cpu().numpy() * 255
    out = Image.fromarray(out.astype(np.uint8))
    return out


if __name__ == '__main__':
    import argparse

    arguments = argparse.ArgumentParser()
    arguments.add_argument('--checkpoint', type=str, default='pretrained_models/style_entity_pretrained_celebs.pth')
    arguments.add_argument('--opt', type=str, default='options/base_setting.yml')
    arguments.add_argument('--target_text', type=str, required=True)
    arguments.add_argument('--align', action='store_true')

    arguments.add_argument('--smooth_num', type=int, default=16)
    arguments.add_argument('--image_path', type=str, required=True)
    arguments.add_argument('--num_style_feat', type=int, default=512)
    arguments.add_argument('--alpha', type=float, default=0.15)
    arguments.add_argument('--seed', type=int, default=None)
    arguments.add_argument('--save_path', type=str, default=None)
    arguments.add_argument('--names', type=str, default='datasets/named_entity_for_inference.txt')
    args = arguments.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    main(args)
