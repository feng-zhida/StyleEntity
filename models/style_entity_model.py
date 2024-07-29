import random
from collections import OrderedDict
from os import path as osp

import numpy as np
import torch
from PIL import Image
from basicsr.archs import build_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger
from clip.clip import tokenize
from torch.nn import functional as F
from torchvision.utils import make_grid
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class StyleEntityModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.logger = get_root_logger()
        self.l2_weight = opt['l2_weight']
        self.num_style_feat = opt['network_g']['num_style_feat']
        if self.is_train:
            self.init_training_settings()
        self.mul_factor = opt['mul_factor']
        self.alpha = opt['alpha']

        self.all_text_features = torch.load(opt['text_features'], map_location='cpu').to(self.device)
        self.max_unique_vectors = opt['max_unique_vectors']
        self.eval_prepare()

        self.nondist_validation(None, 0, None, True)
        self.cl = opt['cl']

    @torch.no_grad()
    def eval_prepare(self):
        self.per_feature_sample = self.opt['per_feature_sample']
        self.eval_names = self.opt['eval_names']
        self.num_eval_features = len(self.eval_names)
        self.clip.eval()
        self.eval_text_features = []
        for name in self.eval_names:
            feat = self.clip.module.clip.encode_text(tokenize(name).to(self.device))
            self.eval_text_features.append(feat.expand(self.per_feature_sample, -1))

        self.eval_text = [name.replace(' ', '_') for name in self.eval_names]
        self.fixed_w_plus = self.get_style_code(self.per_feature_sample)
        with open(self.opt['inference_names_path']) as f:
            self.inference_names = f.read().strip().split('\n')[:16]
            
    def init_training_settings(self):
        # define network net_g
        self.net_g = build_network(self.opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        assert load_path is not None, 'pretrained StyleGAN can not be None'

        param_key = self.opt['path'].get('param_key_g', 'params_ema')
        self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        for p in self.net_g.parameters():
            p.requires_grad = False

        self.clip = build_network(self.opt['clip'])
        self.clip = self.model_to_device(self.clip)
        for p in self.clip.parameters():
            p.requires_grad = False

        self.mapper = build_network(self.opt['mapper'])
        self.mapper = self.model_to_device(self.mapper)
        # self.mapper_ema = build_network(self.opt['mapper'])
        # self.mapper_ema = self.model_to_device(self.mapper_ema)
        self.print_network(self.mapper)
        
        # self.model_ema(0)
        # for p in self.mapper_ema.parameters():
        #     p.requires_grad = False

        self.mapper.train()
        # self.mapper_ema.eval()
        self.clip.eval()
        self.net_g.eval()
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        optim_type = train_opt['optim_mapper'].pop('type')
        lr = train_opt['optim_mapper']['lr']
        self.optimizer = self.get_optimizer(optim_type, self.mapper.parameters(), lr)
        self.optimizers.append(self.optimizer)

    def get_style_code(self, batch):
        # 为每个矩阵生成1-4的不同向量数量
        unique_vectors_per_batch = torch.randint(1, self.max_unique_vectors + 1, (batch,))

        # 初始化一个空的列表用于存放每个矩阵的索引
        indices_list = [torch.randint(
            0, unique_vectors, (18,), device=self.device) for unique_vectors in unique_vectors_per_batch]

        # 生成每个矩阵的随机向量并使用索引进行选择
        vectors_list = [torch.randn(
            unique_vectors, 512, device=self.device
        )[indices] for unique_vectors, indices in zip(unique_vectors_per_batch, indices_list)]

        # 将所有矩阵堆叠成一个批量矩阵
        random_z = torch.stack(vectors_list)
        w_plus = self.net_g.module.style_mlp(random_z)
        return w_plus
    
    def feed_data(self, data):
        self.text_features = self.all_text_features[data].to(self.device)
        self.idx = data.to(self.device)

    def optimize_parameters(self, current_iter):
        batch = self.text_features.size(0)
        loss_dict = OrderedDict()

        self.mapper.train()

        w_plus = self.get_style_code(batch)
        delta_w = self.mapper(self.text_features.float(), w_plus.float()) # B, N, D
        w_plus_prime = w_plus + self.alpha * delta_w

        loss_l2 = F.mse_loss(w_plus, w_plus_prime)
        loss_dict['losses/loss_l2'] = loss_l2

        image, _ = self.net_g([w_plus_prime], input_is_latent=True)
        image_features = self.clip.module.encode_image(self.clip.module.processing_image(image))
        if self.cl:
            sim = self.clip(image_features, self.all_text_features)
            label = self.idx
            acc = (sim.max(1)[1] == label).sum().float() / sim.size(0)
            loss_sim = F.cross_entropy(sim, label)
        else:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
            sim = (image_features * text_features).sum(-1) * self.clip.clip.logit_scale.exp()
            loss_sim = (1 - sim / 100).mean()
        loss_dict['losses/loss_sim'] = loss_sim

        loss_dict['details/acc'] = acc
        # loss_dict['details/acc10'] = acc10
        # loss_dict['details/acc50'] = acc50
        loss = loss_sim + loss_l2 * self.l2_weight
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        # self.model_ema(0.995)

        self.log_dict = self.reduce_loss_dict(loss_dict)
        self.log_dict['lr'] = self.optimizer.param_groups[0]['lr']

#     def model_ema(self, decay=0.999):
#         mapper = self.get_bare_model(self.mapper)

#         net_mapper_params = dict(mapper.named_parameters())
#         net_mapper_ema_params = dict(self.mapper_ema.named_parameters())

#         for k in net_mapper_ema_params.keys():
#             net_mapper_ema_params[k].data.mul_(decay).add_(net_mapper_params[k].data, alpha=1 - decay)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    @torch.no_grad()
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        def get_delta_w(text, w_plus):
            text = tokenize(text).expand(w_plus.size(0), -1).cuda()
            text_features = self.clip.module.clip.encode_text(text)
            delta_w = self.mapper(text_features, w_plus)
            return delta_w
        
        assert dataloader is None, 'Validation dataloader should be None.'
        self.mapper.eval()
        for eval_text_feature, eval_text in zip(self.eval_text_features, self.eval_text):
            save_img_path = osp.join(self.opt['path']['visualization'], f'{eval_text}_{current_iter}.jpg')
            
            delta_w = 0
            for name in self.inference_names:
                tgt = f'{eval_text} {name}'
                src = f'{name}'
                delta_w += get_delta_w(tgt, self.fixed_w_plus) - get_delta_w(src, self.fixed_w_plus)
            delta_w = delta_w / len(self.inference_names)

            w_plus_prime = self.fixed_w_plus + 0.3 * delta_w
            image, _ = self.net_g([w_plus_prime], input_is_latent=True, randomize_noise=False)
            image = image.clip(-1, 1)
            image = image / 2 + .5
            image = make_grid(image, 3).permute(1, 2, 0)
            image = (image.cpu().numpy()*255).round().astype(np.uint8)
            Image.fromarray(image).save(save_img_path)
            if tb_logger is not None:
                tb_logger.add_image(f'samples/{eval_text}', image, global_step=current_iter, dataformats='HWC')

    def get_delta_w(self, text, w_plus):
        text = tokenize(text).cuda()
        text_features = self.clip.module.clip.encode_text(text)
        delta_w = self.mapper(text_features, w_plus)
        return delta_w

    def save(self, epoch, current_iter):
        self.save_network([self.mapper], 'mapper', current_iter, param_key=['params'])
