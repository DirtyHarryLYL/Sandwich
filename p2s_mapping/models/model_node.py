import math
import pickle
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict
from .base import BaseCLIP, ModifiedResNet, TextTransformer, VisionTransformer, LayerNorm, Transformer
from .text_encoder import SentenceTransformer, Hierarchical_Transformer

import geoopt.geoopt.manifolds.poincare.math as pmath
import lorentz as L 
# lorentz from meru: Desai, Karan, et al. "Hyperbolic image-text representations." International Conference on Machine Learning. PMLR, 2023.

class TLayer(nn.Module):
    def __init__(self, num, in_features, out_features, bias=True):
        super(TLayer, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.num          = num
        self.weight       = nn.Parameter(torch.Tensor(num, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features, num))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # input (bz, in, num_node); weight (num_node, out, in); out (bz, out, num_node)
        output = torch.einsum('ilk,kjl->ijk', [input, self.weight.type(input.dtype)]) + self.bias.type(input.dtype)
        return output

    def extra_repr(self):
        return 'num={}, in_features={}, out_features={}, bias={}'.format(
            self.num, self.in_features, self.out_features, self.bias is not None
        )


class Parallel_Trans(nn.Module):
    def __init__(self, num_classes=290):
        super(Parallel_Trans, self).__init__()

        self.num_classes = num_classes
        self.trans = OrderedDict()
        LAYER_SIZE = [512, 512, 512]
        for i in range(1, len(LAYER_SIZE)):
            self.trans['trans%d' % i] = TLayer(self.num_classes, LAYER_SIZE[i - 1], LAYER_SIZE[i], 1)
            if i < len(LAYER_SIZE) - 1:
                self.trans['bn%d' % i]   = nn.BatchNorm1d(LAYER_SIZE[i])
                self.trans['relu%d' % i] = nn.ReLU()
        self.trans = nn.Sequential(self.trans)

    def forward(self, input):
        input = input.permute(0, 2, 1) # input: (bz, num_node, 512) -> (bz, 512, num_node)  
        output = self.trans(input) # (bz, 512, num_node)
        output = output.permute(0, 2, 1)

        return output


class model_for_node_LoL3(BaseCLIP):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # visual encoder
        self.embed_dim = config.dim
        self.visual_config = config.visual
        if self.visual_config.type == 'ViT':
            self.visual_net = VisionTransformer(
                input_resolution=self.visual_config.resolution,
                patch_size=self.visual_config.patch_size,
                width=self.visual_config.width,
                layers=self.visual_config.layers,
                heads=self.visual_config.heads,
                output_dim=self.embed_dim,
            )
        elif self.visual_config.type == 'ResNet':
            self.visual_net = ModifiedResNet(
                layers=self.visual_config.layers,
                output_dim=self.embed_dim,
                heads=self.visual_config.heads,
                input_resolution=self.visual_config.resolution,
                width=self.visual_config.width
            )
        else:
            raise NotImplementedError

        # text encoder
        self.text_config = config.text
        if self.text_config.type == 'transformer':
            self.text_net = TextTransformer(
                embed_dim=self.embed_dim,
                vocab_size=self.text_config.vocab_size,
                context_length=self.text_config.context_length,
                width=self.text_config.width,
                heads=self.text_config.heads,
                layers=self.text_config.layers
            )
        elif self.text_config.type == 'sentence-transformer':
            self.text_net = SentenceTransformer(
                embed_dim=self.embed_dim,
                vocab_size=self.text_config.vocab_size,
                context_length=self.text_config.context_length,
                width=self.text_config.width,
                heads=self.text_config.heads,
                layers=self.text_config.layers
            )
        else:
            raise NotImplementedError

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.config.get("with_L3", False):
            
            # self.logit_scale_node = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            # disentangle
            if not self.config.get("wo_disentangle", False):
                self.num_action = 290
                self.disentangle_weight = nn.Parameter(torch.Tensor(
                    self.num_action, self.embed_dim, self.embed_dim))
                nn.init.kaiming_uniform_(self.disentangle_weight, a=math.sqrt(5))
                # self.trans = Parallel_Trans(num_classes=self.num_action) # disentangle
                if self.config.get("parallel_trans", True):
                    self.trans = Parallel_Trans(num_classes=290)
                else:
                    self.disentangle_bias = nn.Parameter(torch.Tensor(290, config.dim))
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.disentangle_weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(self.disentangle_bias, -bound, bound)

            image_encoder = [
                    ("fc1", nn.Linear(self.embed_dim, self.embed_dim)),
                    ("act1", nn.ReLU()),
                    ("fc2", nn.Linear(self.embed_dim, self.embed_dim)),
                ]
            self.image_encoder = nn.Sequential(OrderedDict(image_encoder))

            if self.config.get("meru", False):
                curv_init = 1.0
                learn_curv = True
                entail_weight = 0.01
                # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
                self.curv = nn.Parameter(
                    torch.tensor(curv_init).log(), requires_grad=learn_curv
                )
                # When learning the curvature parameter, restrict it in this interval to
                # prevent training instability.
                self._curv_minmax = {
                    "max": math.log(curv_init * 10),
                    "min": math.log(curv_init / 10),
                }
                self.entail_weight = entail_weight

                # Learnable scalars to ensure that image/text features have an expected
                # unit norm before exponential map (at initialization).
                self.visual_alpha = nn.Parameter(torch.tensor(self.embed_dim**-0.5).log())
                self.textual_alpha = nn.Parameter(torch.tensor(self.embed_dim**-0.5).log())
                ### ---meru---###
        
            # geometry-hyper
            if self.config.get("hyper_weight", False):
                reduce_dim = [
                    ("fc1", nn.Linear(self.embed_dim, self.embed_dim)),
                    ("act1", nn.ReLU()),
                    ("fc2", nn.Linear(self.embed_dim, 128)),  # (512, 128)
                    ("act2", nn.ReLU()),
                    ("fc3", nn.Linear(128, 2)),
                ]
                self.reduce_dim = nn.Sequential(OrderedDict(reduce_dim))

                ckpt = torch.load("./Data/verbnet.pth.best")
                ckpt["objects"] = np.array(ckpt["objects"])
                ckpt["embeddings"] = np.array(ckpt["embeddings"])

                verbnet_topology = pickle.load(open("./Data/verbnet_topology_898.pkl", "rb"))
                Father2Son, objects = verbnet_topology["Father2Son"], verbnet_topology["objects"]
                objects = np.array(objects)

                idxs = []
                for i, obj in enumerate(objects):
                    idx = np.where(ckpt["objects"] == obj)[0][0]
                    idxs.append(idx)
                idxs = np.unique(np.array(idxs))
                assert len(idxs) == len(objects)

                poincare_embed = ckpt["embeddings"][idxs]
                mapping_node_index = pickle.load(
                    open("./Data/mapping_node_index/704_to_290.pkl", "rb"))
                poincare_embed = poincare_embed[mapping_node_index]
                self.poincare_embed = torch.from_numpy(poincare_embed).cuda().float()
                self.poincare_embed = pmath.expmap0(self.poincare_embed)  # (num_node, 2)
            
            if self.config.get("flag_weight", False):
                from .flag import flag_loss
                self.loss_func_flag = flag_loss()
            
            if self.config.get("wo_semantic", False):
                self.num_action = 290
                node_classifier = [
                    ("fc1", nn.Linear(self.embed_dim, self.embed_dim)),
                    ("act1", nn.ReLU()),
                    ("fc2", nn.Linear(self.embed_dim, self.num_action)),
                ]
                self.node_classifier = nn.Sequential(OrderedDict(node_classifier))

        # if self.config.get("with_Lo", True):
        #     self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_CE = nn.CrossEntropyLoss()
        self.loss_BCE = nn.BCEWithLogitsLoss()

        self.load_checkpoint()
    
    def load_checkpoint(self):
        if self.config.initial.weight is not None:
            print("load checkpoint...")
            state_dict = torch.load(self.config.initial.weight, map_location='cpu')
            if 'state' in state_dict.keys():
                state_dict = state_dict['state']
            if 'module' in [k for k in state_dict.keys()][0]:
                state_dict = {k[7:]:v for k,v in state_dict.items()} # e.g., "module.text_net.transformer.resblocks.11.ln_2.weight" -> "text_net.transformer.resblocks.11.ln_2.weight"
            # state_dict = {k:v for k,v in state_dict.items() if 'disentangle' not in k and 'trans.' not in k}
            if not self.config.initial.get("load_text_pretrain", True): # not load pretrained text encoder
                state_dict = {k:v for k,v in state_dict.items() if 'text_net' not in k}
            if not self.config.initial.get("load_visual_pretrain", True): # not load pretrained visual encoder
                state_dict = {k:v for k,v in state_dict.items() if 'visual_net' not in k}
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print("missing_keys", missing_keys)
            print("unexpected_keys", unexpected_keys)

        if self.config.get("freeze", False):
            if self.config.freeze.get("freeze_text_encoder", False):
                print("###### Setting: freeze_text_encoder. ######")
                for name, param in self.named_parameters():
                    if "text_net" in name:
                        param.requires_grad_(False)
            if self.config.freeze.get("freeze_visual_encoder", False):
                print("###### Setting: freeze_visual_encoder. ######")
                for name, param in self.named_parameters():
                    if "visual_net" in name:
                        param.requires_grad_(False)
            if self.config.freeze.get("freeze_image_mlp_encoder", False):
                print("###### Setting: freeze_image_mlp_encoder. ######")
                for name, param in self.named_parameters():
                    if "image_encoder" in name:
                        param.requires_grad_(False)
            # for name, param in self.named_parameters():
            #     if "logit_scale" in name:
            #         param.requires_grad_(False)
        
        # for name, param in self.named_parameters():
        #     print(name, param.requires_grad)

    def forward(self, batch, pseudo_label_soft = False):
        
        output = {}

        image = batch["image"]
        if isinstance(image, dict) and "whole" in image.keys():
            image = image["whole"]
        elif len(image.shape) == 5:
            image = image.flatten(0, 1)
        bz = image.shape[0]
        image_features = self.encode_image(image)
        
        if self.config.get("with_Lo", True):
            text_action = batch["text_action"]
            text_features = self.encode_text(text_action)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) # -> (bz, 512)
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * (image_features / image_features.norm(dim=-1, keepdim=True)) @ text_features.t()
            logits_per_text = logits_per_image.t()
            output["s"] = logits_per_image
            output["p"] = F.softmax(logits_per_image, dim=1)

        #if "labels_n" in batch.keys() and self.config.get("with_Lo", True):
        if self.config.get("with_Lo",True):
            # image_features = self.image_encoder(image_features)
            if True:#"labels_n" in batch.keys():
                labels = torch.arange(text_action.shape[0]).cuda()
                output['Lo'] = (self.loss_CE(logits_per_image, labels) + self.loss_CE(logits_per_text, labels)) / 2

                # output['p'] = F.softmax(logits_per_image, dim=1)
                # if "labels_r" in batch.keys():
                # output['L4'] = self.loss_func_1(logits_per_image, batch["labels_r"])
                output['loss'] = output['Lo']

        if self.config.get("with_L3", False):
            
            image_features = self.image_encoder(image_features)
            if not self.config.get("wo_disentangle", False):
                image_features = torch.einsum("ik,jkl->ijl",
                                            image_features,
                                            self.disentangle_weight.type(image_features.dtype))  # (bz, num_action, 512)
                # image_features = self.trans(image_features)
                if self.config.get("parallel_trans", True):
                    image_features = self.trans(image_features)
                else:
                    image_features  = image_features + self.disentangle_bias.type(image_features.dtype)

            if self.config.get("meru", False):
                ### ---meru---###
                image_feats = image_features
                image_feats = image_feats * self.visual_alpha.exp()
                image_feats = L.exp_map0(image_feats, self.curv.exp())
                ### ---meru---###
            
            if self.config.get("meru", False):
                text_feats = self.encode_text(batch["text_node"])
                text_feats = text_feats * self.textual_alpha.exp()
                text_feats = L.exp_map0(text_feats, self.curv.exp())
                self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
                _curv = self.curv.exp()
                score = -L.pairwise_dist(image_feats.view(-1, self.embed_dim), text_feats, _curv) # (bz*290, dim), (290,dim)->(bz*290, 290)
                # Clamp temperature such that logits are not scaled more than 100x.
                # ln(100) = ~4.6052
                score = score.view(bz, 290, 290)
                score = torch.stack([torch.diag(score[i]) for i in range(score.shape[0])], 0)
                self.logit_scale_node.data = torch.clamp(self.logit_scale_node.data, max=4.6052)
                _scale = self.logit_scale_node.exp()
                score = _scale * score
                output["s3"] = score
                output["p3"] = torch.sigmoid(score)

                if self.training:
                    output["L3"] = self.loss_BCE(score, batch["labels_n"])
                    ### ---meru---###
                    # Hyperbolic entailment loss: text should entail matching image.
                    image_feats_select = image_feats[batch["labels_n"].bool()]
                    text_feats = text_feats.unsqueeze(0).expand(bz, 290, self.embed_dim)
                    text_feats_select = text_feats[batch["labels_n"].bool()]
                    
                    _angle = L.oxy_angle(text_feats_select, image_feats_select, _curv)
                    _aperture = L.half_aperture(text_feats_select, _curv)
                    entailment_loss = torch.clamp(_angle - _aperture, min=0)
                    entailment_loss = entailment_loss.mean()
                    if self.entail_weight > 0:
                        # print(output["L3"], entailment_loss)
                        output["L3"] = output["L3"] + self.entail_weight * entailment_loss
                    output["loss"] = output["L3"]
                return output
            
            if not self.config.get("wo_semantic", False):
                text_node = batch["text_node"]
                text_features = self.text_net(text_node)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True) # -> (bz, 512)
                score = (image_features / image_features.norm(dim=-1, keepdim=True)) @ text_features.t()
            else:
                score = self.node_classifier(image_features)

            if "labels_n" in batch.keys():
                if self.config.get("flag_weight", False):
                    output['L_flag'] = self.loss_func_flag(score, batch['labels_n']) * self.config.flag_weight

            if not self.config.get("wo_disentangle", False):
                score = torch.stack([torch.diag(score[i]) for i in range(score.shape[0])], 0)
            score = self.logit_scale.exp() * score
            # score = self.logit_scale_node.exp() * score
            pred = torch.sigmoid(score)
            output['p'] = pred

            if "labels_n" in batch.keys():
                if "unsure_mask" in batch.keys():
                    output['L3'] = (batch["unsure_mask"] * F.binary_cross_entropy_with_logits(score, batch["labels_n"], reduce=False)).mean()
                # elif "labels_soft_mask" in batch.keys():
                #     loss_func = nn.BCEWithLogitsLoss()
                #     output['loss'] = loss_func(score, batch["labels_n"])
                else:
                    output['L3'] = self.loss_BCE(score, batch["labels_n"])
                output['loss'] = output['L3']
            
                if self.config.get("hyper_weight", False):
                    # compare poincare distance, position
                    image_features_hyper = self.reduce_dim(image_features.flatten(0, 1)).reshape(image_features.shape[0], -1, 2)  # (bz, num_node, 2)
                    image_features_hyper = image_features_hyper / image_features_hyper.norm(dim=-1, keepdim=True)
                    self.poincare_embed = self.poincare_embed / self.poincare_embed.norm(dim=-1, keepdim=True)
                    # (bz, num_node, 2) * (2, num_node) -> (bz, num_node, num_node)
                    score_hyper = torch.matmul(image_features_hyper, self.poincare_embed.t())
                    score_hyper = torch.stack([torch.diag(score_hyper[i]) for i in range(score_hyper.shape[0])], 0)  # (bz, num_node)
                    output['L_hyper'] = self.loss_BCE(score_hyper, batch["labels_n"])
                    output["L3"] += output['L_hyper']
                
                if self.config.get("flag_weight", False):
                    output["L3"] += output['L_flag']

        if "labels_n" in batch.keys():
            if self.config.get("with_Lo", True):
                if self.config.get("with_L3", False):
                    output["loss"] = output["L3"] + output["Lo"]
            else:
                output["loss"] = output["L3"]
            
        return output
