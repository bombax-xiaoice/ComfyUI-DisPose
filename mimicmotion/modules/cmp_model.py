from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput

import mimicmotion.modules.cmp.models as cmp_models
import mimicmotion.modules.cmp.utils as cmp_utils

import yaml
import os
import torchvision.transforms as transforms


class ArgObj(object):
    def __init__(self):
        pass


class CMP(nn.Module):
    def __init__(self, configfn, load_iter):
        super().__init__()
        args = ArgObj()
        with open(configfn) as f:
            config = yaml.full_load(f)
        for k, v in config.items():
            setattr(args, k, v)
        setattr(args, 'load_iter', load_iter)
        setattr(args, 'exp_path', os.path.dirname(configfn))
        
        self.model = cmp_models.__dict__[args.model['arch']](args.model, dist_model=False)
        self.model.load_state("{}/checkpoints".format(args.exp_path), args.load_iter, False)        
        self.model.switch_to('eval')
        
        self.data_mean = args.data['data_mean']
        self.data_div = args.data['data_div']
        
        self.img_transform = transforms.Compose([
            transforms.Normalize(self.data_mean, self.data_div)])
        
        self.args = args
        self.fuser = cmp_utils.Fuser(args.model['module']['nbins'], args.model['module']['fmax'])
        torch.cuda.synchronize()

    def run(self, image, sparse, mask):
        dtype = image.dtype
        image = image * 2 - 1
        self.model.set_input(image.float(), torch.cat([sparse, mask], dim=1).float(), None)
        try:
            cmp_output = self.model.model(self.model.image_input.to(torch.float16), self.model.sparse_input.to(torch.float16))
        except:
            cmp_output = self.model.model(self.model.image_input.to(torch.float32), self.model.sparse_input.to(torch.float32))
        flow = self.fuser.convert_flow(cmp_output)
        if flow.shape[2] != self.model.image_input.shape[2]:
            flow = nn.functional.interpolate(
                flow, size=self.model.image_input.shape[2:4],
                mode="bilinear", align_corners=True)

        return flow.to(dtype)  # [b, 2, h, w]