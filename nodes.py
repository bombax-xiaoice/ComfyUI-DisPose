import os
import os.path
import folder_paths
import sys
comfy_path = os.path.dirname(folder_paths.__file__)
dispose_path = os.path.join(comfy_path, "custom_nodes", "ComfyUI-DisPose")
dispose_modelpath = os.path.join(dispose_path, "pretrained_weights")
sys.path.append(dispose_path)
from huggingface_hub import hf_hub_download, snapshot_download, try_to_load_from_cache, _CACHED_NO_EXIST
import comfy.utils
import random
import torch
import numpy as np
import typing
import math
from tqdm import tqdm
from comfy import model_management, latent_formats
import latent_preview
import accelerate
import accelerate.utils.memory
import accelerate.hooks
import PIL.Image
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms import PILToTensor
import torchvision
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
import inspect

from einops import rearrange, repeat
import cv2

from mimicmotion.utils.geglu_patch import patch_geglu_inplace
patch_geglu_inplace()

from constants import ASPECT_RATIO
from mimicmotion.modules.controlnet import ControlNetSVDModel
from mimicmotion.pipelines.pipeline_ctrl import Ctrl_Pipeline
from mimicmotion.utils.loader import MimicMotionModel
from mimicmotion.dwpose.preprocess import get_image_pose
from mimicmotion.dwpose.util import draw_pose
from mimicmotion.modules.cmp_model import CMP
from mimicmotion.utils.dift_utils import SDFeaturizer
from mimicmotion.utils.utils import points_to_flows, bivariate_Gaussian, sample_inputs_flow, get_cmp_flow, pose2track
from mimicmotion.dwpose.dwpose_detector import DWposeDetector
from mimicmotion.dwpose.wholebody import Wholebody

class DualUserCpuOffloadHook(accelerate.hooks.UserCpuOffloadHook):
    def __init__(self, hook1, model1, hook2, model2):
        self.hook1 = hook1
        self.model1 = model1
        self.hook2 = hook2
        self.model2 = model2
    
    def offload(self):
        if self.model1.device.type!="meta":
            self.hook1.init_hook(self.model1)
        if self.model2.device.type!="meta":
            self.hook2.init_hook(self.model2)
    
    def remove(self):
        accelerate.hooks.remove_hook_from_module(self.model1)
        accelerate.hooks.remove_hook_from_module(self.model2)

class CallbackWrapper:
    def __init__(self, pipe, device, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        try:
            setattr(pipe, 'load_device', device)
            setattr(pipe, 'model', typing.NewType('PseudoModel',typing.Generic))
            setattr(pipe.model, 'latent_format', latent_formats.SD15())
            self.callback = latent_preview.prepare_callback(pipe, num_inference_steps)
        except:
            self.callback = None

    def do(self, pipe, step, timestep, args):
        self.args = args
        latents = args["latents"].squeeze(0) if args and "latents" in args else None
        self.callback(step, latents, None, self.num_inference_steps)
        return self

    def __call__(self):
        if self.callback:
            return lambda pipe,step,timestep,args:self.do(pipe,step,timestep,args)
        else:
            return None
    
    def pop(self, key, defaultvalue):
        return self.args[key] if key in self.args else defaultvalue
    
class DisPoseLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_dir":("STRING",{"default":"pretrained_weights/stable-video-diffusion-img2vid-xt-1-1"}),
                "ckpt_file":("STRING",{"default":"pretrained_weights/MimicMotion_1-1.pth"}),
                "controlnet_file":("STRING",{"default":"pretrained_weights/DisPose.pth"}),
                "dift_model_dir":("STRING",{"default":"pretrained_weights/stable-diffusion-v1-5"}),
                "cmp_file":("STRING",{"default":"mimicmotion/modules/cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/checkpoints/ckpt_iter_42000.pth.tar"}),
                "dwdetector_file":("STRING",{"default":"pretrained_weights/DWPose/yolox_l.onnx"}),
                "dwpose_file":("STRING",{"default":"pretrained_weights/DWPose/dw-ll_ucoco_384.onnx"}),
                "use_fp16":("BOOLEAN",{"default":True}),
            },
        }

    RETURN_NAMES = ("pipe",)
    RETURN_TYPES = ("DisPosePipeline",)
    FUNCTION = "run"
    CATEGORY = "DisPose"

    def run(self, base_model_dir, ckpt_file, controlnet_file, dift_model_dir, cmp_file, dwdetector_file, dwpose_file, use_fp16):
        pbar = comfy.utils.ProgressBar(6)
        device = model_management.unet_offload_device().type
        
        if not os.path.isabs(base_model_dir) and not os.path.exists(base_model_dir):
            base_model_dir = os.path.join(dispose_path, base_model_dir)
        if not os.path.exists(base_model_dir):
            for file in [
                'feature_extractor/preprocessor_config.json',
                'image_encoder/config.json',
                'image_encoder/model.fp16.safetensors' if use_fp16 else 'image_encoder/model.safetensors',
                'scheduler/scheduler_config.json',
                'unet/config.json',
                'unet/diffusion_pytorch_model.fp16.safetensors' if use_fp16 else 'unet/diffusion_pytorch_model.safetensors',
                'vae/config.json',
                'vae/diffusion_pytorch_model.fp16.safetensors' if use_fp16 else 'vae/diffusion_pytorch_model.safetensors']:
                hf_hub_download('stabilityai/stable-video-diffusion-img2vid-xt-1-1', file, local_dir=base_model_dir)
        mimicmotion_models = MimicMotionModel(base_model_dir)
        pbar.update(1)
        
        if not os.path.isabs(ckpt_file) and not os.path.exists(ckpt_file):
            ckpt_file = os.path.join(dispose_path, ckpt_file)
        if not os.path.exists(ckpt_file):
            hf_hub_download('tencent/MimicMotion', 'MimicMotion_1-1.pth', local_dir=os.path.dirname(ckpt_file))
            if os.path.basename(ckpt_file) != 'MimicMotion_1-1.pth':
                os.symlink(os.path.join(os.path.dirname(ckpt_file), 'MimicMotion_1-1.pth'), ckpt_file)
        mimicmotion_models.load_state_dict(torch.load(ckpt_file, map_location=device), strict=False)
        pbar.update(1)
        
        controlnet = ControlNetSVDModel.from_unet(mimicmotion_models.unet).to(device=mimicmotion_models.unet.device)
        if not os.path.isabs(controlnet_file) and not os.path.exists(controlnet_file):
            controlnet_file = os.path.join(dispose_path, controlnet_file)
        if not os.path.exists(controlnet_file):
            hf_hub_download('lihxxx/DisPose', 'DisPose.pth', local_dir=os.path.dirname(controlnet_file))
            if os.path.basename(controlnet_file) != 'DisPose.pth':
                os.symlink(os.path.join(os.path.dirname(controlnet_file), 'DisPose.pth'), controlnet_file)
        controlnet.load_state_dict(torch.load(controlnet_file, map_location=device),strict=False)
        pipeline = Ctrl_Pipeline(
            vae=mimicmotion_models.vae.to(dtype=torch.float16 if use_fp16 else torch.float32), 
            image_encoder=mimicmotion_models.image_encoder.to(dtype=torch.float16 if use_fp16 else torch.float32), 
            unet=mimicmotion_models.unet.to(dtype=torch.float16 if use_fp16 else torch.float32), 
            controlnet=controlnet.to(dtype=torch.float16 if use_fp16 else torch.float32),
            scheduler=mimicmotion_models.noise_scheduler,
            feature_extractor=mimicmotion_models.feature_extractor, 
            pose_net=mimicmotion_models.pose_net.to(dtype=torch.float16 if use_fp16 else torch.float32)
        )
        pbar.update(1)
        
        if not os.path.isabs(dift_model_dir) and not os.path.exists(dift_model_dir):
            dift_model_dir = os.path.join(dispose_path, dift_model_dir)
        if not os.path.exists(dift_model_dir):
            for file in [
                'feature_extractor/preprocessor_config.json',
                'safety_checker/config.json',
                'safety_checker/model.fp16.safetensors',
                'scheduler/scheduler_config.json',
                'text_encoder/config.json',
                'text_encoder/model.fp16.safetensors',
                'tokenizer/merges.txt',
                'tokenizer/special_tokens_map.json',
                'tokenizer/tokenizer_config.json',
                'tokenizer/vocab.json',
                'unet/config.json',
                'unet/diffusion_pytorch_model.fp16.safetensors',
                'vae/config.json',
                'vae/diffusion_pytorch_model.fp16.safetensors',
                'model_index.json',
                'v1-5-pruned-emaonly.ckpt']:
                hf_hub_download('stable-diffusion-v1-5/stable-diffusion-v1-5', file, local_dir=dift_model_dir)
        dift_model = SDFeaturizer(sd_id = dift_model_dir if os.path.isabs(dift_model_dir) or os.path.exists(dift_model_dir) else os.path.join(dispose_path, dift_model_dir), weight_dtype=torch.float16)
        if not 'cuda' in model_management.text_encoder_offload_device().type:
            dift_model.pipe = dift_model.pipe.to(model_management.text_encoder_offload_device())
            dift_model.null_prompt_embeds = dift_model.null_prompt_embeds.to(model_management.text_encoder_offload_device())
        setattr(pipeline, 'dift_model', dift_model)
        pbar.update(1)
        
        if not os.path.isabs(cmp_file) and not os.path.exists(cmp_file):
            cmp_file = os.path.join(dispose_path, cmp_file)
        if not os.path.exists(cmp_file):
            hf_hub_download('MyNiuuu/MOFA-Video-Hybrid', 'models/cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/checkpoints/ckpt_iter_42000.pth.tar', local_dir=dispose_modelpath)
            os.makedirs(os.path.dirname(cmp_file), exist_ok=True)
            os.symlink(os.path.join(dispose_modelpath, 'models/cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/checkpoints/ckpt_iter_42000.pth.tar'), cmp_file)
        if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(cmp_file)), 'config.yaml')):
            os.symlink(os.path.join(dispose_path, 'mimicmotion/modules/cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow_back/config.yaml'), os.path.join(os.path.dirname(os.path.dirname(cmp_file)), 'config.yaml'))
        cmp = CMP(
            os.path.join(os.path.dirname(os.path.dirname(cmp_file)), 'config.yaml'),
            42000
        )
        cmp.model.model.to(device=device)
        cmp.requires_grad_(False)
        setattr(pipeline, 'cmp', cmp)
        pbar.update(1)
        
        if not os.path.isabs(dwdetector_file) and not os.path.exists(dwdetector_file):
            dwdetector_file = os.path.join(dispose_path, dwdetector_file)
            if not os.path.exists(dwdetector_file):
                hf_hub_download('yzd-v/DWPose', 'yolox_l.onnx', local_dir=os.path.dirname(dwdetector_file))
                if os.path.basename(dwdetector_file) != 'yolox_l.onnx':
                    os.symlink(dwdetector_file, os.path.join(os.path.dirname(dwdetector_file), 'yolox_l.onnx'))
        if not os.path.isabs(dwpose_file) and not os.path.exists(dwpose_file):
            dwpose_file = os.path.join(dispose_path, dwpose_file)
            if not os.path.exists(dwpose_file):
                hf_hub_download('yzd-v/DWPose', 'dw-ll_ucoco_384.onnx', local_dir=os.path.dirname(dwpose_file))
                if os.path.basename(dwpose_file) != 'dw-ll_ucoco_384.onnx':
                    os.symlink(dwpose_file, os.path.join(os.path.basename(dwpose_file), 'dw-ll_ucoco_384.onnx'))
        dwprocessor = DWposeDetector(model_det=dwdetector_file, model_pose=dwpose_file, device=device)
        dwprocessor.pose_estimation = Wholebody(*dwprocessor.args)
        setattr(pipeline, 'dwprocessor', dwprocessor)
        pbar.update(1)
        
        return (pipeline,)

class DisPoseSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe":("DisPosePipeline",{"tooltip":"output from (Down)Loader_DisPose"}),
                "ref_video":("IMAGE",{"tooltip":"video or concatenated images as reference poses, recommend to use VideoHelperSuite"}),
                "ref_image":("IMAGE",{"tooltip":"one image as driving image, keeping everything but pose"}),
                "resolution":("INT",{"default":576,"min":256,"max":640,"step":64, "tooltip":"shorter edge pixels, aspect ratio is 16/9"}),
                "seed":("INT",{"default":42,"tooltip":"set control_after_generation as fixed to make result reproducible"}),
                "tile_size":("INT",{"default":16,"min":1,"max":640,"step":1,"tooltip":"larger size makes better motion quality but requires more vram"}),
                "tile_overlap":("INT",{"default":6,"min":0,"max":64,"step":1,"tooltip":"larger overlap makes better motion quality but increase processing time, vram remains the same"}),
                "num_inference_steps":("INT",{"default":25,"min":1,"max":100,"step":1}),
                "noise_aug_strength":("FLOAT",{"default":0,"min":0.0,"max":1.0,"step":0.05,"tooltip":"larger value makes better motion quality but less alike to ref_image"}),
                "min_guidance_scale":("FLOAT",{"default":2.0,"min":0.1,"max":10.0,"step":0.1}),
                "max_guidance_scale":("FLOAT",{"default":2.0,"min":0.1,"max":10.0,"step":0.1}),
                "cpu_offload":(["none","model","sequential"],{"default":"model","tooltip":"sequence is slowest but consume the least VRAM, model also spare some vram and slightly slower than none"}),
           },
        }

    RETURN_NAMES = ("latent","ref_image","poses")
    RETURN_TYPES = ("LATENT","IMAGE","IMAGE")
    FUNCTION = "run"
    CATEGORY = "DisPose"

    def get_videoframes_pose(self, dwprocessor, frames: torch.Tensor, ref_image: np.ndarray, device=torch.device('cpu')):
        ref_pose = dwprocessor(ref_image)
        ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        ref_keypoint_id = [i for i in ref_keypoint_id \
            if len(ref_pose['bodies']['subset']) > 0 and ref_pose['bodies']['subset'][0][i] >= .0]
        ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]

        height, width, _ = ref_image.shape

        # read input video
        #vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        #sample_stride *= max(1, int(vr.get_avg_fps() / 24))

        #frames = vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()
        detected_poses = [dwprocessor(frm) for frm in tqdm(frames, desc="DWPose")]
        dwprocessor.release_memory()

        detected_bodies = np.stack(
            [p['bodies']['candidate'] for p in detected_poses if p['bodies']['candidate'].shape[0] == 18])[:,
                        ref_keypoint_id]
        # compute linear-rescale params
        ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
        fh, fw, _ = frames[0].shape #vr[0].shape
        ax = ay / (fh / fw / height * width)
        bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
        a = np.array([ax, ay])
        b = np.array([bx, by])
        output_pose = []
        # pose rescale 
        body_point = []
        face_point = []
        for detected_pose in detected_poses:
            detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
            detected_pose['faces'] = detected_pose['faces'] * a + b
            detected_pose['hands'] = detected_pose['hands'] * a + b
            im = draw_pose(detected_pose, height, width)
            output_pose.append(np.array(im))
            body_point.append(detected_pose['bodies'])
            face_point.append(detected_pose['faces'])
        return np.stack(output_pose), body_point, face_point
    
    def preprocess_data(self, ref_video, ref_image, pipe, resolution=576, device=torch.device('cpu'), cpu_offload=False):
        image_pixels = (ref_image.permute(0, 3, 1, 2).squeeze(0)*255.0).to(torch.uint8)
        h, w = image_pixels.shape[-2:]
        ############################ compute target h/w according to original aspect ratio ###############################
        if h>w:
            w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
        elif h==w:
            w_target, h_target = resolution, resolution
        else:
            w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution
        h_w_ratio = float(h) / float(w)
        if h_w_ratio < h_target / w_target:
            h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
        else:
            h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
        image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
        image_pixels = center_crop(image_pixels, [h_target, w_target])
        # h_target, w_target = image_pixels.shape[-2:]
        image_pixels = image_pixels.permute((1, 2, 0)).numpy()
        ##################################### get video flow #################################################
        transform = transforms.Compose(
            [
            
            transforms.Resize((h_target, w_target), antialias=None), 
            transforms.CenterCrop((h_target, w_target)), 
            transforms.ToTensor()
            ]
        )
        
        ref_img = transform(PIL.Image.fromarray(image_pixels))

        ##################################### get image&video pose value #################################################
        # image_pose, ref_point = get_image_pose(image_pixels)
        height, width, _ = image_pixels.shape
        ref_point = pipe.dwprocessor(image_pixels)
        image_pose = np.array(draw_pose(ref_point, height, width))
        # end get_image_pose
        ref_point_body, ref_point_head = ref_point["bodies"], ref_point["faces"]
        video_pose, body_point, face_point = self.get_videoframes_pose(pipe.dwprocessor, (ref_video*255.0).to(dtype=torch.uint8).numpy(force=True), image_pixels, device=device)
        body_point_list = [ref_point_body] + body_point
        face_point_list = [ref_point_head] + face_point

        pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
        image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
        
        #dift_model = SDFeaturizer(sd_id = dift_model_path, weight_dtype=torch.float16)
        model_management.unload_all_models()
        model_management.soft_empty_cache()
        accelerate.utils.memory.clear_device_cache(True)
        try:
            if cpu_offload==True or cpu_offload=="model":
                pipe.dift_model.pipe.maybe_free_model_hooks() if hasattr(pipe.dift_model.pipe, '_all_hooks') else pipe.dift_model.pipe.enable_model_cpu_offload()
            elif cpu_offload=="sequential" and pipe.dift_model.pipe.device!="meta":
                pipe.dift_model.pipe.enable_sequential_cpu_offload()
            elif pipe.dift_model.pipe.device != model_management.text_encoder_device():
                pipe.dift_model.pipe = pipe.dift_model.pipe.to(model_management.text_encoder_device())
            if pipe.dift_model.null_prompt_embeds.device != model_management.text_encoder_device():
                pipe.dift_model.null_prompt_embeds = pipe.dift_model.null_prompt_embeds.to(model_management.text_encoder_device())
            category="human"
            prompt = f'photo of a {category}'
            dift_ref_img = (image_pixels / 255.0 - 0.5) *2
            dift_ref_img = torch.from_numpy(dift_ref_img).to(device, torch.float16)
            dift_feats = pipe.dift_model.forward(dift_ref_img, prompt=prompt, t=[261,0], up_ft_index=[1,2], ensemble_size=8)
            dift_feats = (dift_feats[0].to(device=device), dift_feats[1].to(device=device))
        finally:
            for component in filter(lambda v:isinstance(v, torch.nn.Module), pipe.dift_model.pipe.components.values()):
                accelerate.hooks.remove_hook_from_submodules(component)
            if pipe.dift_model.pipe.device != model_management.text_encoder_offload_device() and pipe.dift_model.pipe.device.type != "meta":
                pipe.dift_model.pipe = pipe.dift_model.pipe.to(model_management.text_encoder_offload_device())
            if pipe.dift_model.null_prompt_embeds.device != model_management.text_encoder_offload_device():
                pipe.dift_model.null_prompt_embeds = pipe.dift_model.null_prompt_embeds.to(model_management.text_encoder_offload_device())

        model_length = len(body_point_list)
        traj_flow = points_to_flows(body_point_list, model_length, h_target, w_target)
        blur_kernel = bivariate_Gaussian(kernel_size=199, sig_x=20, sig_y=20, theta=0, grid=None, isotropic=True)

        for i in range(0, model_length-1):
            traj_flow[i] = cv2.filter2D(traj_flow[i], -1, blur_kernel)

        traj_flow = rearrange(traj_flow, "f h w c -> f c h w") 
        traj_flow = torch.from_numpy(traj_flow)
        traj_flow = traj_flow.unsqueeze(0)
        '''
        cmp = CMP(
            './mimicmotion/modules/cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/config.yaml',
            42000
        ).to(device)
        cmp.requires_grad_(False)
        '''
        pc, ph, pw = ref_img.shape
        poses, poses_subset = pose2track(body_point_list, ph, pw)
        poses = torch.from_numpy(poses).permute(1,0,2)
        poses_subset = torch.from_numpy(poses_subset).permute(1,0,2)

        # pdb.set_trace()
        val_controlnet_image, val_sparse_optical_flow, \
        val_mask, val_first_frame_384, \
            val_sparse_optical_flow_384, val_mask_384 = sample_inputs_flow(ref_img.unsqueeze(0).float(), poses.unsqueeze(0), poses_subset.unsqueeze(0))

        fb, fl, fc, fh, fw = val_sparse_optical_flow.shape

        pipe.cmp.model.model.cuda()
        val_controlnet_flow = get_cmp_flow(
            pipe.cmp, 
            val_first_frame_384.unsqueeze(0).repeat(1, fl, 1, 1, 1).to(device=next(pipe.cmp.model.model.parameters()).device), 
            val_sparse_optical_flow_384.to(device=next(pipe.cmp.model.model.parameters()).device), 
            val_mask_384.to(device=next(pipe.cmp.model.model.parameters()).device)
        ).to(device=device)
        pipe.cmp.model.model.cpu()

        if fh != 384 or fw != 384:
            scales = [fh / 384, fw / 384]
            val_controlnet_flow = F.interpolate(val_controlnet_flow.flatten(0, 1), (fh, fw), mode='nearest').reshape(fb, fl, 2, fh, fw)
            val_controlnet_flow[:, :, 0] *= scales[1]
            val_controlnet_flow[:, :, 1] *= scales[0]
        
        vis_flow = val_controlnet_flow[0]

        return torch.from_numpy(pose_pixels.copy()) / 127.5 - 1, torch.from_numpy(image_pixels) / 127.5 - 1, val_controlnet_flow, val_controlnet_image, body_point_list, dift_feats, traj_flow
    
    def run(self,pipe,ref_video,ref_image,resolution,cpu_offload,seed,tile_size,tile_overlap,num_inference_steps,noise_aug_strength,min_guidance_scale,max_guidance_scale):
        _execution_device = model_management.get_torch_device()
        device = model_management.unet_offload_device() if cpu_offload else _execution_device
        
        pose_pixels, image_pixels, controlnet_flow, controlnet_image, point_list, dift_feats, traj_flow = self.preprocess_data(
            ref_video, ref_image, pipe, resolution=resolution, device=device, cpu_offload=cpu_offload
        )
        #image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
        callback = CallbackWrapper(pipe, device, num_inference_steps)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)    
        if cpu_offload==True or cpu_offload=="model":
            model_management.unload_all_models()
            model_management.soft_empty_cache()
            accelerate.utils.memory.clear_device_cache(True)
            pipe._exclude_from_cpu_offload = ['vae', 'feature_extractor']
            pipe.model_cpu_offload_seq = 'image_encoder->pose_net->controlnet->unet'
            try:
                pipe.maybe_free_model_hooks() if hasattr(pipe, "_all_hooks") else pipe.enable_model_cpu_offload()
                pipe.controlnet._hf_hook.prev_module_hook= DualUserCpuOffloadHook(pipe.pose_net._hf_hook, pipe.pose_net, pipe.unet._hf_hook, pipe.unet) #controlnet and unet are called interleaving
                pipe._all_hooks.append(accelerate.cpu_offload_with_hook(pipe.vae.encoder, execution_device=_execution_device)[1])
                pipe._all_hooks.append(accelerate.cpu_offload_with_hook(pipe.vae.quant_conv, execution_device=_execution_device)[1])
                if pipe.vae.device != model_management.vae_offload_device():
                    pipe.vae.to(model_management.vae_offload_device())
                latent = pipe(
                    [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5], image_pose=pose_pixels, num_frames=pose_pixels.size(0),
                    tile_size=tile_size, tile_overlap=tile_overlap,
                    height=pose_pixels.shape[-2], width=pose_pixels.shape[-1], fps=7,
                    controlnet_flow=controlnet_flow, controlnet_image=controlnet_image, point_list=point_list, dift_feats=dift_feats, traj_flow=traj_flow,
                    noise_aug_strength=noise_aug_strength, num_inference_steps=num_inference_steps,
                    generator=generator, min_guidance_scale=min(min_guidance_scale,max_guidance_scale),
                    max_guidance_scale=max(min_guidance_scale,max_guidance_scale), output_type="latent", device=device, callback_on_step_end=callback()
                ).frames.cpu()
            finally:
                for hook in pipe._all_hooks if hasattr(pipe, '_all_hooks') else []:
                    hook.offload()
                    hook.remove()
                delattr(pipe, '_all_hooks')
        elif cpu_offload=="sequential":
            model_management.unload_all_models()
            model_management.soft_empty_cache()
            accelerate.utils.memory.clear_device_cache(True)
            pipe._exclude_from_cpu_offload = [] #["image_encoder",]
            try:
                pipe.enable_sequential_cpu_offload()
                # hack fix to recover position_ids from meta to cpu 
                pipe.image_encoder.vision_model.embeddings.register_buffer("position_ids", torch.arange(pipe.image_encoder.vision_model.embeddings.num_positions).expand((1, -1)), persistent=False)
                latent = pipe(
                    [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5], image_pose=pose_pixels, num_frames=pose_pixels.size(0),
                    tile_size=tile_size, tile_overlap=tile_overlap,
                    height=pose_pixels.shape[-2], width=pose_pixels.shape[-1], fps=7,
                    controlnet_flow=controlnet_flow, controlnet_image=controlnet_image, point_list=point_list, dift_feats=dift_feats, traj_flow=traj_flow,
                    noise_aug_strength=noise_aug_strength, num_inference_steps=num_inference_steps,
                    generator=generator, min_guidance_scale=min(min_guidance_scale,max_guidance_scale),
                    max_guidance_scale=max(min_guidance_scale,max_guidance_scale), output_type="latent", device=device, callback_on_step_end=callback()
                ).frames.cpu()
            finally:
                for component in filter(lambda v:isinstance(v, torch.nn.Module), pipe.components.values()):
                    accelerate.hooks.remove_hook_from_submodules(component)
        else:
            with torch.autocast(device_type=_execution_device.type):
                latent = pipe(
                    [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5], image_pose=pose_pixels, num_frames=pose_pixels.size(0),
                    tile_size=tile_size, tile_overlap=tile_overlap,
                    height=pose_pixels.shape[-2], width=pose_pixels.shape[-1], fps=7,
                    controlnet_flow=controlnet_flow, controlnet_image=controlnet_image, point_list=point_list, dift_feats=dift_feats, traj_flow=traj_flow,
                    noise_aug_strength=noise_aug_strength, num_inference_steps=num_inference_steps,
                    generator=generator, min_guidance_scale=min(min_guidance_scale,max_guidance_scale),
                    max_guidance_scale=max(min_guidance_scale,max_guidance_scale), output_type="latent", device=_execution_device, callback_on_step_end=callback()
                ).frames.cpu()
        return ({"samples":latent.flatten(0,1)}, image_pixels.permute(0, 2, 3, 1), pose_pixels.permute(0, 2, 3, 1))

class DisPoseDecoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe":("DisPosePipeline",{"tooltip":"output from (Down)Loader_DisPose"}),
                "latent": ("LATENT",),
                "decode_chunk_size":("INT",{"default":1,"min":1,"max":64,"step":1,"tooltip":"no quality impact, larger size is slightly faster but consumes more vram"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "DisPose"

    def run(self,pipe,latent,decode_chunk_size):
        olddevice = latent["samples"].device
        if next(pipe.vae.decoder.parameters()).device != model_management.vae_device():
            try:
                pipe.vae.decoder.to(model_management.vae_device())
            except:
                pipe.vae.decoder.to(torch.device('cpu'))
        if latent["samples"].device != next(pipe.vae.decoder.parameters()).device:
            latent["samples"] = latent["samples"].to(device=next(pipe.vae.decoder.parameters()).device)
        #latents = latents.flatten(0, 1)
        latents = 1 / pipe.vae.config.scaling_factor * latent["samples"] #latents
        forward_vae_fn = pipe.vae._orig_mod.forward if is_compiled_module(pipe.vae) else pipe.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())
        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        pbar = comfy.utils.ProgressBar((latents.shape[0]+decode_chunk_size-1)//decode_chunk_size)
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i: i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = pipe.vae.decode(latents[i: i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame.cpu())
            pbar.update(1)
        frames = torch.cat(frames, dim=0).float()
        
        if next(pipe.vae.decoder.parameters()).device != model_management.vae_offload_device():
            pipe.vae.decoder.to(model_management.vae_offload_device())
        if latent["samples"].device != olddevice:
            latent["samples"] = latent["samples"].to(device=olddevice)
        if frames.device != olddevice:
            frames = frames.to(device=olddevice)
        
        frames = pipe.image_processor.postprocess(frames, "pt")
        # [batch*frames, channels, height, width] -> [batch*frames, height, width, channels]
        return (frames.permute(0, 2, 3, 1),)

NODE_DISPLAY_NAME_MAPPINGS = {
    "DisPoseLoader":"(Down)Loader_DisPose",
    "DisPoseSampler":"Sampler_DisPose",
    "DisPoseDecoder":"Decoder_DisPose",
}

NODE_CLASS_MAPPINGS = {
    "DisPoseLoader":DisPoseLoader,
    "DisPoseSampler":DisPoseSampler,
    "DisPoseDecoder":DisPoseDecoder,
}

