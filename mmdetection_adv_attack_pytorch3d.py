#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 11:57:57 2021

@author: roipo
"""
#%% 

# imports for mmdetection
import torch
import matplotlib.pyplot as plt
import numpy as np
import warnings
import cv2

import mmcv
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
from mmdet.models import build_detector
from mmcv.runner import wrap_fp16_model

import os
import sys

#from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
# Data structures and functions for rendering
# from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,

    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
    TexturesAtlas,
    BlendParams,
    FoVPerspectiveCameras
)

# add path for demo utils functions 

sys.path.append(os.path.abspath(''))

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj
# import torchvision.models as models
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter

from mmcv import Config, DictAction

def preProcessInput(images):
    bg_mask = 1.0*(images[...,3]==0).unsqueeze(-1).repeat(1,1,1,3)
    target_mask =1.0*(images[...,3]>0).unsqueeze(-1).repeat(1,1,1,3)
    images = images[...,:3]
    images = bg_mask*images +target_mask*images/(target_mask*images).max()
    
    images = images.permute([0,-1,1,2])
    # images = images[...,:3].permute([0,-1,1,2])
    
    batch_size = images.shape[0]
    channels = images.shape[1]
    
    images_min = images.view(batch_size, channels, -1).min(2).values.view(batch_size, channels, 1,1)
    
    images = images -images_min
    
    images_max = images.view(batch_size, channels, -1).max(2).values.view(batch_size, channels, 1,1)
    images = images/images_max

    return images

def my_init_detector(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_detector(config.model,train_cfg=config.train_cfg, test_cfg=config.test_cfg)
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if hasattr(checkpoint, 'meta') and 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        elif 'classes' in config:  # <RFL>
            model.CLASSES = config['classes']  # <RFL>
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def chess_board(shape=(512, 512,3), divs=8):
    step = [np.floor(elem / divs) for elem in shape]

    chess = np.zeros((shape))
    for i in range(0, int(np.floor(shape[0] / step[0])), 2):
        for j in range(1, int(np.floor(shape[1] / step[1])), 2):
            chess[(int(i*step[0])) : (int((i+1)*step[0])), (int(j*step[1])) : (int((j+1)*step[1])),:] = 1

    for i in range(1, int(np.floor(shape[0] / step[0])), 2):
        for j in range(0, int(np.floor(shape[1] / step[1])), 2):
            chess[(int(i*step[0])) : (int((i+1)*step[0])), (int(j*step[1])) : (int((j+1)*step[1])), :] = 1

    return chess

def n_chess_board(shape=(512, 512,3), divs=8):
    step = [np.floor(elem / divs) for elem in shape]

    chess = np.zeros([divs*divs]+list(shape))
    k=0
    for i in range(0, int(np.floor(shape[0] / step[0])), 1):
        for j in range(0, int(np.floor(shape[1] / step[1])), 1):
            chess[k,(int(i*step[0])) : (int((i+1)*step[0])), (int(j*step[1])) : (int((j+1)*step[1])),:] = 1
            k+=1
        
    return chess

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)

    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap = "Greys")
    else:
        plt.imshow(np.transpose(npimg, (1,2,0)))

def plots_detected(model ,imgs_for_plots, results, score_thr):
    
    plots=[]
    for img, result in zip(imgs_for_plots, results):
        plots.append(show_result_pyplot(model, img, result, score_thr=score_thr))
        
    # return torch.from_numpy(np.asarray(plots))
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% load mmdetection model
model_config = 'faster_rcnn_r50_fpn_1x_coco.py'
model_ckpt ='run_files/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

obj_filename = 'run_files/dbs/3d_models/person/uploads_files_1837665_GTP_BMan_Jack_07_Stg_Lsn_Adl_Ccs_Gry_Mgr/GTP_BMan_Jack_07_Stg_Lsn_Adl_Ccs_Gry_Mgr.obj'

#%% set fp16 and load a model (object detection)
fp16 =True
cfg = Config.fromfile(model_config)
model = init_detector(model_config, checkpoint=model_ckpt)

if fp16:
    wrap_fp16_model(model)
    data_dtype = torch.half
else:
    data_dtype = torch.float


#%% load 3D model from .obj
mesh = load_objs_as_meshes([obj_filename], device=device)

# vv,ff,aa =load_obj(obj_filename,load_textures=True,create_texture_atlas=True, texture_atlas_size=4)
# mesh=Meshes(verts=[vv],faces=[ff.verts_idx],textures=TexturesAtlas(atlas=[aa.texture_atlas.to(device)]))
# We scale normalize and center the target mesh to fit in a sphere of radius 1 
# centered at (0,0,0). (scale, center) will be used to bring the predicted mesh 
# to its original center and scale.  Note that normalizing the target mesh, 
# speeds up the optimization but is not necessary!
verts = mesh.verts_packed()
N = verts.shape[0]
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
mesh.offset_verts_(-center.expand(N, 3))
mesh.scale_verts_((1.0 / float(scale)));

#%%
num_views = 2 # bacth size
num_views_total =1000

# With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
# So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
elev = torch.linspace(0, 5, num_views_total)
azim = torch.linspace(0, 20, num_views_total)
dist = torch.linspace(3, 3, num_views_total)

R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
# the fas ter coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=800, 
    blur_radius=0, 
    faces_per_pixel=1,
    max_faces_per_bin=150000,
    perspective_correct=False
)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
# -z direction.
lights = PointLights(device=device, location=[[10.0, 10.0, 10.0]])

# blend_params = BlendParams(background_color=(1.,1.,1.))
# blend_params = BlendParams(background_color=(.5,.5,.5))
# Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will 
# interpolate the texture uv coordinates for each vertex, sample from a texture image and 
# apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights,
        # blend_params=blend_params
    )
)

meshes = mesh.extend(num_views)
# meshes=mesh
# images = renderer(meshes, cameras=cameras, lights=lights)

# divs=16
# mask_w=1024
# mask_h=1024

# mask = torch.from_numpy(cb).type(mesh.textures._maps_padded.dtype).to(device)
# mask = torch.ones((divs**2,3,int(mask_w/divs),int(mask_h/divs)), device=device)
delta_1 =torch.nn.Parameter( torch.zeros(meshes.textures.maps_padded()[0].shape,requires_grad=True, device=device))

imm=meshes.textures.maps_padded()[0].cpu().detach()
im =imm -imm.min()
im=im/im.max()
im = (im.numpy()*255).astype(np.uint8)

mask_upper_body = np.zeros(im.shape[:2], np.uint8)
rect = (0,350,500,950)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

cv2.grabCut(im,mask_upper_body,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

stacked_img = np.stack(((1.*(mask_upper_body==3)),)*3, axis=-1).astype(np.float32)
delta_1_masks = torch.from_numpy(stacked_img).to(device)
# mesh.textures.maps_padded().requires_grad =True
delta_1.retain_grad()
# delta_1_masks = (tv.utils.make_grid((mask*delta_1),divs,0)).permute(1,2,0)
# delta_1_masks.requires_grad=True
# delta_1_masks.retain_grad()

source_texture = meshes.textures.maps_padded().clone()
meshes.textures._maps_padded =source_texture +delta_1_masks*delta_1.unsqueeze(0)

#%%

idx = torch.randint(0,num_views_total,[2,num_views])
images = renderer(meshes,cameras = FoVPerspectiveCameras(device=device, R=R[idx[0]], T=T[idx[1]]),lights=lights)

#%%
masks=images[...,-1]
n=masks.shape[0]

bbox=[]
bboxes_ignore=[]
labels=[]
for index in range(n):
    msk=masks[index]
    y,x = torch.where(msk>0)
    bbox.append(torch.zeros((1,4),device=device, dtype=torch.float))
    bboxes_ignore.append(torch.zeros(0,device=device, dtype=torch.float))
    labels.append(torch.zeros((1), device=device, dtype=torch.int64))

    bbox[-1][0,0] = torch.min(x)
    bbox[-1][0,1] = torch.min(y)
    bbox[-1][0,2] = torch.max(x)
    bbox[-1][0,3] = torch.max(y)
    labels[-1][0] = model.CLASSES.index('person')

#%%

#%%
optimizer = torch.optim.Adam([delta_1], lr=0.1)

#%%
img_metas ={'filename': None,
 'ori_shape': (800, 800, 3),
 'img_shape': (800, 800, 3),
 'pad_shape': (800, 800, 3),
 'scale_factor': 1.0}
#%%

verts_shape = meshes.verts_packed().shape
init_color = [238., 133., 28.]
rgb_color = torch.FloatTensor(init_color) / 255.
verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=False)
verts_rgb[...,:] =  torch.FloatTensor(rgb_color).to(device)
cln_test_batch={}


# cln_test_batch['img'] =[images[image_idx,...,:3].unsqueeze(0).permute([0,-1,1,2]).contiguous().clone()]
cln_test_batch['img'] =[images[...,:3].permute([0,-1,1,2]).to(data_dtype).contiguous().clone()]
cln_test_batch['img_metas'] = [[img_metas]*num_views]
imgs_for_plots =cln_test_batch['img'][0].data.cpu().numpy().transpose([0,2,3,1])[...,::-1]*255
with torch.no_grad():
    cln_result = model(return_loss=False, rescale=False, **cln_test_batch)

#%%
image_idx = torch.randint(0, imgs_for_plots.shape[0], [1])[0]

show_result_pyplot(model, imgs_for_plots[image_idx], cln_result[image_idx], score_thr=0.2)
run_folder = 'runs/000/'
writer = SummaryWriter(run_folder)
#%%   
test_batch ={}
lambda_coeff = 0 #10.

log_every_n = 100

cln_train_batch={}
cln_train_batch['img_metas'] = [[img_metas] * num_views]
cln_test_batch['img_metas'] = [[img_metas] * num_views]

for i in range(10000000):
    del masks
    del images
    torch.cuda.empty_cache()
    idx = torch.randint(0,num_views_total,[2,num_views])
    images = renderer(meshes,cameras = FoVPerspectiveCameras(device=device, R=R[idx[0]], T=T[idx[1]]),lights=lights)
    masks=images[...,-1]
    n=masks.shape[0]

    bbox = []
    bboxes_ignore = []
    labels = []
    for index in range(n):
        msk = masks[index]
        y, x = torch.where(msk > 0)
        bbox.append(torch.zeros((1, 4), device=device, dtype=torch.float))
        bboxes_ignore.append(torch.zeros(0, device=device, dtype=torch.float))
        labels.append(torch.zeros((1), device=device, dtype=torch.int64))

        bbox[-1][0, 0] = torch.min(x)
        bbox[-1][0, 1] = torch.min(y)
        bbox[-1][0, 2] = torch.max(x)
        bbox[-1][0, 3] = torch.max(y)
        labels[-1][0] = model.CLASSES.index('person')
    # for index, msk in enumerate(masks):
    #
    #     y,x = torch.where(msk>0)
    #
    #     bbox[index][0,0] = torch.min(x)
    #     bbox[index][0,1] = torch.min(y)
    #     bbox[index][0,2] = torch.max(x)
    #     bbox[index][0,3] = torch.max(y)
    #     labels[index][0] = model.CLASSES.index('person')

    input_batch =  images[...,:3].permute([0,-1,1,2]).to(data_dtype).contiguous()
    # model(return_loss=True, rescale=False, **cln_test_batch)
    # cln_train_batch['gt_bboxes'] = bbox
    # cln_train_batch['gt_labels'] = labels
    cln_train_batch['img'] = [input_batch]

    # qq={
    #     'gt_bboxes': bbox,
    #     'gt_labels': labels}
    # cln_test_batch['gt_labels'] = labels})

    cln_result = model(return_loss=False, rescale=False, **cln_train_batch)
    adversarial_loss =  sum((-torch.log(1. - (head_scores[:,model.CLASSES.index('person') ]) + 1e-6)).sum() for head_scores in model.roi_head.bbox_head.scores)
    # adversarial_loss = (-torch.log(1. -(model.roi_head.bbox_head.scores[0][:,0]) + 1e-6)).sum()

    # adversarial_loss = (-torch.log(1. -(model.roi_head.det_bboxes[0][:,-1]) + 1e-6)).sum()

    # losses_pre_adv, proposal_list = model._get_losses(img=input_batch, img_metas=[img_metas]*num_views,
    #                                                             gt_bboxes=bbox,
    #                                                             gt_labels=labels)
    # adversarial_loss = (1.-torch.tensor(losses_pre_adv['loss_rpn_cls'],requires_grad=True)).sum()
    # adversarial_loss = -sum(_loss.mean() for _loss in losses_pre_adv['loss_rpn_cls'])

    # # loss= -torch.log(F.softmax(model(input_image))[:,847])
    reg_loss = torch.mean(delta_1**2)
    loss = adversarial_loss + lambda_coeff*reg_loss
    # loss = adversarial_loss
    loss = loss/log_every_n

    loss.backward(retain_graph=True)
    print('loss: {:.4f}, adv_loss: {:.4f}, reg_loss: {:.4f}'.format(
        loss,
        adversarial_loss,
        reg_loss))
    

    # delta_1_masks = (tv.utils.make_grid((mask*delta_1),divs,0)).permute(1,2,0)
    # delta_1_masks.requires_grad =True
    # delta_1_masks.retain_grad()


    if i % log_every_n == 0:
        optimizer.step()
        meshes.textures._maps_padded = torch.clamp(source_texture + delta_1_masks * delta_1.unsqueeze(0), 0., 1.)
        optimizer.zero_grad()
        test_batch['img'] =[images[...,:3].permute([0,-1,1,2]).contiguous().clone()]
        test_batch['img_metas'] = [[img_metas]*num_views]
        imgs_for_plots =test_batch['img'][0].data.cpu().numpy().transpose([0,2,3,1])[...,::-1]*255
        with torch.no_grad():
            result = model(return_loss=False, rescale=False, **test_batch)
        plots_detected(model, imgs_for_plots, result, score_thr=0.2)
        # img_grid = tv.utils.make_grid(res_batch.permute([0,-1,1,2])).detach().cpu().to(torch.float32)
        # matplotlib_imshow(img_grid)
        # writer.add_image('adversarial_images', img_grid, i)
        writer.add_scalar('training loss', loss, i)
    

    # meshes.textures._maps_padded =meshes.textures.maps_padded() +ones_mask*delta_1_masks

    # meshes.textures._maps_padded =meshes.textures.maps_padded() + ones_mask*(mask*delta_1 + (1-mask)*delta_2)



    


#%%   

for _ in range(10):
    del images
    torch.cuda.empty_cache()
    images = renderer(meshes, cameras=cameras, lights=lights)
    
    losses_pre_adv, proposals_for_debug = model._get_losses(img= images[image_idx,...,:3].unsqueeze(0).permute([0,-1,1,2]).contiguous(),
                                                            img_metas=data['img_metas'][0],
                                                            gt_bboxes=data['gt_bboxes'][0],
                                                            gt_labels=data['gt_labels'][0])
    # loss= -torch.log(F.softmax(model(input_image))[:,847])
    loss = -1.*losses_pre_adv['loss_cls']
    loss.backward(retain_graph=True)
    print('loss: {}'.format(loss))
    optimizer.step()
    delta_1_masks = (tv.utils.make_grid((mask*delta_1),divs,0)).permute(1,2,0)
    delta_1_masks.retain_grad()
    meshes.textures._maps_padded =meshes.textures.maps_padded() +ones_mask*delta_1_masks

    # meshes.textures._maps_padded =meshes.textures.maps_padded() + ones_mask*(mask*delta_1 + (1-mask)*delta_2)

    # meshes.textures._maps_padded =meshes.textures.maps_padded() + ones_mask*delta_1
    optimizer.zero_grad()
    
#%%


adv_test_batch={}
adv_test_batch['img'] =[images[...,:3].permute([0,-1,1,2]).contiguous().clone()]
# adv_test_batch['img'] =[torch.Tensor(oo).to(device).unsqueeze(0).permute([0,-1,1,2]).contiguous().clone()]
adv_test_batch['img_metas'] = [[img_metas]*1]
imgs_for_plots =adv_test_batch['img'][0].data.cpu().numpy().transpose([0,2,3,1])[...,::-1]
with torch.no_grad():
     attcked_result = model(return_loss=False, rescale=False, **adv_test_batch)

show_result_pyplot(model, imgs_for_plots[0], attcked_result[0], score_thr=0.2)
