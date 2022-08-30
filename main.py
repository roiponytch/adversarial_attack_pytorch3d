import os,sys
import torch
from torch import nn
from torchvision import transforms
import numpy as np
from pytorch3d.io import load_objs_as_meshes, save_obj


from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)

sys.path.insert(0,'/media/Data/hackathon2022/mmdetection')
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.models import build_detector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_config = '/media/Data/hackathon2022/mmdetection/faster_rcnn_r50_fpn_1x_coco.py'
model_ckpt ='/media/Data/hackathon2022/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

obj_filename = '/media/Data/projects/adversarial_live_demo/dbs/3d_models/person/uploads_files_1837665_GTP_BMan_Jack_07_Stg_Lsn_Adl_Ccs_Gry_Mgr/GTP_BMan_Jack_07_Stg_Lsn_Adl_Ccs_Gry_Mgr.obj'

init_color = [238., 133., 28.]  # Initial RGB color
inter_cam = [2.0, 0, 20, 0, 0]
n_views = 1
image_size = 512

dist, elev_start, elev_end, azim_start, azim_end = inter_cam
elev = torch.linspace(elev_start, elev_end, n_views).to(device)
azim = torch.linspace(azim_start, azim_end, n_views).to(device)
R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
cameras = FoVPerspectiveCameras(device=device, R=R.to(device), T=T.to(device)).to(device)

raster_settings_soft = RasterizationSettings(
    image_size=image_size,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Place a point light in front of the object. As mentioned above, the front of
# the cow is facing the -z direction.
lights = PointLights(device=device, location=[[10.0, 10.0, 10.0]])

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings_soft
    ),
    shader=SoftPhongShader(device=device,
                           cameras=cameras,
                           lights=lights))


# target_batch =  torch.LongTensor([target]*self.n_views).to(self.device)

# Load Object
# Initialize Sphere (Source Mesh)
src_mesh = load_objs_as_meshes([obj_filename], device=device)
# We scale normalize and center the target mesh to fit in a sphere of radius 1
# centered at (0,0,0). (scale, center) will be used to bring the predicted mesh
# to its original center and scale.  Note that normalizing the target mesh,
# speeds up the optimization but is not necessary!
verts = src_mesh.verts_packed()
N = verts.shape[0]
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
src_mesh.offset_verts_(-center.expand(N, 3))
src_mesh.scale_verts_((1.0 / float(scale)));

# We will learn to deform the source mesh by offsetting its vertices
# The shape of the deform parameters is equal to the total number of vertices in
# src_mesh
verts_shape = src_mesh.verts_packed().shape
deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)

# We will also learn per vertex colors for our sphere mesh that define texture
# of the mesh
rgb_color = torch.FloatTensor(init_color) / 255.
verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=False)
verts_rgb[...,:] =  torch.FloatTensor(rgb_color).to(device)
verts_rgb.requires_grad = True

opt = torch.optim.Adam([verts_rgb], lr=0.01)
# Loss Function
criterion = torch.nn.CrossEntropyLoss()

model = init_detector(model_config, checkpoint=model_ckpt)
# fp16_cfg = cfg.get('fp16', None)
# if fp16_cfg is not None:
#     wrap_fp16_model(model)
#checkpoint = load_checkpoint(model, model_ckpt, map_location='cpu')
# if args.fuse_conv_bn:
#     model = fuse_conv_bn(model)
# old versions did not save class info in checkpoints, this walkaround is
# for backward compatibility
# if 'CLASSES' in checkpoint.get('meta', {}):
#     model.CLASSES = checkpoint['meta']['CLASSES']
# else:
#     model.CLASSES = dataset.CLASSES
#########################################################
#    PREPARE OPTIMIZATION LOOP
#########################################################
target_cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
imgs_metas = {'filename': None,
              'ori_shape': (800, 800,3),
              'img_shape': (800, 800,3),
              'pad_shape': (800, 800,3),
              'scale_factor': 1.0

}
#
# for i in loop:
#     loss = torch.tensor(0.0, device=self.device)
#     # Initialize optimizer
#     opt.zero_grad()
#
#     # Deform the mesh
#     new_src_mesh = src_mesh.offset_verts(deform_verts)
images_predicted = renderer(src_mesh.extend(n_views), cameras=target_cameras, lights=lights)
a=1