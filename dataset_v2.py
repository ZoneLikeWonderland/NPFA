from re import I
import torch
import torch.nn as nn
import torch.utils.data
import glob
import os
import numpy as np
import os
import torch
import random
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
import pytorch3d
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
import cv2
import torch.nn.functional as F
from utils import *

lm68_14062 = [1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 4390, 3927, 3924, 2608, 3272, 4088, 3443, 268, 493, 1914, 2044, 1401, 3615, 4240, 4114, 2734, 2509, 978, 4527, 4942, 4857, 1140, 2075, 1147, 4269, 3360, 1507, 1542, 1537, 1528, 1518, 1511, 3742, 3751, 3756, 3721, 3725, 3732, 5708, 5695, 2081, 0, 4275, 6200, 6213, 6346, 6461, 5518, 5957, 5841, 5702, 5711, 5533, 6216, 6207, 6470, 5517, 5966]
lm68_na = [1225, 1851, 1052, 367, 1682, 1685, 2159, 1447, 966, 3619, 4308, 3848, 3845, 2566, 3230, 4009, 3401, 268, 493, 1877, 2004, 1401, 3573, 4158, 4035, 2692, 2467, 978, 4443, 4858, 4773, 1140, 2035, 1147, 4187, 3318, 1507, 1541, 1537, 1528, 1518, 1511, 3700, 3709, 3713, 3679, 3683, 3690, 5569, 5559, 2041, 0, 4193, 5920, 5930, 6025, 6105, 5432, 5745, 5664, 5563, 5572, 5444, 5933, 5924, 6110, 5431, 5750]


def lm_mask(lm):

    x_scale = 1
    y_scale = 1
    em_scale = 0.1

    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right

    # cv2.polylines(cv2imgrgb, [np.concatenate((lm_chin[::-1], lm_eyebrow_left, lm_eyebrow_right))], True, (255, 255, 255), 4)

    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    x *= x_scale
    y = np.flipud(x) * [-y_scale, y_scale]
    c = eye_avg + eye_to_mouth * em_scale

    mask = np.zeros((256, 256), dtype=np.float32)
    cv2.fillPoly(mask, [cv2.convexHull(np.concatenate((lm_chin[::-1], lm_eyebrow_left - eye_to_mouth * 0.5, lm_eyebrow_right - eye_to_mouth * 0.5))).astype(int)], (1,))
    return mask


def lm_crop(img: "HxWx3", face_landmarks):
    output_size = 256

    x_scale = 0.75
    y_scale = 1
    em_scale = 0.25

    lowscale = 1

    lm = np.array(face_landmarks, np.float32)
    lm = lm * lowscale
    # cv2imgrgb = cv2imgrgb.copy()
    # mask = np.ones(cv2imgrgb_raw.shape[:2], np.uint8) * 128
    mask = np.ones(img.shape[:2], np.float32) * 0.5

    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right

    # cv2.polylines(cv2imgrgb, [np.concatenate((lm_chin[::-1], lm_eyebrow_left, lm_eyebrow_right))], True, (255, 255, 255), 4)

    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    x *= x_scale
    y = np.flipud(x) * [-y_scale, y_scale]
    c = eye_avg + eye_to_mouth * em_scale

    above_brow = np.random.random() * 0.5
    cv2.fillPoly(mask, [cv2.convexHull(np.concatenate((lm_chin[::-1], lm_eyebrow_left - eye_to_mouth * above_brow, lm_eyebrow_right - eye_to_mouth * above_brow))).astype(int)], (1,))

    t = cv2.getAffineTransform(
        np.array([
            (-1, -1),
            (-1, 1),
            (1, 1),
        ], np.float32),
        np.array([c - x - y, c - x + y, c + x + y], np.float32) / output_size * 2 - 1,
    )
    return t, mask

    cv2imgrgb = cv2.warpAffine(cv2imgrgb_raw, t, (output_size, output_size))
    mask = cv2.warpAffine(mask, t, (output_size, output_size))


class SimpleShader(nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams(background_color=torch.tensor([0, 0, 0], dtype=torch.float32))

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        imgs = hard_rgb_blend(texels, fragments, blend_params)
        return imgs  # (N, H, W, 3) RGBA img


class SimpleRenderer:
    def __init__(self, device) -> None:
        raster_settings = RasterizationSettings(
            image_size=256,
            blur_radius=0.0,
            faces_per_pixel=1,
            # cull_backfaces=True,
            cull_backfaces=False,
        )

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings
            ),
            shader=SimpleShader()
        )
        self.device = device

    def render(self, meshes, aug, dc=None, only_imgs=False, crop=True, return_lm=False, only_imgs_and_lmks=False, imgs_with_transform=False, lm_index=None):
        if isinstance(aug, torch.Tensor):
            aa = aug[:, 0:3]
            T = aug[:, 3:6]
            f = aug[:, 6:7]
            intensity = aug[:, 7:8]
        else:
            aa, T, f, intensity = aug
        if dc is not None:
            raise Exception("dc is deprecated")
        cameras = get_cameras(aa, T, f, device=self.device, dc=dc)
        imgs = self.renderer(meshes, cameras=cameras)
        if only_imgs:
            imgs = imgs.permute(0, 3, 1, 2)[:, :3]
            if imgs_with_transform:
                return imgs, cameras.get_world_to_view_transform().get_matrix()
            return imgs

        if meshes._verts_list[0].shape[0] == 14062:
            lm68 = lm68_14062
        else:
            lm68 = lm68_na

        if lm_index is None:
            points = cameras.transform_points_screen(torch.stack(meshes._verts_list)[:, lm68])
        else:
            points = cameras[lm_index[0]].transform_points_screen(torch.stack(meshes._verts_list[lm_index[1]])[:, lm68])
        if only_imgs_and_lmks:
            imgs = imgs.permute(0, 3, 1, 2)[:, :3]
            return imgs, points[:, :, :2]

        N = aa.shape[0] // 2

        points_numpy = points.detach().cpu().numpy()

        ts = []
        masks = []
        for i in range(imgs.shape[0]):
            lm = points_numpy[i, :, :2]
            t, mask = lm_crop(imgs[i], lm)
            ts.append(t)
            masks.append(mask[None])

        masks = torch.tensor(masks).to(device=imgs.device)

        imgs = imgs.permute(0, 3, 1, 2)[:, :3]

        masks = masks * (imgs > 0).any(dim=1, keepdims=True)

        if crop:
            ts = torch.tensor(ts, dtype=torch.float32)[:N].repeat(2, 1, 1)
            grid = F.affine_grid(ts, imgs.shape, align_corners=False).to(device=imgs.device)
            imgs = F.grid_sample(imgs, grid, align_corners=False)
            masks = F.grid_sample(masks, grid, align_corners=False, mode="nearest")

        if return_lm:
            return imgs, masks, points[:, :, :2]

        return imgs, masks


class PhongRenderer(SimpleRenderer):
    def __init__(self, device) -> None:
        raster_settings = RasterizationSettings(
            image_size=256,
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=True,
        )

        lights = PointLights(device=device, location=[[2, 2, 2]])

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                lights=lights
            )
        )
        self.device = device


def cam_get_aug(
    aa_std=0.1,
    t_mean=[0, 0, 7],
    t_std=[0.2, 0.2, 0.5],
    f_mean=1250,
    f_std=100,
    # d_std=10,
    intensity_std=0.1,
    N=1,
):
    aa = torch.randn((N, 3), dtype=torch.float32) * aa_std
    T = torch.randn((N, 3), dtype=torch.float32)
    T = T * torch.tensor(t_std, dtype=torch.float32)
    T = T + torch.tensor(t_mean, dtype=torch.float32)
    # f = torch.randn((N, 1), dtype=torch.float32) * f_std + f_mean

    f = (f_mean / t_mean[-1]) * T[:, 2:] / ((T[:, 0:1]**2 + T[:, 1:2]**2)**0.5 + 1) + torch.randn((N, 1), dtype=torch.float32) * f_std

    # intensity = torch.randn((N, 1), dtype=torch.float32) * intensity_std + 1
    intensity = torch.ones((N, 1), dtype=torch.float32)
    return aa, T, f, intensity


def get_cameras(aa, T, f, device, dc=None):
    N = aa.shape[0]

    aa = aa.to(device=device)
    T = T.to(device=device)
    f = f.to(device=device)

    R = torch.tensor([[
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ]], dtype=torch.float32, device=device) @ pytorch3d.transforms.axis_angle_to_matrix(aa)

    dx, dy = 0, 0
    K = torch.tensor([[
        [0, 0, 128 - 0.5, 0],
        [0, 0, 128 - 0.5, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ]], dtype=torch.float32, device=device).repeat(N, 1, 1)
    if dc is not None:
        K[:, 0, 2] = K[:, 0, 2] + dc[:, 0]
        K[:, 1, 2] = K[:, 1, 2] + dc[:, 1]
    K[:, 0, 0] = f[:, 0]
    K[:, 1, 1] = f[:, 0]

    cameras = PerspectiveCameras(device=device, R=R, T=T, K=K, in_ndc=False, image_size=[[256, 256]])

    return cameras


def path_get_name(path):
    name = os.path.basename(path)
    if "." in name:
        name = name[:name.find(".")]
    return name


def img_get_aug(res=256, hard=1):
    angle = np.random.normal(0, 1 * hard)
    scale = np.random.normal(1, 0.05 * hard**0.5)
    dc = (np.random.normal(0, 3 * hard), np.random.normal(0, 3 * hard))
    noise = np.random.normal(1, 0.01, size=(res, res, 3))
    light = 1

    return angle, scale, dc, noise, light


def img_aug(img, mask, aug, res=256, lm=None):
    angle, scale, dc, noise, light = aug
    rad = angle / 180 * np.pi

    c = np.array((res / 2, res / 2)) + dc
    x = np.array((np.cos(rad), np.sin(rad))) * res / 2 / scale
    y = x[::-1] * [-1, 1]

    affine_aug = cv2.getAffineTransform(np.array([c, c + x, c + y], np.float32), np.array([(res / 2, res / 2), (res, res / 2), (res / 2, res)], np.float32))

    img = cv2.warpAffine(img, affine_aug, (res, res))
    mask = cv2.warpAffine(mask, affine_aug, (res, res))
    if lm is not None:
        lm = (affine_aug[:, :2] @ lm.T + affine_aug[:, 2:]).T

    img *= noise
    img *= light

    if lm is not None:
        return img, mask, lm

    return img, mask


class MeshDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mesh_folder, tex_folder, image_folder, uv3d_folder, obj_template_path, uv3d_neutral_path,
        device=torch.device("cpu"),
        device_train=torch.device("cpu"),
        need_fake=True, need_real=True, need_uv3d=False,
        single_view=False,
        tex_size=256,
        tex1_folder=None,
        tex2_folder=None,
        tex_neutral_path=None,
        tex1_neutral_path=None,
        tex2_neutral_path=None,
    ) -> None:
        super().__init__()
        self.data = {}

        self.obj_template = load_objs_as_meshes([obj_template_path], device=device)
        self.obj_template2 = self.obj_template.clone()

        if need_fake:
            mesh_paths = sorted(glob.glob(os.path.join(mesh_folder, "*.npy")))
            for mesh_path in mesh_paths:
                item = self.data.get(path_get_name(mesh_path), {})
                item["mesh_path"] = mesh_path
                self.data[path_get_name(mesh_path)] = item

            tex_paths = sorted(glob.glob(os.path.join(tex_folder, "*.png"))) + sorted(glob.glob(os.path.join(tex_folder, "*.jpg")))
            for tex_path in tex_paths:
                item = self.data.get(path_get_name(tex_path), {})
                item["tex_path"] = tex_path
                self.data[path_get_name(tex_path)] = item

        image_subfolders = sorted(glob.glob(os.path.join(image_folder, "*")))
        for image_subfolder in image_subfolders:
            item = self.data.get(path_get_name(image_subfolder), {})
            item["images"] = list(filter(lambda x: not os.path.basename(x).startswith("mask"), sorted(glob.glob(os.path.join(image_subfolder, "*.png")))))
            self.data[path_get_name(image_subfolder)] = item

        if need_uv3d:
            uv3d_paths = sorted(glob.glob(os.path.join(uv3d_folder, "*.bin")))
            for uv3d_path in uv3d_paths:
                item = self.data.get(path_get_name(uv3d_path), {})
                item["uv3d_path"] = uv3d_path
                self.data[path_get_name(uv3d_path)] = item

            uv3d_paths = sorted(glob.glob(os.path.join(uv3d_folder.replace("uv3d", "uv3d2"), "*.bin")))
            for uv3d_path in uv3d_paths:
                item = self.data.get(path_get_name(uv3d_path), {})
                item["uv3d_second_path"] = uv3d_path
                self.data[path_get_name(uv3d_path)] = item

        if tex1_folder is not None and need_fake:
            tex1_paths = sorted(glob.glob(os.path.join(tex1_folder, "*.png"))) + sorted(glob.glob(os.path.join(tex1_folder, "*.jpg")))
            for tex1_path in tex1_paths:
                item = self.data.get(path_get_name(tex1_path), {})
                item["tex1_path"] = tex1_path
                self.data[path_get_name(tex1_path)] = item

            tex2_paths = sorted(glob.glob(os.path.join(tex2_folder, "*.png"))) + sorted(glob.glob(os.path.join(tex2_folder, "*.jpg")))
            for tex2_path in tex2_paths:
                item = self.data.get(path_get_name(tex2_path), {})
                item["tex2_path"] = tex2_path
                self.data[path_get_name(tex2_path)] = item

        self.tex_size = tex_size

        if tex_neutral_path is not None:
            tex_neutral = torch.tensor(cv2.imread(tex_neutral_path)[..., ::-1].astype(np.float32) / 255).permute(2, 0, 1)[None].to(device=device_train)
            tex_neutral = F.interpolate(tex_neutral, size=(self.tex_size, self.tex_size), mode="bilinear", align_corners=False)

            tex_neutral_mask = (tex_neutral > 0).all(dim=1, keepdims=True).to(torch.float32)
            self.tex_neutral_mask = (F.interpolate(tex_neutral_mask, size=(self.tex_size, self.tex_size), mode="bilinear", align_corners=False) == 1).to(torch.float32)

            if tex1_neutral_path is not None:
                tex1_neutral = torch.tensor(cv2.imread(tex1_neutral_path)[..., ::-1].astype(np.float32) / 255).permute(2, 0, 1)[None].to(device=device_train)
                tex1_neutral = F.interpolate(tex1_neutral, size=(self.tex_size, self.tex_size), mode="bilinear", align_corners=False)
                tex2_neutral = torch.tensor(cv2.imread(tex2_neutral_path, cv2.IMREAD_GRAYSCALE)[..., None].astype(np.float32) / 255).permute(2, 0, 1)[None].to(device=device_train)
                tex2_neutral = F.interpolate(tex2_neutral, size=(self.tex_size, self.tex_size), mode="bilinear", align_corners=False)
                tex_neutral = torch.cat([tex_neutral, tex1_neutral, tex2_neutral], dim=1)
            self.tex_neutral = tex_neutral

        item_names = set()

        for key in self.data.keys():
            item_names |= set(self.data[key])
        for key in [i for i in self.data.keys()]:
            for item_name in item_names:
                if item_name not in self.data[key]:
                    print(f"delete {key} due to missing {item_name}",)
                    del self.data[key]
                    break

        self.item_names = item_names

        self.keys = sorted(self.data)

        self.uv3d_neutral = np.fromfile(uv3d_neutral_path, np.float32).reshape(256, 256, 3).transpose(2, 0, 1)
        self.uv3d_mask = np.isfinite(self.uv3d_neutral).all(axis=0, keepdims=True)
        self.uv3d_mask_tensor = torch.tensor(self.uv3d_mask, dtype=torch.float32)[None]

        if "uv3d_second_path" in item_names or True:
            self.uv3d_second_neutral = np.fromfile(uv3d_neutral_path.replace("uv3d", "uv3d2"), np.float32).reshape(256, 256, 3).transpose(2, 0, 1)
            self.uv3d_second_mask = np.isfinite(self.uv3d_second_neutral).all(axis=0, keepdims=True)
            self.uv3d_second_mask_tensor = torch.tensor(self.uv3d_second_mask, dtype=torch.float32)[None]

        self.renderer = SimpleRenderer(device=device)
        self.device = device
        self.device_train = device_train

        self.need_fake = need_fake
        self.need_real = need_real
        self.need_uv3d = need_uv3d
        self.single_view = single_view

    def __len__(self):
        return len(self.keys)

    def get_meshes(self, mesh_tensor=None):
        meshes = self.obj_template2
        if mesh_tensor is not None:
            meshes._verts_list = mesh_tensor
        return meshes

    def tensor_construct_meshes(self, mesh_tensor, tex_tensor):
        meshes = self.obj_template.clone()
        meshes.textures._maps_padded = while_map(tex_tensor[None])
        meshes._verts_list[0] = mesh_tensor
        return meshes

    def get_next_name(self):
        return random.choice(self.keys)

    def name_get_meshes(self, name):
        data = self.data[name]
        mesh_path = data["mesh_path"]
        tex_path = data["tex_path"]

        mesh_tensor = torch.tensor(np.load(mesh_path).astype(np.float32)).to(device=self.device)

        tex_tensor = torch.tensor(cv2.imread(tex_path)[..., ::-1].astype(np.float32) / 255).to(device=self.device)

        meshes = self.tensor_construct_meshes(mesh_tensor, tex_tensor)

        if "tex1_path" in data:
            tex1_path = data["tex1_path"]
            tex1_tensor = torch.tensor(cv2.imread(tex1_path)[..., ::-1].astype(np.float32) / 255).to(device=self.device)

            tex2_path = data["tex2_path"]
            tex2_tensor = torch.tensor(cv2.imread(tex2_path, cv2.IMREAD_GRAYSCALE)[..., None].astype(np.float32) / 255).to(device=self.device)

            tex_tensor = torch.cat([tex_tensor, tex1_tensor, tex2_tensor], dim=2)

        return meshes, tex_tensor.permute(2, 0, 1)

    def name_get_random_image(self, name):
        data = self.data[name]
        image_path = random.choice(data["images"])

        image = cv2.imread(image_path).astype(np.float32)[..., ::-1] / 255
        mask = cv2.imread(
            os.path.join(
                os.path.dirname(image_path),
                "mask_" + os.path.basename(image_path)
            ), cv2.IMREAD_GRAYSCALE
        ).astype(np.float32) / 255

        aug = img_get_aug(hard=2)
        image, mask = img_aug(image, mask, aug)

        return image.transpose(2, 0, 1), mask[None]

    def __getitem__(self, index):
        name1 = self.keys[index]
        name2 = self.get_next_name()

        batch = {}

        if self.need_real:
            image1, mask1 = self.name_get_random_image(name1)
            image2, mask2 = self.name_get_random_image(name2)

            batch["real"] = np.stack([image1, image2])
            batch["real_mask"] = np.stack([mask1, mask2])

        if self.need_fake:

            N = 2

            meshes1, meshes1_tex = self.name_get_meshes(name1)
            meshes2, meshes2_tex = self.name_get_meshes(name2)

            if self.single_view:
                N = 1

            aug = cam_get_aug(N=N, aa_std=torch.tensor([0.05, 0.4, 0.05], dtype=torch.float32), t_std=[0.05, 0.05, 0.2])
            aa, T, f, intensity = aug
            meshes = pytorch3d.structures.meshes.join_meshes_as_batch([meshes1.extend(N), meshes2.extend(N)])
            img, mask = self.renderer.render(
                meshes,
                (aa.repeat(2, 1), T.repeat(2, 1), f.repeat(2, 1), intensity.repeat(2, 1))
            )

            img = img.cpu().numpy()
            mask = mask.cpu().numpy()

            aug1 = img_get_aug(hard=2)
            aug2 = img_get_aug(hard=2)

            for i in range(N * 2):
                # img[i], mask[i] = img_aug(img[i].numpy().transpose(1, 2, 0), mask.numpy().transpose(1, 2, 0), aug1)
                img_i, mask_i = img_aug(img[i].transpose(1, 2, 0), mask[i].transpose(1, 2, 0), aug1 if i % 2 == 0 else aug2)
                img[i] = img_i.transpose(2, 0, 1)
                mask[i] = mask_i[..., None].transpose(2, 0, 1)

            batch["rendered"] = img
            batch["rendered_mask"] = mask
            # batch["tex"] = torch.stack([meshes1_tex, meshes2_tex])
            tex = torch.stack([meshes1_tex, meshes2_tex])
            tex_mask = (tex > 0).all(dim=1, keepdims=True).to(torch.float32)
            batch["tex"] = F.interpolate(tex, size=(self.tex_size, self.tex_size), mode="bilinear", align_corners=False)
            batch["tex_mask"] = (F.interpolate(tex_mask, size=(self.tex_size, self.tex_size), mode="bilinear", align_corners=False) == 1).to(torch.float32)

        if self.need_uv3d:
            uv3d1 = np.fromfile(self.data[name1]["uv3d_path"], np.float32).reshape(256, 256, 3).transpose(2, 0, 1)
            uv3d2 = np.fromfile(self.data[name2]["uv3d_path"], np.float32).reshape(256, 256, 3).transpose(2, 0, 1)

            uv3d = np.stack([uv3d1, uv3d2]) - self.uv3d_neutral
            batch["uv3d_mask"] = np.isfinite(uv3d).all(axis=1, keepdims=True)
            uv3d[np.isnan(uv3d)] = 0
            batch["uv3d"] = uv3d

            if "uv3d_second_path" in self.item_names:
                uv3d1_second = np.fromfile(self.data[name1]["uv3d_second_path"], np.float32).reshape(256, 256, 3).transpose(2, 0, 1)
                uv3d2_second = np.fromfile(self.data[name2]["uv3d_second_path"], np.float32).reshape(256, 256, 3).transpose(2, 0, 1)

                uv3d_second = np.stack([uv3d1_second, uv3d2_second]) - self.uv3d_second_neutral
                batch["uv3d_second_mask"] = np.isfinite(uv3d_second).all(axis=1, keepdims=True)
                uv3d_second[np.isnan(uv3d_second)] = 0
                batch["uv3d_second"] = uv3d_second

        for key in batch:
            if isinstance(batch[key], np.ndarray):
                batch[key] = torch.tensor(batch[key])
            batch[key] = batch[key].to(device=self.device_train)

        return batch
