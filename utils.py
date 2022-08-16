import inspect
import cv2
import numpy as np
import torch
import time
import re
import numpy as np
import os
import glob
import cv2
import shutil
import subprocess
import tqdm
import multiprocessing
import torch.nn.functional as F
import random


def t2n(x):
    return x.detach().cpu().numpy().transpose(1, 2, 0)


def swap_latent(a, b, channel):
    a1 = a.clone()
    b1 = b.clone()
    a1[:, channel], b1[:, channel] = b[:, channel], a[:, channel]
    return a1, b1


def tag_and_hconcat(colslist=None, tag=True, shape_fixed=None):
    names = list(filter(lambda x: x.startswith("show_"), inspect.currentframe().f_back.f_locals.keys()))
    shows = []
    shape = [0, 0]
    for name in names:
        # print(name, type(inspect.currentframe().f_back.f_locals[name]))
        fragment = inspect.currentframe().f_back.f_locals[name].copy()
        if fragment.shape[-1] != 3:
            fragment = cv2.cvtColor(fragment, cv2.COLOR_GRAY2BGR)
        if tag:
            cv2.putText(fragment, name[5:], (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1))
        shows.append(fragment)

        shape[0] = max(shape[0], fragment.shape[0])
        shape[1] = max(shape[1], fragment.shape[1])

    shape = shape if shape_fixed is None else shape_fixed
    for i in range(len(shows)):
        if shows[i].shape[:2] != shape:
            shows[i] = cv2.resize(shows[i], (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    if colslist is not None:
        rows = len(colslist)
        cols = max(colslist)
        show = np.zeros((shape[0] * rows, shape[1] * cols, 3), np.float32)
        i = 0
        for r in range(rows):
            for c in range(colslist[r]):
                if i == len(shows):
                    continue
                show[r * shape[0]:(r + 1) * shape[0], c * shape[1]:(c + 1) * shape[1]] = shows[i]
                i += 1
    else:
        try:
            show = cv2.hconcat(shows)
        except Exception as e:
            for img in shows:
                print("shape", img.shape)
            raise e

    return show


def mesh_diff_jet_rgb(show_mesh, show_mesh_recon, diff_scale=10):
    return cv2.applyColorMap((np.linalg.norm(show_mesh - show_mesh_recon, axis=2) * diff_scale * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1].astype(np.float32) / 255


def assert_loss_finite():
    names = filter(lambda x: x.startswith("loss_"), inspect.currentframe().f_back.f_locals.keys())
    for name in names:
        loss = inspect.currentframe().f_back.f_locals[name]
        if isinstance(loss, torch.Tensor):
            assert torch.isfinite(loss).all(), f"{name} is not finite"


def writer_add_loss(writer, global_step):
    names = filter(lambda x: x.startswith("loss"), inspect.currentframe().f_back.f_locals.keys())
    for name in names:
        loss = inspect.currentframe().f_back.f_locals[name]
        if isinstance(loss, torch.Tensor):
            writer.add_scalar(f'Loss/{name}', loss.item(), global_step)


def update_lr(comment, optimizer_gen):
    try:
        lr = float(open(f"lr/{comment}").read())
        optimizer_gen.param_groups[0]['lr'] = lr
        print(f"set lr = {lr}")
    except BaseException:
        pass


visualizer = None
ply = None


def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def try_save_state(comment, checkpoint_dir, net, identifier=""):
    storage = getattr(try_save_state, f"storage{identifier}", None)
    if storage is None:
        storage = time.time(), 0
    last_save_time, saving = storage
    if time.time() - last_save_time > 60 * 5:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(net.state_dict(),
                   os.path.join(checkpoint_dir, f'{comment}_latest_{saving%2}.pth'))
        last_save_time = time.time()
        saving += 1
    storage = last_save_time, saving
    setattr(try_save_state, f"storage{identifier}", storage)


def dilate1(uv3d, mask):
    mask = mask.to(dtype=torch.float32)
    mask3 = (mask == 1).expand(uv3d.shape[0], 3, -1, -1)
    kernel = torch.tensor([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=torch.float32).to(device=uv3d.device)[None, None]
    mask_sum = F.conv2d(mask, kernel, padding=1)
    # uv3d[torch.isnan(uv3d)] = 0
    uv3d_sum = F.conv2d(uv3d.reshape(uv3d.shape[0] * 3, 1, 256, 256), kernel, padding=1).reshape(uv3d.shape[0], 3, 256, 256)

    mask_sum[mask_sum == 0] = 1
    uv3d_new = uv3d_sum / mask_sum
    uv3d_new[mask3] = uv3d[mask3]
    uv3d = uv3d_new
    return uv3d


def uv3d_construct_mesh(mesh_template, uv3d: "Nx3x256x256", mask: "Nx1x256x256"):
    grid = getattr(uv3d_construct_mesh, "grid", None)
    if grid is None:
        vts = mesh_template.textures.verts_uvs_list()[0]
        n_v = mesh_template._verts_list[0].shape[0]

        v2vt = [set() for i in range(n_v)]
        for face_v, face_vt in zip(mesh_template._faces_list[0].reshape(-1).tolist(), mesh_template.textures.faces_uvs_list()[0].reshape(-1).tolist()):
            v2vt[face_v].add(face_vt)

        v2vt0 = [None for i in range(n_v)]
        for v in range(len(v2vt)):
            for vt in v2vt[v]:
                v2vt0[v] = vts[vt]
                if vts[vt, 0] < 1:
                    break

        v2vt0 = torch.stack(v2vt0)
        grid = v2vt0[None, None] * 2 - 1
        grid = grid.to(device=uv3d.device)
        setattr(uv3d_construct_mesh, "grid", grid)

    # mask = torch.isfinite(uv3d).all(dim=1, keepdims=True)
    mask = mask.to(dtype=torch.float32)
    mask3 = (mask == 1).expand(uv3d.shape[0], 3, -1, -1)
    kernel = torch.tensor([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=torch.float32).to(device=uv3d.device)[None, None]
    mask_sum = F.conv2d(mask, kernel, padding=1)
    # uv3d[torch.isnan(uv3d)] = 0
    uv3d_sum = F.conv2d(uv3d.reshape(uv3d.shape[0] * 3, 1, 256, 256), kernel, padding=1).reshape(uv3d.shape[0], 3, 256, 256)

    mask_sum[mask_sum == 0] = 1
    uv3d_new = uv3d_sum / mask_sum
    uv3d_new[mask3] = uv3d[mask3]
    uv3d = uv3d_new

    transformed_mesh = getattr(uv3d_construct_mesh, "mesh_template", None)
    if transformed_mesh is None:
        # if transformed_mesh is None or len(transformed_mesh._verts_list) != uv3d.shape[0]:
        # transformed_mesh = mesh_template.extend(uv3d.shape[0]).clone().to(device=uv3d.device)
        transformed_mesh = mesh_template.clone().to(device=uv3d.device)
        transformed_mesh.textures._maps_padded = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=uv3d.device)
        setattr(uv3d_construct_mesh, "mesh_template", transformed_mesh.detach())
    transformed_mesh = transformed_mesh.extend(uv3d.shape[0]).clone()

    # transformed_mesh = mesh_template.extend(uv3d.shape[0]).clone().to(device=uv3d.device)
    vertices_pred = F.grid_sample(uv3d, grid.expand(uv3d.shape[0], -1, -1, -1), padding_mode="border", align_corners=False)
    vertices_pred = vertices_pred.reshape(uv3d.shape[0], 3, -1).permute(0, 2, 1)
    for i in range(uv3d.shape[0]):
        transformed_mesh._verts_list[i] = transformed_mesh._verts_list[i] + vertices_pred[i]

    return transformed_mesh


def uv3d_construct_mesh_2(mesh_template, uv3d: "Nx3x256x256", uv3d_second: "Nx3x256x256", mask: "Nx1x256x256", mask_second: "Nx1x256x256", only_mesh=False):

    assert uv3d.shape[1] == 3
    assert uv3d_second.shape[1] == 3
    assert mask.shape[1] == 1
    assert mask_second.shape[1] == 1

    grid0, grid1, grid1_2, v2vtc = getattr(uv3d_construct_mesh_2, "grid", (None, None, None, None))
    if grid0 is None:
        vts = mesh_template.textures.verts_uvs_list()[0]
        n_v = mesh_template._verts_list[0].shape[0]

        v2vt = [set() for i in range(n_v)]
        for face_v, face_vt in zip(mesh_template._faces_list[0].reshape(-1).tolist(), mesh_template.textures.faces_uvs_list()[0].reshape(-1).tolist()):
            v2vt[face_v].add(face_vt)

        # v2vt0 = [None for i in range(n_v)]
        v2vt0 = [torch.tensor([0, 0], dtype=torch.float32, device=vts.device)] * n_v
        v2vt1 = [torch.tensor([0, 0], dtype=torch.float32, device=vts.device)] * n_v
        v2vt1_2 = [torch.tensor([0, 0], dtype=torch.float32, device=vts.device)] * n_v
        v2vtc = [0] * n_v  # (14062)
        for v in range(len(v2vt)):
            second = False
            for vt in v2vt[v]:

                if vts[vt, 0] < 1:
                    v2vt0[v] = vts[vt]
                    v2vtc[v] += 1
                elif vts[vt, 0] > 1:
                    if (v2vt1[v] == 0).all():
                        v2vt1[v] = vts[vt]
                        v2vt1_2[v] = vts[vt]
                    else:
                        v2vt1_2[v] = vts[vt]
                    second = True
            if second:
                v2vtc[v] += 1

        v2vt0 = torch.stack(v2vt0)
        v2vt1 = torch.stack(v2vt1) - torch.tensor([1, 0], dtype=torch.float32, device=vts.device)
        v2vt1_2 = torch.stack(v2vt1_2) - torch.tensor([1, 0], dtype=torch.float32, device=vts.device)
        grid0 = v2vt0[None, None] * 2 - 1
        grid1 = v2vt1[None, None] * 2 - 1
        grid1_2 = v2vt1_2[None, None] * 2 - 1
        grid0 = grid0.to(device=uv3d.device)
        grid1 = grid1.to(device=uv3d.device)
        grid1_2 = grid1_2.to(device=uv3d.device)

        v2vtc = torch.tensor(v2vtc, dtype=torch.float32, device=uv3d.device)[None, :, None]
        assert torch.min(v2vtc) == 1
        assert torch.max(v2vtc) == 2, torch.where(v2vtc == 3)
        setattr(uv3d_construct_mesh_2, "grid", (grid0, grid1, grid1_2, v2vtc))

    uv3d = dilate1(uv3d, mask)
    uv3d_second = dilate1(uv3d_second, mask_second)

    transformed_mesh = getattr(uv3d_construct_mesh, "mesh_template", None)
    if transformed_mesh is None:
        transformed_mesh = mesh_template.clone().to(device=uv3d.device)
        transformed_mesh.textures._maps_padded = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=uv3d.device)
        setattr(uv3d_construct_mesh, "mesh_template", transformed_mesh.detach())
    transformed_mesh = transformed_mesh.extend(uv3d.shape[0]).clone()

    vertices_pred0 = F.grid_sample(uv3d, grid0.expand(uv3d.shape[0], -1, -1, -1), padding_mode="zeros", align_corners=False)
    vertices_pred0 = vertices_pred0.reshape(uv3d.shape[0], 3, -1).permute(0, 2, 1)

    vertices_pred1 = F.grid_sample(uv3d_second, (grid1 if random.random() > 0.5 else grid1_2).expand(uv3d.shape[0], -1, -1, -1), padding_mode="zeros", align_corners=False)
    vertices_pred1 = vertices_pred1.reshape(uv3d.shape[0], 3, -1).permute(0, 2, 1)

    vertices_pred = (vertices_pred0 + vertices_pred1) / v2vtc  # (batch, 14062, 3)

    for i in range(uv3d.shape[0]):
        transformed_mesh._verts_list[i] = transformed_mesh._verts_list[i] + vertices_pred[i]

    if only_mesh:
        return transformed_mesh

    loss_vertices_same_diff = F.mse_loss(vertices_pred0 * (v2vtc - 1), vertices_pred1 * (v2vtc - 1))
    return transformed_mesh, loss_vertices_same_diff


def show_draw_landmark(show, landmark_g, landmark_r):
    show = show.copy()
    landmark_g = landmark_g.detach().cpu().numpy()
    landmark_r = landmark_r.detach().cpu().numpy()
    for k in range(68):
        cv2.circle(show, (int(landmark_g[k, 0] * 256), int(landmark_g[k, 1] * 256)), 1, (0, 1, 0), -1)
        cv2.circle(show, (int(landmark_r[k, 0] * 256), int(landmark_r[k, 1] * 256)), 1, (1, 0, 0), 1)
    return show


# def time_to_show(step, interval=30):
#     last_show_time = getattr(time_to_show, "last_show_time", 0)
#     last_step = getattr(time_to_show, "last_step", None)
#     setattr(time_to_show, "last_step", step)
#     first = last_step != step
#     last_show_step = getattr(time_to_show, "last_show_step", None)
#     if time.time() - last_show_time > interval and first or step == last_show_step:

#         last_show_time = time.time()
#         setattr(time_to_show, "last_show_time", last_show_time)
#         setattr(time_to_show, "last_show_step", step)
#         return True
#     return False


def time_to_show(interval=30):
    last_show_time = getattr(time_to_show, "last_show_time", 0)
    if time.time() - last_show_time > interval:
        last_show_time = time.time()
        setattr(time_to_show, "last_show_time", last_show_time)
        return True
    return False


def mask_or_not(mask):
    return torch.lerp(mask, torch.tensor(1, dtype=torch.float32, device=mask.device), torch.randint(0, 2, (mask.shape[0], 1, 1, 1), dtype=torch.float32, device=mask.device))


# def uv3d_get_grad(uv3d):
#     grad_y = uv3d[:, :, 1:, :] - uv3d[:, :, :-1, :]
#     grad_x = uv3d[:, :, :, 1:] - uv3d[:, :, :, :-1]
#     # return torch.stack([grad_x, grad_y], dim=4)
#     return grad_x, grad_y


def criterion_grad(uv3d, uv3d_second):
    grad_y = uv3d[:, :, 1:, :] - uv3d[:, :, :-1, :]
    grad_x = uv3d[:, :, :, 1:] - uv3d[:, :, :, :-1]

    grad_y_second = uv3d_second[:, :, 1:, :] - uv3d_second[:, :, :-1, :]
    grad_x_second = uv3d_second[:, :, :, 1:] - uv3d_second[:, :, :, :-1]
    return F.mse_loss(grad_x, grad_x_second) + F.mse_loss(grad_y, grad_y_second)


def latent_blend(latent, latent_second):
    return torch.lerp(latent, latent_second, torch.rand((latent.shape[0], *([1] * (len(latent.shape) - 1))), dtype=torch.float32, device=latent.device))


def tqdm_loader(loaders, comment=""):
    return tqdm.tqdm(zip(*loaders), comment, min([len(loader) for loader in loaders]))


def check_loader_latency(position):
    if position == "start":
        now = time.time()
        last_batch_end_time = getattr(check_loader_latency, "last_batch_end_time", 0)
        # if now-last_batch_end_time
        print("latency =", now - last_batch_end_time)
    elif position == "end":
        setattr(check_loader_latency, "last_batch_end_time", time.time())


def while_map(x, black=True):
    # y = torch.ones_like(x, requires_grad=x.requires_grad)
    # y[:, 1:-1, 1:-1, :] = x[:, 1:-1, 1:-1, :]
    x[:, 0, :, :] = 0 if black else 1
    x[:, -1, :, :] = 0 if black else 1
    x[:, :, 0, :] = 0 if black else 1
    x[:, :, -1, :] = 0 if black else 1
    return x
