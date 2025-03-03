import logging
from pathlib import Path
import av
from PIL import Image
import os
from scipy.interpolate import PchipInterpolator
import numpy as np
import pdb
import torch
import torch.nn.functional as F
from torchvision.io import write_video

logger = logging.getLogger(__name__)

@torch.no_grad()
def get_cmp_flow(cmp, frames, sparse_optical_flow, mask):
    '''
        frames: [b, 13, 3, 384, 384] (0, 1) tensor
        sparse_optical_flow: [b, 13, 2, 384, 384] (-384, 384) tensor
        mask: [b, 13, 2, 384, 384] {0, 1} tensor
    '''
    # print(frames.shape)
    dtype = frames.dtype
    b, t, c, h, w = sparse_optical_flow.shape
    assert h == 384 and w == 384
    frames = frames.flatten(0, 1)  # [b*13, 3, 256, 256]
    sparse_optical_flow = sparse_optical_flow.flatten(0, 1)  # [b*13, 2, 256, 256]
    mask = mask.flatten(0, 1)  # [b*13, 2, 256, 256]

    # print(frames.shape)
    # print(sparse_optical_flow.shape)
    # print(mask.shape)

    # assert False

    cmp_flow = []
    for i in range(b*t):
        tmp_flow = cmp.run(frames[i:i+1].float(), sparse_optical_flow[i:i+1].float(), mask[i:i+1].float())  # [b*13, 2, 256, 256]
        cmp_flow.append(tmp_flow)
    cmp_flow = torch.cat(cmp_flow, dim=0)
    cmp_flow = cmp_flow.reshape(b, t, 2, h, w)

    return cmp_flow.to(dtype=dtype)



def sample_optical_flow(A, B, h, w):
    b, l, k, _ = A.shape

    sparse_optical_flow = torch.zeros((b, l, h, w, 2), dtype=B.dtype, device=B.device)
    mask = torch.zeros((b, l, h, w), dtype=torch.uint8, device=B.device)

    x_coords = A[..., 0].long()
    y_coords = A[..., 1].long()

    x_coords = torch.clip(x_coords, 0, h - 1)
    y_coords = torch.clip(y_coords, 0, w - 1)

    b_idx = torch.arange(b)[:, None, None].repeat(1, l, k)
    l_idx = torch.arange(l)[None, :, None].repeat(b, 1, k)

    sparse_optical_flow[b_idx, l_idx, x_coords, y_coords] = B

    mask[b_idx, l_idx, x_coords, y_coords] = 1

    mask = mask.unsqueeze(-1).repeat(1, 1, 1, 1, 2)

    return sparse_optical_flow, mask


@torch.no_grad()
def get_sparse_flow(poses, h, w, t):

    poses = torch.flip(poses, dims=[3])

    pose_flow = (poses - poses[:, 0:1].repeat(1, t, 1, 1))[:, 1:]  # 前向光流
    according_poses = poses[:, 0:1].repeat(1, t - 1, 1, 1)
    
    pose_flow = torch.flip(pose_flow, dims=[3])

    b, t, K, _ = pose_flow.shape

    sparse_optical_flow, mask = sample_optical_flow(according_poses, pose_flow, h, w)

    return sparse_optical_flow.permute(0, 1, 4, 2, 3), mask.permute(0, 1, 4, 2, 3)

def sample_inputs_flow(first_frame, poses, poses_subset):

    pb, pc, ph, pw = first_frame.shape
    
    # print(poses.shape)

    pl = poses.shape[1]

    sparse_optical_flow, mask = get_sparse_flow(poses, ph, pw, pl)

    if ph != 384 or pw != 384:

        first_frame_384 = F.interpolate(first_frame, (384, 384))  # [3, 384, 384]

        poses_384 = torch.zeros_like(poses)
        poses_384[:, :, :, 0] = poses[:, :, :, 0] / pw * 384
        poses_384[:, :, :, 1] = poses[:, :, :, 1] / ph * 384

        sparse_optical_flow_384, mask_384 = get_sparse_flow(poses_384, 384, 384, pl)
    
    else:
        first_frame_384, poses_384 = first_frame, poses
        sparse_optical_flow_384, mask_384 = sparse_optical_flow, mask
    
    controlnet_image = first_frame

    return controlnet_image, sparse_optical_flow, mask, first_frame_384, sparse_optical_flow_384, mask_384

def pose2track(points_list, height, width):
    track_points = np.zeros((18, len(points_list), 2)) # 18 x f x 2
    track_points_subsets = np.zeros((18, len(points_list), 1)) # 18 x f x 2
    for f in range(len(points_list)):
        candidates, subsets, scores = points_list[f]['candidate'], points_list[f]['subset'][0], points_list[f]['score']
        for i in range(18):
            if subsets[i] == -1:
                track_points_subsets[i][f] = -1
            else:
                # track_points[i][f][0] = candidates[i][0]
                # track_points[i][f][1] = candidates[i][1]
                track_points[i][f][0] = max(min(candidates[i][0] * width, width-1), 0)
                track_points[i][f][1] = max(min(candidates[i][1] * height, height-1), 0)
                track_points_subsets[i][f] = i
    
    return track_points, track_points_subsets

def pose2track_batch(points_list, height, width, batch_size):
    track_points = np.zeros((batch_size, 18, len(points_list), 2)) # 18 x f x 2
    track_points_subsets = np.zeros((batch_size, 18, len(points_list), 1)) # 18 x f x 2
    for batch_idx in range(batch_size):
        for f in range(len(points_list)):
            candidates, subsets, scores = points_list[f]['candidate'][batch_idx], points_list[f]['subset'][batch_idx][0], points_list[f]['score'][batch_idx]
            for i in range(18):
                if subsets[i] == -1:
                    track_points_subsets[batch_idx][i][f] = -1
                else:
                    # track_points[i][f][0] = candidates[i][0]
                    # track_points[i][f][1] = candidates[i][1]
                    track_points[batch_idx][i][f][0] = max(min(candidates[i][0] * width, width-1), 0)
                    track_points[batch_idx][i][f][1] = max(min(candidates[i][1] * height, height-1), 0)
                    track_points_subsets[batch_idx][i][f] = i
    
    return track_points, track_points_subsets

def points_to_flows_batch(points_list, model_length, height, width, batch_size):

    track_points, track_points_subsets = pose2track_batch(points_list, height, width, batch_size)
    # model_length = track_points.shape[1]
    input_drag = np.zeros((batch_size, model_length - 1, height, width, 2))
    for batch_idx in range(batch_size):
        for splited_track, points_subset in zip(track_points[batch_idx], track_points_subsets[batch_idx]):
            if len(splited_track) == 1: # stationary point
                displacement_point = tuple([splited_track[0][0] + 1, splited_track[0][1] + 1])
                splited_track = tuple([splited_track[0], displacement_point])
            # interpolate the track
            # splited_track = interpolate_trajectory(splited_track, model_length)
            # splited_track = splited_track[:model_length]
            if len(splited_track) < model_length:
                splited_track = splited_track + [splited_track[-1]] * (model_length -len(splited_track))
            for i in range(model_length - 1):
                if points_subset[i]!=-1:
                    start_point = splited_track[i]
                    end_point = splited_track[i+1]
                    input_drag[batch_idx][i][int(start_point[1])][int(start_point[0])][0] = end_point[0] - start_point[0]
                    input_drag[batch_idx][i][int(start_point[1])][int(start_point[0])][1] = end_point[1] - start_point[1]
    return input_drag

def points_to_flows(points_list, model_length, height, width):
    
    track_points, track_points_subsets = pose2track(points_list, height, width)
    # model_length = track_points.shape[1]
    input_drag = np.zeros((model_length - 1, height, width, 2))

    for splited_track, points_subset in zip(track_points, track_points_subsets):
        if len(splited_track) == 1: # stationary point
            displacement_point = tuple([splited_track[0][0] + 1, splited_track[0][1] + 1])
            splited_track = tuple([splited_track[0], displacement_point])
        # interpolate the track
        # splited_track = interpolate_trajectory(splited_track, model_length)
        # splited_track = splited_track[:model_length]
        if len(splited_track) < model_length:
            splited_track = splited_track + [splited_track[-1]] * (model_length -len(splited_track))
        for i in range(model_length - 1):
            if points_subset[i]!=-1:
                start_point = splited_track[i]
                end_point = splited_track[i+1]
                input_drag[i][int(start_point[1])][int(start_point[0])][0] = end_point[0] - start_point[0]
                input_drag[i][int(start_point[1])][int(start_point[0])][1] = end_point[1] - start_point[1]
    return input_drag

def interpolate_trajectory(points, n_points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    t = np.linspace(0, 1, len(points))

    fx = PchipInterpolator(t, x)
    fy = PchipInterpolator(t, y)

    new_t = np.linspace(0, 1, n_points)

    new_x = fx(new_t)
    new_y = fy(new_t)
    new_points = list(zip(new_x, new_y))

    return new_points


def bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True):
    """Generate a bivariate isotropic or anisotropic Gaussian kernel.
    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.
    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
        isotropic (bool):
    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel

def mesh_grid(kernel_size):
    """Generate the mesh grid, centering at zero.
    Args:
        kernel_size (int):
    Returns:
        xy (ndarray): with the shape (kernel_size, kernel_size, 2)
        xx (ndarray): with the shape (kernel_size, kernel_size)
        yy (ndarray): with the shape (kernel_size, kernel_size)
    """
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)), yy.reshape(kernel_size * kernel_size,
                                                                           1))).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy


def pdf2(sigma_matrix, grid):
    """Calculate PDF of the bivariate Gaussian distribution.
    Args:
        sigma_matrix (ndarray): with the shape (2, 2)
        grid (ndarray): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size.
    Returns:
        kernel (ndarrray): un-normalized kernel.
    """
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel


def sigma_matrix2(sig_x, sig_y, theta):
    """Calculate the rotated sigma matrix (two dimensional matrix).
    Args:
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
    Returns:
        ndarray: Rotated sigma matrix.
    """
    d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))


def save_to_mp4(frames, save_path, fps=7):
    frames = frames.permute((0, 2, 3, 1))  # (f, c, h, w) to (f, h, w, c)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    write_video(save_path, frames, fps=fps)

def read_frames(video_path):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames

def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps


def save_videos_from_pil(pil_images, path, fps=8):
    import av

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=fps)

        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        stream.bit_rate = 10000000   
        stream.options["crf"] = "18"

        for pil_image in pil_images:
            # pil_image = Image.fromarray(image_arr).convert("RGB")
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")

