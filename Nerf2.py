# -*- coding: utf-8 -*-
"""cis580fall2023_projB.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-rK2wAII9cZKQqKPB_9xOpAGx-90PCzI

## CIS 580, Machine Perception, Fall 2023
### Homework 5
#### Due: December 22 2023, 11:59pm ET

Instructions: Create a folder in your Google Drive and place inside this .ipynb file. Open the jupyter notebook with Google Colab. Refrain from using a GPU during implementing and testing the whole thing. You should switch to a GPU runtime only when performing the final training (of the 2D image or the NeRF) to avoid GPU usage runouts.

### Part 1: Fitting a 2D Image
"""

import numpy as np
import os
import gdown
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def positional_encoding(x, num_frequencies=0, incl_input=True):

    """
    Apply positional encoding to the input.

    Args:
    x (torch.Tensor): Input tensor to be positionally encoded.
      The dimension of x is [N, D], where N is the number of input coordinates,
      and D is the dimension of the input coordinate.
    num_frequencies (optional, int): The number of frequencies used in
     the positional encoding (default: 6).
    incl_input (optional, bool): If True, concatenate the input with the
        computed positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """

    results = []
    if incl_input:
        results.append(x)
    # encode input tensor and append the encoded tensor to the list of results.
    #N would be the pixles of the image, D would be the dimension of the input coordinate
    L = num_frequencies

    for i in range(L):
      sine_terms = torch.sin((x*2**i)*torch.pi)
      cosine_terms = torch.cos((x*2**i)*torch.pi)
      results.append(sine_terms)
      results.append(cosine_terms)

    return torch.cat(results, dim=-1)


def get_rays(height, width, intrinsics, w_R_c, w_T_c):

    """
    Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height: the height of an image.
    width: the width of an image.
    intrinsics: camera intrinsics matrix of shape (3, 3).
    w_R_c: Rotation matrix of shape (3,3) from camera to world coordinates.
    w_T_c: Translation vector of shape (3,1) that transforms

    Returns:
    ray_origins (torch.Tensor): A tensor of shape (height, width, 3) denoting the centers of
      each ray. Note that desipte that all ray share the same origin, here we ask you to return
      the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape (height, width, 3) denoting the
      direction of each ray.
    """

    device = intrinsics.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder

   
    ray_origins[:,:] = w_T_c

    K_inv = torch.inverse(intrinsics)

    for i in range(height):
      for j in range(width):
        homo_pix = torch.tensor([j,i,1.0],device = device)
        direction = torch.matmul(w_R_c, torch.matmul(K_inv,homo_pix))
        ray_directions[i,j] = direction
    return ray_origins, ray_directions


def stratified_sampling(ray_origins, ray_directions, near, far, samples):

    """
    Sample 3D points on the given rays. The near and far variables indicate the bounds of sampling range.

    Args:
    ray_origins: Origin of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    ray_directions: Direction of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    near: The 'near' extent of the bounding volume.
    far:  The 'far' extent of the bounding volume.
    samples: Number of samples to be drawn along each ray.

    Returns:
    ray_points: Query 3D points along each ray. Shape: (height, width, samples, 3).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
    """
    height, width, _ = ray_origins.shape
    device = ray_origins.device

    ray_points = []
    depth_points = []
    
    n = samples+1
    for i in range(1,n):
        ti = near + (i)*(far-near)/n
        ray_points.append((ti*ray_directions + ray_origins).unsqueeze(2))
        depth_points.append(torch.tensor([ti]).to(device))

    ray_points = torch.cat(ray_points, dim=2)
    a1 = torch.zeros((height, width, samples)).to(device)
    a2 = torch.stack(depth_points, dim=-1).to(device)
    depth_points = a1 + a2

    return ray_points, depth_points

class nerf_model(nn.Module):

    """
    Define a NeRF model comprising eight fully connected layers and following the
    architecture described in the NeRF paper.
    """

    def __init__(self, filter_size=256, num_x_frequencies=6, num_d_frequencies=3):
        super().__init__()

        self.layers = nn.ModuleDict({
            'layer_1': nn.Linear(3+(3*num_x_frequencies*2),filter_size),
            'layer_2': nn.Linear(filter_size,filter_size),
            'layer_3': nn.Linear(filter_size,filter_size),
            'layer_4': nn.Linear(filter_size,filter_size),
            'layer_5': nn.Linear(filter_size,filter_size),
            'layer_6': nn.Linear(filter_size + 3+(3*num_x_frequencies*2),filter_size),
            'layer_7': nn.Linear(filter_size,filter_size),
            'layer_8': nn.Linear(filter_size,filter_size),
            'layer_9': nn.Linear(filter_size,1),
            'layer_10': nn.Linear(filter_size,filter_size),
            'layer_11': nn.Linear(filter_size + 3+2*3*num_d_frequencies,filter_size//2),
            'layer_12': nn.Linear(filter_size//2,3),
        })
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()



    def forward(self, x, d):
        encoded_posit = x
        encoded_depth = d

        x = self.relu(self.layers['layer_1'](encoded_posit))
        x = self.relu(self.layers['layer_2'](x))
        x = self.relu(self.layers['layer_3'](x))
        x = self.relu(self.layers['layer_4'](x))
        x = self.relu(self.layers['layer_5'](x))
        x = torch.cat((x,encoded_posit),dim=-1)
        x = self.relu(self.layers['layer_6'](x))
        x = self.relu(self.layers['layer_7'](x))
        x = self.relu(self.layers['layer_8'](x))
        sigma = self.layers['layer_9'](x)
        x = self.layers['layer_10'](x)
        x = torch.cat((x,encoded_depth),dim=-1)
        x = self.relu(self.layers['layer_11'](x))
        x = self.layers['layer_12'](x)
        rgb = self.sigmoid(x)

        return rgb, sigma

def get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies):

    def get_chunks(inputs, chunksize = 2**15):
        """
        This fuction gets an array/list as input and returns a list of chunks of the initial array/list
        """
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

    """
    This function returns chunks of the ray points and directions to avoid memory errors with the
    neural network. It also applies positional encoding to the input points and directions before
    dividing them into chunks, as well as normalizing and populating the directions.
    """
    ray_points_batches = []
    ray_directions_batches = []
    device = ray_points.device
    _, _, sample_point_no, _ = ray_points.shape
    ray_directions_norm = torch.norm(ray_directions,dim=-1, keepdim=True)
    ray_directions = ray_directions / ray_directions_norm
    ray_directions_reshape = ray_directions.reshape(-1,3)
    iterate = ray_directions_reshape.unsqueeze(1).repeat(1,sample_point_no,1)

    pt_flatten = ray_points.reshape(-1,sample_point_no,3)

    #concatenating the ray point and direction to a dimension of product of height and width and the sample point number
    pt_dir = torch.cat((pt_flatten,iterate),dim=-1)
    #reshaping
    pt_dir_final = pt_dir.reshape(-1,6)
    #now applying positional encoding
    position = positional_encoding(pt_dir_final[:,:3],num_frequencies=num_x_frequencies, incl_input=True)
    direction = positional_encoding(pt_dir_final[:,3:],num_frequencies=num_d_frequencies, incl_input=True)
    #getting the chunks
    ray_directions_batches = get_chunks(direction, chunksize = 2**15)
    ray_points_batches = get_chunks(position, chunksize = 2**15)


    return ray_points_batches, ray_directions_batches

def volumetric_rendering(rgb, s, depth_points):

    """
    Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
    rgb: RGB color at each query location (X, Y, Z). Shape: (height, width, samples, 3).
    sigma: Volume density at each query location (X, Y, Z). Shape: (height, width, samples).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).

    Returns:
    rec_image: The reconstructed image after applying the volumetric rendering to every pixel.
    Shape: (height, width, 3)
    """
    device = rgb.device
    delta_i = torch.ones_like(depth_points).to(device) * 1e8
    delta_i[..., :-1] = torch.diff(depth_points, dim=-1)
    sig_i = -F.relu(s) * delta_i
    a = torch.exp(sig_i)
    b = torch.cumprod(a, dim=-1)

    # print("Sizes:")
    # print("depth_points:", depth_points.size())
    # print("delta_i:", delta_i.size())
    # print("sig_i:", sig_i.size())
    # print("a:", a.size())
    # print("b:", b.size())

    Transmittance = torch.roll(b, 1, dims=-1)
    Final_S_delta = 1 - a

    # print("Sizes after roll:")
    # print("Transmittance:", Transmittance.size())
    # print("Final_S_delta:", Final_S_delta.size())

    # rec_image = (Transmittance * Final_S_delta[..., None] * rgb).sum(dim=-2)
    rec_image = (Transmittance[..., None] * Final_S_delta[..., None] * rgb).sum(dim=-2)


    # print("Final rec_image size:", rec_image.size())

    return rec_image


def one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies):
    #compute all the rays from the image
    origin_r , direction_r = get_rays(height, width, intrinsics, pose[:3,:3], pose[:3,3])
   #sample the points from the rays
    ray_points, depth_points = stratified_sampling(origin_r, direction_r, near, far, samples)

    #divide data into batches to avoid memory errors
    ray_p_b, ray_dir_b = get_batches(ray_points, direction_r, num_x_frequencies, num_d_frequencies)

    #forward pass the batches and concatenate the outputs at the end
    rgb = []
    sigma = []
    L = len(ray_p_b)
    for i in range(L):
        rgb_b, sigma_b = model.forward(ray_p_b[i],ray_dir_b[i])
        rgb.append(rgb_b)
        sigma.append(sigma_b)
    rgb = torch.cat(rgb,dim=0)
    sigma = torch.cat(sigma,dim=0)
    rgb_final = rgb.reshape(height,width,samples,3)
    sigma_final = sigma.reshape(height,width,samples)

    #apply volumetric rendering to obtain the reconstructed image
    rec_image = volumetric_rendering(rgb_final, sigma_final, depth_points)

    return rec_image

