import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from pcl_helper import *


def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized


def compute_color_histograms(cloud, using_hsv=False):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])

    nbin = 32
    c1 = np.histogram(channel_1_vals, bins=nbin, range=(0,256))
    c2 = np.histogram(channel_2_vals, bins=nbin, range=(0,256))
    c3 = np.histogram(channel_3_vals, bins=nbin, range=(0,256))
    
    # Concatenate the histograms into a single feature vector
    ch_features = np.concatenate((c1[0], c2[0], c3[0])).astype(np.float64)

    # Normalize the result
    norm_features = ch_features / np.sum(ch_features)
    return norm_features


def compute_normal_histograms(normal_cloud):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])
        
    nbin = 32
    n1 = np.histogram(norm_x_vals, bins=nbin, range=(0,256))
    n2 = np.histogram(norm_y_vals, bins=nbin, range=(0,256))
    n3 = np.histogram(norm_z_vals, bins=nbin, range=(0,256))
    
    # Concatenate the histograms into a single feature vector
    n_features = np.concatenate((n1[0], n2[0], n3[0])).astype(np.float64)

    # Normalize the result
    norm_features = n_features / np.sum(n_features)

    return norm_features