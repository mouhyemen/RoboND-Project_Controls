#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

import pickle

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

# ----------------------- HELPER FUNCTIONS -----------------------
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

def voxel_filter(pcloud, leaf=0.01):
    # Voxel Grid Downsampling
    vox    = pcloud.make_voxel_grid_filter()
    leaf_size   = leaf   # leaf/voxel (volume-element) size
    vox.set_leaf_size(leaf_size, leaf_size, leaf_size)
    return vox.filter()  # voxel downsampled point cloud

def passthrough_filter(pcloud, axis='z', axis_min=0.76, axis_max=1.2):
    passthrough = pcloud.make_passthrough_filter()
    passthrough.set_filter_field_name(axis)
    passthrough.set_filter_limits(axis_min, axis_max)
    return passthrough.filter()

def segment_ransac(pcloud, max_dist = 0.01):
    seg_ransac  = pcloud.make_segmenter()
    seg_ransac.set_model_type(pcl.SACMODEL_PLANE)
    seg_ransac.set_method_type(pcl.SAC_RANSAC)
    seg_ransac.set_distance_threshold(max_dist)
    return seg_ransac.segment()

def euclidean_cluster(pcloud, tol=0.02, min_size=40, max_size=4000):
    tree    = pcloud.make_kdtree()
    ec      = pcloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(tol)
    ec.set_MinClusterSize(min_size)
    ec.set_MaxClusterSize(max_size)
    ec.set_SearchMethod(tree)   #Search k-d tree for clusters
    return ec.Extract()  #Indices for each clusters

def color_cluster(pcloud, cluster_indices):
    # Create Cluster-Mask Point Cloud to see each cluster
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, index in enumerate(indices):
            color_cluster_point_list.append([
                pcloud[index][0],
                pcloud[index][1],
                pcloud[index][2],
                rgb_to_float(cluster_color[j])
                ])

    #Create new cloud with all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    return cluster_cloud
# ----------------------------------------------------------------


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    # Convert ROS msg to PCL data
    pcl_cloud = ros_to_pcl(pcl_msg)

    # Voxel Grid Downsampling
    pcl_voxed   = voxel_filter(pcl_cloud)

    # PassThrough Filter
    pcl_passed  = passthrough_filter(pcl_voxed)

    # RANSAC Plane Segmentation
    inliers, coefficients = segment_ransac(pcl_passed) # Extract inliers
    
    outlier_objects = pcl_passed.extract(inliers, negative=True)
    inlier_table    = pcl_passed.extract(inliers, negative=False)

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(outlier_objects)
    cluster_indices = euclidean_cluster(white_cloud)  #Indices for each object cluster on the table

    # Color each individual cluster
    cluster_cloud   = color_cluster(white_cloud, cluster_indices)

    # Convert PCL data to ROS messages
    ros_cloud_objects   = pcl_to_ros(outlier_objects)
    ros_cloud_table     = pcl_to_ros(inlier_table)
    ros_cloud_cluster   = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_obj_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cloud_pub.publish(ros_cloud_cluster)

    # Classify the clusters! (loop through each detected cluster)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster  = outlier_objects.extract(pts_list)
        sample_cloud = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        colorHists  = compute_color_histograms(sample_cloud, using_hsv=True)
        normals     = get_normals(sample_cloud)
        normalHists = compute_normal_histograms(normals)
        
        # Compute the associated feature vector
        feature = np.concatenate((colorHists, normalHists))
        # labeled_features.append([feature, model_name])

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do          = DetectedObject()
        do.label    = label
        do.cloud    = ros_cloud_cluster
        detected_objects.append(do)

    # Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    detected_objects_pub.publish(detected_objects)

if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_obj_pub     = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=2)
    pcl_table_pub   = rospy.Publisher("/pcl_table", PointCloud2, queue_size=2)
    pcl_cloud_pub   = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=2)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=2)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=2)

    # TODO: Load Model From disk
    model = pickle.load(open('/home/mouhyemen/desktop/ros/catkin_ws/src/sensor_stick/scripts/model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()