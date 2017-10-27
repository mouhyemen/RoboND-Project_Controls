#!/usr/bin/env python

# Import modules
from pcl_helper import *
import pcl
import rospy
import struct
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField


# Define functions as required
def ros_to_pcl(ros_cloud):
    """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB
    
        Args:
            ros_cloud (PointCloud2): ROS PointCloud2 message
            
        Returns:
            pcl.PointCloud_PointXYZRGB: PCL XYZRGB point cloud
    """
    points_list = []

    for data in pc2.read_points(ros_cloud, skip_nans=True):
        points_list.append([data[0], data[1], data[2], data[3]])

    pcl_data = pcl.PointCloud_PointXYZRGB()
    pcl_data.from_list(points_list)

    return pcl_data

def pcl_to_ros(pcl_array):
    """ Converts a pcl PointXYZRGB to a ROS PointCloud2 message
    
        Args:
            pcl_array (PointCloud_PointXYZRGB): A PCL XYZRGB point cloud
            
        Returns:
            PointCloud2: A ROS point cloud
    """
    ros_msg = PointCloud2()

    ros_msg.header.stamp = rospy.Time.now()
    ros_msg.header.frame_id = "world"

    ros_msg.height = 1
    ros_msg.width = pcl_array.size

    ros_msg.fields.append(PointField(
                            name="x",
                            offset=0,
                            datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(PointField(
                            name="y",
                            offset=4,
                            datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(PointField(
                            name="z",
                            offset=8,
                            datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(PointField(
                            name="rgb",
                            offset=16,
                            datatype=PointField.FLOAT32, count=1))

    ros_msg.is_bigendian = False
    ros_msg.point_step = 32
    ros_msg.row_step = ros_msg.point_step * ros_msg.width * ros_msg.height
    ros_msg.is_dense = False
    buffer = []

    for data in pcl_array:
        s = struct.pack('>f', data[3])
        i = struct.unpack('>l', s)[0]
        pack = ctypes.c_uint32(i).value

        r = (pack & 0x00FF0000) >> 16
        g = (pack & 0x0000FF00) >> 8
        b = (pack & 0x000000FF)

        buffer.append(struct.pack('ffffBBBBIII', data[0], data[1], data[2], 1.0, b, g, r, 0, 0, 0, 0))

    ros_msg.data = "".join(buffer)

    return ros_msg

def XYZRGB_to_XYZ(XYZRGB_cloud):
    """ Converts a PCL XYZRGB point cloud to an XYZ point cloud (removes color info)
    
        Args:
            XYZRGB_cloud (PointCloud_PointXYZRGB): A PCL XYZRGB point cloud
            
        Returns:
            PointCloud_PointXYZ: A PCL XYZ point cloud
    """
    XYZ_cloud = pcl.PointCloud()
    points_list = []

    for data in XYZRGB_cloud:
        points_list.append([data[0], data[1], data[2]])

    XYZ_cloud.from_list(points_list)
    return XYZ_cloud

def XYZ_to_XYZRGB(XYZ_cloud, color):
    """ Converts a PCL XYZ point cloud to a PCL XYZRGB point cloud
    
        All returned points in the XYZRGB cloud will be the color indicated
        by the color parameter.
    
        Args:
            XYZ_cloud (PointCloud_XYZ): A PCL XYZ point cloud
            color (list): 3-element list of integers [0-255,0-255,0-255]
            
        Returns:
            PointCloud_PointXYZRGB: A PCL XYZRGB point cloud
    """
    XYZRGB_cloud = pcl.PointCloud_PointXYZRGB()
    points_list = []

    float_rgb = rgb_to_float(color)

    for data in XYZ_cloud:
        points_list.append([data[0], data[1], data[2], float_rgb])

    XYZRGB_cloud.from_list(points_list)
    return XYZRGB_cloud

def rgb_to_float(color):
    """ Converts an RGB list to the packed float format used by PCL
    
        From the PCL docs:
        "Due to historical reasons (PCL was first developed as a ROS package),
         the RGB information is packed into an integer and casted to a float"
    
        Args:
            color (list): 3-element list of integers [0-255,0-255,0-255]
            
        Returns:
            float_rgb: RGB value packed as a float
    """
    hex_r = (0xff & color[0]) << 16
    hex_g = (0xff & color[1]) << 8
    hex_b = (0xff & color[2])

    hex_rgb = hex_r | hex_g | hex_b

    float_rgb = struct.unpack('f', struct.pack('i', hex_rgb))[0]

    return float_rgb
# --------------------------------------------------------------


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Convert ROS msg to PCL data
    pcl_cloud = ros_to_pcl(pcl_msg)
    print "1.[ROS->PCL]   Conversion done ..."

    # Voxel Grid Downsampling
    vox         = pcl_cloud.make_voxel_grid_filter()
    LEAF_SIZE   = 0.01   # leaf/voxel (volume-element) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    pcl_voxed   = vox.filter()  # voxel downsampled point cloud
    print "2.[SAMPLE]     Voxel downsampling done ..."

    # PassThrough Filter
    passthrough = pcl_voxed.make_passthrough_filter()
    filter_axis     = 'z'   # create a filter axis
    passthrough.set_filter_field_name(filter_axis)
    axis_min    = 0.76
    axis_max    = 1.2
    passthrough.set_filter_limits(axis_min, axis_max)
    
    pcl_passed  = passthrough.filter()
    print "3.[FILTER]     Passthrough filtering done ..."

    # RANSAC Plane Segmentation
    seg_ransac  = pcl_passed.make_segmenter()
    seg_ransac.set_model_type(pcl.SACMODEL_PLANE)
    seg_ransac.set_method_type(pcl.SAC_RANSAC)
    max_dist    = 0.01

    seg_ransac.set_distance_threshold(max_dist)
    print "4.[SEGMENT]    RANSAC Segmentation done ..."

    # Extract inliers and outliers
    inliers, coefficients = seg_ransac.segment()
    pcl_objects = pcl_passed.extract(inliers, negative=True)
    pcl_table   = pcl_passed.extract(inliers, negative=False)
    print "5.[EXTRACT]    Inliers and Outliers extracted ..."

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(pcl_objects)
    tree = white_cloud.make_kdtree()
    print "6.[CLUSTER]    White Cloud constructed ..."

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.025)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(2000)

    ec.set_SearchMethod(tree)   #Search k-d tree for clusters
    cluster_indices = ec.Extract()  #Indices for each clusters

    # Create Cluster-Mask Point Cloud to see each cluster
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, index in enumerate(indices):
            color_cluster_point_list.append([
                white_cloud[index][0],
                white_cloud[index][1],
                white_cloud[index][2],
                rgb_to_float(cluster_color[j])
                ])

    #Create new cloud with all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)


    # Convert PCL data to ROS messages
    ros_cloud_objects   = pcl_to_ros(pcl_objects)
    ros_cloud_table     = pcl_to_ros(pcl_table)
    ros_cloud_cluster   = pcl_to_ros(cluster_cloud)
    print "7.[PCL->ROS]   Conversion done ..."

    # Publish ROS messages
    pcl_obj_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cloud_pub.publish(ros_cloud_cluster)
    print "8.[PUBLISH]    Object, Table, & Clusters publishing done ..."
    print ""


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    pcl_obj_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=2)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=2)
    pcl_cloud_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=2)

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()

