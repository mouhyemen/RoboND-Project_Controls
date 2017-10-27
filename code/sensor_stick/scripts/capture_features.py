#!/usr/bin/env python
import numpy as np
import pickle
import rospy, progressbar

from sensor_stick.pcl_helper import *
from sensor_stick.training_helper import spawn_model
from sensor_stick.training_helper import delete_model
from sensor_stick.training_helper import initial_setup
from sensor_stick.training_helper import capture_sample
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from sensor_stick.srv import GetNormals
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


if __name__ == '__main__':
    rospy.init_node('capture_node')

    models = [\
       'biscuits',
       'soap2',
       'soap',
       'book',
       'glue',
       'sticky_notes',
       'snacks',
       'eraser'
       ]

    # Disable gravity and delete the ground plane
    initial_setup()
    labeled_features = []

    for count, model_name in enumerate(models):
        num_of_orientations = 50

        print "[MODEL] {}/{}. {}".format(count, len(models), model_name)
        spawn_model(model_name)
        
        # set up the progress bar
        widgets = [
        "Capturing Features:", progressbar.Percentage(), " ", 
        progressbar.Bar(), " ", progressbar.Timer(), " | ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=num_of_orientations, widgets=widgets).start()

        for i in range(num_of_orientations):
            pbar.update(i)
            # make five attempts to get a valid a point cloud then give up
            sample_was_good = False
            try_count = 0
            while not sample_was_good and try_count < 5:
                sample_cloud = capture_sample()
                sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()

                # Check for invalid clouds.
                if sample_cloud_arr.shape[0] == 0:
                    print('Invalid cloud detected instead of model {}'.format(model_name))
                    try_count += 1
                else:
                    sample_was_good = True

            # Extract histogram features
            chists  = compute_color_histograms(sample_cloud, using_hsv=True)
            normals = get_normals(sample_cloud)
            nhists  = compute_normal_histograms(normals)
            feature = np.concatenate((chists, nhists))
            labeled_features.append([feature, model_name])
        pbar.finish()

        delete_model()


    pickle.dump(labeled_features, open('/home/mouhyemen/desktop/ros/catkin_ws/src/sensor_stick/scripts/training_set.sav', 'wb'))

