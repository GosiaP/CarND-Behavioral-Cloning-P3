import csv
import datetime
import re
import numpy as np
import matplotlib.pyplot as plt
import random

# Factor in between steering unit and degree
STEERING_FACTOR = 25.0

# loader for data stored as driving log;
# images + steering angle form center, left and right camera
class DrivingLogLoader:


    def __init__(self, filepath):
        self.file_path = filepath

    # loads data from logfile - only path to images and steering angles
    # for 3 cameras
    def load_log(self):
        with open(self.file_path) as csvfile:
            reader = csv.reader(csvfile, skipinitialspace=True)
            image_list = []
            steering_list = []

            for line in reader:
                image_list.append([line[0], line[1], line[2]])
                steering_center = float(line[3])

                # create adjusted steering measurements for the left
                #  and right camera images
                correction = 0.2  # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                steering_list.append([steering_center, steering_left, steering_right])

        images = np.array(image_list)
        steerings = np.array(steering_list)

        return {'images': images, 'steerings': steerings}

    # shows data as bar graphs
    def show_bar_graph(self, data, title=None):
        x = data[1]
        y = data[0]
        fig, axis = plt.subplots(figsize=(15, 5))
        axis.set_axisbelow(True)
        axis.grid(True, linestyle=":")
        x_bar = np.zeros(len(y))
        for i in range(len(y)):
            x_bar[i] = (x[i] + x[i + 1]) / 2.0
        axis.bar(x_bar, y, width=x[1] - x[0], edgecolor="#000077")
        if title is not None:
            plt.title(title)
        plt.show()

    # filters the driving log data, normalizing feature frequency based on histogram
    def filter_log_data(self, log_data):

        # Calculate histogram with 1° angle resolution
        images = log_data['images']
        steering = log_data['steerings']
        histogram = np.histogram(steering[:, 0:1], bins=np.arange(-1.0, 1.0, 1.0 / STEERING_FACTOR))
        self.show_bar_graph(histogram, title='Steering angle histogram before filtering')

        # calculate keeping probabilities;
        # every history bin shall have not more than 2^0.5 (1.414) times entries
        # than the mean count of all filled bins
        bins_count = np.count_nonzero(histogram[0])
        bins_mean = sum(histogram[0]) / bins_count
        bins_max = np.max(histogram[0])
        bins_target_max = bins_mean * 1.414
        bins_factor = (bins_target_max - bins_mean) / (bins_max - bins_mean)
        keep_probabilities = np.ones(len(histogram[0]))
        for i in range(len(histogram[0])):
            if histogram[0][i] > bins_mean:
                keep_probabilities[i] = (bins_mean + (histogram[0][i] - bins_mean) * bins_factor) / histogram[0][i]

        # filter logged data according to calculated keep probability
        filtered_images = []
        filtered_steering = []
        for i in range(len(steering)):
            steering_center = steering[i][0]
            keep_it = keep_probabilities[int(np.floor((1.0 + steering_center) * STEERING_FACTOR))]
            if random.uniform(0.0, 1.0) <= keep_it:
                filtered_images.append(images[i])
                filtered_steering.append(steering[i])

        result_images = np.array(filtered_images)
        result_steerings = np.array(filtered_steering)

        histogram = np.histogram(result_steerings[:, 0:1], bins=np.arange(-1.0, 1.0, 1.0 / STEERING_FACTOR))
        self.show_bar_graph(histogram, title='Steering angle histogram after filtering')
        return {'images': result_images, 'steerings': result_steerings}

if __name__ == '__main__':

    log_loader = DrivingLogLoader('../3laps_data/driving_log.csv')
    print("Loading driving log..")
    log_data = log_loader.load_log()

    print("Filtering driving log..")
    log_data = log_loader.filter_log_data(log_data)

