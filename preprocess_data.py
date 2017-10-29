import cv2
from sklearn.model_selection import train_test_split
from log_loader import *
from sklearn.utils import shuffle


NR_OF_AUGM_DATA_PER_ITER = 6

# load images and convert them from BGR to RGB color space
def load_image_rgb(image_filename):
    image = cv2.imread("{}".format(image_filename))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Flips image horizontally
def flip_img_horizontal(image):
    image = cv2.flip(image, 1)
    return image

# preprocess image by resizing it to 64x64x3 thereby
# by cropping it from top and bottom as not all of these pixels contain useful information, however.
# The top portion of the image captures trees and hills and sky,
# and the bottom portion of the image captures the hood of the car.
def preprocess(image):
    image = cv2.resize(image, (64, 160))
    image = image[73:image.shape[0] - 23, 0:image.shape[1]]
    image = image.astype(np.float32)
    image_min = np.min(image)
    image_max = np.max(image)
    image = (image - image_min) / (image_max - image_min) - 0.5
    return image


class DataGenerator:

    def __init__(self, data):
        self.data = data
        self.batch_size = 512
        self.test_size = 0.2 #20% of data are used for validation
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            data['images'],
            data['steerings'],
            test_size=self.test_size,
            random_state=0)
        self.image_width = 64
        self.image_height = 64
        self.image_depth = 3

    def get_output_shape(self):
        return self.image_width, self.image_height, self.image_depth

    def create_train_data(self):
        while True:
            yield next(self.create_data(self.x_train, self.y_train))

    def create_validation_data(self):
        while True:
            yield next(self.create_data(self.x_valid, self.y_valid))

    def get_steps_per_epoch(self):
        return np.ceil(len(self.x_train) * NR_OF_AUGM_DATA_PER_ITER / self.batch_size)

    def get_validation_steps(self):
        return np.ceil(len(self.x_valid) * NR_OF_AUGM_DATA_PER_ITER / self.batch_size)

    # create new data using generator concept
    def create_data(self, x_data, y_data):

        x_gen = np.zeros([self.batch_size, self.image_width, self.image_height, self.image_depth])
        y_gen = np.zeros([self.batch_size])

        while True:
            # epoch
            x_data, y_data = shuffle(x_data, y_data)
            count_gen = 0

            for i in range(len(x_data)):

                # create data for center camera

                # 1. image
                center = load_image_rgb(x_data[i][0])
                center_steer = y_data[i][0]
                count_gen = self.create_single_data(center, center_steer, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                # 2. image
                image, steering = self.flip_data(center, center_steer)
                count_gen = self.create_single_data(image, steering, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                # create data for left camera

                left = load_image_rgb(x_data[i][1])
                left_steer = y_data[i][1]

                # 3. image
                count_gen = self.create_single_data(left, left_steer, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                # 4. image
                image, steering = self.flip_data(left, left_steer)
                count_gen = self.create_single_data(image, steering, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                # create data for right camera

                right = load_image_rgb(x_data[i][2])
                right_steer = y_data[i][2]

                # 5. image
                count_gen = self.create_single_data(right, right_steer, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                # 6. image
                image, steering = self.flip_data(right, right_steer)
                count_gen = self.create_single_data(image, steering, x_gen, y_gen, count_gen)
                if count_gen == 0:
                    yield x_gen, y_gen

                # NR_OF_AUGM_DATA_PER_ITER

            # Yield remainder
            if count_gen > 0:
                yield x_gen, y_gen

    def create_single_data(self, image, steering, x_gen, y_gen, count_gen):
        x_gen[count_gen] = preprocess(image)
        y_gen[count_gen] = steering

        count_gen += 1
        if count_gen >= self.batch_size:
            count_gen = 0
        return count_gen

    # flippes image and adapt it steering accordingly
    def flip_data(self, image, steering):
        image = flip_img_horizontal(image)
        return image, -steering


if __name__ == '__main__':
    print("Loading driving log..")
    log_loader = DrivingLogLoader('../3laps_data/driving_log.csv')
    logged_data = log_loader.load_log()
    data_generator = DataGenerator(logged_data)
    x_gen, y_gen = data_generator.create_train_data()
