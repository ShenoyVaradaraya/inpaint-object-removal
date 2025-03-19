import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import laplace
from scipy.ndimage.filters import convolve


class inpainting():
    def __init__(self, img_, mask, patch_size=9):
        self.img = img_.astype('uint8')
        self.mask = mask.round().astype('uint8')
        self.patch_size = patch_size
        self.img_input = None
        self.mask_input = None
        self.front_img = None
        self.confidence_parameter = None
        self.data = None
        self.priority_parameter = None

    def driver(self):
        self.check_inputs()
        self.attributes_initialization()

        start_time = time.time()
        continue_ = True
        while continue_:
            self.hunt_front()

            self.priority_update()

            target_pixel = self.pixel_with_highest_priority()
            find_start_time = time.time()
            source_patch = self.determine_source(target_pixel)
            print('Found best in %f seconds'
                  % (time.time()-find_start_time))

            self.image_updater(target_pixel, source_patch)

            continue_ = not self.completed()

        print('Completion time: %f seconds' % (time.time() - start_time))
        return self.img_input

    def check_inputs(self):
        if self.img.shape[:2] != self.mask.shape:
            raise AttributeError('img_ and mask are not matching in size')

    def _plot_img_(self):
        inverted_mask = 1 - self.mask_input
        inverted_rgb_mask = self.convert_to_rgb(inverted_mask)
        img_ = self.img_input * inverted_rgb_mask

        img_[:, :, 0] += self.front_img * 255

        white_part = (self.mask_input - self.front_img) * 255
        rgb_white_part = self.convert_to_rgb(white_part)
        img_ += rgb_white_part

    def attributes_initialization(self):

        h, w = self.img.shape[:2]

        self.confidence_parameter = (1 - self.mask).astype(float)
        self.data = np.zeros([h, w])

        self.img_input = np.copy(self.img)
        self.mask_input = np.copy(self.mask)

    def hunt_front(self):
        self.front_img = (laplace(self.mask_input) > 0).astype('uint8')

    def priority_update(self):
        self.confidence_update()
        self.data_updater()
        self.priority_parameter = self.confidence_parameter * self.data * self.front_img

    def confidence_update(self):
        confidence1 = np.copy(self.confidence_parameter)
        front_coord = np.argwhere(self.front_img == 1)
        for pt in front_coord:
            patch = self.fetch_patch(pt)
            confidence1[pt[0], pt[1]] = sum(sum(
                self.data_of_patch(self.confidence_parameter, patch)
            ))/self.area_of_patch(patch)

        self.confidence_parameter = confidence1

    def data_updater(self):
        normal = self.normal_matrix()
        grad = self.gradient_matrix()

        normal_grad = normal*grad
        self.data = np.sqrt(
            normal_grad[:, :, 0]**2 + normal_grad[:, :, 1]**2
        ) + 0.001

    def normal_matrix(self):
        x = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        y = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])

        norm_x = convolve(self.mask_input.astype(float), x)
        norm_y = convolve(self.mask_input.astype(float), y)
        norm = np.dstack((norm_x, norm_y))

        h, w = norm.shape[:2]
        norm = np.sqrt(norm_y**2 + norm_x**2) \
                 .reshape(h, w, 1) \
                 .repeat(2, axis=2)
        norm[norm == 0] = 1

        unit_norm = norm/norm
        return unit_norm

    def gradient_matrix(self):
        h, w = self.img_input.shape[:2]

        img_grayed = rgb2gray(self.img_input)
        img_grayed[self.mask_input == 1] = None

        gradient = np.nan_to_num(np.array(np.gradient(img_grayed)))
        gradient_value = np.sqrt(gradient[0]**2 + gradient[1]**2)
        max_grad = np.zeros([h, w, 2])

        front_coord = np.argwhere(self.front_img == 1)
        for pt in front_coord:
            patch = self.fetch_patch(pt)
            patch_y_gradient = self.data_of_patch(gradient[0], patch)
            patch_x_gradient = self.data_of_patch(gradient[1], patch)
            patch_gradient_value = self.data_of_patch(gradient_value, patch)

            patch_max_pos = np.unravel_index(
                patch_gradient_value.argmax(),
                patch_gradient_value.shape
            )

            max_grad[pt[0], pt[1], 0] = \
                patch_y_gradient[patch_max_pos]
            max_grad[pt[0], pt[1], 1] = \
                patch_x_gradient[patch_max_pos]

        return max_grad

    def pixel_with_highest_priority(self):
        pt = np.unravel_index(
            self.priority_parameter.argmax(), self.priority_parameter.shape)
        return pt

    def determine_source(self, target_pixel):
        target = self.fetch_patch(target_pixel)
        h, w = self.img_input.shape[:2]
        patch_h, patch_w = self.patch_shape(target)

        best_match = None
        best_diff = 0

        lab_img_ = rgb2lab(self.img_input)

        for y in range(h - patch_h + 1):
            for x in range(w - patch_w + 1):
                source_patch = [
                    [y, y + patch_h-1],
                    [x, x + patch_w-1]
                ]
                if self.data_of_patch(self.mask_input, source_patch) \
                   .sum() != 0:
                    continue

                diff = self.find_patch_diff(
                    lab_img_,
                    target,
                    source_patch
                )

                if best_match is None or diff < best_diff:
                    best_match = source_patch
                    best_diff = diff
        return best_match

    def image_updater(self, target_pixel, source_patch):
        target_patch = self.fetch_patch(target_pixel)
        pixel_pos = np.argwhere(
            self.data_of_patch(
                self.mask_input,
                target_patch
            ) == 1
        ) + [target_patch[0][0], target_patch[1][0]]
        patch_confidence = self.confidence_parameter[target_pixel[0],
                                                     target_pixel[1]]
        for pt in pixel_pos:
            self.confidence_parameter[pt[0], pt[1]] = patch_confidence

        mask = self.data_of_patch(self.mask_input, target_patch)
        rgb_mask = self.convert_to_rgb(mask)
        source_data = self.data_of_patch(self.img_input, source_patch)
        target_data = self.data_of_patch(self.img_input, target_patch)

        new_data = source_data*rgb_mask + target_data*(1-rgb_mask)

        self.copy_to_patch(
            self.img_input,
            target_patch,
            new_data
        )
        self.copy_to_patch(
            self.mask_input,
            target_patch,
            0
        )

    def fetch_patch(self, pt):
        half_patch_size = (self.patch_size-1)//2
        h, w = self.img_input.shape[:2]
        patch = [
            [
                max(0, pt[0] - half_patch_size),
                min(pt[0] + half_patch_size, h-1)
            ],
            [
                max(0, pt[1] - half_patch_size),
                min(pt[1] + half_patch_size, w-1)
            ]
        ]
        return patch

    def find_patch_diff(self, img_, target_patch, source_patch):
        mask = 1 - self.data_of_patch(self.mask_input, target_patch)
        rgb_mask = self.convert_to_rgb(mask)
        target_data = self.data_of_patch(
            img_,
            target_patch
        ) * rgb_mask
        source_data = self.data_of_patch(
            img_,
            source_patch
        ) * rgb_mask
        squared_distance = ((target_data - source_data)**2).sum()
        euclidean_distance = np.sqrt(
            (target_patch[0][0] - source_patch[0][0])**2 +
            (target_patch[1][0] - source_patch[1][0])**2
        )  # tie-breaker factor
        return squared_distance + euclidean_distance

    def completed(self):
        h, w = self.img_input.shape[:2]
        remaining = self.mask_input.sum()
        total = h * w
        print('%d of %d completed' % (total-remaining, total))
        return remaining == 0

    @staticmethod
    def area_of_patch(patch):
        return (1+patch[0][1]-patch[0][0]) * (1+patch[1][1]-patch[1][0])

    @staticmethod
    def patch_shape(patch):
        return (1+patch[0][1]-patch[0][0]), (1+patch[1][1]-patch[1][0])

    @staticmethod
    def data_of_patch(source, patch):
        return source[
            patch[0][0]:patch[0][1]+1,
            patch[1][0]:patch[1][1]+1
        ]

    @staticmethod
    def copy_to_patch(dest, dest_patch, data):
        dest[
            dest_patch[0][0]:dest_patch[0][1]+1,
            dest_patch[1][0]:dest_patch[1][1]+1
        ] = data

    @staticmethod
    def convert_to_rgb(img_):
        h, w = img_.shape
        return img_.reshape(h, w, 1).repeat(3, axis=2)
