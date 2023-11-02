import numpy as np
# from datetime import datetime
import scipy.io as sio
import os
import hdf5storage
from PIL import Image
import tifffile
from shapely import geometry
import random
from utils import plot_spikes_raster
from utils import get_continous_time_periods, welsh_powell
from utils import CellsCoord
import PIL
from PIL import ImageDraw
from sortedcontainers import SortedDict

# useful on mac to create movie from fig
# matplotlib.use('agg')
# matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import matplotlib.cm as cm
import networkx as nx

# from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import cv2
import time
from datetime import datetime

import matplotlib.pylab as plt


def produce_cell_coord_from_cnn_validated_cells(param):
    # TODO: Make a new version that take either suite2P coords or caiman coords
    path_cnn_classifier = "cell_classifier_results_txt/v_suite_2p"

    # ms_to_use = ["p7_171012_a000_ms", "p8_18_10_24_a005_ms", "p9_18_09_27_a003_ms", "p11_17_11_24_a000_ms",
    #              "p12_171110_a000_ms", "p13_18_10_29_a001_ms"]
    ms_to_use = ["p5_19_03_25_a001_ms", "p7_171012_a000_ms", "p8_18_10_24_a005_ms",
                 "p12_171110_a000_ms", ]

    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_to_use,
                                            param=param,
                                            load_traces=False, load_abf=False,
                                            for_transient_classifier=True)
    coords_to_keep = []
    true_cells = []
    fake_cells = []
    global_cell_index = 0

    for ms in ms_str_to_ms_dict.values():
        path_data = param.path_data

        cnn_file_name = None
        # finding the cnn_file coresponding to the ms
        for (dirpath, dirnames, local_filenames) in os.walk(os.path.join(path_data, path_cnn_classifier)):
            for file_name in local_filenames:
                if file_name.endswith(".txt"):
                    if ms.description.lower() in file_name.lower():
                        cnn_file_name = file_name
                        break
            # looking only in the top directory
            break

        if cnn_file_name is None:
            print(f"{ms.description} no cnn file_name")
            continue

        cell_cnn_predictions = []
        with open(os.path.join(path_data, path_cnn_classifier, cnn_file_name), "r", encoding='UTF-8') as file:
            for nb_line, line in enumerate(file):
                line_list = line.split()
                cells_list = [float(i) for i in line_list]
                cell_cnn_predictions.extend(cells_list)
        cell_cnn_predictions = np.array(cell_cnn_predictions)
        cells_predicted_as_true = np.where(cell_cnn_predictions >= 0.5)[0]

        print(f"ms.coord_obj.coord[0].shape {ms.coord_obj.coord[0].shape}")
        print(f"{ms.description}: n_cells: {ms.coord_obj.n_cells}")
        print(f"{ms.description}: cells_predicted_as_true: {len(cells_predicted_as_true)}")

        for cell in np.arange(ms.coord_obj.n_cells):
            coords_to_keep.append(ms.coord_obj.coord[cell])
            if cell in cells_predicted_as_true:
                true_cells.append(global_cell_index)
            else:
                fake_cells.append(global_cell_index)
            global_cell_index += 1

    print(f"len(coords_to_keep): {len(coords_to_keep)}")
    coords_matlab_style = np.empty((len(coords_to_keep),), dtype=np.object)
    for i in range(len(coords_to_keep)):
        coords_matlab_style[i] = coords_to_keep[i]

    true_cells = np.array(true_cells)
    fake_cells = np.array(fake_cells)

    sio.savemat(os.path.join(param.path_results, "coords_artificial_movie.mat"),
                {"coord": coords_matlab_style, "true_cells": true_cells, "fake_cells": fake_cells})


def shift_cell_coord_to_centroid(centroid, cell_coord, from_matlab=False):
    # it is necessary to remove one, as data comes from matlab, starting from 1 and not 0
    if from_matlab:
        cell_coord = cell_coord - 1
    cell_coord = cell_coord.astype(int)
    coord_list_tuple = []
    for n in np.arange(cell_coord.shape[1]):
        coord_list_tuple.append((cell_coord[0, n], cell_coord[1, n]))

    poly_cell = geometry.Polygon(coord_list_tuple)
    centroid_point = poly_cell.centroid
    # print(f"centroid {centroid} centroid[0] {centroid[0]}")
    # print(f"centroid_point.x {centroid_point.x}, centroid_point.y {centroid_point.y}")
    x_shift = centroid[0] - centroid_point.x
    y_shift = centroid[1] - centroid_point.y
    # print(f"x_shift {x_shift}, y_shift {y_shift}")
    for n in np.arange(cell_coord.shape[1]):
        cell_coord[0, n] = cell_coord[0, n] + x_shift
        cell_coord[1, n] = cell_coord[1, n] + y_shift

    coord_list_tuple = []
    for n in np.arange(cell_coord.shape[1]):
        coord_list_tuple.append((cell_coord[0, n], cell_coord[1, n]))
    poly_cell = geometry.Polygon(coord_list_tuple)

    # cell_coord = cell_coord + 1

    return cell_coord, poly_cell


def change_polygon_centroid(new_centroid, poly_cell):
    centroid_point = poly_cell.centroid
    x_shift = new_centroid[0] - centroid_point.x
    y_shift = new_centroid[1] - centroid_point.y
    coords = poly_cell.exterior.coords
    new_coords = []
    for coord in coords:
        new_coords.append((coord[0] + x_shift, coord[1] + y_shift))
    poly_cell = geometry.Polygon(new_coords)

    return poly_cell


def make_video(images, outvid=None, fps=5, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.

    @param      outvid      output video file_name
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    fourcc = cv2.VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        # if not os.path.exists(image):
        #     raise FileNotFoundError(image)
        # img = imread(image)
        img = image
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = cv2.VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = cv2.resize(img, size)
        vid.write(img)
    vid.release()
    return vid


def fig2data(fig):
    """
    http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())  # "RGBA"


def normalize_array_0_255(img_array):
    minv = np.amin(img_array)
    # minv = 0
    maxv = np.amax(img_array)
    if maxv - minv == 0:
        img_array = img_array.astype(np.uint8)
    else:
        img_array = (255 * (img_array - minv) / (maxv - minv)).astype(np.uint8)
    return img_array


# from https://stackoverflow.com/questions/14435632/impulse-gaussian-and-salt-and-pepper-noise-with-opencv
def noisy(noise_typ, image):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
    :param img_array:
    :return:
    """
    if noise_typ == "gauss":
        mean = 0
        var = 0.1
        sigma = var ** 0.5

        if len(image.shape) == 3:
            row, col, ch = image.shape
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
        else:
            row, col = image.shape
            gauss = np.random.normal(mean, sigma, (row, col))
            gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        if len(image.shape) == 3:
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
        else:
            row, col = image.shape
            gauss = np.random.randn(row, col)
            gauss = gauss.reshape(row, col)
        noisy = image + image * gauss
        return noisy


class CellPiece:

    def __init__(self, id, poly_gon, dimensions, activity_mask=None, frame_mode=False):
        self.id = id
        self.poly_gon = poly_gon
        self.dimensions = dimensions
        self.mask = self.get_mask()
        # TODO: keep a smaller mask to save memory
        self.activity_mask = activity_mask
        # if True, means we work frame by frame
        self.frame_mode = frame_mode
        self.parents = []
        self.daughters = []

    def fill_movie_images(self, images):
        if self.frame_mode:
            images[self.mask] = self.activity_mask[self.mask]
        else:
            images[:, self.mask] = self.activity_mask[:, self.mask]

    def set_activity_mask_from_other(self, other_activity_mask):
        self.activity_mask = np.zeros(other_activity_mask.shape)
        if self.frame_mode:
            self.activity_mask[self.mask] = other_activity_mask[self.mask]
        else:
            self.activity_mask[:, self.mask] = other_activity_mask[:, self.mask]

    def set_activity_mask_from_two_other(self, other_1, other_2):
        self.activity_mask = np.zeros(other_1.shape)
        if self.frame_mode:
            other_1_sum = np.sum(other_1[self.mask])
            other_2_sum = np.sum(other_2[self.mask])
            if other_1_sum >= other_2_sum:
                self.activity_mask[self.mask] = other_1[self.mask]
            else:
                self.activity_mask[self.mask] = other_2[self.mask]
        else:
            for frame in np.arange(other_1.shape[0]):
                other_1_sum = np.sum(other_1[frame, self.mask])
                other_2_sum = np.sum(other_2[frame, self.mask])
                if other_1_sum >= other_2_sum:
                    self.activity_mask[frame, self.mask] = other_1[frame, self.mask]
                else:
                    self.activity_mask[frame, self.mask] = other_2[frame, self.mask]

    def get_mask(self):
        img = PIL.Image.new('1', (self.dimensions[1], self.dimensions[0]), 0)
        try:
            ImageDraw.Draw(img).polygon(list(self.poly_gon.exterior.coords), outline=1,
                                        fill=1)
        except AttributeError:
            # print(f"list(self.poly_gon.coords) {list(self.poly_gon.coords)}")
            ImageDraw.Draw(img).polygon(list(self.poly_gon.coords), outline=1,
                                        fill=1)
        return np.array(img)

    def split(self, other):
        new_pieces = []
        intersection_polygon = self.poly_gon.intersection(other.poly_gon)
        # we need to re-build the activity mask depending on which cell in the most active at each frame
        new_id = self.id + "-" + other.id
        geoms = []
        try:
            geoms.extend(intersection_polygon.geoms)
        except AttributeError:
            geoms.append(intersection_polygon)
        id_ext = ["", "*", ".", "%", "a", "b"]
        for geom_index, geom in enumerate(geoms):
            # testing if there is more than one point in the polygon
            try:
                if len(geom.exterior.coords) < 2:
                    continue
            except AttributeError:
                if len(geom.coords) < 2:
                    continue
            inter_cell_piece = CellPiece(id=new_id + id_ext[geom_index], poly_gon=geom, dimensions=self.dimensions,
                                         frame_mode=self.frame_mode)
            inter_cell_piece.set_activity_mask_from_two_other(self.activity_mask, other.activity_mask)
            inter_cell_piece.parents = [self, other]
            self.daughters.append(inter_cell_piece)
            other.daughters.append(inter_cell_piece)
            new_pieces.append(inter_cell_piece)

        diff_poly_gon = self.poly_gon.difference(other.poly_gon)
        geoms = []
        try:
            geoms.extend(diff_poly_gon.geoms)
        except AttributeError:
            geoms.append(diff_poly_gon)
        for geom_index, geom in enumerate(geoms):
            try:
                if len(geom.exterior.coords) < 2:
                    continue
            except AttributeError:
                if len(geom.coords) < 2:
                    continue
            diff_cell_piece = CellPiece(id=self.id + id_ext[geom_index], poly_gon=geom, dimensions=self.dimensions,
                                        frame_mode=self.frame_mode)
            diff_cell_piece.set_activity_mask_from_other(self.activity_mask)
            diff_cell_piece.parents = [self]
            self.daughters.append(diff_cell_piece)
            new_pieces.append(diff_cell_piece)

        diff_other_poly_gon = other.poly_gon.difference(self.poly_gon)
        geoms = []
        try:
            geoms.extend(diff_other_poly_gon.geoms)
        except AttributeError:
            geoms.append(diff_other_poly_gon)
        for geom_index, geom in enumerate(geoms):
            try:
                if len(geom.exterior.coords) < 2:
                    continue
            except AttributeError:
                if len(geom.coords) < 2:
                    continue
            diff_other_cell_piece = CellPiece(id=other.id + id_ext[geom_index], poly_gon=geom,
                                              dimensions=other.dimensions,
                                              frame_mode=self.frame_mode)
            diff_other_cell_piece.set_activity_mask_from_other(other.activity_mask)
            diff_other_cell_piece.parents = [other]
            other.daughters.append(diff_other_cell_piece)
            new_pieces.append(diff_other_cell_piece)

        return new_pieces

    def update_activity_mask(self, activity_mask=None):
        if len(self.parents) == 0:
            if activity_mask is not None:
                self.activity_mask = activity_mask
            return
        for parent in self.parents:
            parent.update_activity_mask()
        if len(self.parents) == 1:
            self.set_activity_mask_from_other(self.parents[0].activity_mask)
        if len(self.parents) == 2:
            self.set_activity_mask_from_two_other(self.parents[0].activity_mask, self.parents[1].activity_mask)

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


def get_mask(dimensions, poly_gon):
    img = PIL.Image.new('1', (dimensions[1], dimensions[0]), 0)
    try:
        ImageDraw.Draw(img).polygon(list(poly_gon.exterior.coords), outline=1,
                                    fill=1)
    except AttributeError:
        ImageDraw.Draw(img).polygon(list(poly_gon.coords), outline=1,
                                    fill=1)
    return np.array(img)


def get_weighted_activity_mask_for_a_cell(mask, soma_mask, n_pixels, n_pixels_soma):
    mu, sigma = 100, 15  # mean and standard deviation
    mu_soma = random.randint(50, 70)
    sigma_soma = mu_soma // 6

    weighted_mask = np.zeros(mask.shape)

    weighted_mask = weighted_mask.reshape(mask.shape[0] * mask.shape[1])  # flattening
    n_pixels = np.sum(mask)
    weighted_mask[(mask.reshape(mask.shape[0] * mask.shape[1])) > 0] = \
        np.random.normal(loc=mu, scale=sigma, size=n_pixels)
    if len(np.where(weighted_mask < 0)[0]) > 0:
        print(f"weighted_mask < 0 {len(np.where(weighted_mask < 0)[0])}")
    weighted_mask[weighted_mask < 0] = 0
    weighted_mask = weighted_mask.reshape((mask.shape[0], mask.shape[1]))  # back to original shape
    # print(f"weighted_mask {np.sum(weighted_mask)}")

    if soma_mask is not None:
        weighted_soma_mask = np.zeros(soma_mask.shape)

        weighted_soma_mask = weighted_soma_mask.reshape(soma_mask.shape[0] * soma_mask.shape[1])  # flattening
        weighted_soma_mask[(soma_mask.reshape(soma_mask.shape[0] * soma_mask.shape[1])) > 0] = \
            np.random.normal(loc=mu_soma, scale=sigma_soma, size=n_pixels_soma)
        if len(np.where(weighted_soma_mask < 0)[0]) > 0:
            print(f"weighted_soma_mask < 0 {len(np.where(weighted_soma_mask < 0)[0])}")
        weighted_soma_mask[weighted_soma_mask < 0] = 0
        weighted_soma_mask = weighted_soma_mask.reshape(
            (soma_mask.shape[0], soma_mask.shape[1]))  # back to original shape
        # print(f"weighted_soma_mask {np.sum(weighted_soma_mask)}")

        weighted_mask[soma_mask] = weighted_soma_mask[soma_mask]
        # print(f"weighted_mask with soma {np.sum(weighted_mask)}")

    return weighted_mask


class MovieConstructor:
    """
    Used to construct the movie, will build the frames, deal with overlapping and pixels intensity
    """

    def __init__(self, coord_obj, traces, dimensions, baseline, soma_geoms, vessels):
        """

        Args:
            coord_obj:
            traces:
            dimensions:
            baseline:
            soma_geoms:
            vessels:
        """
        self.n_frames = traces.shape[1]
        self.n_cells = coord_obj.n_cells
        self.soma_indices = np.arange(len(soma_geoms))
        self.baseline_traces = np.min(traces)
        self.same_weight_for_all_frame = True
        self.weighted_masks = dict()
        self.dimensions = dimensions
        self.traces = traces
        self.baseline = baseline
        self.vessels = vessels
        self.soma_masks = dict()
        self.coord_obj = coord_obj
        self.masks = dict()
        self.n_pixels = dict()
        self.n_pixels_soma = dict()
        self.with_vessels = (len(vessels) > 0)
        self.masks_vessel = []
        self.raw_traces = np.zeros((self.n_cells, self.n_frames))
        # indicate if we identify already the cell pieces
        self.cell_pieces_computed = False
        self.parent_cell_pieces = dict()
        self.all_cell_pieces = []

        if self.with_vessels:
            vessels_index = np.arange(len(vessels))
            random.shuffle(vessels_index)
            n_vessels = len(vessels)
            for vessel_index in vessels_index[:n_vessels]:
                vessel = vessels[vessel_index]
                self.masks_vessel.append(get_mask(dimensions=dimensions, poly_gon=vessel))

        # change_polygon_centroid(new_centroid, poly_cell)
        for cell in np.arange(self.n_cells):
            # print(f"construct_movie_images begins, cell {cell}")
            self.masks[cell] = coord_obj.get_cell_mask(cell, dimensions)  # (dimensions[1], dimensions[0])
            self.n_pixels[cell] = np.sum(self.masks[cell])

            add_soma = not (cell % 5 == 0)

            if add_soma:
                # first we pick a soma for this cell
                random.shuffle(self.soma_indices)
                soma_geom = soma_geoms[self.soma_indices[0]]
                # then we want to move it in the cell
                # centroid_x = random.randint(-1, 1)
                # centroid_y = random.randint(-1, 1)
                centroid_x = 0
                centroid_y = 0
                cell_poly = coord_obj.cells_polygon[cell]
                centroid_x += cell_poly.centroid.x
                centroid_y += cell_poly.centroid.y
                soma_geom = change_polygon_centroid(new_centroid=(centroid_x, centroid_y), poly_cell=soma_geom)

                soma_mask = get_mask(dimensions=dimensions, poly_gon=soma_geom)
                self.soma_masks[cell] = soma_mask
                self.n_pixels_soma[cell] = np.sum(soma_mask)
            else:
                soma_mask = None
                self.soma_masks[cell] = None
                self.n_pixels_soma[cell] = None

            if self.same_weight_for_all_frame:
                self.weighted_masks[cell] = get_weighted_activity_mask_for_a_cell(mask=self.masks[cell],
                                                                                  soma_mask=soma_mask,
                                                                                  n_pixels=self.n_pixels[cell],
                                                                                  n_pixels_soma=self.n_pixels_soma[
                                                                                      cell])
                # del soma_mask
                # del self.masks[cell]

    def get_frame(self, frame):
        """

        Args:
            frame:

        Returns:

        """
        cell_pieces = set()
        # update_activity_mask(self, activity_mask)
        for cell in np.arange(self.n_cells):

            if not self.same_weight_for_all_frame:
                weighted_mask = get_weighted_activity_mask_for_a_cell(mask=self.masks[cell],
                                                                      soma_mask=self.soma_masks[cell],
                                                                      n_pixels=self.n_pixels[cell],
                                                                      n_pixels_soma=self.n_pixels_soma[cell])
            else:
                weighted_mask = self.weighted_masks[cell]
            amplitude = self.traces[cell, frame]
            if amplitude == self.baseline_traces:
                weighted_mask_tmp = np.copy(weighted_mask)
                if self.soma_masks[cell] is not None:
                    weighted_mask_tmp[self.soma_masks[cell]] = 0
                activity_mask = weighted_mask_tmp * (amplitude / np.sum(weighted_mask))
                del weighted_mask_tmp
            else:
                activity_mask = weighted_mask * (amplitude / np.sum(weighted_mask))
            del weighted_mask

            if self.cell_pieces_computed:
                self.parent_cell_pieces[cell].update_activity_mask(activity_mask=activity_mask)
            else:
                cell_piece = CellPiece(id=f"{cell}", poly_gon=self.coord_obj.cells_polygon[cell],
                                       activity_mask=activity_mask, dimensions=self.dimensions, frame_mode=True)
                self.parent_cell_pieces[cell] = cell_piece
                cell_pieces.add(cell_piece)
                self.all_cell_pieces.append(cell_piece)

        frame_image = np.ones((self.dimensions[0], self.dimensions[1]))
        frame_image *= self.baseline

        if self.with_vessels:
            for mask_vessel in self.masks_vessel:
                frame_image[mask_vessel] = self.baseline / 2

        # then we collect all pieces of cell, by piece we mean part of the cell with no overlap and part
        # with one or more intersect, and get it as a polygon object
        if not self.cell_pieces_computed:
            while len(cell_pieces) > 0:
                cell_piece = cell_pieces.pop()
                no_intersections = True
                for other_index, other_cell_piece in enumerate(cell_pieces):
                    # check if they interesect and not just touches
                    if cell_piece.poly_gon.intersects(other_cell_piece.poly_gon) and \
                            (not cell_piece.poly_gon.touches(other_cell_piece.poly_gon)):
                        no_intersections = False
                        cell_pieces.remove(other_cell_piece)
                        # then we split those 2 cell_pieces in 3, and loop again
                        new_cell_pieces = cell_piece.split(other_cell_piece)
                        cell_pieces.update(new_cell_pieces)
                        self.all_cell_pieces.extend(new_cell_pieces)
                        break
                if no_intersections:
                    cell_piece.fill_movie_images(frame_image)
            self.cell_pieces_computed = True
        else:
            for cell_piece in self.all_cell_pieces:
                # picking only cell_piece with no daugter, as recursively it will update the parent until to
                # to get to the top of the tree, cell with no parents, that have been updated earlier
                if len(cell_piece.daughters) == 0:
                    cell_piece.update_activity_mask()
                cell_piece.fill_movie_images(frame_image)

        for cell in np.arange(self.n_cells):
            self.raw_traces[cell, frame] = np.mean(frame_image[self.masks[cell]])

        return frame_image


def construct_movie_images(coord_obj, traces, dimensions, baseline, soma_geoms, vessels, param,
                           n_pixels_by_cell=None):
    print("construct_movie_images begins")
    cell_pieces = set()
    n_frames = traces.shape[1]
    n_cells = coord_obj.n_cells
    soma_indices = np.arange(len(soma_geoms))
    baseline_traces = np.min(traces)

    # change_polygon_centroid(new_centroid, poly_cell)
    for cell in np.arange(n_cells):
        # print(f"construct_movie_images begins, cell {cell}")
        mask = coord_obj.get_cell_mask(cell, dimensions)
        n_pixels = np.sum(mask)

        add_soma = not (cell % 5 == 0)

        same_weight_for_all_frame = True

        if add_soma:
            # first we pick a soma for this cell
            random.shuffle(soma_indices)
            soma_geom = soma_geoms[soma_indices[0]]
            # then we want to move it in the cell
            # centroid_x = random.randint(-1, 1)
            # centroid_y = random.randint(-1, 1)
            centroid_x = 0
            centroid_y = 0
            cell_poly = coord_obj.cells_polygon[cell]
            centroid_x += cell_poly.centroid.x
            centroid_y += cell_poly.centroid.y
            soma_geom = change_polygon_centroid(new_centroid=(centroid_x, centroid_y), poly_cell=soma_geom)

            soma_mask = get_mask(dimensions=dimensions, poly_gon=soma_geom)
            n_pixels_soma = np.sum(soma_mask)
        else:
            soma_mask = None
            n_pixels_soma = None

        if same_weight_for_all_frame:
            weighted_mask = get_weighted_activity_mask_for_a_cell(mask=mask, soma_mask=soma_mask,
                                                                  n_pixels=n_pixels, n_pixels_soma=n_pixels_soma)
            # del soma_mask
            del mask

        # TODO: decrease size of the mask so it just fit the cell
        # TODO: and add coord to where the top right of the mask starts
        activity_mask = np.zeros((n_frames, dimensions[0], dimensions[1]))

        for frame in np.arange(n_frames):
            if not same_weight_for_all_frame:
                weighted_mask = get_weighted_activity_mask_for_a_cell(mask=mask, soma_mask=soma_mask,
                                                                      n_pixels=n_pixels, n_pixels_soma=n_pixels_soma)

            amplitude = traces[cell, frame]
            if amplitude == baseline_traces:
                weighted_mask_tmp = np.copy(weighted_mask)
                weighted_mask_tmp[soma_mask] = 0
                activity_mask[frame] = weighted_mask_tmp * (amplitude / np.sum(weighted_mask))
                del weighted_mask_tmp
            else:
                activity_mask[frame] = weighted_mask * (amplitude / np.sum(weighted_mask))
        del weighted_mask
        cell_pieces.add(CellPiece(id=f"{cell}", poly_gon=coord_obj.cells_polygon[cell],
                                  activity_mask=activity_mask, dimensions=dimensions))

    images = np.ones((n_frames, dimensions[0], dimensions[1]))
    images *= baseline

    if (len(vessels) > 0) and (param.n_vessels > 0):
        vessels_index = np.arange(len(vessels))
        random.shuffle(vessels_index)
        n_vessels = min(param.n_vessels, len(vessels))
        for vessel_index in vessels_index[:n_vessels]:
            vessel = vessels[vessel_index]
            mask_vessel = get_mask(dimensions=dimensions, poly_gon=vessel)
            images[:, mask_vessel] = baseline / 2

    # then we collect all pieces of cell, by piece we mean part of the cell with no overlap and part
    # with one or more intersect, and get it as a polygon object

    #
    while len(cell_pieces) > 0:
        cell_piece = cell_pieces.pop()
        no_intersections = True
        for other_index, other_cell_piece in enumerate(cell_pieces):
            # check if they interesect and not just touches
            if cell_piece.poly_gon.intersects(other_cell_piece.poly_gon) and \
                    (not cell_piece.poly_gon.touches(other_cell_piece.poly_gon)):
                no_intersections = False
                cell_pieces.remove(other_cell_piece)
                # then we split those 2 cell_pieces in 3, and loop again
                new_cell_pieces = cell_piece.split(other_cell_piece)
                cell_pieces.update(new_cell_pieces)
                break
        if no_intersections:
            cell_piece.fill_movie_images(images)

    print("construct_movie_images is over")

    return images


def build_somas(coord_obj, dimensions):
    soma_geoms = []

    for cell, poly_gon in coord_obj.cells_polygon.items():
        img = PIL.Image.new('1', (dimensions[0], dimensions[1]), 0)
        ImageDraw.Draw(img).polygon(list(poly_gon.exterior.coords), outline=1,
                                    fill=1)
        img = np.array(img)
        n_pixel = np.sum(img)

        n_trial = 0
        max_trial = 200
        while True:
            if n_trial > max_trial:
                # print("point break")
                break
            n_trial += 1
            distances = np.arange(-3, -0.3, 0.2)
            random.shuffle(distances)
            soma = poly_gon.buffer(distances[0])
            if hasattr(soma, 'geoms') or (soma.exterior is None):
                # means its a MultiPolygon object
                continue
            img = PIL.Image.new('1', (dimensions[0], dimensions[1]), 0)
            if len(list(soma.exterior.coords)) < 2:
                continue
            ImageDraw.Draw(img).polygon(list(soma.exterior.coords), outline=1,
                                        fill=1)
            img = np.array(img)
            n_pixel_soma = np.sum(img)
            ratio = n_pixel / n_pixel_soma
            if (ratio > 3) and (ratio < 5):
                # print(f"soma {cell}: {n_pixel} {n_pixel_soma} {str(np.round(ratio, 2))}")
                soma_geoms.append(soma)
                break
    return soma_geoms


def give_values_on_linear_line_between_2_points(x_coords, y_coords):
    # x_coords = [100, 400]
    # y_coords = [240, 265]
    # print(f"x_coords {x_coords} y_coords {y_coords}")
    # Calculate the coefficients. This line answers the initial question.
    coefficients = np.polyfit(x_coords, y_coords, 1)

    # 'a =', coefficients[0]
    # 'b =', coefficients[1]

    # Let's compute the values of the line...
    polynomial = np.poly1d(coefficients)
    x_axis = np.arange(x_coords[0], x_coords[1] + 1)
    y_axis = polynomial(x_axis)

    return y_axis


def exponential_decay_formula(t, a, k, c):
    """
    Exponential decay formula
    Code probably copy from an online source. Sorry i don't have the reference
    :param t: time that has passed
    :param a: initial value (amount before measuring growth or decay)
    :param k: continuous growth rate (also called constant of proportionality)
    (k > 0, the amount is increasing (growing); k < 0, the amount is decreasing (decaying))
    :param c: lowest value
    :return:
    """
    return a * np.exp(k * t) + c


def finding_growth_rate(t, a, end_value):
    """
    Find the growth rate.
    Code probably copy from an online source. Sorry i don't have the reference
    Args:
        t: time
        a:
        end_value:

    Returns:

    """
    k = np.log(end_value / a) / t
    return k


def produce_vessels(vessels_imgs_dir, path_results=None):
    """
    Produce vessels polygons based on images in vessels_imgs_dir
    Args:
        vessels_imgs_dir: directory from which load the images containing the vessels
        path_results: If not None, the path where to save to vessels produced

    Returns: coord_list a list of 2D array (n_coord x 2): each column represents the x and y coordinates of each point
            of the polygon (vessel)
            dimensions_list a List of 1D array with 2 values, integer, representing the height and width of the movie

    """
    vessels_imgs = []
    # finding the cnn_file coresponding to the ms
    for (dirpath, dirnames, local_filenames) in os.walk(os.path.join(vessels_imgs_dir)):
        for file_name in local_filenames:
            if file_name.endswith(".tiff") or file_name.endswith(".tif"):
                vessel_img = mpimg.imread(os.path.join(vessels_imgs_dir, file_name))
                vessels_imgs.append(vessel_img)
                print(f"vessel_img.shape {vessel_img.shape}")
        # looking only in the top directory
        break

    # we want to produce a polygon from the border of the vessels
    # and be able to save the coords in a file to load it later
    # we also save the size of the window
    # vessel_img.shape[0] represent the height of the image
    # vessel_img.shape[1] represent the width of the image
    # 255 means white, 0 means black
    coord_array_list = []
    dimensions_list = []
    for vessel_img in vessels_imgs:
        coord_list = []
        height = vessel_img.shape[0]
        width = vessel_img.shape[1]
        for y in np.arange(height):
            x_pixels = (np.where(vessel_img[y, :] == 0))[0]
            n_pixels = len(x_pixels)
            if n_pixels > 0:
                if (y == 0) or (y == height - 1):
                    for x in x_pixels:
                        coord_list.append((x, y))
                else:
                    coord_list.append((x_pixels[0], y))
                    coord_list.append((x_pixels[-1], y))
        coord_array = np.zeros((len(coord_list), 2), dtype="uint16")
        for index, coord in enumerate(coord_list):
            coord_array[index, 0] = coord[0]
            coord_array[index, 1] = coord[1]
        coord_array_list.append(coord_array)
        dimensions = np.array([height, width]).astype(int)
        dimensions_list.append(dimensions)

    dict_to_save = dict()
    for index, coord_array in enumerate(coord_array_list):
        dict_to_save[f"coord_{index}"] = coord_array
        dict_to_save[f"dimensions_{index}"] = dimensions_list[index]

    if path_results is not None:
        sio.savemat(os.path.join(path_results, "vessels_coord.mat"), dict_to_save)

    return coord_array_list, dimensions_list


def plot_all_cells_on_map(coord_obj, path_results, save_formats="pdf"):
    """
    Plot all cells contour on a map using welsh powell algorithm to color cell that intersect
    with a different color
    Args:
        coord_obj: instance of CellsCoord object
        path_results: (string), directory where to save the figure
        save_formats: string or list of string, formats in which to save the figure

    Returns:

    """
    # we want to color cells that overlap with different colors
    n_cells = len(coord_obj.coords)
    # white, http://doc.instantreality.org/tools/color_calculator/
    # white: 1, 1, 1
    # red: 1, 0, 0
    isolated_cell_color = (1, 0, 0, 1.0)
    isolated_group = []
    cells_groups_colors = []
    cells_groups_edge_colors = []
    cells_groups_alpha = []
    cells_groups = []

    # building networkx graph
    graphs = []
    cells_added = []
    for cell in np.arange(n_cells):
        if cell in cells_added:
            continue
        # welsh_powell
        n_intersect = len(coord_obj.intersect_cells[cell])
        if n_intersect == 0:
            isolated_group.append(cell)
            cells_added.append(cell)
        else:
            graph = nx.Graph()
            cells_to_expend = [cell]
            edges = set()
            while len(cells_to_expend) > 0:
                if cells_to_expend[0] not in cells_added:
                    cells_added.append(cells_to_expend[0])
                    n_intersect = len(coord_obj.intersect_cells[cells_to_expend[0]])
                    if n_intersect > 0:
                        for inter_cell in coord_obj.intersect_cells[cells_to_expend[0]]:
                            min_c = min(inter_cell, cells_to_expend[0])
                            max_c = max(inter_cell, cells_to_expend[0])
                            edges.add((min_c, max_c))
                            cells_to_expend.append(inter_cell)
                cells_to_expend = cells_to_expend[1:]
            graph.add_edges_from(list(edges))
            graphs.append(graph)
    cells_by_color_code = dict()
    max_color_code = 0
    for graph in graphs:
        # dict that give for each cell a color code
        col_val = welsh_powell(graph)
        for cell, color_code in col_val.items():
            if color_code not in cells_by_color_code:
                cells_by_color_code[color_code] = []
            cells_by_color_code[color_code].append(cell)
            max_color_code = max(max_color_code, color_code)

    for color_code, cells in cells_by_color_code.items():
        if len(cells) == 0:
            continue
        cells_groups.append(cells)
        cells_groups_colors.append(cm.nipy_spectral(float(color_code + 1) / (max_color_code + 1)))
        cells_groups_edge_colors.append("white")
        cells_groups_alpha.append(0.8)
    cells_groups.append(isolated_group)
    cells_groups_colors.append(isolated_cell_color)
    cells_groups_alpha.append(1)
    cells_groups_edge_colors.append("white")
    avg_cell_map_img = None

    if isinstance(save_formats, str):
        save_formats = [save_formats]

    coord_obj.plot_cells_map(path_results=path_results,
                             data_id="simulated_movie", show_polygons=False,
                             fill_polygons=False,
                             title_option="all cells", connections_dict=None,
                             cells_groups=cells_groups,
                             img_on_background=avg_cell_map_img,
                             cells_groups_colors=cells_groups_colors,
                             cells_groups_edge_colors=cells_groups_edge_colors,
                             with_edge=True, cells_groups_alpha=cells_groups_alpha,
                             dont_fill_cells_not_in_groups=False,
                             with_cell_numbers=True, save_formats=save_formats)


class SimulatedMovieGenerator:
    """
    Class that will handle the generation of simulated calcium imaging movie
    """

    def __init__(self, n_frames, path_results,
                 dimensions=(120, 120), with_mvt=False,
                 raster_dur_from_image=None,
                 n_cells_of_interest=16,
                 n_overlap_by_cell_range=(1, 4),
                 non_overlap_by_cell_range=(2, 10),
                 range_n_transient_cells_of_interest=(2, 4),
                 range_n_transient_overlapping_cells=(8, 16),
                 range_n_transient_other_cells=(2, 16),
                 range_n_transient_img_cells=(3,12),
                 time_str=None, use_only_valid_cells=True):
        """

        Args:
            dimensions: tuple of 2 int, represent the dimension of the movie in pixels, (x, y)
            with_mvt: boolean, is True, means some movement will be added, otherwise it would produce a movie with
            perfect motion correction
            n_overlap_by_cell_range: (tuple of 2 int) bottom and upper range of how many cells will overlap
            our cell "of interest" aimed to train the classifier
            non_overlap_by_cell_range:  (tuple of 2 int) bottom and upper range of  how many cells will be near
            our cells "of interest" aimed to train the classifier, but without any overlap
            n_cells_of_interest: (int) cells of interest, are the cell for which we want to control how many overlaps,
             their activity the aim is to use them to train the classifier
            range_n_transient_cells_of_interest: bottom and upper range of the number of transients over 1000 frames
            for our cells of interest
            raster_dur_from_image: None or list of image file name, should be white & black image
            range_n_transient_overlapping_cells: bottom and upper range of the number of transients over 1000 frames
            for cells that overlap our cell of interest
            range_n_transient_other_cells: bottom and upper range of the number of transients over 1000 frames
            range_n_transient_img_cells: bottom and upper range of the number of transients over 1000 frames when using
            images to build the raster
            for cells that are not of interest or overlap them
            use_only_valid_cells: Cells models could be classify as Valid or False (using a classifier for exemple).
            This allow for the movie to be generated with ROIs that would have the rightly segmented cell
            path_results: Directory in which to save the results
            time_str (string): id (meant to be a timestamps string representation) in order to  give a unique id to files.
        """
        # use as id to identify files produced
        if time_str is not None:
            self.time_str = time_str
        else:
            self.time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        self.dimensions = dimensions
        self.with_mvt = with_mvt
        # list of shapely Polygon instances
        self.vessels = []
        # list of coords used to produce the cells contours
        self.cell_models = None
        # boolean array or list of the same size of cell_models that precise if the cell is valid or not
        # could stay to None, in that case any cell in cell_models could be used
        self.is_valid_cell = None

        # if not None, n_frames will be changed in build_raster_dur
        self.raster_dur_from_images = raster_dur_from_image

        self.n_frames = n_frames

        self.path_results = path_results

        # --------- generate_artificial_map section ---------

        # meaning how many valid cells for a non valid cell
        # only works if some cell models of non valid cell are given
        # non valid cell could be determine using a classifier specialize in this task
        self.ratio_valid_to_non_valid_cells = 4
        self.use_only_valid_cells = use_only_valid_cells
        # how many cells will overlap our cell "of interest" aimed to train the classifier
        self.n_overlap_by_cell_range = n_overlap_by_cell_range
        # how many cells will be near our cell "of interest" aimed to train the classifier, but without any overlap
        self.non_overlap_by_cell_range = non_overlap_by_cell_range
        self.padding = 5
        # cells of interest, are the cell for which we want to control how many overlaps, their activity
        # the aim is to use them to train the classifier
        self.n_cells_of_interest = n_cells_of_interest

        # list that contains the coordinates of the cells that are part of the simulated movie
        self.map_coords = []
        # indicate if coods have been computed using matlab and coordinate starts at 1
        # initialize in load_cell_coords() method
        self.cell_coords_from_matlab = False
        # list of int representing the indices of the cells of interest (index matching map_coords dimensions)
        self.cells_with_overlap = []
        # list of int representing the indices of the cells overlapping the cells of interest
        # (index matching map_coords dimensions). Useful because we might want to increase the fire
        # rate of those cells
        self.overlapping_cells = []

        # --------- building the raster_dur ---------

        self.raster_dur = None

        # ratio to based the n transient on a 1000 frame movie
        ratio_transients = self.n_frames / 1000
        # number of transient over the simulated movie depending on the type of cell
        self.range_n_transient_cells_with_overlap = (int(range_n_transient_cells_of_interest[0] * ratio_transients),
                                                     int(range_n_transient_cells_of_interest[1] * ratio_transients))
        self.range_n_transient_overlapping_cells = (int(range_n_transient_overlapping_cells[0] * ratio_transients),
                                                    int(range_n_transient_overlapping_cells[1] * ratio_transients))
        self.range_n_transient_other_cells = (int(range_n_transient_other_cells[0] * ratio_transients),
                                              int(range_n_transient_other_cells[1] * ratio_transients))
        self.range_n_transient_img_cells = (int(range_n_transient_img_cells[0] * ratio_transients),
                                              int(range_n_transient_img_cells[1] * ratio_transients))

        # ---- building traces ------
        # if True, cell baseline (when not active) will have the intensity than the background
        self.same_baseline_from_cell_than_background = True
        self.raw_traces = None

        # ---- producing movie ----------
        self.shaking_frames = None

    def generate_movie(self):
        """
        Generate the movie by buimding first the cells map, then generate ting the raster then the traces
        and then the movie.
        Before calling this method vessels must have been created or loaded if necessary and the cells model
        must be loaded as well.
        Returns:

        """
        # first we check a few conditions

        self.__generate_artificial_map()
        self.__build_raster_dur()
        self.save_raster_dur_for_gui()
        self.__produce_movie()
        self.save_traces()

        file_name_txt = 'artificial_cells_listing.txt'

        with open(os.path.join(self.path_results, file_name_txt), "w", encoding='UTF-8') as file:
            file.write(f"Targets cells: {', '.join(list(map(str, self.cells_with_overlap)))}" + '\n')
            file.write(f"Shaking frames: {', '.join(list(map(str, self.shaking_frames)))}" + '\n')

        self.save_cell_coords()

    def save_cell_coords(self):
        """
        Save the cell coordinate so far in the same output as CaImAn
        Returns:

        """
        coords_caiman_style = np.empty((len(self.map_coords),), dtype=np.object)
        for i in range(len(self.map_coords)):
            coords_caiman_style[i] = self.map_coords[i]
        sio.savemat(os.path.join(self.path_results, "map_coords.mat"), {"coord_python": coords_caiman_style})

    def __generate_artificial_map(self):
        """
        Use the cells models to generate a map of the dimensions given when instanciating SimulatedMovieGenerator.
        It will create a "square" for each cell of interest that will be in the center of this square,
        this square will be filled with a certain number of cells overlapping the cell of interest
        and other cell that will not overlapping it but might overlap other cells.
        All the parameters can be change in the __init__ of the class
        Returns: None

        """
        # padding is used to to add mvts
        dimensions_without_padding = self.dimensions
        # dimensions = (dimensions_without_padding[0] + (self.padding * 2),
        #               dimensions_without_padding[1] + (self.padding * 2))
        # model cells, then we'll put cells around with some overlaping
        # n_cells = 16
        sub_window_size = (30, 30)
        # cell padding
        x_padding = 1  # sub_window_size[1] // 6
        y_padding = 1  # sub_window_size[0] // 6
        # centroids of the cells of interest
        centroids = []
        line = 0
        col = 0
        max_lines = dimensions_without_padding[0] // sub_window_size[0]
        max_cols = dimensions_without_padding[1] // sub_window_size[1]
        x_borders = []
        y_borders = []
        for c in np.arange(self.n_cells_of_interest):
            x_borders.append((col * sub_window_size[1] + self.padding, (col + 1) * sub_window_size[1] + self.padding))
            y_borders.append((line * sub_window_size[0] + self.padding, (line + 1) * sub_window_size[0] + self.padding))
            centroids.append((int((col + 0.5) * sub_window_size[1] + self.padding),
                              int((line + 0.5) * sub_window_size[0] + self.padding)))
            line += 1
            if (line % max_lines) == 0:
                line = 0
                col += 1

        # coords_to_use = coords_to_use
        if self.is_valid_cell is not None:
            coords_true_cells = []
            for cell_index, is_valid in self.is_valid_cell:
                if is_valid:
                    coords_true_cells.append(self.cell_models[cell_index])

            coords_fake_cells = []
            for cell_index, is_valid in self.is_valid_cell:
                if not is_valid:
                    coords_fake_cells.append(self.cell_models[cell_index])
        else:
            coords_true_cells = self.cell_models
            coords_fake_cells = []

        # cells of interest, with some overlaps
        cells_with_overlap = []
        # key is an int (one of the cells_with_overlap), and value an int corresponding
        overlapping_cells = dict()
        map_coords = []
        # in order to randomly choose a true cell coord
        true_cells_index = np.arange(len(coords_true_cells))
        random.shuffle(true_cells_index)
        # in order to randomly choose a fake cell coord
        fake_cells_index = np.arange(len(coords_fake_cells))
        random.shuffle(fake_cells_index)
        cell_index = 0
        n_non_target_cells_added = 0

        cells_coord_to_add = []

        for c in np.arange(self.n_cells_of_interest):
            cell_coord = coords_true_cells[true_cells_index[0]]
            true_cells_index = true_cells_index[1:]
            # we center the cell and change its coordinates
            centroid = centroids[c]
            cell_coord, poly_main_cell = shift_cell_coord_to_centroid(centroid=centroid,
                                                                      cell_coord=cell_coord,
                                                                      from_matlab=self.cell_coords_from_matlab)
            cells_with_overlap.append(cell_index)
            main_cell_index = cell_index
            overlapping_cells[main_cell_index] = []
            map_coords.append(cell_coord)
            cell_index += 1

            # we decide how many cells will be overlaping it (more like intersect)
            n_overlaps = random.randint(self.n_overlap_by_cell_range[0], self.n_overlap_by_cell_range[1])
            n_non_overlaps = random.randint(self.non_overlap_by_cell_range[0], self.non_overlap_by_cell_range[1])
            n_over_added = 0
            n_non_over_added = 0
            centroids_added = []
            max_n_overall_trial = 200
            n_overall_trial = 0

            while (n_over_added < n_overlaps) or (n_non_over_added < n_non_overlaps):
                if n_overall_trial >= max_n_overall_trial:
                    print("n_overall_trial >= max_n_overall_trial")
                    break
                n_overall_trial += 1
                # adding a fake cell one every self.ratio_valid_to_non_valid_cells cells
                if (not self.use_only_valid_cells) and (len(fake_cells_index) > 0) and \
                        (n_non_target_cells_added % (self.ratio_valid_to_non_valid_cells + 1) == 0):
                    over_cell_coord = coords_fake_cells[fake_cells_index[0]]
                    fake_cells_index = fake_cells_index[1:]
                else:
                    over_cell_coord = coords_true_cells[true_cells_index[0]]
                    true_cells_index = true_cells_index[1:]
                new_centroid_x_values = np.concatenate((np.arange(x_borders[c][0] + x_padding, centroid[0] - 2),
                                                        np.arange(centroid[0] + 2, x_borders[c][1] + 1 - x_padding)))
                new_centroid_y_values = np.concatenate((np.arange(y_borders[c][0] + y_padding, centroid[1] - 2),
                                                        np.arange(centroid[1] + 2, y_borders[c][1] + 1 - y_padding)))
                not_added = True
                # one cell might be too big to fit in, then we give up and go to the next window
                max_n_trial = 1000
                n_trial = 0
                while not_added:
                    if n_trial >= max_n_trial:
                        print("n_trial >= max_n_trial")
                        break
                    n_trial += 1
                    # random x and y for centroid
                    np.random.shuffle(new_centroid_x_values)
                    np.random.shuffle(new_centroid_y_values)
                    x = new_centroid_x_values[0]
                    y = new_centroid_y_values[1]
                    # first we want this centroid to be at least 2 pixels away of any added centroid
                    if (x, y) in centroids_added:
                        continue
                    to_close = False
                    for centr in centroids_added:
                        if (abs(x - centr[0]) <= 2) and (abs(y - centr[1]) <= 2):
                            to_close = True
                            break
                    if to_close:
                        continue
                    cell_coord, poly_new_cell = shift_cell_coord_to_centroid(centroid=(x, y),
                                                                             cell_coord=over_cell_coord,
                                                                             from_matlab=self.cell_coords_from_matlab)
                    # first we need to make sure the cell don't go out of the frame
                    minx, miny, maxx, maxy = np.array(list(poly_new_cell.bounds))
                    if (minx <= self.padding) or (miny <= self.padding) or (maxx >= dimensions_without_padding[1]) or \
                            (maxy >= dimensions_without_padding[0]):
                        continue
                    # if intersects and not just touches (means commun border)
                    if poly_main_cell.intersects(poly_new_cell) and (not poly_main_cell.touches(poly_new_cell)) \
                            and n_over_added < n_overlaps:
                        n_over_added += 1
                        not_added = False
                    elif n_non_over_added < n_non_overlaps:
                        n_non_over_added += 1
                        not_added = False
                    if not not_added:
                        cells_coord_to_add.append(cell_coord)
                        centroids_added.append((x, y))
                        # cell_index += 1
                        n_non_target_cells_added += 1
                    if n_trial >= max_n_trial:
                        print("n_trial >= max_n_trial")
                        break

        random.shuffle(cells_coord_to_add)
        for cell_coord_to_add in cells_coord_to_add:
            map_coords.append(cell_coord_to_add)

        print(f"cells_with_overlap {cells_with_overlap}")
        self.map_coords = map_coords
        self.cells_with_overlap = cells_with_overlap
        self.overlapping_cells = overlapping_cells

    def save_raster_dur_for_gui(self):
        """
        Save the raster dur that was generated in the path_results defined in the constructor
        Returns: None

        """
        if self.raster_dur is None:
            return

        spike_nums = np.zeros(self.raster_dur.shape, dtype="int8")
        peak_nums = np.zeros(self.raster_dur.shape, dtype="int8")
        for cell in np.arange(self.raster_dur.shape[0]):
            binary_raster = np.copy(self.raster_dur[cell])
            binary_raster[binary_raster >= 1] = 1
            active_periods = get_continous_time_periods(binary_raster)
            for period in active_periods:
                spike_nums[cell, period[0]] = 1
                peak_nums[cell, period[1]] = 1
        sio.savemat(os.path.join(self.path_results, "gui_data.mat"), {'Bin100ms_spikedigital_Python': spike_nums,
                                                                      'LocPeakMatrix_Python': peak_nums})

    def __build_raster_dur(self):
        """
        Build the raster that will be used to produce the simulated movie
        Returns: a 2d-array (int8) of n_cells x n_frames

        """
        n_cells = len(self.map_coords)
        print(f"n_cells = {n_cells}")
        if self.raster_dur_from_images is not None:
            total_n_frames = 0
            raster_dur = np.zeros((n_cells, 0), dtype="int8")
            for img_file_name in self.raster_dur_from_images:
                frame_img = np.array(Image.open(img_file_name))
                width = frame_img.shape[1]
                height = frame_img.shape[0]
                print(f"Width {width}, height {height}")
                # we want to scale the height to the number of cells
                n_frames = int((width * n_cells) / height)
                total_n_frames += n_frames
                frame_img = np.array(Image.open(img_file_name).convert('L').resize((n_frames, n_cells)))
                threshold = np.percentile(frame_img, 5)
                # print(f"threshold {threshold}")
                binary_version = False
                if binary_version:
                    threshold = 128

                    frame_img[frame_img <= threshold] = 1
                    frame_img[frame_img > threshold] = 0
                else:
                    threshold = np.percentile(frame_img, 99)
                    print(f"threshold {threshold}")
                    frame_img_copy = np.copy(frame_img)
                    frame_img[frame_img_copy <= 200] = 1  # 120
                    frame_img[frame_img_copy <= 120] = 2  # 90
                    frame_img[frame_img_copy <= 70] = 3
                    frame_img[frame_img_copy > 200] = 0  # 120

                frame_img = frame_img.astype("int8")
                raster_dur = np.hstack((raster_dur, frame_img[::-1, :]))
            self.n_frames = total_n_frames
            # frame_img = np.array(Image.open(raster_dur_from_image).convert('L').resize((self.n_frames, n_cells)))
            # frame_img = imageio.imread(raster_dur_from_image)
            # frame_img = np.array(frame_img)
            # frame_img = cv2.resize(frame_img,
            #                        (self.n_frames, n_cells))
            # plt.imshow(frame_img, cmap=cm.Greys)
            # plt.show()
            # binarize image
            add_noise = True
            if add_noise:
                for cell in np.arange(n_cells):
                    # n_transient = random.randint(3, 12)
                    n_transient = random.randint(self.range_n_transient_img_cells[0],
                                                 self.range_n_transient_img_cells[1])
                    onsets = np.zeros(0, dtype="int16")
                    for transient in np.arange(n_transient):
                        while True:
                            onset = random.randint(0, self.n_frames - 20)
                            sub_array = np.abs(onsets - onset)
                            if (len(onsets) == 0) or (np.min(sub_array) > 3):
                                onsets = np.append(onsets, [onset])
                                break
                    # useless
                    onsets.sort()
                    for transient in np.arange(n_transient):
                        onset = onsets[transient]
                        duration_transient = random.randint(1, 5)
                        raster_dur[cell, onset:onset + duration_transient] = 1

            print(f"raster_dur.shape {raster_dur.shape}")

            fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=False,
                                    figsize=(10, 5), dpi=200)
            ax1.imshow(raster_dur[::-1], cmap=cm.Greys)
            plt.show()
            fig.savefig(os.path.join(f'{self.path_results}', f'gray_level_raster.png'),
                        format=f"png")
        else:
            raster_dur = np.zeros((n_cells, self.n_frames), dtype="int8")
            for cell in np.arange(n_cells):
                if cell in self.cells_with_overlap:
                    n_transient = random.randint(self.range_n_transient_cells_with_overlap[0],
                                                 self.range_n_transient_cells_with_overlap[1])
                elif cell in self.overlapping_cells:
                    n_transient = random.randint(self.range_n_transient_overlapping_cells[0],
                                                 self.range_n_transient_overlapping_cells[1])
                else:
                    n_transient = random.randint(self.range_n_transient_other_cells[0],
                                                 self.range_n_transient_other_cells[1])
                onsets = np.zeros(0, dtype="int16")
                for transient in np.arange(n_transient):
                    while True:
                        onset = random.randint(0, self.n_frames - 20)
                        sub_array = np.abs(onsets - onset)
                        if (len(onsets) == 0) or (np.min(sub_array) > 3):
                            onsets = np.append(onsets, [onset])
                            break
                # useless
                onsets.sort()
                for transient in np.arange(n_transient):
                    onset = onsets[transient]
                    duration_transient = random.randint(1, 8)
                    raster_dur[cell, onset:onset + duration_transient] = 1

        plot_spikes_raster(spike_nums=raster_dur, path_results=self.path_results,
                           time_str=self.time_str,
                           spike_train_format=False,
                           title=f"raster plot test",
                           file_name=f"spike_nums__dur_artificial",
                           y_ticks_labels_size=4,
                           save_raster=True,
                           without_activity_sum=False,
                           sliding_window_duration=1,
                           show_sum_spikes_as_percentage=True,
                           show_raster=False,
                           plot_with_amplitude=False,
                           spike_shape="|",  # o
                           spike_shape_size=1,  # 4
                           cell_spikes_color='cornflowerblue',
                           activity_sum_plot_color="red",
                           save_formats="pdf")

        self.raster_dur = raster_dur

    def build_traces(self, n_pixels_by_cell, baseline, use_traces_for_amplitude=None):
        n_cells = self.raster_dur.shape[0]
        n_frames = self.raster_dur.shape[1]

        decay_factor = 10
        traces = np.ones((n_cells, n_frames))
        original_baseline = baseline
        # cell_baseline_ratio: by how much increasing the baseline of the cell comparing to the baseline of map
        # The ratio change for each cell, goes from 1.4 to 1.6
        cell_baseline_ratio_values = np.arange(1.7, 1.9, 0.1)
        for cell in np.arange(n_cells):
            if self.same_baseline_from_cell_than_background:
                baseline = original_baseline * n_pixels_by_cell[cell]
            else:
                random.shuffle(cell_baseline_ratio_values)
                cell_baseline_ratio = cell_baseline_ratio_values[0]
                baseline = original_baseline * n_pixels_by_cell[cell] * cell_baseline_ratio
            traces[cell] *= baseline
            binary_raster = np.copy(self.raster_dur[cell])
            binary_raster[binary_raster >= 1] = 1
            active_periods = get_continous_time_periods(binary_raster)
            for period in active_periods:
                last_frame = min(period[1] + 1, n_frames - 1)
                max_raster_value = np.max(self.raster_dur[cell, period[0]:last_frame + 1])
                len_period = last_frame - period[0]
                x_coords = [period[0], last_frame]
                low_amplitude = traces[cell, period[0]]
                if use_traces_for_amplitude is not None:
                    amplitude_max = np.max(use_traces_for_amplitude[cell, period[0]:last_frame + 1])
                else:
                    amplitude_max = 2 * max_raster_value
                    # if len_period <= 2:
                    #     amplitude_max = random.randint(2, 2*max_raster_value) # (2, 5)
                    # elif len_period <= 5:
                    #     amplitude_max = random.randint(3, 3*max_raster_value) # (3, 8)
                    # else:
                    #     amplitude_max = random.randint(5, 4*max_raster_value) # (5, 10)
                    amplitude_max *= n_pixels_by_cell[cell]
                amplitude_max += low_amplitude
                y_coords = [low_amplitude, amplitude_max]
                traces_values = give_values_on_linear_line_between_2_points(x_coords, y_coords)
                try:
                    traces[cell, period[0]:last_frame + 1] = traces_values
                except ValueError:
                    print(f"traces[cell, period[0]:last_frame + 1] {traces[cell, period[0]:last_frame + 1]}, "
                          f"traces_values {traces_values}, period[0] {period[0]}, last_frame + 1 {last_frame + 1}")
                    raise Exception("ValueError: could not broadcast input array from shape (3) into shape (2)")
                if (last_frame + 1) == n_frames:
                    continue
                len_decay = max(len_period * decay_factor, 12)
                growth_rate = finding_growth_rate(t=len_decay, a=amplitude_max, end_value=baseline)
                traces_decay_values = exponential_decay_formula(t=np.arange(len_decay), a=amplitude_max,
                                                                k=growth_rate, c=0)

                if last_frame + len_decay <= n_frames:
                    traces[cell, last_frame:last_frame + len_decay] = traces_decay_values
                else:
                    offset = (last_frame + len_decay) - n_frames
                    traces[cell, last_frame:] = traces_decay_values[:-offset]

        z_score_traces = np.copy(traces)
        for cell in np.arange(n_cells):
            if np.std(z_score_traces[cell]) > 0:
                z_score_traces[cell] = (z_score_traces[cell] - np.mean(z_score_traces[cell])) / np.std(
                    z_score_traces[cell])

        plot_spikes_raster(spike_nums=self.raster_dur,
                           display_spike_nums=False,
                           display_traces=True,
                           traces=z_score_traces,
                           raster_face_color="white",
                           path_results=self.path_results,
                           spike_train_format=False,
                           title=f"traces",
                           file_name=f"traces_artificial",
                           y_ticks_labels_size=4,
                           save_raster=True,
                           without_activity_sum=True,
                           show_raster=False,
                           plot_with_amplitude=False,
                           spike_shape="o",
                           spike_shape_size=0.4,
                           save_formats="pdf")

        return traces

    def __produce_movie(self, use_traces_for_amplitude=None, file_name=None):
        """
        Generate the movie based on the cell contour map, the raster and the traces generated previously
        Args:
            use_traces_for_amplitude:
            file_name:

        Returns:

        """
        start_time = time.time()
        n_frames = self.raster_dur.shape[1]
        n_cells = self.raster_dur.shape[0]
        dimensions = self.dimensions

        coord_obj = CellsCoord(coords=self.map_coords, nb_col=dimensions[0],
                               nb_lines=dimensions[1], from_matlab=False)

        # build polygons representing somas
        soma_geoms = build_somas(coord_obj, dimensions=dimensions)

        plot_all_cells_on_map(coord_obj=coord_obj, path_results=self.path_results,
                              save_formats="pdf")

        n_pixels_by_cell = dict()
        for cell in np.arange(n_cells):
            mask = coord_obj.get_cell_mask(cell, dimensions)
            n_pixels = np.sum(mask)
            n_pixels_by_cell[cell] = n_pixels

        # default value at rest for a pixel
        baseline = 1
        traces = self.build_traces(n_pixels_by_cell=n_pixels_by_cell, baseline=baseline,
                                   use_traces_for_amplitude=use_traces_for_amplitude)

        # images = construct_movie_images(coord_obj=coord_obj, traces=traces, dimensions=dimensions,
        #                                 n_pixels_by_cell=n_pixels_by_cell, baseline=baseline, soma_geoms=soma_geoms,
        #                                 vessels=vessels, param=param)
        # cells_activity_mask
        noise_str = "gauss"
        # in ["s&p", "poisson", "gauss", "speckle"]:
        if file_name is None:
            file_name = "simulated_movie"
        outvid_tiff = os.path.join(self.path_results,
                                   f"{file_name}_{self.time_str}.tiff")

        # used to add mvt
        # images_with_padding = np.ones((images.shape[0], dimensions[0], dimensions[1]))
        # images_with_padding *= baseline

        # images_mask = np.zeros((images.shape[1], images.shape[2]), dtype="uint8")
        # print(f"images_with_padding.shape {images_with_padding.shape}")
        # print(f"images_mask.shape {images_mask.shape}")

        # add movement to the movie if with_mvt is set to True
        x_shift = 0
        y_shift = 0
        shaking_frames = []
        if self.with_mvt:
            # adding movement
            shaking_rate = 1 / 60
            x_shift_range = (-2, 2)
            y_shift_range = (-2, 2)
            n_continuous_shaking_frames_range = (1, 5)
            shake_it_when_it_fired = True
            if shake_it_when_it_fired:
                shaking_frames = []
                # we put a frame contained in each active period of target cells or overlaping cells
                cells = list(self.cells_with_overlap) + list(self.overlapping_cells)
                for cell in cells:
                    periods = get_continous_time_periods(self.raster_dur[cell])
                    for period in periods:
                        shaking_frames.append(random.randint(max(period[0] - 1, 0), period[1]))
                shaking_frames = np.unique(shaking_frames)
            else:
                shaking_frames = np.arange(n_frames)
            random.shuffle(shaking_frames)
            shaking_frames = shaking_frames[:int(n_frames * shaking_rate)]
            shaking_frames_to_concat = np.zeros(0, dtype='int16')
            for frame in shaking_frames:
                n_to_add = random.randint(n_continuous_shaking_frames_range[0], n_continuous_shaking_frames_range[1])
                shaking_frames_to_concat = np.concatenate(
                    (shaking_frames_to_concat, np.arange(frame + 1, frame + n_to_add)))
            shaking_frames = np.concatenate((shaking_frames, shaking_frames_to_concat))
            # doing it for the saving on file
            shaking_frames = np.unique(shaking_frames)
            np.ndarray.sort(shaking_frames)

        print(f"Saving the tiff using MovieConstructor")
        with tifffile.TiffWriter(outvid_tiff) as tiff:
            movie_constructor = MovieConstructor(coord_obj=coord_obj, traces=traces, dimensions=dimensions,
                                                 baseline=baseline, soma_geoms=soma_geoms,
                                                 vessels=self.vessels)
            for frame in np.arange(n_frames):
                img_array = movie_constructor.get_frame(frame=frame)
                image = np.ones((dimensions[0], dimensions[1]))
                if self.with_mvt:
                    shaked = False
                    if frame in shaking_frames:
                        x_shift = random.randint(x_shift_range[0], x_shift_range[1])
                        y_shift = random.randint(y_shift_range[0], y_shift_range[1])
                        shaked = True
                last_y = (-self.padding + y_shift) if (-self.padding + y_shift) < 0 else img_array.shape[0]
                last_x = (-self.padding + x_shift) if (-self.padding + x_shift) < 0 else img_array.shape[1]
                if self.padding > 0:
                    image[self.padding + y_shift:last_y, self.padding + x_shift:last_x] = \
                        img_array[self.padding:-self.padding, self.padding:-self.padding]
                    img_array = image
                # else:
                #     image = img_array
                img_array = noisy(noise_str, img_array)
                img_array = normalize_array_0_255(img_array)
                # tiff.save(img_array, compress=6)
                tiff.write(img_array)
                # images[frame] = img_array
                if self.with_mvt:
                    if shaked:
                        x_shift = 0
                        y_shift = 0

            # for frame, img_array in enumerate(images):
            #     if param.with_mvt:
            #         shaked = False
            #         if frame in shaking_frames:
            #             x_shift = random.randint(x_shift_range[0], x_shift_range[1])
            #             y_shift = random.randint(y_shift_range[0], y_shift_range[1])
            #             shaked = True
            #     last_y = (-padding + y_shift) if (-padding + y_shift) < 0 else images.shape[1]
            #     last_x = (-padding + x_shift) if (-padding + x_shift) < 0 else images.shape[2]
            #     images[frame, padding + y_shift:last_y, padding + x_shift:last_x] = \
            #         img_array[padding:-padding, padding:-padding]
            #     img_array = images[frame]
            #     img_array = noisy(noise_str, img_array)
            #     img_array = normalize_array_0_255(img_array)
            #     tiff.save(img_array, compress=6)
            #     images[frame] = img_array
            #     if param.with_mvt:
            #         if shaked:
            #             x_shift = 0
            #             y_shift = 0
        self.raw_traces = movie_constructor.raw_traces
        stop_time = time.time()
        print(f"Time to complete produce_movie(): "
              f"{np.round(stop_time - start_time, 3)} s")

        print(f"n shaking frames : {len(shaking_frames)}")
        self.shaking_frames = shaking_frames

    def save_traces(self, output_formats="npy"):
        """
        Save traces
        Args:
            output_formats: (string or list of string) npy or mat (if matlab, the traces variable name will be raw_traces

        Returns:

        """
        if isinstance(output_formats, str):
            output_formats = [output_formats]
        for output_format in output_formats:
            if output_format == "mat":
                sio.savemat(os.path.join(self.path_results, "simulated_traces.mat"), {'raw_traces': self.raw_traces})
            else:
                np.save(os.path.join(self.path_results, "simulated_traces.mat"), self.raw_traces)

    def produce_and_load_vessels(self, vessels_imgs_dir, n_vessels_max=None, path_results=None):
        """

        Args:
            path_results: indicate the directory where to save the vessels produced. If None, the vessels won't be
            saved.
            vessels_imgs_dir:

        Returns:

        """
        coord_list, dimensions_list = produce_vessels(path_results=path_results, vessels_imgs_dir=vessels_imgs_dir)

        self.__add_vessels(coord_list=coord_list, dimensions_list=dimensions_list, n_vessels_max=n_vessels_max)

    def load_vessels(self, vessels_dir, n_vessels_max):
        """
        Load vessels from data generated previously
        Args:
            vessels_dir: Directory that contain the data file allowing to load vessels to the movie.
            n_vessels_max: max number of vessels that can be loaded. If None, no limit
        Returns:

        """
        # vessels_dir = "artificial_movie_generator"
        file_names = []
        # loop files in the directory, keeping the one containg vessels data
        for (dirpath, dirnames, local_filenames) in os.walk(os.path.join(vessels_dir)):
            for file_name in local_filenames:
                if file_name.endswith(".mat") and ("vessel" in file_name.lower()):
                    file_names.append(file_name)
            # looking only in the top directory
            break

        # for each file, we construct the dimensions list and the coordinates list
        for file_name in file_names:
            if (n_vessels_max is not None) and len(self.vessels) >= n_vessels_max:
                break
            data = hdf5storage.loadmat(os.path.join(vessels_dir, file_name))
            coord_list = []
            dimensions_list = []
            index = 0
            while True:
                if f"coord_{index}" in data:
                    coord_list.append(data[f"coord_{index}"])
                    dimensions_list.append(data[f"dimensions_{index}"])
                else:
                    break
                index += 1

            self.__add_vessels(coord_list=coord_list, dimensions_list=dimensions_list, n_vessels_max=n_vessels_max)

    def load_cell_coords(self, data_file, from_matlab):
        """
        Load cell coords based on model from the data_file path
        Args:
            data_file: path and file name of the file containing model of cell contours.
            from_matlab: means the cells coordinates have been computed with matlab and starts at 1

        Returns:

        """
        data = hdf5storage.loadmat(data_file)

        self.cell_models = data["coord"][0]
        self.cell_coords_from_matlab = from_matlab

        print(f"len(self.available_cell_coords) {len(self.cell_models)}")

    def __add_vessels(self, coord_list, dimensions_list, n_vessels_max=None):
        """
        Add vessels in the movie based on the the coordinates and dimensions list given
        Args:
            coord_list: List of 2D array (n_coord x 2): each column represents the x and y coordinates of each point
            of the polygon (vessel)
            dimensions_list: List of 1D array with 2 values, integer, representing the height and width of the movie
            n_vessels_max: max number of vessels that can be loaded. If None, no limit
        Returns:

        """
        dimensions_to_fit = self.dimensions
        for index, coord in enumerate(coord_list):
            if (n_vessels_max is not None) and len(self.vessels) >= n_vessels_max:
                break
            coord_tuple_list = []
            dimensions = dimensions_list[index]

            ratio_x = dimensions_to_fit[0] / dimensions[0]
            ratio_y = dimensions_to_fit[1] / dimensions[1]

            for (x, y) in coord:
                coord_tuple_list.append((x * ratio_x, y * ratio_y))
            self.vessels.append(geometry.Polygon(coord_tuple_list))
        # print(f"vessel added, len vessels: {len(self.vessels)}")


if __name__ == '__main__':

    # root path, just used to avoid copying the path everywhere
    root_path = '../data'

    # path to calcium imaging data
    data_path = root_path

    raster_dur_img_path = os.path.join(data_path, "raster_dur_images")

    # file containing cell models, based on suite2p segmentation from a few
    # 2-photon calcium imaging recorded in the CA1 pyramidal layer on pups
    cell_models_file = os.path.join(data_path,
                                    "cell_models_simulated_movie_suite2p.mat")

    # path of the directory where the results will be save
    # a directory will be created each time the prediction is run
    # the directory name will be the date and time at which the analysis has been run
    # the predictions will be in this directory.
    path_results = os.path.join("/Users/pappyhammer/Documents/academique/python_code_output/synthetic_raster_plot_outputs")
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    path_results = os.path.join(path_results, time_str)
    if not os.path.isdir(path_results):
        os.mkdir(path_results)

    # #### First step: create an instance of SimulatedMovieGenerator and decide of the parameters
    # #### (dimensions, number of frames, transients' rate etc...)

    """
    dimensions: tuple of 2 int, represent the dimension of the movie in pixels, (x, y)
    n_overlap_by_cell_range: (tuple of 2 int) bottom and upper range of how many cells will overlap
                our cell "of interest" aimed to train the classifier
    non_overlap_by_cell_range:  (tuple of 2 int) bottom and upper range of  how many cells will be near
                our cells "of interest" aimed to train the classifier, but without any overlap
    n_cells_of_interest: (int) cells of interest, are the cell for which we want to control how many overlaps,
                 their activity the aim is to use them to train the classifier
    range_n_transient_cells_of_interest: bottom and upper range of the number of transients over 1000 frames
                for our cells of interest
    range_n_transient_overlapping_cells: bottom and upper range of the number of transients over 1000 frames
                for cells that overlap our cell of interest
    range_n_transient_other_cells: bottom and upper range of the number of transients over 1000 frames
                for cells that are not of interest or overlap them
    with_mvt: boolean, is True, means some motion will be added, otherwise it would produce a movie with 
                perfect motion correction  
    path_results: Directory in which to save the results
    time_str (string): id (meant to be a timestamps string representation) in order to  give a unique id to files. 
    """

    img_file_names = []
    img_file_names_dict = SortedDict()
    # loop files in the directory, keeping the one containg vessels data
    for (dirpath, dirnames, local_filenames) in os.walk(raster_dur_img_path):
        for file_name in local_filenames:
            if (file_name.endswith(".png") or file_name.endswith(".jpg") or file_name.endswith(".jpeg")) and (
            not file_name.startswith(".")):
                img_file_names_dict[file_name] = file_name
                # img_file_names.append(file_name)
        # looking only in the top directory
        break

    img_file_names = list(img_file_names_dict.keys())
    print(f"img_file_names {img_file_names}")
    raster_dur_from_image = [os.path.join(raster_dur_img_path, f) for f in img_file_names]
    movie_generator = SimulatedMovieGenerator(dimensions=(120, 120),
                                              n_cells_of_interest=16,
                                              n_overlap_by_cell_range=(1, 4),
                                              non_overlap_by_cell_range=(2, 10),
                                              range_n_transient_cells_of_interest=(2, 4),
                                              range_n_transient_overlapping_cells=(8, 16),
                                              range_n_transient_other_cells=(2, 16),
                                              range_n_transient_img_cells=(3, 15),
                                              n_frames=1500,
                                              raster_dur_from_image=raster_dur_from_image,
                                              with_mvt=False,
                                              time_str=time_str,
                                              path_results=path_results)


    # ## OPTIONAL CODE FOR INCLUDING VESSELS ###
    """
    (optional) This code allows you to create vessels based on tiff images. You can find examples on our gitlab, 
    it should be a picture with a white background and the vessel should be in black. 
    """
    load_vessels = False

    if load_vessels:
        vessels_imgs_dir = os.path.join(data_path, "vessels_tiff_imgs")

        # you can indicate a path where to save the newly created vessels using path_results to load
        # them directly another time
        # n_vessels_max is the number of vessels to add, if the directory contains more
        # than n_vessels_max

        movie_generator.produce_and_load_vessels(vessels_imgs_dir=vessels_imgs_dir,
                                                 n_vessels_max=2, path_results=None)
        # ## Code to load vessels already created

        # vessels_dir is the directory that contains the vessels
        # n_vessels_max is the number of vessels to add, if the directory contains more
        # than n_vessels_max
        vessels_dir = data_path
        movie_generator.load_vessels(vessels_dir=vessels_dir, n_vessels_max=2)

    # ## END VESSELS ###

    # ## Then it is necessary to load some cell models that will be use to generate the contour map.
    # from_matlab: boolean that indicates if the coordinates have been generated (indexing starts at 1 then)
    # using matlab code and so starting from 1 and not 0.
    movie_generator.load_cell_coords(data_file=cell_models_file, from_matlab=False)

    """
    Finally, we generate the movie.

    The automatic steps are:

    1. Generating a contour map (a figure will be created to represent it as well as a file with the coordinates)
    2. Generating a for each transients and building a raster 
    (figure will be created as well as files to be opened in the GUI to check the activity)
    3. Generating the traces (figure created as well)
    4. Generating the movie (tiff file created)

    """

    movie_generator.generate_movie()
