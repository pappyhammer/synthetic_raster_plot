
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib
from shapely import geometry
import PIL
from PIL import ImageDraw
import math
import scipy.stats as stats
# from matplotlib.colors import NoNorm
import matplotlib.cm as cm
import networkx as nx
import matplotlib.colors as plt_colors
from shapely.geometry import MultiPoint, LineString
import os

BREWER_COLORS = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
                 '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
                 '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
                 '#74add1', '#4575b4', '#313695']

class CellsCoord:

    def __init__(self, coords=None, pixel_masks=None, nb_lines=None, nb_col=None,
                 from_matlab=False, invert_xy_coord=False):  # from_suite_2p=False, from_fiji=False
        """

        Args:
            coords: list of array of 2 lines (x & y) and n columns, n being the number of coordinate to set the contour of
            the cell. if None, then pixel_masks must be not None
            pixel_masks: a list of list of tuples of int representing the coordinates of the pixels in a given cell.
                len(pixel_masks) == number of cells.
            nb_lines: If None, set it using the max coordinate in coords
            nb_col:If None, set it using the max coordinate in coords
            from_matlab: Means coordinates have been computed on matlab and indexing starts at 1 so
            invert_xy_coord: if True, invert the xy coords
        """

        if coords is None and pixel_masks is None:
            raise Exception("Coords or pixel_masks must be given to instanciate CellsCoord")

        # dict of tuples, key is the cell #, cell center coords x and y (x and y are inverted for imshow)
        self.center_coord = dict()
        # shapely polygons
        self.cells_polygon = dict()
        # first key is an int representing the number of the cell, and value is a list of cells it interesects
        self.intersect_cells = dict()
        # for garbage collector issue only
        self.cell_contour = None

        # contour coords
        self.coords = coords

        # ---------------------------
        # we clean coords if not None
        # ---------------------------
        if self.coords is not None:
            for cell, coord in enumerate(self.coords):
                # coord = np.array(coord)
                # print(f"coord {coord}")
                if from_matlab:
                    # it is necessary to remove one, as data comes from matlab, starting from 1 and not 0
                    coord = coord - 1
                # in case it would be floats
                coord.astype(int, copy=False)
                if invert_xy_coord:
                    # invert xy lines
                    coord = np.flipud(coord)
                self.coords[cell] = coord

        self.pixel_masks = pixel_masks

        # ---------------------------
        # we clean pixel_masks if not None
        # we change the structure of so each element
        # is an np.array of 2d with 2 lines and n columns
        # n being the number of pixels
        # ---------------------------
        if self.pixel_masks is not None:
            tmp_pixel_masks = self.pixel_masks
            self.pixel_masks = []
            to_substract = 0
            if from_matlab:
                to_substract = 1
            for pixel_mask in tmp_pixel_masks:
                pix_array = np.zeros((2, len(pixel_mask)), dtype="int16")
                for index_pix, pix in enumerate(pixel_mask):
                    if invert_xy_coord:
                        pix_array[0, index_pix] = int(pix[1] - to_substract)
                        pix_array[1, index_pix] = int(pix[0] - to_substract)
                    else:
                        pix_array[0, index_pix] = int(pix[0] - to_substract)
                        pix_array[1, index_pix] = int(pix[1] - to_substract)
                self.pixel_masks.append(pix_array)

        # --------------------------------------------
        # Number of lines and columsn (height & width)
        # of the Field Of View (FOV)
        # --------------------------------------------
        if (nb_lines is None) or (nb_col is None):
            # in that case we take the maximum coordinates to fix the size of the FOV adding some padding
            self.nb_lines = 0
            self.nb_col = 0
            padding = 2

            if self.coords is not None:
                for cell, coord in enumerate(self.coords):
                    self.nb_col = max(self.nb_col, np.max(coord[0, :]) + padding)
                    self.nb_lines = max(self.nb_lines, np.max(coord[1, :]) + padding)
            else:
                for pixel_mask in self.pixel_masks:
                    # raise Exception("TOTO")
                    self.nb_col = max(self.nb_col, np.max(pixel_mask[0, :]) + padding)
                    self.nb_lines = max(self.nb_lines, np.max(pixel_mask[1, :]) + padding)
        else:
            self.nb_lines = nb_lines
            self.nb_col = nb_col

        if self.coords is not None:
            self.n_cells = len(self.coords)
        else:
            self.n_cells = len(self.pixel_masks)

        # -----------------------------------------------------------
        # Using pixels_masks to build contour coords if doesn't exist
        # -----------------------------------------------------------
        if self.pixel_masks is not None:
            if self.coords is None:
                # first we build self.coords from self.pixel_masks
                self.coords = []
                for cell, pixel_mask in enumerate(self.pixel_masks):
                    # we use pixel_masks to build the coords, using convex_hull
                    list_points_coord = list(zip(pixel_mask[0], pixel_mask[1]))
                    convex_hull = MultiPoint(list_points_coord).convex_hull
                    if isinstance(convex_hull, LineString):
                        coord_shapely = MultiPoint(list_points_coord).convex_hull.coords
                    else:
                        coord_shapely = MultiPoint(list_points_coord).convex_hull.exterior.coords
                    self.coords.append(np.array(coord_shapely).transpose())

        # ----------------------------------------------------------------------
        # Going through contour coords and building polygons and centers of mass
        # ----------------------------------------------------------------------
        # print(f"self.coords {self.coords}")
        for cell, coord in enumerate(self.coords):
            if coord.shape[0] == 0:
                print(f'Error: {cell} coord.shape {coord.shape}')
                continue

            self.build_cell_polygon_from_contour(cell=cell)

            # poly = MultiPoint(list_points).convex_hull

            use_centroid = True

            if use_centroid:
                self.center_coord[cell] = self.cells_polygon[cell].centroid.coords[0]
            else:
                bw = np.zeros((self.nb_col, self.nb_lines), dtype="int8")
                # we could use morphology.binary_fill_holes to build pixel_mask from coord
                # but let's keep pixel_mask to None if not passed as argument
                # morphology.binary_fill_holes(input
                bw[coord[0, :], coord[1, :]] = 1
                c_x, c_y = ndimage.center_of_mass(bw)
                self.center_coord[cell] = (c_x, c_y)

            # if (cell == 0) or (cell == 159):
            #     print(f"cell {cell} fig")
            #     fig, ax = plt.subplots(nrows=1, ncols=1,
            #                            gridspec_kw={'height_ratios': [1]},
            #                            figsize=(5, 5))
            #     ax.imshow(bw)
            #     plt.show()
            #     plt.close()

        # --------------------------------------------------------------
        # building the intersections dictionnary using coords
        # --------------------------------------------------------------
        for cell_1 in np.arange(self.n_cells - 1):
            if cell_1 not in self.intersect_cells:
                if cell_1 not in self.cells_polygon:
                    continue
                self.intersect_cells[cell_1] = set()
            for cell_2 in np.arange(cell_1 + 1, self.n_cells):
                if cell_2 not in self.cells_polygon:
                    continue
                if cell_2 not in self.intersect_cells:
                    self.intersect_cells[cell_2] = set()
                poly_1 = self.cells_polygon[cell_1]
                poly_2 = self.cells_polygon[cell_2]
                # if it intersects and not only touches if adding and (not poly_1.touches(poly_2))
                # try:
                if poly_1.intersects(poly_2):
                    self.intersect_cells[cell_2].add(cell_1)
                    self.intersect_cells[cell_1].add(cell_2)
                # except shapely.errors.TopologicalError:
                #     print(f"cell_1 {cell_1}, cell_2 {cell_2}")
                #     print(f"cell_1 {poly_1.is_valid}, cell_2 {poly_2.is_valid}")
                #     poly_1 = poly_1.buffer(0)
                #     poly_2 = poly_2.buffer(0)
                #     print(f"cell_1 {poly_1.is_valid}, cell_2 {poly_2.is_valid}")
                #     raise Exception("shapely.errors.TopologicalError")

    def build_raw_traces_from_movie(self, movie, without_overlap=False, buffer_overlap=1):
        """
        Return a 2d array representing the fluoresence signal raw trace for each cell
        Args:
            movie: 3d array n_frames x len_y x len_x
            without_overlap: (bool) if True, means the trace will be build only from the pixels from this cell
            buffer_overlap: indicate from how much pixels increasing the size of overlaping cell

        Returns: A 2d array (n_cells * n_frames) of float

        """
        buffer_overlap = max(0, buffer_overlap)
        raw_traces = np.zeros((self.n_cells, movie.shape[0]))
        for cell in np.arange(self.n_cells):
            tmp_buffer_overlap = buffer_overlap
            while True:
                mask = self.get_cell_mask(cell=cell,
                                          dimensions=(movie.shape[1], movie.shape[2]),
                                          without_overlap=without_overlap,
                                          buffer_overlap=tmp_buffer_overlap)
                n_pixels_in_cell = np.sum(mask)
                if (n_pixels_in_cell > 2) or tmp_buffer_overlap == 0:
                    break
                tmp_buffer_overlap -= 1

            # print(f"n_pixels_in_cell for cell {cell}: {n_pixels_in_cell}")
            if n_pixels_in_cell > 0:
                raw_traces[cell, :] = np.mean(movie[:, mask], axis=1)
        return raw_traces

    def build_cell_polygon_from_contour(self, cell):
        """
        Build the (shapely) polygon representing a given cell using its contour's coordinates.
        Args:
            cell:

        Returns:

        """
        coord = self.coords[cell]
        # make a list of tuple representing x, y coordinates of the contours points
        coord_list_tuple = list(zip(coord[0], coord[1]))

        # buffer(0) or convex_hull could be used if the coords are a list of points not
        # in the right order. However buffer(0) return a MultiPolygon with no coords available.
        if len(coord_list_tuple) < 3:
            list_points = []
            for coords in coord_list_tuple:
                list_points.append(geometry.Point(coords))
            self.cells_polygon[cell] = geometry.LineString(list_points)
        else:
            self.cells_polygon[cell] = geometry.Polygon(coord_list_tuple)  # .convex_hull # buffer(0)

        # self.coords[cell] = np.array(self.cells_polygon[cell].exterior.coords).transpose()

    def get_cell_mask(self, cell, dimensions, without_overlap=False, buffer_overlap=1):
        """
        Return the mask of the pixels of the cell
        :param cell:
        :param dimensions: height x width
        :param without_overlap: if True, means with return only the pixel belonging to this cell
            buffer_overlap: indicate from how much pixels increasing the size of overlaping cell
        :return: binary 2d array (movie dimension), with 1 for the pixels belonging to the cell
        """
        poly_gon = self.cells_polygon[cell]
        img = PIL.Image.new('1', (dimensions[1], dimensions[0]), 0)
        if (self.pixel_masks is not None) and (not without_overlap):
            img = np.array(img)
            pix_array = self.pixel_masks[cell]
            for index in np.arange(pix_array.shape[1]):
                img[pix_array[1, index], pix_array[0, index]] = 1
            # if without_overlap and (cell in self.intersect_cells) and (len(self.intersect_cells[cell]) > 0):
            #     for overlaping_cell in self.intersect_cells[cell]:
            #         overlaping_poly_gon = self.cells_polygon[overlaping_cell]
            #         # increasing the size of the cell of 1
            #         if buffer_overlap > 0:
            #             overlaping_poly_gon = overlaping_poly_gon.buffer(buffer_overlap)
            #         if isinstance(overlaping_poly_gon, geometry.LineString):
            #             ImageDraw.Draw(img).polygon(list(overlaping_poly_gon.coords), outline=0,
            #                                         fill=0)
            #         else:
            #             ImageDraw.Draw(img).polygon(list(overlaping_poly_gon.exterior.coords), outline=0,
            #                                         fill=0)
                    # pix_array = self.pixel_masks[overlaping_cell]
                    # for index in np.arange(pix_array.shape[1]):
                    #     img[pix_array[1, index], pix_array[0, index]] = 0
        else:
            if isinstance(poly_gon, geometry.LineString):
                ImageDraw.Draw(img).polygon(list(poly_gon.coords), outline=1,
                                            fill=1)
            else:
                ImageDraw.Draw(img).polygon(list(poly_gon.exterior.coords), outline=1,
                                            fill=1)
            if without_overlap and (cell in self.intersect_cells) and (len(self.intersect_cells[cell]) > 0):
                for overlaping_cell in self.intersect_cells[cell]:
                    overlaping_poly_gon = self.cells_polygon[overlaping_cell]
                    if buffer_overlap > 0:
                        overlaping_poly_gon = overlaping_poly_gon.buffer(buffer_overlap)
                    if isinstance(overlaping_poly_gon, geometry.LineString):
                        ImageDraw.Draw(img).polygon(list(overlaping_poly_gon.coords), outline=0,
                                                    fill=0)
                    else:
                        ImageDraw.Draw(img).polygon(list(overlaping_poly_gon.exterior.coords), outline=0,
                                                    fill=0)
        return np.array(img)

    def match_cells_indices(self, coord_obj, path_results, plot_title_opt=""):
        """

        :param coord_obj: another instanc of coord_obj
        :return: a 1d array, each index corresponds to the index of a cell of coord_obj, and map it to an index to self
        or -1 if no cell match
        """
        mapping_array = np.zeros(len(coord_obj.coord), dtype='int16')
        for cell in np.arange(len(coord_obj.coord)):
            c_x, c_y = coord_obj.center_coord[cell]
            distances = np.zeros(len(self.coords))
            for self_cell in np.arange(len(self.coords)):
                self_c_x, self_c_y = self.center_coord[self_cell]
                # then we calculte the cartesian distance to all other cells
                distances[self_cell] = math.sqrt((self_c_x - c_x) ** 2 + (self_c_y - c_y) ** 2)
            if np.min(distances) <= 2:
                mapping_array[cell] = np.argmin(distances)
            else:
                mapping_array[cell] = -1
        plot_result = True
        if plot_result:
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   gridspec_kw={'height_ratios': [1]},
                                   figsize=(20, 20))

            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")

            # dark blue
            other_twin_color = list((0.003, 0.313, 0.678, 1.0))
            n_twins = 0
            # red
            other_orphan_color = list((1, 0, 0, 1.0))
            n_other_orphans = 0
            # light blue
            self_twin_color = list((0.560, 0.764, 1, 1.0))
            # green
            self_orphan_color = list((0.278, 1, 0.101, 1.0))
            n_self_orphans = 0
            with_edge = True
            edge_line_width = 1
            z_order_cells = 12
            for cell in np.arange(len(coord_obj.coord)):
                xy = coord_obj.coord[cell].transpose()
                if with_edge:
                    line_width = edge_line_width
                    edge_color = "white"
                else:
                    edge_color = "white"
                    line_width = 0
                # allow to set alpha of the edge to 1
                if mapping_array[cell] >= 0:
                    # dark blue
                    face_color = other_twin_color
                    n_twins += 1
                else:
                    # red
                    face_color = other_orphan_color
                    n_other_orphans += 1
                face_color[3] = 0.8
                face_color = tuple(face_color)
                cell_contour = patches.Polygon(xy=xy,
                                               fill=True, linewidth=line_width,
                                               facecolor=face_color,
                                               edgecolor=edge_color,
                                               zorder=z_order_cells)  # lw=2
                ax.add_patch(cell_contour)
            for cell in np.arange(len(self.coords)):
                xy = self.coords[cell].transpose()
                if with_edge:
                    line_width = edge_line_width
                    edge_color = "white"
                else:
                    edge_color = "white"
                    line_width = 0
                # allow to set alpha of the edge to 1
                if cell in mapping_array:
                    # light blue
                    face_color = self_twin_color
                else:
                    # green
                    face_color = self_orphan_color
                    n_self_orphans += 1
                face_color[3] = 0.8
                face_color = tuple(face_color)
                # drawing the cell
                cell_contour = patches.Polygon(xy=xy,
                                               fill=True, linewidth=line_width,
                                               facecolor=face_color,
                                               edgecolor=edge_color,
                                               zorder=z_order_cells)  # lw=2
                ax.add_patch(cell_contour)
                with_cell_numbers = True
                if with_cell_numbers:
                    self.plot_text_cell(cell=cell, cell_numbers_color="black", ax=ax, text_size=6)
            fontsize = 12
            plt.text(x=190, y=180,
                     s=f"{n_twins}", color=self_twin_color, zorder=22,
                     ha='center', va="center", fontsize=fontsize, fontweight='bold')
            plt.text(x=190, y=185,
                     s=f"{n_self_orphans}", color=self_orphan_color, zorder=22,
                     ha='center', va="center", fontsize=fontsize, fontweight='bold')
            plt.text(x=190, y=190,
                     s=f"{n_twins}", color=other_twin_color, zorder=22,
                     ha='center', va="center", fontsize=fontsize, fontweight='bold')
            plt.text(x=190, y=195,
                     s=f"{n_other_orphans}", color=other_orphan_color, zorder=22,
                     ha='center', va="center", fontsize=fontsize, fontweight='bold')
            ax.set_ylim(0, self.nb_lines)
            ax.set_xlim(0, self.nb_col)
            ylim = ax.get_ylim()
            # invert Y
            ax.set_ylim(ylim[::-1])
            plt.setp(ax.spines.values(), color="black")
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            save_format = "png"
            fig.savefig(f'{path_results}/cells_map_{plot_title_opt}.{save_format}',
                        format=f"{save_format}",
                        facecolor=fig.get_facecolor())
            plt.close()
        return mapping_array

    def plot_cells_map(self, path_results, data_id, use_pixel_masks=False, title_option="", connections_dict=None,
                       background_color=(0, 0, 0, 1), default_cells_color=(1, 1, 1, 1.0),
                       default_edge_color="white",
                       dont_fill_cells_not_in_groups=False,
                       link_connect_color="white", link_line_width=1,
                       cell_numbers_color="dimgray", show_polygons=False,
                       cells_to_link=None, edge_line_width=2, cells_alpha=1.0,
                       fill_polygons=True, cells_groups=None, cells_groups_colors=None,
                       cells_groups_alpha=None,
                       cells_to_hide=None, img_on_background=None,
                       real_size_image_on_bg=True,
                       cells_groups_edge_colors=None, with_edge=False,
                       with_cell_numbers=False, text_size=6, save_formats="png",
                       save_plot=True, return_fig=False, ax_to_use=None,
                       verbose=False,
                       use_welsh_powell_coloring=False, dpi=300):
        """

        Args:
            path_results:
            data_id:
            use_pixel_masks:
            title_option:
            connections_dict: key is an int representing a cell number, and value is a dict representing the cells it
            connects to. The key is a cell is connected too, and the value represent the strength of the connection
            (like how many times it connects to it)
            background_color:
            default_cells_color:
            default_edge_color:
            dont_fill_cells_not_in_groups:
            link_connect_color:
            link_line_width:
            cell_numbers_color:
            show_polygons:
            cells_to_link:
            edge_line_width:
            cells_alpha:
            fill_polygons:
            cells_groups:
            cells_groups_colors:
            cells_groups_alpha:
            cells_to_hide:
            img_on_background:
            real_size_image_on_bg: if True, the size of the figure will respect the original size of the background
            image
            cells_groups_edge_colors:
            with_edge:
            with_cell_numbers:
            text_size:
            save_formats:
            save_plot:
            return_fig:
            ax_to_use:
            verbose: if True, some informations will be printed along the way
            use_welsh_powell_coloring: if True, use welsh powell algorithm to color all cells that intersect with
        different color. In that case, cancel cell_groups arguments.
            dpi:

        Returns:

        """

        if use_pixel_masks and (self.pixel_masks is None):
            print(f"No pixel_masks available in plot_cells_map() and use_pixel_masks argument set to True")
            return

        cells_center = self.center_coord
        n_cells = len(self.coords)
        if cells_to_hide is None:
            cells_to_hide = []

        if use_welsh_powell_coloring:
            if verbose:
                print("Welsh Powell coloring:")
            isolated_cell_color = default_cells_color
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
                n_intersect = len(self.intersect_cells[cell])
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
                            n_intersect = len(self.intersect_cells[cells_to_expend[0]])
                            if n_intersect > 0:
                                for inter_cell in self.intersect_cells[cells_to_expend[0]]:
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
                # cells_groups_colors.append(cm.nipy_spectral(float(color_code + 1) / (max_color_code + 1)))
                color_hex = BREWER_COLORS[color_code % len(BREWER_COLORS)]
                # putting it with value from 0.0 to 1.0
                color_rgb = plt_colors.hex2color(color_hex)
                # adding the alpha
                color_rgba = [c for c in color_rgb]
                color_rgba.append(cells_alpha)
                cells_groups_colors.append(color_rgba)
                cells_groups_edge_colors.append(default_edge_color)
                cells_groups_alpha.append(0.8)
            cells_groups.append(isolated_group)
            cells_groups_colors.append(isolated_cell_color)
            cells_groups_alpha.append(1)
            cells_groups_edge_colors.append("white")
            if verbose:
                print(f"Isolated cells: {len(isolated_group)}")
                print(f"Grouped cells: {self.n_cells - len(isolated_group)}")

        cells_in_groups = []
        if cells_groups is not None:
            for group_id, cells_group in enumerate(cells_groups):
                cells_in_groups.extend(cells_group)
        cells_in_groups = np.array(cells_in_groups)
        cells_not_in_groups = np.setdiff1d(np.arange(n_cells), cells_in_groups)

        if ax_to_use is None:
            if (img_on_background is None) or (real_size_image_on_bg is False):
                fig, ax = plt.subplots(nrows=1, ncols=1,
                                       gridspec_kw={'height_ratios': [1]},
                                       figsize=(20, 20), dpi=dpi)
                fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
                plt.tight_layout()
            else:
                # then we want the figure to respect the size of the image on background
                height, width = img_on_background.shape

                # What size does the figure need to be in inches to fit the image?
                figsize = width / float(dpi), height / float(dpi)

                # Create a figure of the right size with one axes that takes up the full figure
                fig = plt.figure(figsize=figsize)
                ax = fig.add_axes([0, 0, 1, 1])

                text_size = 0.05
                edge_line_width = 0.1

            fig.patch.set_facecolor(background_color)
            ax.set_facecolor(background_color)
        else:
            ax = ax_to_use

        if img_on_background is not None:
            img_on_background = img_on_background - np.min(img_on_background)
            # print(f"np.mean(img_on_background) {np.mean(img_on_background)} {np.max(img_on_background)}")
            n_bits = 8
            # np.transpose(img_on_background)
            ax.imshow(img_on_background, cmap=plt.get_cmap("gray"))  # , vmin=0,
            # vmax=np.max(img_on_background), interpolation='nearest') # vmax= 4096 math.pow(2, n_bits)-1

        if use_pixel_masks:
            cells_imshow_alpha = 0.2 if (img_on_background is not None) else 1
            bg_imshow_alpha = 0 if (img_on_background is not None) else 1

            self.add_cells_using_pixel_masks_on_ax(ax, cells_groups, cells_not_in_groups, cells_to_hide,
                                                   default_cells_color, cells_groups_colors, with_cell_numbers,
                                                   cell_numbers_color, text_size,
                                                   background_color, cells_imshow_alpha=cells_imshow_alpha,
                                                   bg_imshow_alpha=bg_imshow_alpha)
        else:
            self.add_cells_using_polygons_on_ax(ax, cells_groups, cells_not_in_groups, cells_to_hide, with_edge,
                                                edge_line_width, default_cells_color,
                                                default_edge_color, cells_groups_edge_colors, cells_groups_colors,
                                                cells_groups_alpha, cells_alpha, with_cell_numbers,
                                                cell_numbers_color, text_size, dont_fill_cells_not_in_groups)

        ax.set_ylim(0, self.nb_lines)
        ax.set_xlim(0, self.nb_col)
        ylim = ax.get_ylim()
        # invert Y
        ax.set_ylim(ylim[::-1])

        if connections_dict is not None:
            zorder_lines = 15
            for neuron in connections_dict.keys():
                # plot a line to all out of the neuron
                for connected_neuron, nb_connexion in connections_dict[neuron].items():
                    line_width = link_line_width + np.log(nb_connexion)

                    c_x = cells_center[neuron][0]
                    c_y = cells_center[neuron][1]
                    c_x_c = cells_center[connected_neuron][0]
                    c_y_c = cells_center[connected_neuron][1]

                    line = plt.plot((c_x, c_x_c), (c_y, c_y_c), linewidth=line_width, c=link_connect_color,
                                    zorder=zorder_lines)[0]
        # print(f"(self.cells_groups is not None) {(self.cells_groups is not None)} show_polygons {show_polygons}")
        if (cells_groups is not None) and show_polygons:
            for group_id, cells in enumerate(cells_groups):
                points = np.zeros((2, len(cells)))
                for cell_id, cell in enumerate(cells):
                    c_x, c_y = cells_center[cell]
                    points[0, cell_id] = c_x
                    points[1, cell_id] = c_y
                # finding the convex_hull for each group
                xy = convex_hull(points=points)
                # xy = xy.transpose()
                # print(f"xy {xy}")
                # xy is a numpy array with as many line as polygon point
                # and 2 columns: x and y coords of each point
                face_color = list(cells_groups_colors[group_id])
                # changing alpha
                face_color[3] = 0.3
                face_color = tuple(face_color)
                # edge alpha will be 1
                poly_gon = patches.Polygon(xy=xy,
                                           fill=fill_polygons, linewidth=0, facecolor=face_color,
                                           edgecolor=cells_groups_colors[group_id],
                                           zorder=15, lw=3)
                ax.add_patch(poly_gon)

        # plt.title(f"Cells map {data_id} {title_option}")

        # ax.set_frame_on(False)

        # invert Y
        ax.set_ylim(ylim[::-1])
        plt.setp(ax.spines.values(), color=background_color)
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)

        # ax.xaxis.set_ticks_position('none')
        # ax.yaxis.set_ticks_position('none')
        #  :param plot_option: if 0: plot n_out and n_int,
        #  if 1 only n_out, if 2 only n_in, if 3: only n_out with dotted to
        # show the commun n_in and n_out, if 4: only n_in with dotted to show the commun n_in and n_out,
        if ax_to_use is None:
            if save_plot:
                if isinstance(save_formats, str):
                    save_formats = [save_formats]
                for save_format in save_formats:
                    fig.savefig(f'{path_results}/{data_id}_cell_maps_{title_option}.{save_format}',
                                format=f"{save_format}",
                                facecolor=fig.get_facecolor())
            if return_fig:
                return fig
            else:
                plt.close()

    def add_cells_using_pixel_masks_on_ax(self, ax, cells_groups, cells_not_in_groups, cells_to_hide,
                                          default_cells_color, cells_groups_colors, with_cell_numbers,
                                          cell_numbers_color, text_size,
                                          background_color, cells_imshow_alpha, bg_imshow_alpha):
        """
        Using pixel mask if it exists
        Args:
            ax:
            cells_groups:
            cells_not_in_groups:
            cells_to_hide:
            default_cells_color:
            cells_groups_colors:
            with_cell_numbers:
            cell_numbers_color:
            text_size:
            background_color:

        Returns:

        """
        if self.pixel_masks is None:
            return

        use_multiple_imshow = False

        background_color = [code for code in background_color]
        background_color[-1] = bg_imshow_alpha
        cmap_colors = [background_color]
        # 0 represents the background
        color_index = 1
        cells_displayed = []
        if not use_multiple_imshow:
            img = np.zeros((self.nb_col, self.nb_lines), dtype="int8")
        else:
            default_cells_color = [code for code in default_cells_color]
            default_cells_color[-1] = cells_imshow_alpha

        if cells_groups is not None:
            for group_index, cell_group in enumerate(cells_groups):
                for cell in cell_group:
                    if cell in cells_to_hide:
                        continue
                    if use_multiple_imshow:
                        img = np.zeros((self.nb_col, self.nb_lines), dtype="int8")
                        cmap_colors = [background_color]
                        color_index = 1

                    cells_displayed.append(cell)
                    pixel_mask = self.pixel_masks[cell]
                    img[pixel_mask[0, :], pixel_mask[1, :]] = color_index
                    face_color = list(cells_groups_colors[group_index])
                    face_color[-1] = cells_imshow_alpha

                    if use_multiple_imshow:
                        cmap_colors.append(face_color)
                        cmap = matplotlib.colors.ListedColormap(cmap_colors)
                        ax.imshow(np.transpose(img), cmap=cmap, interpolation='nearest')
                if not use_multiple_imshow:
                    cmap_colors.append(face_color)
                    color_index += 1

        for cell in cells_not_in_groups:
            if cell in cells_to_hide:
                continue

            if use_multiple_imshow:
                img = np.zeros((self.nb_col, self.nb_lines), dtype="int8")
                cmap_colors = [background_color]
                color_index = 1

            cells_displayed.append(cell)
            pixel_mask = self.pixel_masks[cell]
            img[pixel_mask[0, :], pixel_mask[1, :]] = color_index

            cmap_colors.append(default_cells_color)
            if use_multiple_imshow:
                cmap = matplotlib.colors.ListedColormap(cmap_colors)
                ax.imshow(np.transpose(img), cmap=cmap, interpolation='nearest')
            else:
                color_index += 1

        if not use_multiple_imshow:
            cmap = matplotlib.colors.ListedColormap(cmap_colors)
            ax.imshow(np.transpose(img), cmap=cmap,
                      interpolation='nearest')  # , origin='lower') alpha=cells_imshow_alpha,

        if with_cell_numbers:
            for cell in cells_displayed:
                self.plot_text_cell(cell=cell, cell_numbers_color=cell_numbers_color, ax=ax,
                                    text_size=text_size)

    def add_cells_using_polygons_on_ax(self, ax, cells_groups, cells_not_in_groups, cells_to_hide, with_edge,
                                       edge_line_width, default_cells_color,
                                       default_edge_color, cells_groups_edge_colors, cells_groups_colors,
                                       cells_groups_alpha, cells_alpha, with_cell_numbers,
                                       cell_numbers_color, text_size, dont_fill_cells_not_in_groups):
        """
        Add cells to a matplolib ax using the polygons representation. Arguments give parameters to apply
        Args:
            ax:
            cells_groups:
            cells_not_in_groups:
            cells_to_hide:
            with_edge:
            edge_line_width:
            default_cells_color:
            default_edge_color:
            cells_groups_edge_colors:
            cells_groups_colors:
            cells_groups_alpha:
            cells_alpha:
            with_cell_numbers:
            cell_numbers_color:
            text_size:
            dont_fill_cells_not_in_groups:

        Returns:

        """
        z_order_cells = 12
        if cells_groups is not None:
            for group_index, cell_group in enumerate(cells_groups):
                for cell in cell_group:
                    if cell in cells_to_hide:
                        continue

                    xy = self.coords[cell].transpose()
                    if with_edge:
                        line_width = edge_line_width
                        if cells_groups_edge_colors is None:
                            edge_color = default_edge_color
                        else:
                            edge_color = cells_groups_edge_colors[group_index]
                    else:
                        edge_color = cells_groups_colors[group_index]
                        line_width = 0
                    # allow to set alpha of the edge to 1
                    face_color = list(cells_groups_colors[group_index])
                    # changing alpha
                    if cells_groups_alpha is not None:
                        face_color[3] = cells_groups_alpha[group_index]
                    else:
                        face_color[3] = cells_alpha
                    face_color = tuple(face_color)
                    # TODO: a solution to explore to plot pixel by pixel and not using the polygon version
                    #  https://stackoverflow.com/questions/39753282/scatter-plot-with-single-pixel-marker-in-matplotlib
                    self.cell_contour = patches.Polygon(xy=xy,
                                                        fill=True, linewidth=line_width,
                                                        facecolor=face_color,
                                                        edgecolor=edge_color,
                                                        zorder=z_order_cells)  # lw=2
                    ax.add_patch(self.cell_contour)
                    if with_cell_numbers:
                        self.plot_text_cell(cell=cell, cell_numbers_color=cell_numbers_color, ax=ax,
                                            text_size=text_size)

        for cell in cells_not_in_groups:
            if cell in cells_to_hide:
                continue
            xy = self.coords[cell].transpose()
            # face_color = default_cells_color
            # if dont_fill_cells_not_in_groups:
            #     face_color = None
            self.cell_contour = patches.Polygon(xy=xy,
                                                fill=not dont_fill_cells_not_in_groups,
                                                linewidth=0, facecolor=default_cells_color,
                                                edgecolor=default_edge_color,
                                                zorder=z_order_cells, lw=edge_line_width)
            ax.add_patch(self.cell_contour)

            if with_cell_numbers:
                self.plot_text_cell(cell=cell, cell_numbers_color=cell_numbers_color, ax=ax,
                                    text_size=text_size)

    def plot_text_cell(self, cell, ax, cell_numbers_color, text_size):
        """
        Plot the cell number on the cell
        Args:
            cell: integer
            ax: matplolib axis
            cell_numbers_color: color of the text
            text_size: text size (float)

        Returns:

        """
        fontsize = text_size
        if cell >= 100:
            if fontsize > 2.5:
                fontsize -= 2
        elif cell >= 10:
            if fontsize > 2:
                fontsize -= 1

        c_x_c = self.center_coord[cell][0]
        c_y_c = self.center_coord[cell][1]

        if fontsize < 0.5:
            fontweight = 'ultralight'
        else:
            fontweight = 'bold'

        ax.text(x=c_x_c, y=c_y_c,
                s=f"{cell}", color=cell_numbers_color, zorder=22,
                ha='center', va="center", fontsize=fontsize, fontweight=fontweight)

    def get_cell_new_coord_in_source(self, cell, minx, miny):
        coord = self.coords[cell]
        # coords = coords - 1
        coord = coord.astype(int)
        n_coord = len(coord[0, :])
        xy = np.zeros((n_coord, 2))
        for n in np.arange(n_coord):
            # shifting the coordinates in the square size_square+1
            xy[n, 0] = coord[0, n] - minx
            xy[n, 1] = coord[1, n] - miny
        return xy

    def scale_polygon_to_source(self, poly_gon, minx, miny):
        coords = list(poly_gon.exterior.coords)
        scaled_coords = []
        for coord in coords:
            scaled_coords.append((coord[0] - minx, coord[1] - miny))
        # print(f"scaled_coords {scaled_coords}")
        return geometry.Polygon(scaled_coords)

    def get_source_profile(self, cell, tiff_movie, traces, peak_nums, spike_nums,
                           pixels_around=0, bounds=None, buffer=None, with_full_frame=False):
        """
        Return the source profile of a cell
        :param cell:
        :param pixels_around:
        :param bounds: how much padding around the cell pretty much, coordinate of the frame covering the source profile
        4 int list
        :param buffer:
        :param with_full_frame:  Average the full frame
        :return:
        """
        # print("get_source_profile")
        len_frame_x = tiff_movie[0].shape[1]
        len_frame_y = tiff_movie[0].shape[0]

        # determining the size of the square surrounding the cell
        poly_gon = self.cells_polygon[cell]
        if bounds is None:
            minx, miny, maxx, maxy = np.array(list(poly_gon.bounds)).astype(int)
        else:
            minx, miny, maxx, maxy = bounds

        if with_full_frame:
            minx = 0
            miny = 0
            maxx = len_frame_x - 1
            maxy = len_frame_y - 1
        else:
            minx = max(0, minx - pixels_around)
            miny = max(0, miny - pixels_around)
            maxx = min(len_frame_x - 1, maxx + pixels_around)
            maxy = min(len_frame_y - 1, maxy + pixels_around)

        len_x = maxx - minx + 1
        len_y = maxy - miny + 1

        # mask used in order to keep only the cells pixel
        # the mask put all pixels in the polygon, including the pixels on the exterior line to zero
        scaled_poly_gon = self.scale_polygon_to_source(poly_gon=poly_gon, minx=minx, miny=miny)
        img = PIL.Image.new('1', (len_x, len_y), 1)
        if buffer is not None:
            scaled_poly_gon = scaled_poly_gon.buffer(buffer)
        ImageDraw.Draw(img).polygon(list(scaled_poly_gon.exterior.coords), outline=0, fill=0)
        mask = np.array(img)
        # mask = np.ones((len_x, len_y))
        # cv2.fillPoly(mask, scaled_poly_gon, 0)
        # mask = mask.astype(bool)

        source_profile = np.zeros((len_y, len_x))

        # selectionning the best peak to produce the source_profile
        peaks = np.where(peak_nums[cell, :] > 0)[0]
        threshold = np.percentile(traces[cell, peaks], 95)
        selected_peaks = peaks[np.where(traces[cell, peaks] > threshold)[0]]
        # max 10 peaks, min 5 peaks
        if len(selected_peaks) > 10:
            p = 10 / len(peaks)
            threshold = np.percentile(traces[cell, peaks], (1 - p) * 100)
            selected_peaks = peaks[np.where(traces[cell, peaks] > threshold)[0]]
        elif (len(selected_peaks) < 5) and (len(peaks) > 5):
            p = 5 / len(peaks)
            threshold = np.percentile(traces[cell, peaks], (1 - p) * 100)
            selected_peaks = peaks[np.where(traces[cell, peaks] > threshold)[0]]

        # print(f"threshold {threshold}")
        # print(f"n peaks: {len(selected_peaks)}")

        onsets_frames = np.where(spike_nums[cell, :] > 0)[0]
        pos_traces = np.copy(traces)
        pos_traces += abs(np.min(traces))
        for peak in selected_peaks:
            tmp_source_profile = np.zeros((len_y, len_x))
            onsets_before_peak = np.where(onsets_frames <= peak)[0]
            if len(onsets_before_peak) == 0:
                # shouldn't arrive
                continue
            onset = onsets_frames[onsets_before_peak[-1]]
            # print(f"onset {onset}, peak {peak}")
            frames_tiff = tiff_movie[onset:peak + 1]
            for frame_index, frame_tiff in enumerate(frames_tiff):
                tmp_source_profile += (frame_tiff[miny:maxy + 1, minx:maxx + 1] * pos_traces[cell, onset + frame_index])
            # averaging
            tmp_source_profile = tmp_source_profile / (np.sum(pos_traces[cell, onset:peak + 1]))
            source_profile += tmp_source_profile
        if len(selected_peaks) > 0:
            source_profile = source_profile / len(selected_peaks)

        return source_profile, minx, miny, mask

    def get_transient_profile(self, cell, transient, tiff_movie, traces,
                              pixels_around=0, bounds=None):
        len_frame_x = tiff_movie[0].shape[1]
        len_frame_y = tiff_movie[0].shape[0]

        # determining the size of the square surrounding the cell
        if bounds is None:
            poly_gon = self.cells_polygon[cell]
            minx, miny, maxx, maxy = np.array(list(poly_gon.bounds)).astype(int)
        else:
            minx, miny, maxx, maxy = bounds

        minx = max(0, minx - pixels_around)
        miny = max(0, miny - pixels_around)
        maxx = min(len_frame_x - 1, maxx + pixels_around)
        maxy = min(len_frame_y - 1, maxy + pixels_around)

        len_x = maxx - minx + 1
        len_y = maxy - miny + 1

        transient_profile = np.zeros((len_y, len_x))
        frames_tiff = tiff_movie[transient[0]:transient[-1] + 1]
        # print(f"transient[0] {transient[0]}, transient[1] {transient[1]}")
        # now we do the weighted average
        raw_traces = np.copy(traces)
        # so the lowest value is zero
        raw_traces += abs(np.min(raw_traces))

        for frame_index, frame_tiff in enumerate(frames_tiff):
            transient_profile += (
                    frame_tiff[miny:maxy + 1, minx:maxx + 1] * raw_traces[cell, transient[0] + frame_index])
        # averaging
        transient_profile = transient_profile / (np.sum(raw_traces[cell, transient[0]:transient[-1] + 1]))

        return transient_profile, minx, miny

    def corr_between_source_and_transient(self, cell, transient, source_profile_dict, tiff_movie, traces,
                                          source_profile_corr_dict=None,
                                          pixels_around=1):
        """
        Measure the correlation (pearson) between a source and transient profile for a giveb cell
        :param cell:
        :param transient:
        :param source_profile_dict should contains cell as key, and results of get_source_profile avec values
        :param pixels_around:
        :param source_profile_corr_dict: if not None, used to save the correlation of the source profile, f
        for memory and computing proficiency
        :return:
        """
        # print('corr_between_source_and_transient')
        poly_gon = self.cells_polygon[cell]

        # Correlation test
        bounds_corr = np.array(list(poly_gon.bounds)).astype(int)
        # looking if this source has been computed before for correlation
        if (source_profile_corr_dict is not None) and (cell in source_profile_corr_dict):
            source_profile_corr, mask_source_profile = source_profile_corr_dict[cell]
        else:
            source_profile_corr, minx_corr, \
            miny_corr, mask_source_profile, xy_source = source_profile_dict[cell]
            # normalizing
            source_profile_corr = source_profile_corr - np.mean(source_profile_corr)
            # we want the mask to be at ones over the cell
            mask_source_profile = (1 - mask_source_profile).astype(bool)
            if source_profile_corr_dict is not None:
                source_profile_corr_dict[cell] = (source_profile_corr, mask_source_profile)

        transient_profile_corr, minx_corr, miny_corr = self.get_transient_profile(cell=cell,
                                                                                  transient=transient,
                                                                                  tiff_movie=tiff_movie, traces=traces,
                                                                                  pixels_around=pixels_around,
                                                                                  bounds=bounds_corr)
        transient_profile_corr = transient_profile_corr - np.mean(transient_profile_corr)

        pearson_corr, pearson_p_value = stats.pearsonr(source_profile_corr[mask_source_profile],
                                                       transient_profile_corr[mask_source_profile])

        return pearson_corr



def welsh_powell(graph):
    """
        implementation of welsh_powell algorithm
        https://github.com/MUSoC/Visualization-of-popular-algorithms-in-Python/blob/master/Graph%20Coloring/graph_coloring.py
        Args:
            graph: instance of networkx graph

        Returns:

    """
    # sorting the nodes based on it's valency
    node_list = sorted(graph.nodes(), key=lambda x: graph.degree(x))
    # dictionary to store the colors assigned to each node
    col_val = {}
    # assign the first color to the first node
    col_val[node_list[0]] = 0
    # Assign colors to remaining N-1 nodes
    for node in node_list[1:]:
        available = [True] * len(graph.nodes())  # boolean list[i] contains false if the node color 'i' is not available

        # iterates through all the adjacent nodes and marks it's color as unavailable, if it's color has been set already
        for adj_node in graph.neighbors(node):
            if adj_node in col_val.keys():
                col = col_val[adj_node]
                available[col] = False
        clr = 0
        for clr in range(len(available)):
            if available[clr] == True:
                break
        col_val[node] = clr

    return col_val


def get_continous_time_periods(binary_array):
    """
    take a binary array and return a list of tuples representing the first and last position(included) of continuous
    positive period
    This code was copied from another project or from a forum, but i've lost the reference.
    :param binary_array:
    :return:
    """
    binary_array = np.copy(binary_array).astype("int8")
    n_times = len(binary_array)
    d_times = np.diff(binary_array)
    # show the +1 and -1 edges
    pos = np.where(d_times == 1)[0] + 1
    neg = np.where(d_times == -1)[0] + 1

    if (pos.size == 0) and (neg.size == 0):
        if len(np.nonzero(binary_array)[0]) > 0:
            return [(0, n_times-1)]
        else:
            return []
    elif pos.size == 0:
        # i.e., starts on an spike, then stops
        return [(0, neg[0])]
    elif neg.size == 0:
        # starts, then ends on a spike.
        return [(pos[0], n_times-1)]
    else:
        if pos[0] > neg[0]:
            # we start with a spike
            pos = np.insert(pos, 0, 0)
        if neg[-1] < pos[-1]:
            #  we end with aspike
            neg = np.append(neg, n_times - 1)
        # NOTE: by this time, length(pos)==length(neg), necessarily
        h = np.matrix([pos, neg])
        if np.any(h):
            result = []
            for i in np.arange(h.shape[1]):
                if h[1, i] == n_times-1:
                    result.append((h[0, i], h[1, i]))
                else:
                    result.append((h[0, i], h[1, i]-1))
            return result
    return []


def plot_spikes_raster(spike_nums=None, title=None, file_name=None,
                       time_str=None,
                       spike_train_format=False,
                       y_ticks_labels=None,
                       y_ticks_labels_size=None,
                       y_ticks_labels_color="white",
                       x_ticks_labels_color="white",
                       x_ticks_labels=None,
                       x_ticks_labels_size=None,
                       x_ticks=None,
                       hide_x_labels=False,
                       figure_background_color="black",
                       without_ticks=True,
                       save_raster=False,
                       show_raster=False,
                       plot_with_amplitude=False,
                       activity_threshold=None,
                       save_formats="png",
                       span_area_coords=None,
                       span_area_colors=None,
                       span_area_only_on_raster=True,
                       alpha_span_area=0.5,
                       cells_to_highlight=None,
                       cells_to_highlight_colors=None,
                       color_peaks_activity=False,
                       horizontal_lines=None,
                       horizontal_lines_colors=None,
                       horizontal_lines_sytle=None,
                       horizontal_lines_linewidth=None,
                       vertical_lines=None,
                       vertical_lines_colors=None,
                       vertical_lines_sytle=None,
                       vertical_lines_linewidth=None,
                       scatters_on_traces=None,
                       scatters_on_traces_marker="*",
                       scatters_on_traces_size=5,
                       sliding_window_duration=1,
                       show_sum_spikes_as_percentage=False,
                       span_cells_to_highlight=None,
                       span_cells_to_highlight_colors=None,
                       spike_shape="|",
                       spike_shape_size=10,
                       raster_face_color='black',
                       cell_spikes_color='white',
                       activity_sum_plot_color="white",
                       activity_sum_face_color="black",
                       y_lim_sum_activity=None,
                       seq_times_to_color_dict=None,
                       link_seq_categories=None,
                       link_seq_color=None, min_len_links_seq=3,
                       link_seq_line_width=1, link_seq_alpha=1,
                       jitter_links_range=1,
                       display_link_features=True,
                       seq_colors=None, debug_mode=False,
                       axes_list=None,
                       SCE_times=None,
                       ylabel=None,
                       without_activity_sum=False,
                       spike_nums_for_activity_sum=None,
                       spikes_sum_to_use=None,
                       size_fig=None,
                       cmap_name="jet", traces=None,
                       display_traces=False,
                       display_spike_nums=True,
                       traces_lw=0.3,
                       path_results=None,
                       without_time_str_in_file_name=False,
                       desaturate_color_according_to_normalized_amplitude=False,
                       lines_to_display=None,
                       lines_color="white",
                       lines_width=1,
                       lines_band=0,
                       lines_band_color="white",
                       use_brewer_colors_for_traces=False,
                       dpi=100
                       ):
    """
    Plot or save a raster given a 2d array either binary representing onsets, peaks or rising time, or made of float
    to represents traces or encoding in onset/peaks/rising time a value.
    :param spike_nums: np.array of 2D, axis=1 (lines) represents the cells, the columns representing the spikes
    It could be binary, or containing the amplitude, if amplitudes values should be display put plot_with_amplitude
    to True
    :param spike_train_format: if True, means the data is a list of np.array, and then spike_nums[i][j] is
    a timestamps value as float
    :param title: title to be plot
    :param file_name: name of the file if save_raster is True
    :param save_raster: if True, the plot will be save. To do so param should not be None and contain a variable
    path_results that will indicated where to save file_name
    :param show_raster: if True, the plot will be shown
    :param plot_with_amplitude: to display a color bar representing the content values.
    :param activity_threshold: Int representing a threshold that will be display as a red line on the sum of activity
    subplot.
    :param save_formats: string or list of string representing the formats in which saving the raster.
    Exemple: "pdf" or ["pdf", "png"]
    :param span_area_coords: List of list of tuples of two float representing coords (x, x) of span band with a color
    corresponding to the one in span_area_colors
    :param span_area_colors: list of colors, same len as span_area_coords
    :param span_area_only_on_raster: if True, means the span won't be on the sum of activity on the sub-plot as well
    :param cells_to_highlight: cells index to span (y-axis) with special spikes color, list of int
    :param cells_to_highlight_colors: cells colors to span, same len as cells_to_span, list of string
    :param color_peaks_activity: if True, will span to the color of cells_to_highlight_colors each time at which a cell
    among cells_to_highlight will spike on the activity peak diagram
    :param horizontal_lines: list of float, representing the y coord at which trace horizontal lines
    :param horizontal_lines_colors: if horizontal_lines is not None, will set the colors of each line,
    list of string or color code
    :param horizontal_lines_style: give the style of the lines, string
    :param vertical_lines: list of float, representing the x coord at which trace vertical lines
    :param vertical__lines_colors: if horizontal_lines is not None, will set the colors of each line,
    list of string or color code
    :param vertical__lines_style: give the style of the lines, string
    :param vertical_lines_linewidth: linewidth of vertical_lines
    :param raster_face_color: the background color of the raster
    :param cell_spikes_color: the color of the spikes of the raster
    :param spike_shape: shape of the spike, "|", "*", "o"
    :param spike_shape_size: use for shape != of "|"
    :param seq_times_to_color_dict: None or a dict with as the key a tuple of int representing the cell index,
    and as a value a list of set, each set composed of int representing the times value at which the cell spike should
    be colored. It will be colored if there is indeed a spike at that time otherwise, the default color will be used.
    :param seq_colors: A dict, with key a tuple represening the indices of the seq and as value of colors,
    a color, should have the same keys as seq_times_to_color_dict
    :param link_seq_color: if not None, give the color with which link the spikes from a sequence. If not None,
    seq_colors will be ignored. could be a dict with key same tuple as seq_times_to_color_dict or a string and then
    we use the same color for all seq
    :param min_len_links_seq: minimum len of a seq for the links to be drawn
    :param axes_list if not None, give a list of axes that will be used, and be filled, but no figure will be created
    or saved then. Doesn't work yet is show_amplitude is True
    :param SCE_times:  a list of tuple corresponding to the first and last index of each SCE,
    (last index being included in the SCE). Will display the position of the SCE and their number above the activity
    diagram. If None, the overall time will be displayed. Need to be adapted to the format spike_numw or
    spike_train. Equivalent to span_are_coords
    :param without_activity_sum: if True, don't plot the sum of activity diagram, valid only if axes_list is not None
    :param spike_nums_for_activity_sum: if different that the one given for the raster, should be the
    same second dimension
    :param spikes_sum_to_use: an array of 1D, that will be use to display the sum of activity,
    :param size_fig: tuple of int
    :param cmap_name: "jet" by default, used if with_amplitude for the colormap
    :param traces if not None and display_traces is True, will display traces instead of a raster
    :param display_traces, if True display traces
    :param display_spike_nums, if False, won't display a raster using spike_nums
    :param traces_lw, default 0.3,  linewidth of the traces
    :param path_results: indicate where to save the plot, replace the param.path_results if it exists
    :param desaturate_color_according_to_normalized_amplitude: if True, spike_nums should be filled with float between
    0 and 1, representing the amplitude of the spike. And if a color is given for a cell, then it will be desaturate
    according to this value
    :param lines_to_display, dict that takes for a key a tuple of int representing 2 cells, and as value a list of tuple of 2 float
    representing the 2 extremities of a line between those 2 cells. By defualt, no lines
    :param lines_color="white": colors of lines_to_display
    :param lines_width=1: width of lines_to_display
    :param lines_band=0: if > 0, display a band around the line with transparency
    :param lines_band_color="white"
    :return:
    """

    # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
    # + 11 diverting
    brewer_colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
                     '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
                     '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
                     '#74add1', '#4575b4', '#313695']
    brewer_colors = brewer_colors[::-1]

    if (spike_nums is None) and (traces is None):
        return

    if display_traces:
        if traces is None:
            return

    if spike_nums_for_activity_sum is None:
        spike_nums_for_activity_sum = spike_nums

    if plot_with_amplitude and spike_train_format:
        # not possible for now
        return

    if spike_nums is None:
        n_cells = len(traces)
    else:
        n_cells = len(spike_nums)

    if axes_list is None:
        if size_fig is None:
            size_fig = (15, 8)
        if not plot_with_amplitude:
            if without_activity_sum:
                fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=False,
                                        figsize=size_fig, dpi=dpi)
            else:
                sharex = True  # False if (SCE_times is None) else True
                fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=sharex,
                                               gridspec_kw={'height_ratios': [10, 2]},
                                               figsize=size_fig, dpi=dpi)
            fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
        else:
            fig = plt.figure(figsize=size_fig, dpi=dpi)
            fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'h_pad': 1})
            outer = gridspec.GridSpec(1, 2, width_ratios=[100, 1])  # , wspace=0.2, hspace=0.2)
        fig.patch.set_facecolor(figure_background_color)
    else:
        if without_activity_sum:
            ax1 = axes_list[0]
        else:
            ax1, ax2 = axes_list

    if plot_with_amplitude:
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                 subplot_spec=outer[0], height_ratios=[10, 2])
        # inner.tight_layout(fig, pad=0.1)
        ax1 = fig.add_subplot(inner[0])  # plt.Subplot(fig, inner[0])
        min_value = np.min(spike_nums)
        max_value = np.max(spike_nums)
        step_color_value = 0.1
        colors = np.r_[np.arange(min_value, max_value, step_color_value)]
        mymap = plt.get_cmap("jet")
        # get the colors from the color map
        my_colors = mymap(colors)

        # colors = plt.cm.hsv(y / float(max(y)))
        scalar_map = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=min_value, vmax=max_value))
        # # fake up the array of the scalar mappable. Urgh…
        scalar_map._A = []

    # -------- end plot with amplitude ---------

    ax1.set_facecolor(raster_face_color)

    min_time = 0
    max_time = 0

    max_n_color = 10
    if display_traces:
        n_times = len(traces[0, :])
        zorder_traces = 21 + len(traces)
        for cell, trace in enumerate(traces):
            if use_brewer_colors_for_traces:
                color = brewer_colors[cell % len(brewer_colors)]
            else:
                color = cm.nipy_spectral(((cell % max_n_color) + 1) / (max_n_color + 1))
            ax1.plot(np.arange(n_times), trace + cell, lw=traces_lw, color=color, zorder=zorder_traces)
            if scatters_on_traces is not None:
                times_to_scatter = np.where(scatters_on_traces[cell, :])[0]
                ax1.scatter(times_to_scatter, trace[times_to_scatter] + cell,
                            color="white",
                            marker=scatters_on_traces_marker,
                            s=scatters_on_traces_size, zorder=zorder_traces - 1)
            zorder_traces -= 1
            line_beg_x = -1
            line_end_x = n_times + 1
            ax1.hlines(cell, line_beg_x, line_end_x, lw=0.1, linestyles="dashed", color=color, zorder=5)
    if cells_to_highlight is not None:
        cells_to_highlight = np.array(cells_to_highlight)
    if display_spike_nums:
        for cell, spikes in enumerate(spike_nums):
            if spike_train_format:
                if cell == 0:
                    min_time = np.min(spikes)
                else:
                    min_time = int(np.min((min_time, np.min(spikes))))
                max_time = int(np.ceil(np.max((max_time, np.max(spikes)))))
            # print(f"Neuron {y}, total spikes {len(np.where(neuron)[0])}, "
            #       f"nb > 2: {len(np.where(neuron>2)[0])}, nb < 2: {len(np.where(neuron[neuron<2])[0])}")
            if display_traces:
                # same color as traces
                color_neuron = cm.nipy_spectral(((cell % max_n_color) + 1) / (max_n_color + 1))
            else:
                color_neuron = cell_spikes_color
            if cells_to_highlight is not None:
                if cell in cells_to_highlight:
                    index = np.where(cells_to_highlight == cell)[0][0]
                    color_neuron = cells_to_highlight_colors[index]
            if spike_train_format:
                neuron_times = spikes
            else:
                neuron_times = np.where(spikes > 0)[0]
            if spike_shape != "|":
                if plot_with_amplitude:
                    ax1.scatter(neuron_times, np.repeat(cell, len(neuron_times)),
                                color=scalar_map.to_rgba(spikes[spikes > 0]),
                                marker=spike_shape,
                                s=spike_shape_size, zorder=20)
                elif desaturate_color_according_to_normalized_amplitude:
                    n_spikes = len(neuron_times)
                    colors_list = [sns.desaturate(x, p) for x, p in
                                   zip([color_neuron] * n_spikes, spikes[neuron_times])]
                    ax1.scatter(neuron_times, np.repeat(cell, len(neuron_times)),
                                color=colors_list,
                                marker=spike_shape,
                                s=spike_shape_size, zorder=20)
                else:
                    if display_traces:
                        y_values = traces[cell, neuron_times] + cell
                    else:
                        y_values = np.repeat(cell, len(neuron_times))
                    ax1.scatter(neuron_times, y_values, color=color_neuron, marker=spike_shape,
                                s=spike_shape_size, zorder=20)
            else:
                if desaturate_color_according_to_normalized_amplitude:
                    n_spikes = len(neuron_times)
                    colors_list = [sns.desaturate(x, p) for x, p in
                                   zip([color_neuron] * n_spikes, spikes[neuron_times])]
                    ax1.vlines(neuron_times, cell - .5, cell + .5, color=colors_list,
                               linewidth=spike_shape_size, zorder=20)
                elif plot_with_amplitude:
                    ax1.vlines(neuron_times, cell - .5, cell + .5, color=scalar_map.to_rgba(spikes[spikes > 0]),
                               linewidth=spike_shape_size, zorder=20)
                else:
                    ax1.vlines(neuron_times, cell - .5, cell + .5, color=color_neuron, linewidth=spike_shape_size,
                               zorder=20)
    if lines_to_display is not None:
        """
        lines_to_display=None,
                       lines_color="white",
                       lines_width=1,
                       lines_band=0,
                       lines_band_color="white"
                       dict that takes for a key a tuple of int representing 2 cells, and as value a list of tuple of 2 float
    representing the 2 extremities of a line between those 2 cells. By defualt, no lines
        """
        for cells_tuple, spike_times_list in lines_to_display.items():
            for spike_times in spike_times_list:
                ax1.plot(list(spike_times), list(cells_tuple),
                         color=lines_color,
                         linewidth=lines_width, zorder=30, alpha=1)
                if lines_band > 0:
                    xy = np.zeros((4, 2))
                    xy[0, 0] = spike_times[0] - lines_band
                    xy[0, 1] = cells_tuple[0]
                    xy[1, 0] = spike_times[1] - lines_band
                    xy[1, 1] = cells_tuple[1]
                    xy[2, 0] = spike_times[1] + lines_band
                    xy[2, 1] = cells_tuple[1]
                    xy[3, 0] = spike_times[0] + lines_band
                    xy[3, 1] = cells_tuple[0]
                    band_patch = patches.Polygon(xy=xy,
                                                 fill=True,
                                                 # linewidth=line_width,
                                                 facecolor=lines_band_color,
                                                 # edgecolor=edge_color,
                                                 alpha=0.4,
                                                 zorder=10)  # lw=2
                    ax1.add_patch(band_patch)
                    # ax1.fill_between(list(spike_times), np.array(cells_tuple)-lines_band,
                    #                  np.array(cells_tuple)+lines_band, facecolor=lines_band_color, alpha=0.4,
                    #                  zorder=10)
    if seq_times_to_color_dict is not None:
        seq_count = 0
        links_labels = []
        links_labels_color = []
        links_labels_y_coord = []
        if jitter_links_range > 0:
            nb_jitters = 10
            indices_rand_x = np.linspace(-jitter_links_range, jitter_links_range, nb_jitters)
            np.random.shuffle(indices_rand_x)
        for seq_indices, seq_times_list in seq_times_to_color_dict.items():
            nb_seq_times = 0
            for times_list_index, times_list in enumerate(seq_times_list):
                x_coord_to_link = []
                y_coord_to_link = []
                for time_index, t in enumerate(times_list):
                    cell_index = seq_indices[time_index]
                    # in case of this seq of cell would be used in a zoom version of the raster
                    if cell_index >= n_cells:
                        continue
                    # first we make sure the cell does spike at the given time
                    if spike_train_format:
                        if t not in spike_nums[cell_index]:
                            continue
                    else:
                        pass
                        if spike_nums[cell_index, t] == 0:
                            cell_for_msg = cell_index
                            if y_ticks_labels is not None:
                                cell_for_msg = y_ticks_labels[cell_index]
                            print(f"Not there: seq {times_list_index} cell {cell_for_msg} - {cell_index}, "
                                  f"time {t}")
                            continue
                        # print(f"## There: seq {times_list_index} cell {cell_index}, time {t}")
                    if link_seq_color is not None:
                        x_coord_to_link.append(t)
                        y_coord_to_link.append(cell_index)
                    else:
                        # if so, we draw the spike
                        if spike_shape != "|":
                            ax1.scatter(t, cell_index, color=seq_colors[seq_indices],
                                        marker=spike_shape,
                                        s=spike_shape_size, zorder=20)
                        else:
                            ax1.vlines(t, cell_index - .5, cell_index + .5, color=seq_colors[seq_indices],
                                       linewidth=1, zorder=20)
                if (link_seq_color is not None) and (len(x_coord_to_link) >= min_len_links_seq):
                    if isinstance(link_seq_color, str):
                        color_to_use = link_seq_color
                    elif isinstance(link_seq_color, dict):
                        color_to_use = link_seq_color[seq_indices]
                    else:
                        color_to_use = link_seq_color[seq_count % len(link_seq_color)]
                    x_coord_to_link = np.array(x_coord_to_link)
                    if jitter_links_range > 0:
                        jitter_to_add = indices_rand_x[seq_count % nb_jitters]
                    else:
                        jitter_to_add = 0
                    ax1.plot(x_coord_to_link + jitter_to_add, y_coord_to_link,
                             color=color_to_use,
                             linewidth=link_seq_line_width, zorder=30, alpha=link_seq_alpha)
                    nb_seq_times += 1
            if nb_seq_times > 0:
                category = ""
                if link_seq_categories is not None:
                    category = "*" * link_seq_categories[seq_indices]
                links_labels.append(f"l{len(seq_indices)}, r{nb_seq_times} {category}")
                links_labels_color.append(color_to_use)
                links_labels_y_coord.append((seq_indices[0] + seq_indices[-1]) / 2)
            seq_count += 1

    if display_traces:
        ax1.set_ylim(-0.5, n_cells + 4)
    else:
        ax1.set_ylim(-0.5, n_cells)
    if y_ticks_labels is not None:
        ax1.set_yticks(np.arange(n_cells))
        ax1.set_yticklabels(y_ticks_labels)
    if (x_ticks_labels is not None) and (x_ticks is not None):
        ax1.set_xticks(x_ticks)
        ax1.tick_params('x', length=2, width=0.5, which='both')
        ax1.set_xticklabels(x_ticks_labels, rotation=45)  # ha="right", va="center
    if x_ticks_labels_size is not None:
        ax1.xaxis.set_tick_params(labelsize=x_ticks_labels_size)
    if y_ticks_labels_size is not None:
        ax1.yaxis.set_tick_params(labelsize=y_ticks_labels_size)
    else:
        if n_cells < 50:
            y_ticks_labels_size = 5
        elif n_cells < 100:
            y_ticks_labels_size = 4
        elif n_cells < 200:
            y_ticks_labels_size = 3
        elif n_cells < 400:
            y_ticks_labels_size = 1
        else:
            y_ticks_labels_size = 0.1
        ax1.yaxis.set_tick_params(labelsize=y_ticks_labels_size)
    if without_ticks:
        if x_ticks is not None:
            ax1.tick_params(axis='y', which='both', length=0)
        else:
            ax1.tick_params(axis='both', which='both', length=0)

    if seq_times_to_color_dict is not None:
        if link_seq_color is not None:
            ax_right = ax1.twinx()
            ax_right.set_frame_on(False)
            ax_right.set_ylim(-1, n_cells)
            ax_right.set_yticks(links_labels_y_coord)
            # clusters labels
            ax_right.set_yticklabels(links_labels)
            ax_right.yaxis.set_ticks_position('none')
            if y_ticks_labels_size > 1:
                y_ticks_labels_size -= 1
            else:
                y_ticks_labels_size -= 0.5
            ax_right.yaxis.set_tick_params(labelsize=y_ticks_labels_size)
            # ax_right.yaxis.set_tick_params(labelsize=2)
            for index in np.arange(len(links_labels)):
                ax_right.get_yticklabels()[index].set_color(links_labels_color[index])

    if spike_train_format:
        n_times = int(math.ceil(max_time - min_time))
    else:
        if spike_nums is None:
            n_times = traces.shape[1]
        else:
            n_times = len(spike_nums[0, :])

    # draw span to highlight some periods
    if span_area_coords is not None:
        if len(span_area_coords) != len(span_area_colors):
            raise Exception("span_area_coords and span_area_colors are not the same size")
        for index, span_area_coord in enumerate(span_area_coords):
            for coord in span_area_coord:
                if span_area_colors is not None:
                    color = span_area_colors[index]
                else:
                    color = "lightgrey"
                ax1.axvspan(coord[0], coord[1], alpha=alpha_span_area, facecolor=color, zorder=1)

    if (span_cells_to_highlight is not None):
        for index, cell_to_span in enumerate(span_cells_to_highlight):
            ax1.axhspan(cell_to_span - 0.5, cell_to_span + 0.5, alpha=0.4,
                        facecolor=span_cells_to_highlight_colors[index])

    if horizontal_lines is not None:
        line_beg_x = 0
        line_end_x = 0
        if spike_train_format:
            line_beg_x = min_time - 1
            line_end_x = max_time + 1
        else:
            line_beg_x = -1
            line_end_x = n_times + 1
        if horizontal_lines_linewidth is None:
            ax1.hlines(horizontal_lines, line_beg_x, line_end_x, color=horizontal_lines_colors, linewidth=2,
                       linestyles=horizontal_lines_sytle)
        else:
            ax1.hlines(horizontal_lines, line_beg_x, line_end_x, color=horizontal_lines_colors,
                       linewidth=horizontal_lines_linewidth,
                       linestyles=horizontal_lines_sytle)

    if vertical_lines is not None:
        line_beg_y = 0
        line_end_y = n_cells - 1
        ax1.vlines(vertical_lines, line_beg_y, line_end_y, color=vertical_lines_colors,
                   linewidth=vertical_lines_linewidth,
                   linestyles=vertical_lines_sytle)

    if spike_train_format:
        ax1.set_xlim(min_time - 1, max_time + 1)
    else:
        ax1.set_xlim(-1, n_times + 1)
    # ax1.margins(x=0, tight=True)

    if not without_activity_sum or hide_x_labels:
        ax1.get_xaxis().set_visible(False)

    if title is not None:
        ax1.set_title(title)
    # Give x axis label for the spike raster plot
    # ax.xlabel('Frames')
    # Give y axis label for the spike raster plot
    if ylabel is not None:
        ax1.set_ylabel(ylabel)

    # ax1.spines['left'].set_color(y_ticks_labels_color)
    # ax1.spines['bottom'].set_color(x_ticks_labels_color)
    # ax1.yaxis.label.set_color(y_ticks_labels_color)
    if isinstance(y_ticks_labels_color, list):
        for xtick, color in zip(ax1.get_yticklabels(), y_ticks_labels_color):
            xtick.set_color(color)
    else:
        ax1.tick_params(axis='y', colors=y_ticks_labels_color)
    ax1.tick_params(axis='x', colors=x_ticks_labels_color)

    if (axes_list is not None) and without_activity_sum:
        return

    if not without_activity_sum:
        # ################################################################################################
        # ################################ Activity sum plot part ################################
        # ################################################################################################
        if (sliding_window_duration >= 1) and (spikes_sum_to_use is None):
            sum_spikes = np.zeros(n_times)
            if spike_train_format:
                windows_sum = np.zeros((n_cells, n_times), dtype="int16")
                # one cell can participate to max one spike by window
                # if value is True, it means this cell has already been counted
                cell_window_participation = np.zeros((n_cells, n_times), dtype="bool")
                for cell, spikes_train in enumerate(spike_nums_for_activity_sum):
                    for spike_time in spikes_train:
                        # first determining to which windows to add the spike
                        spike_index = int(spike_time - min_time)
                        first_index_window = np.max((0, spike_index - sliding_window_duration))
                        if np.sum(cell_window_participation[cell, first_index_window:spike_index]) == 0:
                            windows_sum[cell, first_index_window:spike_index] += 1
                            cell_window_participation[cell, first_index_window:spike_index] = True
                        else:
                            for t in np.arange(first_index_window, spike_index):
                                if cell_window_participation[cell, t] is False:
                                    windows_sum[cell, t] += 1
                                    cell_window_participation[cell, t] = True
                sum_spikes = np.sum(windows_sum, axis=0)
                if debug_mode:
                    print("sliding window over")
                # for index, t in enumerate(np.arange(int(min_time), int((np.ceil(max_time) - sliding_window_duration)))):
                #     # counting how many cell fire during that window
                #     if (index % 1000) == 0:
                #         print(f"index {index}")
                #     sum_value = 0
                #     t_min = t
                #     t_max = t + sliding_window_duration
                #     for spikes_train in spike_nums:
                #         # give the indexes
                #         # np.where(np.logical_and(spikes_train >= t, spikes_train < t_max))
                #         spikes = spikes_train[np.logical_and(spikes_train >= t, spikes_train < t_max)]
                #         nb_spikes = len(spikes)
                #         if nb_spikes > 0:
                #             sum_value += 1
                #     sum_spikes[index] = sum_value
                # sum_spikes[(n_times - sliding_window_duration):] = sum_value
            else:
                for t in np.arange(0, (n_times - sliding_window_duration)):
                    # One spike by cell max in the sum process
                    sum_value = np.sum(spike_nums_for_activity_sum[:, t:(t + sliding_window_duration)], axis=1)
                    sum_spikes[t] = len(np.where(sum_value)[0])
                sum_spikes[(n_times - sliding_window_duration):] = len(np.where(sum_value)[0])
        elif spikes_sum_to_use is None:
            if spike_train_format:
                pass
            else:
                binary_spikes = np.zeros((n_cells, n_times), dtype="int8")
                for spikes, spikes in enumerate(spike_nums_for_activity_sum):
                    binary_spikes[spikes, spikes > 0] = 1
                # if (param is not None) and (param.bin_size > 1):
                #     sum_spikes = np.mean(np.split(np.sum(binary_spikes, axis=0), n_times // param.bin_size), axis=1)
                #     sum_spikes = np.repeat(sum_spikes, param.bin_size)
                # else:
                sum_spikes = np.sum(binary_spikes, axis=0)
        else:
            sum_spikes = spikes_sum_to_use

        if spike_train_format:
            x_value = np.arange(min_time, max_time)
        else:
            x_value = np.arange(n_times)

        if plot_with_amplitude:
            ax2 = fig.add_subplot(inner[1], sharex=ax1)

        ax2.set_facecolor(activity_sum_face_color)

        # sp = UnivariateSpline(x_value, sum_spikes, s=240)
        # ax2.fill_between(x_value, 0, smooth_curve(sum_spikes), facecolor="black") # smooth_curve(sum_spikes)
        if show_sum_spikes_as_percentage:
            if debug_mode:
                print("using percentages")
            sum_spikes = sum_spikes / n_cells
            sum_spikes *= 100
            if activity_threshold is not None:
                activity_threshold = activity_threshold / n_cells
                activity_threshold *= 100

        ax2.fill_between(x_value, 0, sum_spikes, facecolor=activity_sum_plot_color, zorder=10)
        if activity_threshold is not None:
            line_beg_x = 0
            line_end_x = 0
            if spike_train_format:
                line_beg_x = min_time - 1
                line_end_x = max_time + 1
            else:
                line_beg_x = -1
                line_end_x = len(spike_nums_for_activity_sum[0, :]) + 1
            ax2.hlines(activity_threshold, line_beg_x, line_end_x, color="red", linewidth=1, linestyles="dashed")

        # draw span to highlight some periods
        if (span_area_coords is not None) and (not span_area_only_on_raster):
            for index, span_area_coord in enumerate(span_area_coords):
                for coord in span_area_coord:
                    if span_area_colors is not None:
                        color = span_area_colors[index]
                    else:
                        color = "lightgrey"
                    ax2.axvspan(coord[0], coord[1], alpha=0.5, facecolor=color, zorder=1)

        # early born
        if (cells_to_highlight is not None) and color_peaks_activity:
            for index, cell_to_span in enumerate(cells_to_highlight):
                ax2.vlines(np.where(spike_nums_for_activity_sum[cell_to_span, :])[0], 0, np.max(sum_spikes),
                           color=cells_to_highlight_colors[index],
                           linewidth=2, linestyles="dashed", alpha=0.2)

        # ax2.yaxis.set_visible(False)
        ax2.set_frame_on(False)
        ax2.get_xaxis().set_visible(True)
        if y_lim_sum_activity is not None:
            ax2.set_ylim(y_lim_sum_activity[0], y_lim_sum_activity[1])
        else:
            ax2.set_ylim(0, np.max(sum_spikes))
        if spike_train_format:
            ax2.set_xlim(min_time - 1, max_time + 1)
        else:
            if spike_nums_for_activity_sum is not None:
                ax2.set_xlim(-1, len(spike_nums_for_activity_sum[0, :]) + 1)
            else:
                ax2.set_xlim(-1, len(spikes_sum_to_use) + 1)

        if SCE_times is not None:
            ax_top = ax2.twiny()
            ax_top.set_frame_on(False)
            if spike_train_format:
                ax_top.set_xlim(min_time - 1, max_time + 1)
            else:
                if spike_nums_for_activity_sum is not None:
                    ax_top.set_xlim(-1, len(spike_nums_for_activity_sum[0, :]) + 1)
                else:
                    ax_top.set_xlim(-1, len(spikes_sum_to_use) + 1)
            xticks_pos = []
            for times_tuple in SCE_times:
                xticks_pos.append(times_tuple[0])
            ax_top.set_xticks(xticks_pos)
            ax_top.xaxis.set_ticks_position('none')
            ax_top.set_xticklabels(np.arange(len(SCE_times)))
            ax_top.tick_params(axis='x', colors=x_ticks_labels_color)
            plt.setp(ax_top.xaxis.get_majorticklabels(), rotation=90)
            if len(SCE_times) > 30:
                ax_top.xaxis.set_tick_params(labelsize=3)
            elif len(SCE_times) > 50:
                ax_top.xaxis.set_tick_params(labelsize=2)
            elif len(SCE_times) > 100:
                ax_top.xaxis.set_tick_params(labelsize=1)
            elif len(SCE_times) > 300:
                ax_top.xaxis.set_tick_params(labelsize=0.5)
            else:
                ax_top.xaxis.set_tick_params(labelsize=4)
        # print(f"max sum_spikes {np.max(sum_spikes)}, mean  {np.mean(sum_spikes)}, median {np.median(sum_spikes)}")

        if without_ticks:
            ax2.tick_params(axis='both', which='both', length=0)
        # ax2.yaxis.label.set_color(y_ticks_labels_color)
        if isinstance(y_ticks_labels_color, list):
            for xtick, color in zip(ax2.get_yticklabels(), y_ticks_labels_color):
                xtick.set_color(color)
        else:
            ax2.tick_params(axis='y', colors=y_ticks_labels_color)
        ax2.tick_params(axis='x', colors=x_ticks_labels_color)

        if (x_ticks_labels is not None) and (x_ticks is not None):
            ax2.set_xticks(x_ticks)
            ax2.tick_params('x', length=2, width=0.5, which='both')
            ax2.set_xticklabels(x_ticks_labels, rotation=45)  # ha="right", va="center
        if x_ticks_labels_size is not None:
            ax2.xaxis.set_tick_params(labelsize=x_ticks_labels_size)

    # color bar section
    if plot_with_amplitude:
        inner_2 = gridspec.GridSpecFromSubplotSpec(1, 1,
                                                   subplot_spec=outer[1])  # , wspace=0.1, hspace=0.1)
        ax3 = fig.add_subplot(inner_2[0])  # plt.Subplot(fig, inner_2[0])
        cb = fig.colorbar(scalar_map, cax=ax3)
        cb.ax.tick_params(axis='y', colors="white")
    if axes_list is None:
        if save_raster and (path_results is not None):
            # transforming a string in a list
            if isinstance(save_formats, str):
                save_formats = [save_formats]
            if time_str is None:
                time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
            for save_format in save_formats:
                if without_time_str_in_file_name:
                    fig.savefig(os.path.join(f'{path_results}', f'{file_name}.{save_format}'),
                                format=f"{save_format}",
                                facecolor=fig.get_facecolor())
                else:
                    fig.savefig(os.path.join(f'{path_results}', f'{file_name}_{time_str}.{save_format}'),
                                format=f"{save_format}",
                                facecolor=fig.get_facecolor())
        # Display the spike raster plot
        if show_raster:
            plt.show()
        plt.close()


def _angle_to_point(point, centre):
    '''calculate angle in 2-D between points and x axis'''
    delta = point - centre
    res = np.arctan(delta[1] / delta[0])
    if delta[0] < 0:
        res += np.pi
    return res


def area_of_triangle(p1, p2, p3):
    '''calculate area of any triangle given co-ordinates of the corners'''
    return np.linalg.norm(np.cross((p2 - p1), (p3 - p1))) / 2.


def convex_hull(points, smidgen=0.0075):
    '''
    from: https://stackoverflow.com/questions/17553035/draw-a-smooth-polygon-around-data-points-in-a-scatter-plot-in-matplotlib
    Calculate subset of points that make a convex hull around points
    Recursively eliminates points that lie inside two neighbouring points until only convex hull is remaining.

    :Parameters:
    points : ndarray (2 x m)
    array of points for which to find hull
    use pylab to show progress?
    smidgen : float
    offset for graphic number labels - useful values depend on your data range

    :Returns:
    hull_points : ndarray (2 x n)
    convex hull surrounding points
    '''

    n_pts = points.shape[1]
    # assert(n_pts > 5)
    centre = points.mean(1)

    angles = np.apply_along_axis(_angle_to_point, 0, points, centre)
    pts_ord = points[:, angles.argsort()]

    pts = [x[0] for x in zip(pts_ord.transpose())]
    prev_pts = len(pts) + 1
    k = 0
    while prev_pts > n_pts:
        prev_pts = n_pts
        n_pts = len(pts)
        i = -2
        while i < (n_pts - 2):
            Aij = area_of_triangle(centre, pts[i], pts[(i + 1) % n_pts])
            Ajk = area_of_triangle(centre, pts[(i + 1) % n_pts], pts[(i + 2) % n_pts])
            Aik = area_of_triangle(centre, pts[i], pts[(i + 2) % n_pts])
            if Aij + Ajk < Aik:
                del pts[i + 1]
            i += 1
            n_pts = len(pts)
        k += 1
    return np.asarray(pts)
