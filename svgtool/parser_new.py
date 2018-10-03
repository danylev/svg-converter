import re

import numpy as np
from matplotlib import pyplot as plt

from xml.etree import ElementTree as ET

from collections import deque
from svg.path import Line, Arc, QuadraticBezier, CubicBezier, parse_path
from svg.path.path import Move

from math import pi, cos, sin
from copy import deepcopy

MOVETO_MARKERS = ('m', 'M')
CLOSEPATH_MARKERS = ('z', 'Z')
LINETO_MARKERS = ('l', 'L',)
HORIZONTAL_MARKERS = ('h', 'H',)
VERTICAL_MARKERS = ('v', 'V')
CUBIC_MARKERS = ('c', 'C')
QUADRATIC_MARKERS = ('q', 'Q')
SMOOTH_C_MARKERS = ('s', 'S')
SMOOTH_Q_MARKERS = ('t', 'T')
ELLIPTIC_MARKER = ('a', 'A')

MARKERS = (
        MOVETO_MARKERS +
        CLOSEPATH_MARKERS +
        LINETO_MARKERS +
        CUBIC_MARKERS +
        ELLIPTIC_MARKER +
        HORIZONTAL_MARKERS +
        VERTICAL_MARKERS +
        SMOOTH_C_MARKERS +
        SMOOTH_Q_MARKERS +
        QUADRATIC_MARKERS
)


def to_absolute(current_point, *args):
    return list(map(lambda x: np.add(current_point, x), args))


def centroid(matrix):
    import ipdb
    ipdb.set_trace()
    length = len(matrix)
    summed = matrix.sum(axis=0)
    return summed / length


class SimpleElement:

    def __init__(self, path):
        self.root_path = path
        self.num_of_paths = self.get_paths
        self.subpath_list = self.get_subpaths
        self.curve_list = self.get_curves
        self.path = parse_path(path)
        self.points = []

    def __str__(self):
        return self.root_path

    def __len__(self):
        return self.num_of_paths


    @property
    def get_paths(self):
        subpaths = 0
        for marker in MOVETO_MARKERS:
            subpaths += self.root_path.count(marker)
        return subpaths

    @property
    def get_subpaths(self):
        path_start = 0
        paths = []
        for index, element in enumerate(self.root_path[1:]):
            if element in MOVETO_MARKERS:
                paths.append(self.root_path[path_start:index].strip())
                path_start = index
        paths.append(self.root_path[path_start:len(self.root_path)].strip())
        return paths

    @property
    def get_curves(self):
        curves = deque()
        for path in self.subpath_list:
            re_pattern = ''.join(MARKERS)
            command_list = zip(re.findall(f"[{re_pattern}]", path), re.split(f"[{re_pattern}]", path)[1:])
            curves.append(list(command_list))
        return curves

    def parse_element(self):
        for primitive in self.path:
            method = self.get_parser(type(primitive))
            method(primitive)

    def get_parser(self, element_type):
        parsers = {
            Move: 'move_parser',
            Line: 'line_parser',
            Arc: 'curve_parser',
            CubicBezier: 'curve_parser',
            QuadraticBezier: 'curve_parser',
        }
        return self.__getattribute__(parsers.get(element_type))

    def move_parser(self, primitive, storage=None):
        if storage is None:
            storage = self.points
        storage.append(np.array([primitive.start.real, primitive.start.imag]))

    def line_parser(self, primitive, storage=None):
        if storage is None:
            storage = self.points
        storage.append(np.array([primitive.end.real, primitive.end.imag]))

    def curve_parser(self, primitive, storage=None, splits=15):
        if storage is None:
            storage = self.points
        for t in (x * (1/splits) for x in range(1, splits)):
            point = primitive.point(t)
            storage.append(np.array([point.real, point.imag]))

    def normalize(self):
        self.points = list(map(lambda x: np.multiply(x, np.array([1, -1])), self.points))

    def scale_xy(self, x_scale=1, y_scale=1):
        self.points = list(map(lambda x: np.multiply(x, np.array([x_scale, y_scale])), self.points))


class SvgParser:
    ELEMENT_MARKER = '{http://www.w3.org/2000/svg}path'
    PATH_MARKER = 'd'
    BASE_CTM = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def __init__(self, file, width=None, height=None, scale_x=None, scale_y=None, rotate=None, radians=False, svg=None):
        self.tree = ET.parse(file)
        self.root = self.tree.getroot()
        self.coords = list(map(float, self.root.attrib.get('viewBox').split()))
        self.width = self.coords[2] - self.coords[0]
        self.height = self.coords[3] - self.coords[1]
        self.measurement = self.root.attrib.get('height')[-2:]
        self.element_list = []

        # Transform arguments
        if width and height:
            self.field = (width, height)
        self.scale_x = scale_x
        self.scale_y = scale_y
        if not radians:
            self.rotate = (rotate * pi) / 180
        else:
            self.rotate = rotate

    def __str__(self):
        return self.tree

    @property
    def num_of_paths(self):
        if not self.element_list:
            self.parse()
        return sum(map(len, self.element_list))

    @property
    def paths(self):
        path_container = [item.get(self.PATH_MARKER) for item in self.root.iter(self.ELEMENT_MARKER)]
        return path_container

    def parse(self):
        for element in self.paths:
            figure = SimpleElement(element)
            self.element_list.append(figure)
        for element in self.element_list:
            element.parse_element()
            element.normalize()
            if self.scale_x or self.scale_y:
                element.scale_xy(x_scale=(self.scale_x or 1), y_scale=(self.scale_y or 1))
        if self.rotate:
            # Reshape into vertical polygon, very demanding operation, especially for long text\figures
            merged = np.concatenate([element.points for element in self.element_list])
            polygon = np.array([np.reshape(x, (2, 1)) for x in merged])
            cent = np.reshape(centroid(polygon), (2, 1))
            lenght = len(merged)
            import ipdb
            ipdb.set_trace()
            tile = np.tile(cent, (lenght//2, 1))
            rotation_matrix = np.array([[cos(self.rotate), -sin(self.rotate)],
                                        [sin(self.rotate), cos(self.rotate)]])
            new_points = rotation_matrix * (polygon - tile) + tile
            print('lul')


if __name__ == '__main__':
    some_svg = SvgParser('../example (1).svg', scale_x=5, rotate=90)
    some_svg.parse()
    for figure in some_svg.element_list:
        points = figure.points
        verticals = [np.reshape(x, (2, 1)) for x in points]
        polygon = np.concatenate(list(verticals))
        plt.scatter(*(zip(*points)))
    import ipdb
    ipdb.set_trace()
    # plt.axis([65, 95, 89, 100])
    plt.show()
    print('Finish')
