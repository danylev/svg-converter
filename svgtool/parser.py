import re

import numpy as np
from matplotlib import pyplot as plt

from xml.etree import ElementTree as ET

from collections import deque
from svg.path import Line, Arc, QuadraticBezier, CubicBezier, Path, parse_path
from svg.path.path import Move

from math import pi, cos, sin
from copy import deepcopy
from scipy import dot

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
        self.to_print = None

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

    def __init__(self, file, out=None, viewbox=None, width=None, height=None, plot=False,
                 scale_x=None, scale_y=None, rotate=None, radians=False):
        self.out = out
        self.plot = plot
        self.tree = ET.parse(file)
        self.root = self.tree.getroot()
        self.coords = list(map(float, self.root.attrib.get('viewBox').split()))
        self.width = self.coords[2] - self.coords[0]
        self.height = self.coords[3] - self.coords[1]
        self.viewbox = viewbox
        self.measurement = self.root.attrib.get('height')[-2:]
        self.element_list = []
        self.points = []
        # Transform arguments
        if width and height:
            self.field = (width, height)
        self.scale_x = scale_x
        self.scale_y = scale_y
        if not radians and rotate:
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

    def smart_filling(self, original_size, original_viewbox, requested_viewbox):
        new_size = np.array((abs(requested_viewbox[0] - requested_viewbox[2]),
                             abs(requested_viewbox[1] - requested_viewbox[3])))
        import ipdb
        ipdb.set_trace()
        scale = 0
        shift = new_size
        return shift, scale

    def parse(self):
        for element in self.paths:
            if sum([element.count(marker) for marker in MOVETO_MARKERS]) > 1:
                moveto_indexes = [i for i, point in enumerate(element) if point in MOVETO_MARKERS]
                moveto_indexes.append(len(element))
                start = 0
                for moveto in moveto_indexes[1:]:
                    figure = SimpleElement(element[start:moveto])
                    self.element_list.append(figure)
            else:
                figure = SimpleElement(element)
                self.element_list.append(figure)
        for element in self.element_list:
            element.parse_element()
            element.normalize()
            if self.scale_x or self.scale_y:
                element.scale_xy(x_scale=(self.scale_x or 1), y_scale=(self.scale_y or 1))

        self.points = np.concatenate([element.points for element in self.element_list])
        cent = centroid(self.points)
        tile = np.tile(cent, (len(self.points), 1))
        min, max = np.amin(self.points, axis=0), np.amax(self.points, axis=0)
        original_viewbox = (min, max)
        original_size = np.absolute((max-min))
        if self.viewbox:
            shift_matrix, scale_matrix = self.smart_filling(original_size, original_viewbox, self.viewbox)
            # dot(scale_matrix * (self.points - shift_matrix))
        import ipdb
        ipdb.set_trace()
        if self.rotate:
            # Reshape into vertical polygon, very demanding operation, especially for long text\figures
            rotation_matrix = np.array([[cos(self.rotate), -sin(self.rotate)], [sin(self.rotate), cos(self.rotate)]])
            self.points = dot((self.points-tile), rotation_matrix) + tile
        if self.plot:
            plt.scatter(*zip(*self.points))
            plt.show()
        if self.out:
            with open(self.out, 'w') as fd:
                start = 0
                number_of_elements = 0
                for element in self.element_list:
                    number_of_elements += len(element.points)
                    x_point, y_point = self.points[start]
                    fd.write(f'{{"command":"move","prm":{{"path":"line","movement":0,"speed":5000.0,"x":{x_point},"y":{y_point}}}}}\n')
                    fd.write(f'{{"command":"move","prm":{{"path":"line","movement":1,"speed":5000.0,"z":-1}}}}\n')
                    for point in self.points[start:number_of_elements]:
                        x_point, y_point = point
                        fd.write(f'{{"command":"move","prm":{{"path":"line","movement":0,"speed":5000.0,"x":{x_point},"y":{y_point}}}}}\n')
                    fd.write(f'{{"command":"move","prm":{{"path":"line","movement":1,"speed":5000.0,"z":1}}}}\n')
                    start = number_of_elements


if __name__ == '__main__':
    some_svg = SvgParser('../example (1).svg', out='out2.txt', viewbox=(13.7, -3.0, 17.7, 3), plot=True)
    some_svg.parse()
    import ipdb
    ipdb.set_trace()
