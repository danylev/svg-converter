import re

import numpy as np
from matplotlib import pyplot as plt

from xml.etree import ElementTree as ET

from collections import deque

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


def quadratice_sanitizer(points):
    print(points)
    if len(points) == 2 and sum(map(lambda x: len(x.split(',')), points)):
        return points
    else:
        return ['0,0', '0,0']


def curve_is_flat(points, accuracy=0.1):
    """
    Check if line already flat enough and we can finish iterations
    :param points: points of bezier curve
    :param accuracy: accuracy
    :return: bool - end of iterations
    """
    dx = points[-1][0] - points[0][0]
    dy = points[-1][1] - points[0][1]
    height_left = abs((points[1][0] - points[3][0]) * dy - (points[1][1] - points[3][1]) * dx)
    height_right = abs((points[2][0] - points[3][0]) * dy - (points[2][1] - points[3][1]) * dx)
    return bool((height_left + height_right) ** 2 < accuracy * (dx ** 2 + dy ** 2))


def _cubic_path_segmentation(points, current_point, storage):
    """
    Paul de Casteljau Divides and Conquers algorithm
    :param points: list of points - both control points and end point
    :param current_point: start of curve
    :param storage: storage where to append points
    :return: point segmentation for cubic bezier
    """
    left_half = (current_point + points[0]) / 2  # also first left controlpoint
    middle = (points[0] + points[1]) / 2
    right_half = points[1] + points[2] / 2  # also second right controlpoint
    left_second_controlpoint = (left_half + middle) / 2
    rigth_first_contolpoint = (middle + right_half) / 2
    mutual_point = (left_second_controlpoint + rigth_first_contolpoint) / 2  # new point on curve
    if curve_is_flat(points=[current_point, *points]):
        storage.append(mutual_point)
        return  # hand back control
    else:
        _cubic_path_segmentation([left_half, left_second_controlpoint, mutual_point], current_point, storage)
        _cubic_path_segmentation([rigth_first_contolpoint, right_half, points[-1]], mutual_point, storage)


def _naive_path_segmentation(points, current_point, storage, dissections=50):
    for t in range(1, dissections+1):
        t = t / dissections
        storage.append(
            (1-t) ** 3 * current_point +
            3 * (1-t) ** 2 * t * points[0] +
            3 * (1-t) * t ** 2 * points[1] +
            t ** 3 * points[2]
        )


class SimpleElement:

    approximation_methods = {
        'naive': _naive_path_segmentation,
        'castekjau': _cubic_path_segmentation,
    }

    APPROXIMATION_METHOD = 'naive'

    def __init__(self, path, ctm):
        self.root_path = path
        self.num_of_paths = self.get_paths
        self.subpath_list = self.get_subpaths
        self.transformation_matrix = ctm
        self.curve_list = self.get_curves
        self.figure_points = []
        self.z_end_point = None
        self.current_point = None

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

    def convert_to_simple_paths(self):
        self.current_point = None
        for path in self.curve_list:
            self.figure_points.append([])
            for curve in path:
                command = curve[0]
                data = curve[1].strip()
                self.add_paths_via_method(command, data, deepcopy(self.current_point))

    @staticmethod
    def get_method(command):
        method_mapping = {
            MOVETO_MARKERS: 'moveto_path',
            CLOSEPATH_MARKERS: 'close_path',
            LINETO_MARKERS: 'lineto_path',
            CUBIC_MARKERS: 'curveto_path',
            ELLIPTIC_MARKER: 'elliptic_path',
            VERTICAL_MARKERS: 'vertical_path',
            HORIZONTAL_MARKERS: 'horizontal_path',
            QUADRATIC_MARKERS: 'quadratic_path',
        }

        for marker in method_mapping:
            if command in marker:
                return method_mapping.get(marker)
        return method_mapping.get(LINETO_MARKERS)

    def approximation_function(self, name):
        return self.approximation_methods.get(name, _naive_path_segmentation)

    def add_paths_via_method(self, command, data, current_point):
        method = self.get_method(command)
        draw_command = getattr(self, method)
        draw_command(command, data, current_point)

    def moveto_path(self, command, data, current_point):
        key_point, lineto_points = self.movento_info(data)
        if command is 'M':
            current_point = np.array(self.convert_point(key_point))
        else:
            if self.z_end_point is None:
                current_point = np.array(self.convert_point(key_point))
            else:
                current_point = self.z_end_point + np.array(self.convert_point(key_point))
        self.figure_points[-1].append(deepcopy(current_point))
        self.current_point = deepcopy(current_point)
        self.z_end_point = deepcopy(current_point)
        if lineto_points:
            self.lineto_path('l', lineto_points, deepcopy(self.current_point))

    @staticmethod
    def movento_info(data):
        data = data.split(' ')
        key_point = data[0]
        lineto_points = ' '.join(data[1:])
        return key_point, lineto_points

    def lineto_path(self, command, data, current_point):
        points = data.split(' ')
        if command is 'L':
            for point in points:
                current_point = np.array(self.convert_point(point))
                self.figure_points[-1].append(deepcopy(current_point))
        else:
            for point in points:
                current_point += np.array(self.convert_point(point))
                self.figure_points[-1].append(deepcopy(current_point))

    def horizontal_path(self, command, data, current_point):
        lengths = data.split(' ')
        if command is 'H':
            for length in lengths:
                point = ','.join((length, str(current_point[0])))
                current_point = np.array(self.convert_point(point))
                self.figure_points[-1].append(deepcopy(current_point))
        else:
            for length in lengths:
                point = ','.join((length, str(0)))
                current_point += np.array(self.convert_point(point))
                self.figure_points[-1].append(deepcopy(current_point))
        self.current_point = deepcopy(current_point)

    def vertical_path(self, command, data, current_point):
        lengths = data.split(' ')
        if command is 'V':
            for length in lengths:
                point = ','.join((str(current_point[0]), length))
                current_point = np.array(self.convert_point(point))
                self.figure_points[-1].append(deepcopy(current_point))
        else:
            for length in lengths:
                point = ','.join((str(0), length))
                current_point += np.array(self.convert_point(point))
                self.figure_points[-1].append(deepcopy(current_point))
        self.current_point = deepcopy(current_point)

    def curveto_path(self, command, data, current_point):
        if isinstance(data, str):
            data = data.split()
        curveto_points = data[:3]
        next_curveto = data[3:]
        if command is 'C':
            self.approximation_function('naive')(curveto_points, deepcopy(current_point), self.figure_points[-1])
        else:
            curveto_points = to_absolute(deepcopy(current_point), *curveto_points)
            self.approximation_function('naive')(curveto_points, deepcopy(current_point), self.figure_points[-1])
        self.current_point = deepcopy(self.figure_points[-1][-1])
        if next_curveto:
            self.curveto_path(command, next_curveto, deepcopy(self.current_point))

    def quadratic_path(self, command, data, current_point):
        if isinstance(data, str):
            data = data.split()
        quadratic_points = data[:2]
        next_quadratic = data[2:]
        curveto_points = self.to_cubic_bezier(quadratic_points, deepcopy(current_point))
        self.curveto_path('C', curveto_points, deepcopy(current_point))
        if next_quadratic:
            self.quadratic_path(command, next_quadratic, deepcopy(self.current_point))

    def close_path(self, command, data, _):
        assert command in 'zZ'
        assert data is ''
        self.figure_points[-1].append(deepcopy(self.z_end_point))

    @staticmethod
    def convert_point(raw_data):
        return np.array(tuple(map(float, raw_data.split(','))))

    def to_cubic_bezier(self, quadratic_points, reference_point):
        quadratic_points = quadratice_sanitizer(quadratic_points)
        curve_start = deepcopy(reference_point)
        curve_end = self.convert_point(quadratic_points[1]) + curve_start
        control_point = self.convert_point(quadratic_points[0]) + curve_start
        first_control_point = curve_start + (2/3) * (control_point - curve_start)
        second_control_point = curve_end + (2/3) * (control_point - curve_end)
        return [first_control_point, second_control_point, curve_end]


class SvgParser:
    ELEMENT_MARKER = '{http://www.w3.org/2000/svg}path'
    PATH_MARKER = 'd'
    BASE_CTM = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def __init__(self, file, width=None, height=None, scale_x=1, scale_y=1, rotate=0, radians=False, svg=None):
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
    def transformation_matrix(self):
        scale_tm = np.array([[self.scale_x, 0, 0],
                             [0, self.scale_y, 0],
                             [0, 0, 1]])
        rotation_tm = np.array([[cos(self.rotate), -sin(self.rotate), 0],
                                [sin(self.rotate), cos(self.rotate), 0],
                                [0, 0, 1]])
        ctm = self.BASE_CTM * scale_tm * rotation_tm
        return ctm

    @property
    def paths(self):
        path_container = [item.get(self.PATH_MARKER) for item in self.root.iter(self.ELEMENT_MARKER)]
        return path_container

    def parse(self):
        for element in self.paths:
            self.element_list.append(SimpleElement(element, ctm=self.transformation_matrix))
        return self.element_list


if __name__ == '__main__':
    some_svg = SvgParser('../example (1).svg', scale_x=2, scale_y=0.5, rotate=90)
    some_svg.parse()
    # elm = some_svg.element_list
    for element in some_svg.element_list:
        element.convert_to_simple_paths()
        for subpath in element.figure_points:
            subpath = 
            plt.scatter(*zip(*subpath))
    #     figure.convert_to_simple_paths()
    #     for paths in figure.figure_points:
    #         for subpath in figure:
    #             plt.scatter(*zip(*subpath   ))
    plt.show()
    import ipdb
    ipdb.set_trace()
    print('Finish')
