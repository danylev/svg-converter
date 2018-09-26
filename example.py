import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import json


# we define a few functions
# we always start at 1" above the reference z at the writing point
# we always end the wites or erases at this state too


# the amount that the erase head is longer
erase_delta = 2.
# next we write the gcode(for Dorna)
f = open('dorna.txt', 'w')

# this function generates the paths for a filename and plots the results
def get_paths(file_name, x_max, y_max, x_min, y_min):
    paths = []
    path_x = []
    path_y = []

    # first we read the svg file
    tree = ET.parse(file_name)
    root = tree.getroot()
    for path in root.iter('{http://www.w3.org/2000/svg}path'):

        d = path.attrib['d']
        d = d.replace(',', ' ')
        d = d.split()

        cur_x = 0.
        cur_y = 0.

        start_x = 0.
        start_y = 0.

        current_cmd = None

        i = 0
        while i < len(d):
            try:
                float(d[i])
            except ValueError:
                current_cmd = d[i]
                i += 1
                # when there is an m, a new path has started
            if current_cmd == 'm':

                if path_x:
                    paths.append([-np.array(path_x), np.array(path_y)])
                    path_x = []
                    path_y = []
                    # we should start a new path here
                    # we should read x and y as the first points
                cur_x += float(d[i])
                path_x.append(cur_x)
                i += 1

                cur_y += float(d[i])
                path_y.append(cur_y)
                i += 1

                start_x = cur_x
                start_y = cur_y


            elif current_cmd == 'M':

                if path_x:
                    paths.append([-np.array(path_x), np.array(path_y)])
                    path_x = []
                    path_y = []
                    # we should start a new path here
                    # we should read x and y as the first points
                cur_x = float(d[i])
                path_x.append(cur_x)
                i += 1

                cur_y = float(d[i])
                path_y.append(cur_y)
                i += 1

                start_x = cur_x
                start_y = cur_y


            elif current_cmd == 'v':

                cur_y += float(d[i])
                path_x.append(cur_x)
                path_y.append(cur_y)
                i += 1

            elif current_cmd == 'q':
                p0_x = 0
                p0_y = 0

                p1_x = float(d[i])
                i += 1

                p1_y = float(d[i])
                i += 1

                p2_x = float(d[i])
                i += 1

                p2_y = float(d[i])
                i += 1

                for t in np.linspace(0, .8, 5):
                    p_x = (1 - t) ** 2 * p0_x + 2 * t * (1 - t) * p1_x + t ** 2 * p2_x
                    p_y = (1 - t) ** 2 * p0_y + 2 * t * (1 - t) * p1_y + t ** 2 * p2_y
                    path_x.append(cur_x + p_x)
                    path_y.append(cur_y + p_y)

                cur_x += p2_x
                cur_y += p2_y

            elif current_cmd == 'Q':
                p0_x = 0
                p0_y = 0

                p1_x = float(d[i])
                i += 1

                p1_y = float(d[i])
                i += 1

                p2_x = float(d[i])
                i += 1

                p2_y = float(d[i])
                i += 1

                for t in np.linspace(0, .8, 5):
                    p_x = (1 - t) ** 2 * p0_x + 2 * t * (1 - t) * p1_x + t ** 2 * p2_x
                    p_y = (1 - t) ** 2 * p0_y + 2 * t * (1 - t) * p1_y + t ** 2 * p2_y
                    path_x.append(cur_x + p_x)
                    path_y.append(cur_y + p_y)

                cur_x = p2_x
                cur_y = p2_y


            elif current_cmd == 'c':
                p0_x = 0
                p0_y = 0

                p1_x = float(d[i])
                i += 1

                p1_y = float(d[i])
                i += 1

                p2_x = float(d[i])
                i += 1

                p2_y = float(d[i])
                i += 1

                p3_x = float(d[i])
                i += 1

                p3_y = float(d[i])
                i += 1

                for t in np.linspace(0, .8, 5):
                    p_x = (1 - t) ** 3 * p0_x + 3 * t * (1 - t) ** 2 * p1_x + 3 * t * (
                                                                                      1 - t) ** 2 * p2_x + t ** 3 * p3_x
                    p_y = (1 - t) ** 3 * p0_y + 3 * t * (1 - t) ** 2 * p1_y + 3 * t * (
                                                                                      1 - t) ** 2 * p2_y + t ** 3 * p3_y
                    path_x.append(cur_x + p_x)
                    path_y.append(cur_y + p_y)

                cur_x += p3_x
                cur_y += p3_y

            elif current_cmd == 'C':
                p0_x = 0
                p0_y = 0

                p1_x = float(d[i])
                i += 1

                p1_y = float(d[i])
                i += 1

                p2_x = float(d[i])
                i += 1

                p2_y = float(d[i])
                i += 1

                p3_x = float(d[i])
                i += 1

                p3_y = float(d[i])
                i += 1

                for t in np.linspace(0, .8, 5):
                    p_x = (1 - t) ** 3 * p0_x + 3 * t * (1 - t) ** 2 * p1_x + 3 * t * (
                                                                                      1 - t) ** 2 * p2_x + t ** 3 * p3_x
                    p_y = (1 - t) ** 3 * p0_y + 3 * t * (1 - t) ** 2 * p1_y + 3 * t * (
                                                                                      1 - t) ** 2 * p2_y + t ** 3 * p3_y
                    path_x.append(cur_x + p_x)
                    path_y.append(cur_y + p_y)

                cur_x = p3_x
                cur_y = p3_y

            elif current_cmd == 'l':
                cur_x += float(d[i])
                path_x.append(cur_x)
                i += 1

                cur_y += float(d[i])
                path_y.append(cur_y)
                i += 1

            elif current_cmd == 'z' or current_cmd == 'Z':
                path_x.append(start_x)
                path_y.append(start_y)
                cur_x = start_x
                cur_y = start_y

            elif current_cmd == 'h':

                cur_x += float(d[i])
                path_x.append(cur_x)
                path_y.append(cur_y)
                i += 1

            elif current_cmd == 'H':

                cur_x = float(d[i])
                path_x.append(cur_x)
                path_y.append(cur_y)
                i += 1

            elif current_cmd == 'V':

                cur_y = float(d[i])
                path_x.append(cur_x)
                path_y.append(cur_y)
                i += 1

            else:  # other commands not supported
                print(current_cmd)

    paths.append([-np.array(path_x), np.array(path_y)])

    # now we should scale the paths
    # we first find the min and max of the paths

    x_path_min = 100000
    x_path_max = -100000
    y_path_min = 100000
    y_path_max = -100000
    for pairs in paths:
        if min(pairs[0]) < x_path_min:
            x_path_min = min(pairs[0])

        if max(pairs[0]) > x_path_max:
            x_path_max = max(pairs[0])

        if min(pairs[1]) < y_path_min:
            y_path_min = min(pairs[1])

        if max(pairs[1]) > y_path_max:
            y_path_max = max(pairs[1])

    # next we scale the pairs
    for pairs in paths:
        print(pairs)
        pairs[0] = (x_max - x_min) * (pairs[0] - x_path_min) / (x_path_max - x_path_min) + x_min
        pairs[1] = (y_max - y_min) * (pairs[1] - y_path_min) / (y_path_max - y_path_min) + y_min

    for path_pair in paths:
        plt.plot(path_pair[1], path_pair[0])
    plt.show()
    return paths



def write_text(file_name, x_max, y_max, x_min, y_min):
    paths = get_paths(file_name, x_max, y_max, x_min, y_min)
    for path_pair in paths[:1]:
        print(path_pair)
        #cmd = json.dumps({"command": "move", "x": path_pair[0][0], "y": path_pair[1][0]})
        cmd = json.dumps({"command": "move", "prm": {"path": "line", "movement": 0, "speed": 5000.0, "x": path_pair[0][0], "y": path_pair[1][0]}})
        f.write(cmd + '\n')
        #cmd = json.dumps({"command": "move", "delta": 1, "z": -1})
        cmd = json.dumps({"command": "move", "prm": {"path": "line", "movement": 1, "speed": 5000.0, "z": -1}})
        f.write(cmd + '\n')
        for i in range(len(path_pair[0])):
            #cmd = json.dumps({"command": "move", "x": path_pair[0][i], "y": path_pair[1][i]})
            cmd = json.dumps({"command": "move", "prm": {"path": "line", "movement": 0, "speed": 5000.0, "x": path_pair[0][i], "y": path_pair[1][i]}})
            f.write(cmd + '\n')
        #cmd = json.dumps({"command": "move", "delta": 1, "z": 1})
        cmd = json.dumps({"command": "move", "prm": {"path": "line", "movement": 1, "speed": 5000.0, "z": 1}})
        f.write(cmd + '\n')


def erase_text(file_name, x_max, y_max, x_min, y_min):
    # first we make the toolholder at the correct height
    #cmd = json.dumps({"command": "move", "delta": 1, "z": erase_delta})
    cmd = json.dumps({"command": "move", "prm": {"path": "line", "movement": 1, "speed": 5000.0, "z": erase_delta}})
    f.write(cmd + '\n')
    #cmd = json.dumps({"command": "move", "delta": 1, "j4": 180})
    cmd = json.dumps({"command": "move", "prm": {"path": "joint", "movement": 1, "speed": 5000.0, "j4": 180.0}})
    f.write(cmd + '\n')
    # next write the text
    write_text(file_name, x_max, y_max, x_min, y_min)

    # finally move the tool holder to the right position
    #cmd = json.dumps({"command": "move", "delta": 1, "j4": -180})
    cmd = json.dumps({"command": "move", "prm": {"path": "joint", "movement": 1, "speed": 5000.0, "j4": -180.0}})
    f.write(cmd + '\n')
    #cmd = json.dumps({"command": "move", "delta": 1, "z": -erase_delta})
    cmd = json.dumps({"command": "move", "prm": {"path": "line", "movement": 1, "speed": 5000.0, "z": -erase_delta}})
    f.write(cmd + '\n')





# now lets write the same thing  and delete it:
x_max = 17.7
y_max = 3.0
x_min = 13.7
y_min = -3.0


write_text('example.svg', x_max, y_max, x_min, y_min)
#erase_text('hello.svg', x_max, y_max, x_min, y_min)
#write_text('imdorna.svg', x_max, y_max, x_min, y_min)


f.close()