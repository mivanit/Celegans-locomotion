from typing import List
from math import degrees

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Circle, Rectangle, Wedge  # type: ignore
from matplotlib.collections import PatchCollection  # type: ignore

from software.Lines.LineSet import LineSet
from software.Lines.collision_object import CollisionType


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = plt.Figure()
        super(MplCanvas, self).__init__(self.fig)
        self.axes = self.fig.add_axes([0.05, 0.05, 0.9, 0.9])
        self.memo = {}

    def plot(self, line_list: LineSet):
        if line_list.list == []:
            print("No avaliable data to plot")  # TODO: jumping window
        else:
            print("search finish, number is", len(line_list.list))
            self._initialize_canvas(line_list)
            for line in line_list.list:
                if line.visible:
                    self.axes.plot(line.data['x'][:, 0], line.data['y'][:, 0],
                                   color=(line.color[0]/255, line.color[1]/255, line.color[2]/255))
                    foodpos_x: float = float(line.params["ChemoReceptors"]["foodPos"]["x"])
                    foodpos_y: float = float(line.params["ChemoReceptors"]["foodPos"]["y"])
                    self.axes.plot(foodpos_x, foodpos_y, 'x',color=(line.color[0]/255, line.color[1]/255, line.color[2]/255))
                    print("plot successfully", line.folder_dir, np.shape(line.data['x'][:, 0]))
                    # print(self.axes.lines)
            self.axes.set_aspect('equal')
            self.fig.canvas.draw()

    def _initialize_canvas(self, line_list: LineSet, line_id=0):
        self.axes.cla()
        self._plot_collision(line_list=line_list, line_id=line_id)
        self._plot_foodPos(line_list=line_list, line_id=line_id)
        # TODO: exam the existence of foodpos or coll

    def _plot_collision(self, line_list: LineSet, line_id=0):
        if line_list.list[line_id].collobjs is None:
            print(f"The {line_id} st line has no collision")
        else:
            plot_objs: List[Patch] = []
            for obj in line_list.list[line_id].collobjs:
                if obj.coll_type == CollisionType.Box_Ax:
                    plot_objs.append(Rectangle(
                        xy=[obj['bound_min_x'], obj['bound_min_y']],
                        width=obj['bound_max_x'] - obj['bound_min_x'],
                        height=obj['bound_max_y'] - obj['bound_min_y'],
                        fill=True,
                    ))
                elif obj.coll_type == CollisionType.Disc:
                    plot_objs.append(Wedge(
                        center=[obj['centerpos_x'], obj['centerpos_y']],
                        r=obj['radius_outer'],
                        theta1=degrees(obj['angle_min']),
                        theta2=degrees(obj['angle_max']),
                        width=obj['radius_outer'] - obj['radius_inner'],
                        fill=True,
                    ))
            pc: PatchCollection = PatchCollection(
                plot_objs,
                facecolor='red',
                alpha=0.5,
                edgecolor='red',
            )
            self.axes.add_collection(pc)

    def _plot_foodPos(self, line_list: LineSet, line_id=0, fmt: str = 'x', label: str = None,
                      maxdist_disc: bool = True):
        if line_list.list[line_id].params and "ChemoReceptors" in line_list.list[line_id].params:
            if "DISABLED" not in line_list.list[line_id].params["ChemoReceptors"]:
                foodpos_x: float = float(line_list.list[line_id].params["ChemoReceptors"]["foodPos"]["x"])
                foodpos_y: float = float(line_list.list[line_id].params["ChemoReceptors"]["foodPos"]["y"])
                self.axes.plot(foodpos_x, foodpos_y, fmt, label=label)
                # self.axes.plot(-foodpos_x, foodpos_y, fmt, label=label)
                if maxdist_disc:
                    if "max_distance" in line_list.list[line_id].params["ChemoReceptors"]:
                        self.axes.add_patch(Circle(
                            (foodpos_x, foodpos_y),
                            radius=line_list.list[line_id].params["ChemoReceptors"]["max_distance"],
                            alpha=0.1,
                            color='green',
                        ))
                    else:
                        print('couldnt find "max_distance"')
                return (foodpos_x, foodpos_y)
        else:
            print(f"The {line_id} st line has no food pos")
