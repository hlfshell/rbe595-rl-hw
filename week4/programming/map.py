from typing import List, Tuple

from PIL import Image, ImageDraw

OPEN = 0
OBSTACLE = 1
ROBOT = 2


class Map:
    def __init__(self, goal: Tuple[int, int] = (10, 7), robot=(40, 7)):
        self.goal = goal
        self.robot = robot
        self.__init_map__()

    def __init_map__(self):
        self.map: List[List[int]] = []

        with open("./map.dat", "r") as file:
            lines = file.readlines()

            for index, line in enumerate(lines):
                self.map.append([])
                for char in line:
                    if char == "\n":
                        continue
                    self.map[index].append(int(char))

    def shape(self) -> Tuple[int, int]:
        return (len(self.map), len(self.map[0]))

    def print(self):
        for row in self.map:
            for col in row:
                print(col, end="")
            print()

    def draw(self, pixels_per=10):
        h, w = self.shape()
        img = Image.new("RGB", (w * pixels_per, h * pixels_per))
        draw = ImageDraw.Draw(img)

        for r, row in enumerate(self.map):
            for c, cell in enumerate(row):
                if (c, r) == self.goal:
                    color = "red"
                elif (c, r) == self.robot:
                    color = "green"
                elif cell == OPEN:
                    color = "white"
                elif cell == OBSTACLE:
                    color = "black"

                upper_left = (c * pixels_per, r * pixels_per)
                bottom_right = (upper_left[0] + pixels_per, upper_left[1] + pixels_per)
                draw.rectangle([upper_left, bottom_right], color)

        return img

    def set_robot(self, xy):
        self.robot = xy

    # def get_neighbors(
    #     self, xy: Tuple[int, int]
    # ) -> Tuple[int, int, int, int, int, int, int, int]:
    #     """
    #     get_neighbors takes a given xy coordinate, then returns the
    #     neighbors in order from top around clockwise:
    #     812
    #     7#3
    #     654
    #     If the edge is queried, it will return obstacles instead
    #     of wrapping or failing.
    #     """
    #     # 0,-1  1,-1    1,0     1,1     0,1
    #     # -1,1  -1,0    -1,-1
    #     for y_diff in [-1, 0, 1]:
    #         for
