import numpy as np

class Line:

    def __init__(self, line):
        self.rho = line[0]
        self.theta = line[1]

        # id also reflect the order of the line relative to the origin of image
        self.id = None

        # list of intersections coordinates [(x,y), (x,y), ...]
        self.intersections = []

        # id of the intersection line
        self.intersection_line_ids = []

    def add_intersection(self, coordinates, line_id):
        if len(coordinates) != 2:
            raise ValueError("Coordinates should have 2 values (x, y)")

        self.intersections.append(coordinates)
        self.intersection_line_ids.append(line_id)

    def set_id(self, id):
        self.id = id

    def get_line(self):
        return (self.rho, self.theta)

    def get_intersections(self):
        return np.asarray(self.intersections, dtype=int)



