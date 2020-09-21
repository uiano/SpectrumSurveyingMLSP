from utilities import project_to_interval, mat_argmin
import numpy as np

class RectangularGrid():
    """
        The origin is on the bottom-left entry of the grid.
    """

    def __init__(
            self,
            # area_side_length=None,
            gridpoint_spacing=None,  # Distance between adjacent grid points
            num_points_x=None,
            num_points_y=None,
            height=None):
        # Input check: check that mandatory arguments were provided
        assert gridpoint_spacing
        assert num_points_x
        assert num_points_y
        assert height, "Argument `height` must be provided"

        self.num_points_x = num_points_x
        self.num_points_y = num_points_y
        self.gridpoint_spacing = gridpoint_spacing

        # create a grid
        v_x_coords = np.arange(0, self.num_points_x) * self.gridpoint_spacing
        v_y_coords = np.arange(self.num_points_y - 1, -1,
                               step=-1) * self.gridpoint_spacing

        x_coords, y_coords = np.meshgrid(v_x_coords, v_y_coords, indexing='xy')
        z_coords = height * np.ones((self.num_points_y, self.num_points_x))

        self.t_coordinates = np.array([x_coords, y_coords, z_coords])

        self.m_all_distances = self.get_distance_matrix()

    #  def all_coordinates(self):
    #     # returns a  two-row array of grid i.e rearrange corresponding x an y coordinates
    #     return m_grid_cordinates = np.array([self.x_coords.ravel(), self.y_coords.ravel()])

    def four_nearest_gridpoint_indices(self, v_point):
        """returns a list with the indices of the 4 nearest grid points to
            `v_point`. Two or four of these points may coincide.
        """
        row_before, row_after, col_before, col_after = self.nearest_rows_and_cols(
            v_point)

        l_points = [(row_before, col_before), (row_before, col_after),
                    (row_after, col_after), (row_after, col_before)]

        return l_points

    def nearest_rows_and_cols(self, v_point):
        col_before = np.floor(v_point[0] / self.gridpoint_spacing)
        col_after = np.ceil(v_point[0] / self.gridpoint_spacing)

        row_before = self.num_points_y - 1 - np.ceil(
            v_point[1] / self.gridpoint_spacing)
        row_after = self.num_points_y - 1 - np.floor(
            v_point[1] / self.gridpoint_spacing)

        # Ensure within limits
        col_before = int(project_to_interval(col_before, 0, self.num_points_x))
        col_after = int(project_to_interval(col_after, 0, self.num_points_x))
        row_before = int(project_to_interval(row_before, 0, self.num_points_y))
        row_after = int(project_to_interval(row_after, 0, self.num_points_y))

        return row_before, row_after, col_before, col_after

    def point_as_convex_combination(self, v_point):
        """This function may be used for interpolation.
        Returns:
        `l_point_inds`: list with at most 3 tuples of two indices,
        each corresponding to one of the grid points that are
        nearest to v_point.
        'l_coef': list of coefficients adding up to 1 such that the
        x and y coordinates (1st two entries) of \sum_ind l_coef[ind]
        l_point_inds[ind] equal the x and y coordinates of v_point.
        """

        def point_inds_to_coefficients(l_point_inds, v_point):
            # Find convex combination coefficients -> system of equations
            m_points = np.array(
                [self.indices_to_point(tp_inds) for tp_inds in l_point_inds])
            m_A = np.vstack((m_points[:, 0:2].T, np.ones(
                (1, len(l_point_inds)))))
            v_b = np.array([v_point[0], v_point[1], 1])

            # if np.linalg.matrix_rank(m_A) < len(l_point_inds):
            #     set_trace()

            # v_coefficients = np.linalg.solve(m_A, v_b)
            v_coefficients = np.linalg.pinv(m_A) @ v_b  # Use pinv since there may be repeated rows

            # avoid numerical issues
            eps = 1e-6
            v_coefficients[np.logical_and(v_coefficients < 0, v_coefficients > -eps)] = 0

            if np.logical_or(
                    v_coefficients > 1, v_coefficients <
                                        0).any() or np.abs(np.sum(v_coefficients) - 1) > 1e-3:
                print(v_coefficients)
                # set_trace()
            return v_coefficients

        # debug_code
        # v_point = [51, 137,0]

        row_before, row_after, col_before, col_after = self.nearest_rows_and_cols(
            v_point)

        # Boundary cases
        if row_before == row_after:
            if col_before == col_after:
                return [(row_before, col_before)], [1]
            else:
                # The point lies in a segment between two points
                l_point_inds = [(row_before, col_before),
                                (row_before, col_after)]
        else:
            if col_before == col_after:
                l_point_inds = [(row_before, col_before),
                                (row_after, col_before)]
            else:
                # General case
                l_point_inds = [(row_after, col_before),
                                (row_before, col_after)]

                # Out of the other two points, find the one that is the closest
                ind_pt_1 = (row_before, col_before)
                ind_pt_2 = (row_after, col_after)
                d1 = np.linalg.norm(self.indices_to_point(ind_pt_1) - v_point)
                d2 = np.linalg.norm(self.indices_to_point(ind_pt_2) - v_point)
                if d1 < d2:
                    l_point_inds.append(ind_pt_1)
                else:
                    l_point_inds.append(ind_pt_2)

        v_coefficients = point_inds_to_coefficients(l_point_inds, v_point)

        return l_point_inds, list(v_coefficients)

    def get_distance_to_grid_points(self, v_point):
        """ returns an self.num_points_x x self.num_points_y whose
        (i,j)-th entry is the distance from the (i,j)-th grid point to
        v_point.
           """

        #
        # to get the distance between grid point and transmitter
        t_repeated_point = np.repeat(np.repeat(np.reshape(v_point, (3, 1, 1)),
                                               self.num_points_y,
                                               axis=1),
                                     self.num_points_x,
                                     axis=2)
        return np.linalg.norm(self.t_coordinates - t_repeated_point,
                              ord=2,
                              axis=0)

    def get_distance_matrix(self):
        """Returns an (self.num_points_x * self.num_points_y) x
        (self.num_points_x * self.num_points_y) matrix with the
        distance between each pair of grid points.
        """
        m_all_points = np.reshape(self.t_coordinates, (1, 3, -1))
        tcol_all_points = np.transpose(m_all_points, axes=(1, 2, 0))

        tcol_all_points_repeated = np.repeat(tcol_all_points,
                                             self.num_points_x *
                                             self.num_points_y,
                                             axis=2)

        m_all_distances = np.linalg.norm(
            tcol_all_points_repeated -
            np.transpose(tcol_all_points_repeated, axes=(0, 2, 1)),
            ord=2,
            axis=0)

        return m_all_distances

    def nearest_gridpoint_inds(self, point):
        """Returns a tuple with the indices of the grid point closest to `point`"""

        distance_to_grid_points = self.get_distance_to_grid_points(point)
        return mat_argmin(distance_to_grid_points)

    def indices_to_point(self, inds):
        """Returns a length-3 vector with the coordinates of the point with 2D
indices `inds`."""

        return self.t_coordinates[:, inds[0], inds[1]]

    def random_point_in_the_area(self):
        # The implementation may be improved
        x_coord = np.random.uniform((min(self.t_coordinates[0, 0])),
                                    (max(self.t_coordinates[0, 0])))
        y_coord = np.random.uniform((min(self.t_coordinates[1, :, 0])),
                                    (max(self.t_coordinates[1, :, 0])))
        z_coord = self.t_coordinates[2, 0, 0]

        return np.array([x_coord, y_coord, z_coord])

    def random_points_in_the_area(self, num_points=1):
        """Returns a matrix with `num_points` columns. Each column is a random
        point.
        FUT: change the implementation: generate the points here and
        invoke this function from `random_point_in_the_area`.
        """

        m_points = np.full((3, num_points), fill_value=None, dtype=float)
        for ind_point in range(num_points):
            m_points[:, ind_point] = self.random_point_in_the_area()

        return m_points


    def random_points_on_the_grid(self, num_points=1):
        
        pass
    
    def min_x(self):
        return self.t_coordinates[0, 0, 0]

    def max_x(self):
        return self.t_coordinates[0, 0, -1]

    def min_y(self):
        return self.t_coordinates[1, -1, 0]

    def max_y(self):
        return self.t_coordinates[1, 0, 0]

    def z_value(self):
        return self.t_coordinates[2, 0, 0]

    def num_points_x(self):
        return self.t_coordinates.shape[2]

    def num_points_y(self):
        return self.t_coordinates.shape[1]


class MapGenerator():
    
    def precompute(self, num_maps, filename):
    
        pass

class MapGeneratorFromFile(MapGenerator):
    
    def __init__(self, filename):

        self.filename = filename

    def generate_map(self):
        """Returns a map from file `self.filename`. """
        pass
