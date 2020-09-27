import numpy as np
import abc
import matplotlib.pyplot as plt
from utilities import FifoUniqueQueue, mat_argmax, mat_argmin
from scipy.ndimage import convolve

from queue import Queue, PriorityQueue
from IPython.core.debugger import set_trace


class RoutePlanner():
    name_on_figs = "(name not set by child class)"

    def __init__(self,
                 grid=None,
                 initial_location=None,
                 dist_between_measurements=None,
                 debug_code=0):

        assert grid
        assert dist_between_measurements

        self.grid = grid
        self.debug_code = debug_code
        self.dist_between_measurements = dist_between_measurements
        # Necessary to save `initial_location` to reset().
        self.initial_location = initial_location

        self.reset()

        if self.debug_code == 1:
            self.l_waypoints = [self.previous_waypoint]

            self.l_meas_locs = []
            num_locations = 50
            for ind_location in range(num_locations):
                meas_location = self.next_measurement_location(None)
                self.l_meas_locs.append(meas_location)

            m_waypoints = np.array(self.l_waypoints)
            plt.plot(m_waypoints[:, 0],
                     m_waypoints[:, 1],
                     '+-',
                     label="waypoints",
                     color="black")

            m_meas_locs = np.array(self.l_meas_locs)
            plt.plot(m_meas_locs[:, 0],
                     m_meas_locs[:, 1],
                     'o',
                     label="measurements",
                     color="blue")
            plt.show()

    def reset(self):
        """ Used e. g. in Monte Carlo"""

        if self.initial_location is not None:
            self.previous_waypoint = self.initial_location
        else:
            self.previous_waypoint = self.grid.random_point_in_the_area()

        # Always take a measurement at the starting point. --> already done
        self.l_next_measurement_locations = [self.previous_waypoint]
        self.remaining_dist = self.dist_between_measurements

    def next_measurement_location(self, m_uncertainty):

        while not self.l_next_measurement_locations:
            # Obtain next measurement location
            next_waypoint = self.next_waypoint(m_uncertainty)

            self.l_next_measurement_locations, self.remaining_dist = \
                self.measurement_locations_from_waypoints(
                    self.previous_waypoint,
                    next_waypoint,
                    self.remaining_dist)

            self.previous_waypoint = next_waypoint

        next_location = self.l_next_measurement_locations.pop(0)

        # set_trace()
        # print(f"route_planning: next_location = {next_location}")

        return next_location

    def measurement_locations_from_waypoints(self, previous_waypoint,
                                             next_waypoint, remaining_dist):
        """Args:
        Returns:

        - "l_locations": list of locations between `previous_waypoint`
          and `next_waypoint` if any. First location is at distance
          `remaining_dist` from `previous_waypoint`. `next_waypoint`
          is never included.
        - "remaining_dist": distance to next measurement point after
          the UAV is at point `next_waypoint`.
        """
        dist_waypoints = np.linalg.norm(next_waypoint - previous_waypoint)

        if dist_waypoints <= remaining_dist:
            return [], remaining_dist - dist_waypoints

        num_points = np.ceil(
            (dist_waypoints - remaining_dist) / self.dist_between_measurements)

        v_distances = \
            remaining_dist + np.arange(num_points) * self.dist_between_measurements
        remaining_dist = self.dist_between_measurements - (dist_waypoints -
                                                           v_distances[-1])

        unit_vector = (next_waypoint - previous_waypoint) / dist_waypoints

        l_locations = [
            previous_waypoint + dist * unit_vector for dist in v_distances
        ]

        return l_locations, remaining_dist

    @abc.abstractmethod
    def next_waypoint(self, d_map_estimate):
        pass

    def plot_path(self,
                  start,
                  l_path,
                  axis=None,
                  m_node_costs=None,
                  label="",
                  color="white"):

        if axis is None:
            fig, axis = plt.subplots(nrows=1, ncols=1)

        if m_node_costs is not None:
            im = axis.imshow(
                #            im = plt.imshow(
                m_node_costs,
                # interpolation='spline16',
                cmap='jet',
                # extent=[0, m_node_costs.shape[1], m_node_costs.shape[0], 0],
                extent=np.array(
                    [0, m_node_costs.shape[1], m_node_costs.shape[0], 0]) -
                       0.5,
                # origin="upper"
                vmax=1,  # m_uncertainty_map.max(),
                vmin=0,  # m_uncertainty_map.min()
            )
            # fig.colorbar(im)
            plt.colorbar(im)

        m_path = np.array([start] + l_path)
        axis.plot(
            # plt.plot(
            m_path[:, 1],
            m_path[:, 0],
            '+-',
            label=label,
            color=color,
        )

        axis.plot(
            m_path[0, 1],
            m_path[0, 0],
            'o',
            label=label,
            color=color,
        )

        return axis


class MinimumCostPlanner(RoutePlanner):
    name_on_figs = "Min. cost planner"

    d_actions = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
        "UPLEFT": (-1, -1),
        "UPRIGHT": (-1, 1),
        "DOWNLEFT": (1, -1),
        "DOWNRIGHT": (1, 1),
    }

    l_next_waypoints = []

    def __init__(self,
                 metric=None,  # can be "power_variance" or "service_entropy"
                 grid=None,
                 **kwargs):

        assert metric
        assert grid

        self.metric = metric
        self.name_on_figs = f"{self.name_on_figs} ({metric})"
        # number of times at each grid point
        self.m_visited_points = np.zeros((grid.num_points_y, grid.num_points_x))

        super().__init__(grid=grid, **kwargs)

    def next_waypoint(self, d_map_estimate):

        if not self.l_next_waypoints:

            t_inds_current = self.grid.nearest_gridpoint_inds(
                self.previous_waypoint)

            # Choose a destination
            t_inds_destination, m_uncertainty = self.next_destination(t_inds_current, d_map_estimate)

            # Find shortest path
            m_node_costs = 1 / (m_uncertainty + 0.01)
            l_path_inds = self.shortest_path(m_node_costs, t_inds_current,
                                             t_inds_destination)

            # Turn indices to coordinates
            self.l_next_waypoints = [
                self.grid.indices_to_point(inds) for inds in l_path_inds
            ]

            if self.debug_code == 2:
                self.plot_path(t_inds_current, l_path_inds, m_node_costs=1 / m_node_costs)
                plt.show()
                # set_trace()

        if not self.l_next_waypoints:
            set_trace()

        return self.l_next_waypoints.pop(0)

    def next_destination(self, t_inds_current, d_map_estimate):
        """Returns the indices of the next destination grid point."""

        if self.metric == "power_variance":
            m_uncertainty = np.sum(d_map_estimate["t_power_map_norm_variance"], 0)
        elif self.metric == "service_entropy":
            m_uncertainty = np.sum(d_map_estimate["t_service_map_entropy"], 0)
            # m_uncertainty = np.max(d_map_estimate["t_service_map_entropy"],0)
        else:
            raise Exception("Invalide metric")

        # Spatial filter
        kernel = np.ones((3, 3))
        m_mod_uncertainty = convolve(m_uncertainty, kernel)

        # Modified uncertainty
        m_mod_uncertainty = m_mod_uncertainty * (1 / (1 + self.m_visited_points))
        m_mod_uncertainty[t_inds_current] = 0  # prevent remaining in the same point

        t_inds_destination = mat_argmax(m_mod_uncertainty)
        if t_inds_destination == t_inds_current:
            set_trace()
            # print("already at point of maximum uncertainty, choosing next waypoint randomly")
            # t_inds_destination = (np.random.randint(0, high=m_uncertainty.shape[0]),
            #                       np.random.randint(0, high=m_uncertainty.shape[0])
            #                       )

        self.m_visited_points[t_inds_destination] += 1

        return t_inds_destination, m_mod_uncertainty

    def shortest_path(self, m_node_costs=None, start=None, destination=None):
        """Bellman-Ford algorithm, aka breath-first search (BFS)
        [bertsekas2005]. Provides a path with minimum cost."
        Arguments:
        `m_node_costs`: Ny x Nx matrix whose (i,j)-th entry is the
        cost of traversing the grid point (i,j). Its dimensions Ny, Nx
        define a rectangular grid.
        Returns:
        `l_path` : None if a path does not exist. Else, a list of grid point
        indices corresponding to a shortest path between `start` and
        `destination`, which are grid node indices. `start` is not
        included in the returned list unless start==destination, but
        `destination` is always included.
        """

        self.m_node_costs = m_node_costs  # Required by other methods

        def is_in_grid(t_inds):
            return (t_inds[0] < m_node_costs.shape[0]) \
                   and (t_inds[1] < m_node_costs.shape[1])

        assert is_in_grid(start)
        assert is_in_grid(destination)

        if start == destination:
            return [destination]

        queue = FifoUniqueQueue()  # OPEN queue in [bertsekas2005]. It
        # contains nodes that are candidates
        # for being in the shortest path to
        # other nodes.
        queue.put(start)
        m_cost = np.full(m_node_costs.shape, fill_value=np.infty, dtype=float)
        m_cost[start] = 0

        # keys: state. values: previous_state
        d_branches = {}
        while not queue.empty():

            current_point = queue.get()
            # print(f"point={current_point}")

            for str_action in self._possible_actions(current_point):
                next_point = self._next_state(current_point, str_action)

                new_cost = m_cost[current_point] + self.transition_cost(
                    current_point, next_point)

                # UPPER in [bertsekas2005] is m_cost[destination]
                if new_cost < min(m_cost[next_point], m_cost[destination]):

                    d_branches[next_point] = current_point
                    m_cost[next_point] = new_cost

                    if next_point != destination:
                        queue.put(next_point)

        if m_cost[destination] < np.infty:
            if self.debug_code:
                print("Route found")
            state = destination
            l_path = [state]
            while d_branches[state] != start:
                previous_state = d_branches[state]
                l_path.append(previous_state)
                state = previous_state

            return l_path[::-1]

        else:
            set_trace()
            print("Route not found")
            return None

    def approximate_bfs_shortest_path(self, m_node_costs=None, start=None, destination=None):
        """It is an approximate Dijkstra algorithm because we do not revisit
        nodes. It is assumed that the distance stored for a visited
        node is the distance of the sortest path. I think this makes
        sense if our metric is Euclidean distance on a regular graph,
        but not with node costs.
        [Daniel] I adapted this from the flying car degree.
        Arguments:
        `m_node_costs`: Ny x Nx matrix whose (i,j)-th entry is the
        cost of traversing the grid point (i,j). Its dimensions Ny, Nx
        define a rectangular grid.
        Returns:
        `l_path` : None if a path does not exist. Else, a list of grid point
        indices corresponding to a shortest path between `start` and
        `destination`, which are grid node indices. `start` is not
        included in the returned list unless start==destination, but
        `destination` is always included.
        """

        self.m_node_costs = m_node_costs  # Required by other methods

        def is_in_grid(t_inds):
            return (t_inds[0] < m_node_costs.shape[0]) \
                   and (t_inds[1] < m_node_costs.shape[1])

        assert is_in_grid(start)
        assert is_in_grid(destination)

        if start == destination:
            return [destination]

        queue = PriorityQueue()
        queue.put((0, start))
        s_visited = {start}

        # keys: state. values: previous_state
        d_branches = {}
        b_found = False
        while not queue.empty():

            cost_so_far, current_point = queue.get()
            # print(f"point={current_point}")

            for str_action in self._possible_actions(current_point):
                next_point = self._next_state(current_point, str_action)

                if next_point not in s_visited:
                    s_visited.add(next_point)
                    new_cost = cost_so_far + self.transition_cost(
                        current_point, next_point)
                    queue.put((new_cost, next_point))
                    d_branches[next_point] = current_point

                    if next_point == destination:
                        b_found = True
                        print("Route found")
                        break

            if b_found:
                break

        if b_found:

            state = destination
            l_path = [state]
            while d_branches[state] != start:
                previous_state = d_branches[state]
                l_path.append(previous_state)
                state = previous_state

            return l_path[::-1]

        else:
            set_trace()
            print("Route not found")
            return None

    def transition_cost(self, point_1, point_2):
        # approximates the integral of the cost

        dist = np.linalg.norm(np.array(point_2) - np.array(point_1))

        cost_1 = self.m_node_costs[point_1[0], point_1[1]]
        cost_2 = self.m_node_costs[point_2[0], point_2[1]]
        return dist / 2 * cost_1 + dist / 2 * cost_2

    def _possible_actions(self, state):
        """
        Arguments:
        `state`: tuple with the indices of the considered grid point.
        Returns:
        list of possible actions at state `state`.

        """

        max_row_index = self.m_node_costs.shape[0] - 1
        max_col_index = self.m_node_costs.shape[1] - 1

        l_actions = []
        for str_action in self.d_actions:

            candidate_entry = self._next_state(state, str_action)

            # Check if in grid
            if candidate_entry[0] >= 0 and \
                    candidate_entry[0] <= max_row_index and \
                    candidate_entry[1] >= 0 and \
                    candidate_entry[1] <= max_col_index:
                l_actions.append(str_action)

        return l_actions

    def _next_state(self, state, str_action):

        v_movement = self.d_actions[str_action]
        return (state[0] + v_movement[0], state[1] + v_movement[1])

    def path_cost(self, m_node_costs, start, l_path):

        # Returns the sum of the cost of all nodes in l_path.

        self.m_node_costs = m_node_costs

        cost = 0
        if len(l_path) == 0:
            return 0
        prev_state = start
        for state in l_path:
            cost += self.transition_cost(prev_state, state)
            prev_state = state

        return cost


class IndependentUniformPlanner(RoutePlanner):
    name_on_figs = "Indep. Uniform Planner "

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def next_waypoint(self, d_map_estimate):
        """Returns length-3 vector with the coordinates of a random point in
        the area (not nec. on the grid)
        """
        # random coordinates in the range of x_coords, y_coords, and z_coords

        waypoint = self.grid.random_point_in_the_area()

        if self.debug_code == 1:
            self.l_waypoints.append(waypoint)

        return waypoint


class GridPlanner(RoutePlanner):
    name_on_figs = "Grid Planner"
    l_next_waypoints = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def next_waypoint(self, d_map_estimate):
        if not self.l_next_waypoints:
            # stay `self.dist_between_measurements/2` away from the limits
            max_x = self.grid.max_x() - self.dist_between_measurements / 2
            min_x = self.grid.min_x() + self.dist_between_measurements / 2

            max_y = self.grid.max_y() - self.dist_between_measurements / 2
            min_y = self.grid.min_y() + self.dist_between_measurements / 2

            # Coordinates of the turning points:
            eps = 1e-10  # to force arange to include the upper limit if needed
            v_x_coords = np.arange(min_x, max_x + eps, step=self.dist_between_measurements)
            # v_y_coords = np.arange(min_y,max_y+eps, step=self.dist_between_measurements)
            v_y_coords = np.array([max_y, min_y])  # v_y_coords[::-1] # Flip it

            # Now form the sequence of turning points
            # Just repeat each entry of v_x_coords
            v_seq_x_coords = np.vstack((v_x_coords, v_x_coords)).T.ravel()
            # Similar for Y, but with a shift
            v_seq_y_coords = np.tile(v_y_coords, int(np.ceil(len(v_x_coords) / 2)))
            v_seq_y_coords = np.vstack((v_seq_y_coords, v_seq_y_coords)).T.ravel()
            v_seq_y_coords = v_seq_y_coords[0:len(v_seq_x_coords)]
            # v_seq_y_coords = np.concatenate( ([v_seq_y_coords[-1]] , v_seq_y_coords[0:-1]))
            v_seq_y_coords = np.concatenate((v_seq_y_coords[1:], [v_seq_y_coords[0]]))

            v_seq_z_coords = np.full(v_seq_y_coords.shape, fill_value=self.grid.z_value(), dtype=float)

            m_points = np.vstack((v_seq_x_coords, v_seq_y_coords, v_seq_z_coords))

            self.l_next_waypoints = list(m_points.T)

        return self.l_next_waypoints.pop(0)


class SquareSpiralGridPlanner(RoutePlanner):
    name_on_figs = "Spiral Grid Planner"
    l_next_waypoints = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def next_waypoint(self, d_map_estimate):
        if not self.l_next_waypoints:
            # stay `self.dist_between_measurements` away from the limits
            max_x = self.grid.max_x()
            min_x = self.grid.min_x()

            max_y = self.grid.max_y()
            min_y = self.grid.min_y()

            # initial point in top left corner
            v_x_coords = min_x
            v_y_coords = max_y
            count_in_corner_grid = 1
            count_in_spiral = 1
            count = 0
            while 1:
                if count_in_corner_grid == 1:
                    next_v_x_coords = min_x + count_in_spiral * self.dist_between_measurements
                    next_v_y_coords = max_y - count_in_spiral * self.dist_between_measurements
                elif count_in_corner_grid == 2:
                    next_v_x_coords = max_x - count_in_spiral * self.dist_between_measurements
                    next_v_y_coords = max_y - count_in_spiral * self.dist_between_measurements

                elif count_in_corner_grid == 3:
                    next_v_x_coords = max_x - count_in_spiral * self.dist_between_measurements
                    next_v_y_coords = min_y + count_in_spiral * self.dist_between_measurements
                elif count_in_corner_grid == 4:
                    next_v_x_coords = min_x + (count_in_spiral + 1) * self.dist_between_measurements
                    next_v_y_coords = min_y + count_in_spiral * self.dist_between_measurements

                if count_in_corner_grid % 4 == 0:
                    count_in_spiral += 1
                    count_in_corner_grid = 1
                else:
                    count_in_corner_grid += 1

                count += 1
                if count == 200:
                    break

                v_x_coords = np.append(v_x_coords, next_v_x_coords)
                v_y_coords = np.append(v_y_coords, next_v_y_coords)

            v_z_coords = np.full(v_y_coords.shape, fill_value=self.grid.z_value(), dtype=float)

            m_points = np.vstack((v_x_coords, v_y_coords, v_z_coords))

            self.l_next_waypoints = list(m_points.T)

        return self.l_next_waypoints.pop(0)


class RandomPlanner(RoutePlanner):
    name_on_figs = "Random Planner"
    l_previous_waypoints = None

    def __init__(self,
                 dist_between_waypoints=None,
                 **kwargs):
        super().__init__(**kwargs)
        assert dist_between_waypoints
        self.dist_between_waypoints = dist_between_waypoints

    def next_waypoint(self, d_map_estimate):

        max_x = self.grid.max_x()
        min_x = self.grid.min_x()

        max_y = self.grid.max_y()
        min_y = self.grid.min_y()
        slope_in_radian = np.random.choice(np.arange(0, 2 * np.pi, np.pi / 6))  # list of angle spaced by 30 degree

        # slope_in_radian = np.random.uniform(0, 2 * np.pi)

        if self.l_previous_waypoints is None:
            waypoint = self.grid.random_point_in_the_area()
            self.l_previous_waypoints = waypoint

        else:

            x_coord = self.l_previous_waypoints[0] + self.dist_between_waypoints * np.cos(slope_in_radian)

            if x_coord > max_x:
                x_coord = max_x
            elif x_coord < min_x:
                x_coord = min_x
            else:
                x_coord = x_coord

            y_coord = self.l_previous_waypoints[1] + self.dist_between_waypoints * np.sin(slope_in_radian)
            if y_coord > max_y:
                y_coord = max_y
            elif y_coord < min_y:
                y_coord = min_y

            z_coord = self.l_previous_waypoints[2]

            waypoint = np.array([x_coord, y_coord, z_coord])
            self.l_previous_waypoints = waypoint

        return waypoint


def tests():
    def random_point(num_points_x, num_points_y):

        return (np.random.randint(low=0, high=num_points_y),
                np.random.randint(low=0, high=num_points_x))

    num_points_x = 20
    num_points_y = 10

    rp = MinimumCostPlanner(grid=1,
                            dist_between_measurements=1,
                            initial_location=(0, 0))
    m_node_costs = np.ones((num_points_y, num_points_x))
    # m_node_costs = m_node_costs + np.triu(m_node_costs)
    for row in range(num_points_y):
        for col in range(num_points_x):
            m_node_costs[row, col] = np.exp(-((row - num_points_y / 2) ** 2) /
                                            (num_points_y) ** 2 -
                                            ((col - num_points_x / 2) ** 2) /
                                            (num_points_x / 2) ** 2)

    while True:
        # start = (1, 6)  #
        start = random_point(num_points_x, num_points_y)
        # destination = (11, 6)
        destination = random_point(num_points_x, num_points_y)

        # approximately minimum cost
        l_path_cost_appr = rp.approximate_bfs_shortest_path(m_node_costs=m_node_costs,
                                                            start=start,
                                                            destination=destination)
        cost_appr = rp.path_cost(m_node_costs, start, l_path_cost_appr)
        print(f"Cost of approximate min cost path {cost_appr}")

        # shortest path in distance
        l_path_nodes = rp.approximate_bfs_shortest_path(m_node_costs=np.ones(
            (num_points_y, num_points_x)),
            start=start,
            destination=destination)

        print(f"Cost min dist path {rp.path_cost(m_node_costs, start, l_path_nodes)}")

        # minimum cost
        l_path_bf = rp.shortest_path(m_node_costs=m_node_costs,
                                     start=start,
                                     destination=destination)

        cost_bf = rp.path_cost(m_node_costs, start, l_path_bf)
        print(f"Cost BF path {cost_bf}")

        axis = rp.plot_path(start,
                            l_path_cost_appr,
                            m_node_costs=m_node_costs,
                            label="approx cost",
                            color="white")

        rp.plot_path(start,
                     l_path_nodes,
                     axis=axis,
                     label="nodes",
                     color="blue")

        rp.plot_path(start,
                     l_path_bf,
                     axis=axis,
                     label="BF",
                     color="red")

        if cost_appr < cost_bf:
            set_trace()

        plt.show()
    # set_trace()


if __name__ == "__main__":
    tests()

# TESTs
# m_node_costs = np.ones((num_points_y, num_points_x))
# m_node_costs[:, 5:] = 2
# start = (1, 6)  #random_point(num_points_x, num_points_y)
# destination = (11, 6)  #random_point(num_points_x, num_points_y)