from utilities import np, empty_array
import scipy


class Sampler():
    def __init__(self, grid, noise_std=0, interpolation_method="linear"):

        self.grid = grid
        self.noise_std = noise_std
        self.interpolation_method = interpolation_method

    def sample_map_multiple_loc(self, t_map, m_sample_locations):
        """m_sample_location: num_points x 3 matrix with the 3D locations of
        the sampling points.
        Returns:
        num_points x num_channels matrix with the measurements.
        """
        l_samples = []
        for v_location in m_sample_locations:
            l_samples.append(self.sample_map(t_map, v_location))

        return np.array(l_samples)

    def sample_map(self, t_map, v_sample_location):
        """Args:
           `t_map` is num_sources x self.grid.num_points_y x
        self.grid.num_points_x tensor with the power at each channel
        and location.
        Returns:
           length- num_sources vector where the i-th entry is the
           power measured at point `v_sample_location`. It is computed
           by interpolating `t_map[i,:,:]` and adding a zero-mean
           Gaussian random variable with std self.noise_std
           FUT: improve noise model.
           """
        # find the nearest grid point to the sample point
        # then find the power from t_map at that grid
        num_sources = t_map.shape[0]

        # OLD
        # v_sample_power = np.full(shape=(num_sources, ),
        #                          fill_value=None,
        #                          dtype=float)
        # for ind_source in range(num_sources):
        #     v_sample_power[ind_source] = t_map[ind_source, min_dist_index[0], min_dist_index[1]] + \
        #         np.random.normal(loc=0, scale=self.noise_std)

        if self.interpolation_method == "avg_nearest":
            l_four = self.grid.four_nearest_gridpoint_indices(
                v_sample_location)
            m_four_values = empty_array((num_sources, 4))
            for ind_source in range(num_sources):
                for ind_point in range(4):
                    m_four_values[ind_source, ind_point] = \
                        t_map[ind_source, l_four[ind_point][0], l_four[ind_point][1]]

            v_sample_power_interpolated = np.mean(m_four_values, axis=1) + \
                                          np.random.normal(size=(num_sources,), loc=0, scale=self.noise_std)

        elif self.interpolation_method == "linear":
            l_three, l_coef = self.grid.point_as_convex_combination(
                v_sample_location)
            v_sample_power_interpolated = empty_array((num_sources,))
            for ind_source in range(num_sources):
                l_three_values = [
                    t_map[ind_source, point_inds[0], point_inds[1]]
                    for point_inds in l_three
                ]
                v_sample_power_interpolated[ind_source] = np.dot(
                    l_coef, l_three_values)

        elif self.interpolation_method == "splines":

            x = self.grid.t_coordinates[0, 0, :]
            y = self.grid.t_coordinates[1, ::-1, 0]
            v_sample_power_interpolated = empty_array((num_sources,))
            for ind_source in range(num_sources):
                z = t_map[ind_source, ::-1, :].T
                interpolator = scipy.interpolate.RectBivariateSpline(x, y, z)
                v_sample_power_interpolated[ind_source] = interpolator.ev(
                    v_sample_location[0], v_sample_location[1])

        # print(v_sample_power)
        # print(v_sample_power_interpolated)
        # print(m_four_values)

        return v_sample_power_interpolated