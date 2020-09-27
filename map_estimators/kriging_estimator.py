from utilities import empty_array
import scipy
import numpy as np
import abc
from map_estimators.map_estimator import MapEstimator
from IPython.core.debugger import set_trace


class KrigingEstimator(MapEstimator):
    """
    `metric`: indicates the estimated metric:
      - "power": metric = received power
      - "service": metric = posterior probability of received power >= `self.min_service_power`.
    `f_shadowing_covariance` is a function of a distance argument that returns
    the covariance between the map values at two points separated by
    that distance.
    `f_shadowing_covariance`: if None, map assumed zero mean. Else, this is a
    function of distance.
    """

    def __init__(
            self,
            grid,
            # metric="power",  # values: "power", "service"
            f_shadowing_covariance=None,
            f_mean=None,            
            dumping_factor=0.001,
            **kwargs
    ):

        self.grid = grid
        # self.metric = metric
        self.f_shadowing_covariance = f_shadowing_covariance
        self.f_mean = f_mean
        #self.min_service_power = min_service_power
        self.dumping_factor = dumping_factor

        self.m_all_measurements = None  # num_sources x num_measurements_so_far
        self.m_all_measurement_loc = np.zeros(
            (3, 0))  # 3 x num_measurements_so_far

        super().__init__(**kwargs)
        
    def reset(self):
        """ Clear buffers to start estimating again."""
        self.m_all_measurements = None  # num_sources x num_measurements_so_far
        self.m_all_measurement_loc = np.zeros(
            (3, 0))  # 3 x num_measurements_so_far

    @abc.abstractmethod
    def store_measurement(self, v_measurement_loc, v_measurement):
        pass

    @abc.abstractmethod
    def estimate_power_one_channel(self, ind_channel):
        pass

    def estimate(self, v_measurement_loc, v_measurement):
        """Args:
            `v_measurement` : number source- length vector whose i-th
            entry denotes the received power at the i-th
            v_measurement_loc location transmitted by i-th power
            source.
           Returns:
           -`d_map_estimate`: dictionary whose fields are num_sources
           x self.grid.num_points_y x self.grid.num_points_x
           tensors. They are:
           "t_power_map_estimate" :  tensor whose (i,j,k)-th entry is the
           estimated power of the i-th channel at grid point (j,k).
           "t_power_map_norm_variance" : Contains the variance a
           posteriori normalized to the interval (0,1).
           "t_service_map_estimate": [only if self.min_service_power
           is not None]. Each entry contains the posterior probability
           that the power is above `self.min_service_power`.
           "t_service_map_entropy": [only if self.min_service_power
           is not None] Entropy of the service indicator.
        """
        num_sources = len(v_measurement)
        # get the measurements
        self.store_measurement(v_measurement_loc, v_measurement)

        # Now estimate metric per channel
        t_power_map_est = empty_array(
            (num_sources, self.grid.num_points_y, self.grid.num_points_x))
        t_power_map_var = empty_array(
            (num_sources, self.grid.num_points_y, self.grid.num_points_x))
        if self.min_service_power is not None:
            t_service_map_est = empty_array(
                (num_sources, self.grid.num_points_y, self.grid.num_points_x))
            t_service_map_ent = empty_array(
                (num_sources, self.grid.num_points_y, self.grid.num_points_x))
        for ind_src in range(num_sources):
            t_power_map_est[ind_src, :, :], t_power_map_var[ind_src, :, :] = \
                self.estimate_power_one_channel(ind_src)
            if self.min_service_power is not None:
                t_service_map_est[ind_src, :, :], t_service_map_ent[ind_src, :, :] = \
                    self.estimate_service_one_channel(t_power_map_est[ind_src, :, :], t_power_map_var[ind_src, :, :])
                # self.estimate_service_one_channel(self.m_all_measurements[ind_src, :], ind_src)

        # m_entropy = np.mean(t_power_map_var, axis=0)

        d_map_estimate = {
            "t_power_map_estimate":
                t_power_map_est,
            "t_power_map_norm_variance":
                t_power_map_var / self.f_shadowing_covariance(0)  # ensure in (0,1)
        }
        if self.min_service_power:
            d_map_estimate["t_service_map_estimate"] = t_service_map_est
            d_map_estimate["t_service_map_entropy"] = t_service_map_ent

        return d_map_estimate


class BatchKrigingEstimator(KrigingEstimator):

    name_on_figs = "(KrigingEstimator)"
    def __init__(
            self,
            *args,
            **kwargs):
        super(BatchKrigingEstimator, self).__init__(*args,
                                                    **kwargs)
        self.reset()

    def reset(self):
        super().reset()

    def store_measurement(self, v_measurement_loc, v_measurement):
        """Args:
            `v_measurement` : number source- length vector whose i-th
            entry denotes the received power at the i-th
            v_measurement_loc location transmitted by i-th power
            source.
           Returns:
           -`m_all_measurement_loc`: buffered measurement locations whose
           j-th column represents (x, y, z) coordinate of measurement location
           "m_all_measurements" :  buffered measurements whose (i, j)-th
           entry denotes the received power at j-th m_all_measurement_loc
           transmitted by i-th power source.
        """

        num_sources = len(v_measurement)
        if self.m_all_measurements is None:
            self.m_all_measurements = np.reshape(v_measurement,
                                                 (num_sources, 1))
            self.m_all_measurement_loc = np.reshape(v_measurement_loc,
                                                    (3, 1))
        else:
            self.m_all_measurements = np.hstack(
                (self.m_all_measurements,
                 np.reshape(v_measurement, (num_sources, 1))))
            self.m_all_measurement_loc = np.hstack(
                (self.m_all_measurement_loc, np.reshape(v_measurement_loc,
                                                        (3, 1))))

    def estimate_power_one_channel(self, ind_channel):

        num_grid_points = self.grid.t_coordinates.shape[
                              1] * self.grid.t_coordinates.shape[2]

        v_measurements = self.m_all_measurements[ind_channel, :]
        num_measurements = v_measurements.shape[0]

        if not self.f_mean:
            # average mean = 0
            mean_pobs = np.zeros(shape=(
                num_measurements,
                1))  # mean of power at the locations of the observations

            # mean of power at the locations without observations
            mean_pnobs = np.zeros(shape=(num_grid_points, 1))
        else:
            # mean of power at the locations with observations

            mean_pobs = np.reshape(
                self.f_mean(ind_channel, self.m_all_measurement_loc),
                (num_measurements, 1))

            # mean of power at the locations with observations
            m_all_grid_points = np.reshape(self.grid.t_coordinates, (1, 3, -1))
            mean_pnobs = np.zeros(shape=(num_grid_points, 1))

            for ind_num_grid_points in range(num_grid_points):
                mean_pnobs[ind_num_grid_points] = \
                    self.f_mean(ind_channel, np.reshape(
                        (m_all_grid_points[0, :, ind_num_grid_points]), (3, 1)))

        m_location = self.m_all_measurement_loc

        # First obtain the partitioned covariance matrix (theorem 10.2 of Kay, page 324)
        cov_obs_obs = np.zeros(shape=(num_measurements, num_measurements))
        for ind_measurement in range(num_measurements):
            v_distance = np.linalg.norm(
                m_location - np.expand_dims(m_location[:, ind_measurement], 1),
                ord=2,
                axis=0)
            cov_obs_obs[ind_measurement, :] = self.f_shadowing_covariance(
                v_distance)

        cov_obs_nobs = np.zeros(shape=(num_measurements, num_grid_points))
        m_location_transposed = m_location.T
        for ind_measurement in range(0, num_measurements):
            cov_obs_nobs[ind_measurement, :] = self.f_shadowing_covariance(
                self.grid.get_distance_to_grid_points(
                    m_location_transposed[ind_measurement, :])).reshape(
                1, num_grid_points)

        cov_nobs_obs = cov_obs_nobs.T

        cov_nobs_nobs = self.f_shadowing_covariance(
            self.grid.get_distance_matrix())

        # Dumping factor
        cov_obs_obs += self.dumping_factor * np.eye(num_measurements)

        v_coef = (np.linalg.inv(cov_obs_obs) @ (v_measurements.reshape(
            (num_measurements, 1)) - mean_pobs))

        # MMSE estimator for power at each spatial (for here at every grid point) P(y/x)
        m_estimated_pow_of_source = np.reshape(
            (mean_pnobs + cov_nobs_obs @ v_coef),
            (self.grid.num_points_y, self.grid.num_points_x))

        m_estimated_cov = cov_nobs_nobs - np.matmul(
            cov_nobs_obs,
            (np.matmul(np.linalg.inv(cov_obs_obs), cov_obs_nobs)))

        # (i,j)-th entry contains the posterior variance of the power at the (i,j)-th grid point
        m_variance = np.reshape(
            (m_estimated_cov.diagonal()),
            (self.grid.num_points_y, self.grid.num_points_x))

        if (m_variance < 0).any():
            set_trace()
        # m_uncertainty = np.zeros((self.grid.num_points_x, self.grid.num_points_y))

        return m_estimated_pow_of_source, m_variance


class OnlineKrigingEstimator(KrigingEstimator):
    def __init__(
            self,
            *args,
            **kwargs):
        super(OnlineKrigingEstimator, self).__init__(*args,
                                                     **kwargs)
        self.reset()

    def reset(self):

        super().reset()
        self.prev_mean = []
        self.prev_cov = []
        self.count_ind_channel = 0

    def store_measurement(self, v_measurement_loc, v_measurement):
        """Args:
            `v_measurement` : number source- length vector whose i-th
            entry denotes the received power at the i-th
            v_measurement_loc location transmitted by i-th power
            source.
           Returns:
           -`m_all_measurement_loc`: measurement location whose
           j-th column represents (x, y, z) coordinate of measurement location
           "m_all_measurements" :  measurements whose j-th
           entry denotes the received power at j-th m_all_measurement_loc
           transmitted by i-th power source.
        """
        num_sources = len(v_measurement)

        self.m_all_measurements = np.reshape(v_measurement,
                                             (num_sources, 1))
        self.m_all_measurement_loc = np.reshape(v_measurement_loc,
                                                (3, 1))


    def estimate_power_one_channel(self, ind_channel):
        num_grid_points = self.grid.t_coordinates.shape[
                              1] * self.grid.t_coordinates.shape[2]

        v_measurements = self.m_all_measurements[ind_channel, :]
        num_measurements = v_measurements.shape[0]

        num_sources = len(self.m_all_measurements)

        if not self.f_mean:
            # average mean = 0
            mean_pobs = np.zeros(shape=(
                num_measurements, 1))  # mean of power at the locations of the observations

            # mean of power at the locations without observations
            mean_pnobs = np.zeros(shape=(num_grid_points, 1))
        else:
            # mean of power at the locations with observations
            mean_pobs = self.f_mean(ind_channel, self.m_all_measurement_loc)

            # mean of power at the locations with observations
            m_all_grid_points = np.reshape(self.grid.t_coordinates, (1, 3, -1))
            mean_pnobs = np.zeros(shape=(num_grid_points, 1))

            for ind_num_grid_points in range(num_grid_points):
                mean_pnobs[ind_num_grid_points] = \
                    self.f_mean(ind_channel, np.reshape((m_all_grid_points[0, :, ind_num_grid_points]), (3, 1)))

        v_location = self.m_all_measurement_loc.T

        # First obtain the partitioned covariance matrix (theorem 10.2 of Kay, page 324)

        cov_nobs_nobs = self.f_shadowing_covariance(
            self.grid.get_distance_matrix())

        cov_obs_nobs = self.f_shadowing_covariance(self.grid.get_distance_to_grid_points(
            v_location)).reshape(1, num_grid_points)

        # Computing vector a_t and scalars b_t and lambda_t in the draft
        v_at = np.linalg.inv(cov_nobs_nobs) @ cov_obs_nobs.reshape(-1, 1)
        b_t = mean_pobs - (cov_obs_nobs @ np.linalg.inv(cov_nobs_nobs)) @ mean_pnobs
        lambda_t = self.f_shadowing_covariance(0) - (
                cov_obs_nobs @ (np.linalg.inv(cov_nobs_nobs) @ cov_obs_nobs.reshape(-1, 1)))

        if self.count_ind_channel < num_sources:
            # for initial at time t = 0
            self.prev_cov.append(cov_nobs_nobs)
            self.prev_mean.append(mean_pnobs)
            self.count_ind_channel += 1

        # MMSE estimator for power at each spatial (for here at every grid point)
        m_estimated_cov = np.linalg.inv(np.linalg.inv(self.prev_cov[ind_channel]) +
                                        (v_at * v_at.reshape(1, -1)) / lambda_t)

        v_estimated_mean_power = m_estimated_cov @ (((v_measurements - b_t)
                                                     / lambda_t) * v_at +
                                                    np.linalg.inv(self.prev_cov[ind_channel]) @ self.prev_mean[
                                                        ind_channel])

        # Store the current mean_power and covariance for next measurement
        self.prev_cov[ind_channel] = m_estimated_cov
        self.prev_mean[ind_channel] = v_estimated_mean_power

        m_estimated_mean_power = np.reshape(v_estimated_mean_power,
                                            (self.grid.num_points_y, self.grid.num_points_x))

        # (i,j)-th entry contains the posterior variance of the power at the (i,j)-th grid point
        m_variance = np.reshape(
            (m_estimated_cov.diagonal()),
            (self.grid.num_points_y, self.grid.num_points_x))

        if (m_variance < 0).any():
            # set_trace()
            m_variance = np.copy(m_variance)
            m_variance[m_variance < 0] = 1e-6

        # m_uncertainty = np.zeros((self.grid.num_points_x, self.grid.num_points_y))

        return m_estimated_mean_power, m_variance
