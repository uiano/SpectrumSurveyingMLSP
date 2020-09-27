import numpy as np
import abc
from scipy.stats import norm

class MapEstimator():
    def __init__(
            self,
            min_service_power=None,  # Minimum rx power to consider that there service at a point
    ):

        self.min_service_power = min_service_power

    @abc.abstractmethod
    def estimate(self, v_measurement_loc, v_measurement):
        pass

    def estimate_service_one_channel(self, m_mean_power, m_variance):
        """Returns:
        `m_service`: num_point_y x num_points_x matrix where the
        (i,j)-th entry is the probability that the power at grid point
        (i,j) is greater than `self.min_service_power`.
        """
        def entropy_bernoulli(m_service):
            # Avoid log2(0):
            m_service_copy = np.copy(m_service)
            mb_zero_entropy = np.logical_or((m_service_copy == 0),
                                            (m_service_copy == 1))
            m_service_copy[mb_zero_entropy] = .5  # dummy value

            m_entropy = -(1 - m_service_copy) * np.log2(1 - m_service_copy) - (
                m_service_copy) * np.log2(m_service_copy)
            m_entropy[mb_zero_entropy] = 0

            return m_entropy

        m_service = 1 - norm.cdf(self.min_service_power, m_mean_power,
                                 np.sqrt(m_variance))

        m_entropy = entropy_bernoulli(m_service)

        return m_service, m_entropy
