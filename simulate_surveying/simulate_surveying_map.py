from utilities import np, empty_array
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt



def simulate_surveying_map(
        map,
        num_measurements=1,
        min_service_power=None,
        map_generator=None,
        route_planner=None,
        sampler=None,
        estimator=None,
        num_measurements_to_plot=None  # plot maps every this # of meas.
):
    # Metric placeholders
    v_mse = empty_array((num_measurements,))
    v_norm_avg_power_variance = empty_array((num_measurements,))
    if min_service_power:
        # Check external error in parameter setting
        assert min_service_power == estimator.min_service_power
        v_service_error_rate = empty_array((num_measurements,))
        v_avg_service_entropy = empty_array((num_measurements,))

    m_all_measurement_loc = np.zeros((3, 0))  # 3 x num_measurements_so_far

    d_map_estimate = None
    for ind_measurement in range(num_measurements):
        if ind_measurement % 10 == 0:
            print(f"ind_measurement={ind_measurement}")

        # Measurement collection
        v_measurement_location = route_planner.next_measurement_location(
            d_map_estimate)
        m_all_measurement_loc = np.hstack(
            (m_all_measurement_loc, np.reshape(v_measurement_location,
                                               (3, 1))))
        # print(f"v_measurement_location={v_measurement_location}")
        # 1D vector of length num_sources
        v_measurement = sampler.sample_map(map, v_measurement_location)

        # Estimation
        d_map_estimate = estimator.estimate(v_measurement_location, v_measurement)

        # Performance evaluation
        v_mse[ind_measurement] = np.linalg.norm(
            map - d_map_estimate["t_power_map_estimate"]) ** 2 / map.size

        v_norm_avg_power_variance[ind_measurement] = np.mean(
            d_map_estimate["t_power_map_norm_variance"])

        # Prob. error
        if min_service_power:
            assert "t_service_map_estimate" in d_map_estimate.keys()
            v_service_error_rate[ind_measurement] = np.sum(
                (d_map_estimate["t_service_map_estimate"] > 0.5) !=
                (map > estimator.min_service_power)) / map.size
            v_avg_service_entropy[ind_measurement] = np.mean(
                d_map_estimate["t_service_map_entropy"])

        if num_measurements_to_plot and (ind_measurement % num_measurements_to_plot == 0):
            map_generator.compare_maps(map,
                                       d_map_estimate,
                                       min_service_power=min_service_power,
                                       m_measurement_loc=m_all_measurement_loc)
            plt.show()

    d_metrics = {
        "v_mse": v_mse,
        "v_norm_avg_power_variance": v_norm_avg_power_variance
    }
    if min_service_power:
        d_metrics["v_service_error_rate"] = v_service_error_rate
        d_metrics["v_avg_service_entropy"] = v_avg_service_entropy

    return d_map_estimate, m_all_measurement_loc, d_metrics


def simulate_surveying_montecarlo(num_mc_iterations=None,
                                  map_generator=None,
                                  route_planner=None,
                                  estimator=None,
                                  parallel_proc=False,
                                  **kwargs):
    """Returns a dictionary similar to the one returned by simulate_surveying_map.
    """
    assert num_mc_iterations
    assert map_generator
    assert estimator

    def simulate_one_run(ind_run):
        map = map_generator.generate_map()
        estimator.reset()
        route_planner.reset()
        _, _, d_metrics = simulate_surveying_map(map,
                                                 estimator=estimator,
                                                 route_planner=route_planner,
                                                 **kwargs)
        v_map_norm_one_run = np.linalg.norm(map) ** 2
        return d_metrics, v_map_norm_one_run

    if parallel_proc:
        num_cores = int(multiprocessing.cpu_count() / 2)
        all_metrics_and_norms = Parallel(n_jobs=num_cores)(delayed(simulate_one_run)(i)
                                                           for i in range(num_mc_iterations))
        all_metrics_and_norms_ar = np.array(all_metrics_and_norms)
        ld_metrics = all_metrics_and_norms_ar[:, 0]
        v_map_norm = all_metrics_and_norms_ar[:, 1]
    else:
        ld_metrics = []
        v_map_norm = empty_array((num_mc_iterations,))  # Used for normalization
        for ind_mc_iteration in range(num_mc_iterations):
            d_metrics, v_map_norm_ind = simulate_one_run(ind_mc_iteration)
            ld_metrics.append(d_metrics)
            v_map_norm[ind_mc_iteration] = v_map_norm_ind

    # Average the metrics
    def avg_metric(str_metric):
        l_vals = [d_metrics[str_metric] for d_metrics in ld_metrics]
        m_vals = np.array(l_vals)
        return np.mean(m_vals, 0)

    map = map_generator.generate_map()
    d_metrics = {"v_mse": avg_metric("v_mse"),
                 "v_nmse": avg_metric("v_mse") / (np.mean(v_map_norm) / map.size),
                 "v_norm_avg_power_variance": avg_metric("v_norm_avg_power_variance")}

    if "v_service_error_rate" in ld_metrics[0].keys():
        d_metrics["v_service_error_rate"] = avg_metric("v_service_error_rate")
        d_metrics["v_avg_service_entropy"] = avg_metric(
            "v_avg_service_entropy")

    return d_metrics
