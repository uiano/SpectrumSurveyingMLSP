import gsim
from gsim.gfigure import GFigure
import matplotlib.pyplot as plt
from map_generators.map_generator import RectangularGrid, np
from map_generators.correlated_shadowing_generator import CorrelatedShadowingGenerator
from samplers.sampler import Sampler
from map_estimators.kriging_estimator import BatchKrigingEstimator, OnlineKrigingEstimator
from route_planners.route_planner import SquareSpiralGridPlanner, MinimumCostPlanner, \
    IndependentUniformPlanner, RandomPlanner, GridPlanner
from simulate_surveying.simulate_surveying_map import simulate_surveying_map, \
    simulate_surveying_montecarlo


class ExperimentSet(gsim.AbstractExperimentSet):
    """###################################################################
    10. 
    ###################################################################
    """

    """
    Monte Carlo comparison of several route planners -> candidate figure.
    """
    def experiment_1001(args):
        ''' region of interest is square with l =100, speed of EM (C), Freq, Po, antenna height
            'grid_length' is length of square region of interest
            ' grid_size is the spacing of grid in ROI
            'v_source_power' is the transmitted power in watt
        '''

        np.random.seed(0)

        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=10,
            num_points_x=30,  #20,  #12
            num_points_y=25,  #18,  #10
            height=20)
        print("grid constructed")

        # 1. Map generator
        num_sources = 2
        source_height = 10
        v_source_power = 10 * np.ones(shape=(num_sources, ))
        m_source_locations = grid.random_points_in_the_area(num_sources)
        m_source_locations[2, :] = source_height
        print(f"sl={m_source_locations}")
        #f_shadowing_covariance = CorrelatedShadowingGenerator.gaussian_correlation(stdev=10)
        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=3,  # 1
            shadowing_correlation_dist=50  #40, #15
        )
        map_generator = CorrelatedShadowingGenerator(
            grid=grid,
            m_source_locations=m_source_locations,
            v_source_power=v_source_power,
            frequency=2.4e9,
            f_shadowing_covariance=f_shadowing_covariance)

        # 2. Sampling
        initial_location = grid.indices_to_point((0, 0))
        dist_between_measurements = 5
        l_route_planners = [
            GridPlanner(grid=grid,
                        initial_location=initial_location,
                        dist_between_measurements=dist_between_measurements),
            SquareSpiralGridPlanner(grid=grid,
                                    initial_location=initial_location,
                                    dist_between_measurements=dist_between_measurements),
            IndependentUniformPlanner(
                grid=grid,
                initial_location=initial_location,
                dist_between_measurements=dist_between_measurements),
            MinimumCostPlanner(
                grid=grid,
                metric="power_variance",
                #metric="service_entropy",
                initial_location=initial_location,
                dist_between_measurements=dist_between_measurements),
            MinimumCostPlanner(
                grid=grid,
                metric="service_entropy",
                initial_location=initial_location,
                dist_between_measurements=dist_between_measurements),
        ]
        sampler = Sampler(
            grid,
            #interpolation_method="avg_nearest",
            interpolation_method="splines",
        )

        # 3. Estimator
        min_service_power = 5
        estimator = BatchKrigingEstimator(
            grid,
            min_service_power=min_service_power,
            f_shadowing_covariance=f_shadowing_covariance,
            f_mean=map_generator.mean_at_channel_and_location)

        # Simulation
            
        F = ExperimentSet.compare_route_planners_montecarlo(num_mc_iterations=50,# 40,
                                                        num_measurements=150, #300,
                                                        min_service_power=min_service_power,
                                                        map_generator=map_generator,
                                                        l_route_planners=l_route_planners,
                                                        sampler=sampler,
                                                        estimator=estimator)

       
        return F

    
    """
    Experiments used to test the online estimator." 
    """
    def experiment_1002(args):
        ''' region of interest is square with l =100, speed of EM (C), Freq, Po, antenna height
            'grid_length' is length of square region of interest
            ' grid_size is the spacing of grid in ROI
            'v_source_power' is the transmitted power in watt
        '''

        # 0. Grid    
        grid = RectangularGrid(
            gridpoint_spacing=10,
            num_points_x=30,  # 12
            num_points_y=25,  # 10
            height=20)
        print("grid constructed")

        # 1. Map generator
        num_sources = 2
        source_height = 10
        v_source_power = 10 * np.ones(shape=(num_sources,))
        m_source_locations = grid.random_points_in_the_area(num_sources)
        m_source_locations[2, :] = source_height
        print(f"sl={m_source_locations}")
        # f_shadowing_covariance = CorrelatedShadowingGenerator.gaussian_correlation(stdev=10)
        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=3.0, shadowing_correlation_dist=200)
        map_generator = CorrelatedShadowingGenerator(
            grid=grid,
            m_source_locations=m_source_locations,
            v_source_power=v_source_power,
            frequency=2.4e9,
            f_shadowing_covariance=f_shadowing_covariance)

        # 2. Sampling
        initial_location = grid.indices_to_point((0, 0))
        dist_between_measurements = 5
        dist_between_waypoints = 50
        l_route_planners = [
            GridPlanner(grid=grid,
                        initial_location=initial_location,
                        dist_between_measurements=dist_between_measurements),

            SquareSpiralGridPlanner(grid=grid,
                                    initial_location=initial_location,
                                    dist_between_measurements=dist_between_measurements),
            RandomPlanner(grid=grid,
                          initial_location=initial_location,
                          dist_between_measurements=dist_between_measurements,
                          dist_between_waypoints=dist_between_waypoints),
            IndependentUniformPlanner(
                grid=grid,
                initial_location=initial_location,
                dist_between_measurements=dist_between_measurements),
            MinimumCostPlanner(grid=grid,
                               # metric="power_variance",
                               metric="service_entropy",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements),
        ]
        sampler = Sampler(grid)

        # 3. Estimator
        min_service_power = 5
        estimators = [BatchKrigingEstimator(
            grid,
            min_service_power=min_service_power,
            f_shadowing_covariance=f_shadowing_covariance,
            f_mean=map_generator.mean_at_channel_and_location),
            OnlineKrigingEstimator(
                grid,
                min_service_power=min_service_power,
                f_shadowing_covariance=f_shadowing_covariance,
                f_mean=map_generator.mean_at_channel_and_location)]

        # Simulation
        def experiment_1a():
            """Plot maps after every measurement."""
            global b_do_not_plot_power_maps
            b_do_not_plot_power_maps = False  # True
            route_planner = l_route_planners[4]
            route_planner.debug_level = 0
            map = map_generator.generate_map()

            d_map_estimate, m_all_measurement_loc, d_metrics = \
                simulate_surveying_map(
                    map,
                    num_measurements=150,
                    min_service_power=min_service_power,
                    map_generator=map_generator,
                    route_planner=route_planner,
                    sampler=sampler,
                    estimator=estimators[1],
                    num_measurements_to_plot=10)

            ld_metrics = [(route_planner, d_metrics)]
            return map_generator.plot_metrics(ld_metrics)

        def experiment_1b():
            """Compares the route planners for a single realization of the map. """
            map = map_generator.generate_map()

            ld_metrics = []
            for route_planner in l_route_planners[3:5]:
                estimators[1].reset()
                d_map_estimate, m_all_measurement_loc, d_metrics = \
                    simulate_surveying_map(
                        map,
                        num_measurements=80,
                        min_service_power=min_service_power,
                        map_generator=map_generator,
                        route_planner=route_planner,
                        sampler=sampler,
                        estimator=estimators[1])

                ld_metrics.append((route_planner, d_metrics))

                # # Plot maps
                map_generator.compare_maps(map,
                                           d_map_estimate,
                                           min_service_power=min_service_power,
                                           m_measurement_loc=m_all_measurement_loc)

            map_generator.plot_metrics(ld_metrics)

        def experiment_1d():
            "Monte Carlo with Online Estimator and Batch Estimator"

            ld_metrics = []
            ld_planners = l_route_planners[3 : 5]
            for ind_estimator in range(len(estimators)):

                for route_planner in ld_planners:
                    d_metrics = simulate_surveying_montecarlo(
                        num_mc_iterations=5,
                        num_measurements=20,
                        min_service_power=min_service_power,
                        map_generator=map_generator,
                        route_planner=route_planner,
                        sampler=sampler,
                        estimator=estimators[ind_estimator])
                    ld_metrics.append((route_planner, d_metrics))

            map_generator.plot_metrics(ld_metrics)

        def experiment_1e():
            """Plot maps after every measurement. for both estimators"""
            global b_do_not_plot_power_maps
            b_do_not_plot_power_maps = False  # True
            route_planner = l_route_planners[4]
            route_planner.debug_level = 0
            map = map_generator.generate_map()

            for ind_estimator in range(len(estimators)):
                d_map_estimate, m_all_measurement_loc, d_metrics = \
                    simulate_surveying_map(
                        map,
                        num_measurements=50,
                        min_service_power=min_service_power,
                        map_generator=map_generator,
                        route_planner=route_planner,
                        sampler=sampler,
                        estimator=estimators[ind_estimator],
                        num_measurements_to_plot=10)

                ld_metrics = [(route_planner, d_metrics)]
                map_generator.plot_metrics(ld_metrics)

        def experiment_1f():
            """Compares the route planners for both estimators of one map realization. """
            map = map_generator.generate_map()

            ld_metrics = []
            for route_planner in l_route_planners[3:5]:
                for ind_estimator in range(len(estimators)):
                    estimators[1].reset()
                    d_map_estimate, m_all_measurement_loc, d_metrics = \
                        simulate_surveying_map(
                            map,
                            num_measurements=10,
                            min_service_power=min_service_power,
                            map_generator=map_generator,
                            route_planner=route_planner,
                            sampler=sampler,
                            estimator=estimators[ind_estimator])

                    ld_metrics.append((route_planner, d_metrics))


        F = experiment_1a()
        #plt.show()
        return F
        #experiment_1b()
        #experiment_1c()
        #experiment_1d()
        #experiment_1e()
        #experiment_1f()



    """
    Experiment used to plot the true and estimated maps, true and estimated service maps, and uncertainty maps ." 
    """
    def experiment_1003(args):
        ''' region of interest is square with l =100, speed of EM (C), Freq, Po, antenna height
            'grid_length' is length of square region of interest
            ' grid_size is the spacing of grid in ROI
            'v_source_power' is the transmitted power in watt
        '''

        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=10,
            num_points_x=30,  # 12
            num_points_y=25,  # 10
            height=20)
        print("grid constructed")

        # 1. Map generator
        num_sources = 2
        source_height = 10
        v_source_power = 10 * np.ones(shape=(num_sources,))
        m_source_locations = grid.random_points_in_the_area(num_sources)
        m_source_locations[2, :] = source_height
        print(f"sl={m_source_locations}")
        # f_shadowing_covariance = CorrelatedShadowingGenerator.gaussian_correlation(stdev=10)
        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=3.0, shadowing_correlation_dist=200)
        map_generator = CorrelatedShadowingGenerator(
            grid=grid,
            m_source_locations=m_source_locations,
            v_source_power=v_source_power,
            frequency=2.4e9,
            f_shadowing_covariance=f_shadowing_covariance)

        # 2. Sampling
        initial_location = grid.indices_to_point((0, 0))
        dist_between_measurements = 5
        dist_between_waypoints = 50
        l_route_planners = [
            GridPlanner(grid=grid,
                        initial_location=initial_location,
                        dist_between_measurements=dist_between_measurements),

            SquareSpiralGridPlanner(grid=grid,
                                    initial_location=initial_location,
                                    dist_between_measurements=dist_between_measurements),
            RandomPlanner(grid=grid,
                          initial_location=initial_location,
                          dist_between_measurements=dist_between_measurements,
                          dist_between_waypoints=dist_between_waypoints),
            IndependentUniformPlanner(
                grid=grid,
                initial_location=initial_location,
                dist_between_measurements=dist_between_measurements),
            MinimumCostPlanner(grid=grid,
                               # metric="power_variance",
                               metric="service_entropy",
                               initial_location=initial_location,
                               dist_between_measurements=dist_between_measurements),
        ]
        sampler = Sampler(grid)

        # 3. Estimator
        min_service_power = 5
        estimators = [BatchKrigingEstimator(
            grid,
            min_service_power=min_service_power,
            f_shadowing_covariance=f_shadowing_covariance,
            f_mean=map_generator.mean_at_channel_and_location),
            OnlineKrigingEstimator(
                grid,
                min_service_power=min_service_power,
                f_shadowing_covariance=f_shadowing_covariance,
                f_mean=map_generator.mean_at_channel_and_location)]

        #
        """Plot maps after every measurement."""
        global b_do_not_plot_power_maps
        b_do_not_plot_power_maps = False  # True
        route_planner = l_route_planners[4]
        route_planner.debug_level = 0
        map = map_generator.generate_map()

        d_map_estimate, m_all_measurement_loc, d_metrics = \
            simulate_surveying_map(
                map,
                num_measurements=190,
                min_service_power=min_service_power,
                map_generator=map_generator,
                route_planner=route_planner,
                sampler=sampler,
                estimator=estimators[1],
                num_measurements_to_plot=10)

        ld_metrics = [(route_planner, d_metrics)]
        plt.show()
        return map_generator.plot_metrics(ld_metrics)
    

    """
    MC experiment that compares probability of service error and uncertainty for multiple route planners. Online estimator.
    """
    def experiment_1004(args):
        ''' region of interest is square with l =100, speed of EM (C), Freq, Po, antenna height
            'grid_length' is length of square region of interest
            ' grid_size is the spacing of grid in ROI
            'v_source_power' is the transmitted power in watt
        '''

        np.random.seed(0)

        # 0. Grid
        grid = RectangularGrid(
            gridpoint_spacing=10,
            num_points_x=30,  #20,  #12
            num_points_y=25,  #18,  #10
            height=20)
        print("grid constructed")

        # 1. Map generator
        num_sources = 2
        source_height = 10
        v_source_power = 10 * np.ones(shape=(num_sources, ))
        m_source_locations = grid.random_points_in_the_area(num_sources)
        m_source_locations[2, :] = source_height
        print(f"sl={m_source_locations}")
        #f_shadowing_covariance = CorrelatedShadowingGenerator.gaussian_correlation(stdev=10)
        f_shadowing_covariance = CorrelatedShadowingGenerator.gudmundson_correlation(
            shadowing_std=3,  # 1
            shadowing_correlation_dist=50  #40, #15
        )
        map_generator = CorrelatedShadowingGenerator(
            grid=grid,
            m_source_locations=m_source_locations,
            v_source_power=v_source_power,
            frequency=2.4e9,
            f_shadowing_covariance=f_shadowing_covariance)

        # 2. Sampling
        initial_location = grid.indices_to_point((0, 0))
        dist_between_measurements = 5
        dist_between_waypoints = 50
        l_route_planners = [
            GridPlanner(grid=grid,
                        initial_location=initial_location,
                        dist_between_measurements=dist_between_measurements),
            SquareSpiralGridPlanner(grid=grid,
                                    initial_location=initial_location,
                                    dist_between_measurements=dist_between_measurements),
            IndependentUniformPlanner(
                grid=grid,
                initial_location=initial_location,
                dist_between_measurements=dist_between_measurements),
            MinimumCostPlanner(
                grid=grid,
                metric="service_entropy",
                initial_location=initial_location,
                dist_between_measurements=dist_between_measurements),
        ]
        sampler = Sampler(
            grid,
            #interpolation_method="avg_nearest",
            interpolation_method="splines",
        )

        # 3. Estimator
        min_service_power = 5
        estimator = OnlineKrigingEstimator(
                grid,
                min_service_power=min_service_power,
                f_shadowing_covariance=f_shadowing_covariance,
                f_mean=map_generator.mean_at_channel_and_location)

        # Simulation
            
        F = ExperimentSet.compare_route_planners_montecarlo(num_mc_iterations=1000,# 40,
                                                        num_measurements=120, #300,
                                                        min_service_power=min_service_power,
                                                        map_generator=map_generator,
                                                        l_route_planners=l_route_planners,
                                                        sampler=sampler,
                                                        estimator=estimator)

       
        return F

    def compare_route_planners_montecarlo(num_mc_iterations, num_measurements,
                                          min_service_power, map_generator,
                                          l_route_planners, sampler,
                                          estimator):
        "Monte Carlo"

        ld_metrics = []
        for route_planner in l_route_planners:

            d_metrics = simulate_surveying_montecarlo(
                num_mc_iterations=num_mc_iterations,
                num_measurements=num_measurements,
                min_service_power=min_service_power,
                map_generator=map_generator,
                route_planner=route_planner,
                sampler=sampler,
                estimator=estimator)
            ld_metrics.append((route_planner.name_on_figs, d_metrics))

        F = map_generator.plot_metrics(ld_metrics)

        return F
