import numpy as np
from numpy import pi
import gtsam
from gtsam import Pose3
from gtsam.symbol_shorthand import X

class PoseGraph:
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        
        # add prior factor to graph
        pose_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [1*pi/180, 1*pi/180, 1*pi/180, 0.3, 0.3, 0.3]))  # std, (1 deg, input must be radian), 0.3
        self.priorFactor = gtsam.PriorFactorPose3(X(0), Pose3(), pose_noise_model)
        self.graph.add(self.priorFactor)
        
        # create initial estimate for zero locaiton
        self.initialEstimate = gtsam.Values()
        self.initialEstimate.insert(X(0), Pose3())
        self.edge_list = []

    def add_factor(self, i, j, pose_from_j_to_i, cov_j_conditional_on_i):
        """ add BetweenFactorPose3 to the graph

        :param pose_from_j_to_i: pose from camera j to camera i
        :param cov_j_conditional_on_i: covariance of camera j conditional on camera i
        :return:
        """
        self.edge_list.append((i,j))
        noise_model = gtsam.noiseModel.Gaussian.Covariance(cov_j_conditional_on_i)
        factor = gtsam.BetweenFactorPose3(X(i), X(j), pose_from_j_to_i, noise_model)
        self.graph.add(factor)
        # add initial estimate if needed:
        if X(j) not in self.initialEstimate.keys():
            pose_from_i_to_world = self.initialEstimate.atPose3(X(i))
            pose_from_j_to_world = pose_from_i_to_world.compose(pose_from_j_to_i)
            self.initialEstimate.insert(X(j), pose_from_j_to_world)

    def optimize(self):
        """
       :return:
           values - gtsam.Values, containing estimations of camera locations
           marginals - gtsam.Marginals, allows extraction of conditional covariance
       """
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initialEstimate)
        values = optimizer.optimize()
        marginals = gtsam.Marginals(self.graph, values)
        return values, marginals
