# ID-Fits
# Copyright (c) 2015 Institut National de l'Audiovisuel, INA, All rights reserved.
# 
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3.0 of the License, or (at your option) any later version.
# 
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this library.


import numpy as np
from nose.tools import *

from landmarks_datasets import BioId
from PythonWrapper.tree_based_regression import ForestRegressorTraining as CppForestRegressorTraining
from PythonWrapper.tree_based_regression import AlignmentMethod, AlignmentMethodTraining

import pyximport
install = pyximport.install()
from tree_based_regression_training import CyForestRegressorTraining
pyximport.uninstall(*install)


path = "models/alignment/"

class TestRandomForestTraining:
    @classmethod
    def setup_class(cls):
        dataset = BioId()
        cls.images = dataset.loadImages()
        cls.ground_truth = dataset.loadGroundTruth()

    def setup(self):
        self.random_forest_training = CppForestRegressorTraining(5, 5)
        self.reference = CyForestRegressorTraining(5, 5)
        self.training_images = self.images[:500]
        self.training_landmarks = self.ground_truth[:500]
        self.mean_shape = np.mean(self.training_landmarks, axis=0)
        self.delta_s = self.training_landmarks - self.mean_shape
        self.L = self.mean_shape.shape[0]

    """
    def test_transformation_computation(self):
        x_bar = self.mean_shape
        x_i = self.training_landmarks[0]

        R = self.random_forest_training.computeTransformation(x_i, x_bar)
        s = 1/np.linalg.norm(R[0,:2])
        t = np.mean(x_bar, axis=0).reshape(2,1) - ((s**2)*R.T).dot(np.mean(x_i, axis=0).reshape(2,1))
        square_error = np.sum([(x_bar[l].reshape(2,1) - ((s**2)*(R.T).dot(x_i[l].reshape(2,1)) + t))**2 for l in range(self.L)])
       
        A = np.zeros((2*self.L, 4), dtype=np.float64)

        for l in range(self.L):
            A[2*l] = np.asarray([x_i[l,0], -x_i[l,1], 1, 0])
            A[2*l+1] = np.asarray([x_i[l,1], x_i[l,0], 0, 1])
        result, square_error_ref = np.linalg.lstsq(A, x_bar.reshape(2*self.L, 1))[:2]
        
        s_ref = np.linalg.norm(result[:2])
        R_ref = np.array([result[0], -result[1], result[1], result[0]]).reshape(2,2) / s_ref
        t_ref = np.array([result[2], result[3]]).reshape(2,1)

        assert_almost_equal(square_error, square_error_ref[0])
        assert_almost_equal(s, s_ref)
        assert_true((abs(s*R.T-R_ref) < 1e-5).all())
        assert_true((abs(t-t_ref) < 1e-5).all())
    """

    def test_generate_random_features(self):
        random_features = self.random_forest_training.generateRandomFeatures(0.25)
        assert_equal(len(random_features), 500)

    """
    def test_train_tree(self):
        self.random_forest_training.trainTree(0, self.mean_shape, self.training_landmarks[:,:], self.delta_s[:,:], self.training_images, 25.0)
    """

    """
    def test_train(self):
        random_forest = self.random_forest_training.train(0, self.training_landmarks[:,:], self.delta_s[:,:], self.training_images, 25.0)
        assert_equal(len(random_forest), 5)
    """

class TestAlignmentMethod:
    @classmethod
    def setup_class(cls):
        dataset = BioId()
        cls.images = dataset.loadImages()
        cls.ground_truth = dataset.loadGroundTruth()
        cls.bounding_boxes = dataset.loadBoundingBoxes()

    def setup(self):
        self.test_images = self.images[900:]
        self.test_landmarks = self.ground_truth[900:]
        self.test_bounding_boxes = self.bounding_boxes[900:]

    def test_local_trees_regressors_align(self):
        local_trees_regressor = AlignmentMethod(path + "local_trees_regression_model_bioid_small.txt")
        for i in range(len(self.test_images)):
            local_trees_regressor.align(self.test_bounding_boxes[i], self.test_images[i])

    def test_lbf_regressors_align(self):
        lbf_regressor = AlignmentMethod(path + "lbf_regression_model_bioid_small.txt", method="lbf")
        for i in range(len(self.test_images)):
            lbf_regressor.align(self.test_bounding_boxes[i], self.test_images[i])


class TestAlignmentMethodTraining:
    @classmethod
    def setup_class(cls):
        dataset = BioId()
        cls.images = dataset.loadImages()
        cls.ground_truth = dataset.loadGroundTruth()
        cls.bounding_boxes = dataset.loadBoundingBoxes()

    def setup(self):
        self.training_images = self.images[:100]
        self.training_landmarks = self.ground_truth[:100]
        self.mean_shape = np.mean(self.training_landmarks, axis=0)
        self.s0 = np.vstack(tuple(self.mean_shape[np.newaxis, ...] for i in range(self.training_landmarks.shape[0])))

    def test_local_trees_regressors_train(self):
        local_regressor_training = AlignmentMethodTraining(1, 3, 300/20, 5, sampled_random_features_number=100)
        local_regressor_training.train(self.s0, self.training_landmarks, self.bounding_boxes, self.training_images, (0,1), "local_trees_regression_model_bioid_small.txt")

    def test_lbf_regressors_train(self):
        lbf_training = AlignmentMethodTraining(1, 3, 300/20, 5, method="lbf", sampled_random_features_number=100)
        lbf_training.train(self.s0, self.training_landmarks, self.bounding_boxes, self.training_images, (36,45), "lbf_regression_model_bioid_small.txt")

    """
    def test_local_trees_regressors_train_strange_criteria(self):
        local_regressor_training = AlignmentMethodTraining(1, 3, 300/20, 5, sampled_random_features_number=100)
        local_regressor_training.train(self.s0, self.training_landmarks, self.bounding_boxes, self.training_images, (0,1), "local_trees_regression_model_bioid_small.txt", "test")
    """

    def test_local_trees_regressors_train_small_dimensions(self):
        s0 = self.s0[:5,:2,:]
        training_landmarks = self.training_landmarks[:5,:2,:]
        bounding_boxes = self.bounding_boxes[:5]
        training_images = self.training_images[:5]

        local_regressor_training = AlignmentMethodTraining(1, 1, 3, 2, sampled_random_features_number=100)
        local_regressor_training.train(s0, training_landmarks, bounding_boxes, training_images, (0,1), "local_trees_regression_model_bioid_very_small.txt")

    def test_lbf_regressors_train_small_dimensions(self):
        s0 = self.s0[:5,:2,:]
        training_landmarks = self.training_landmarks[:5,:2,:]
        bounding_boxes = self.bounding_boxes[:5]
        training_images = self.training_images[:5]

        lbf_training = AlignmentMethodTraining(1, 1, 3, 2, method="lbf", sampled_random_features_number=100)
        lbf_training.train(s0, training_landmarks, bounding_boxes, training_images, (36,45), "lbf_regression_model_bioid_very_small.txt")
