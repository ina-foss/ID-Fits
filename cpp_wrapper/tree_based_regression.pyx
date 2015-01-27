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


from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair

import numpy as np
cimport numpy as np

from opencv_types cimport *



ctypedef pair[float[2], float[2]] _Feature


cdef extern from "ForestBasedRegression.h":

    cdef cppclass _BinaryTree "BinaryTree":
        float index1_[2]
        float index2_[2]
        int pos_
        _BinaryTree* left_
        _BinaryTree* right_


    cdef cppclass _TreeRegressor "TreeRegressor":
        _TreeRegressor(_BinaryTree&, int)
        void getOutput(float*, const _Mat&, const _Mat&)


    cdef cppclass _ForestRegressor "ForestRegressor":
        void getLocalBinaryFeature(_Mat&, const _Mat&, const _Mat&)


    cdef cppclass _AlignmentMethod "AlignmentMethod":
        _Mat align(const _Rect&, const _Mat&) const


    cdef cppclass _LocalRegressorAlignment "LocalRegressorAlignment" (_AlignmentMethod):
        _LocalRegressorAlignment(const string&)


    cdef cppclass _LocalBinaryFeatureAlignment "LocalBinaryFeatureAlignment" (_AlignmentMethod):
        _LocalBinaryFeatureAlignment(const string&)



cdef extern from "ForestBasedRegressionTraining.h":

    cdef enum _node_separation_criteria "node_separation_criteria_t":
        LS, VAR, MEAN_NORM, NORMALIZED_LS, NORMALIZED_VAR


    cdef cppclass _ForestRegressorTraining "ForestRegressorTraining":
        _ForestRegressorTraining(int, int)
        void setSampledRandomFeaturesNumber(int)
        void setNodeSeparationCriteria(_node_separation_criteria)
        void generateRandomFeatures(float, _Feature*)


    cdef cppclass _AlignmentMethodTraining "AlignmentMethodTraining":
        _ForestRegressorTraining& getForestRegressorTraining()
        void train(const vector[_Mat]&, const vector[_Mat]&, const vector[_Rect]&, const vector[_Mat]&, int, int, const string&)


    cdef cppclass _LocalBinaryFeatureTraining "LocalBinaryFeatureTraining" (_AlignmentMethodTraining):
        _LocalBinaryFeatureTraining(int, int, int, int)


    cdef cppclass _LocalRegressorAlignmentTraining "LocalRegressorAlignmentTraining" (_AlignmentMethodTraining):
        _LocalRegressorAlignmentTraining(int, int, int, int)

        

cdef class BinaryTree:

    cdef _BinaryTree *thisptr


    def __cinit__(self):
        self.thisptr = new _BinaryTree()


    def __dealloc__(self):
        del self.thisptr


    cdef clone(self, _BinaryTree* tree):
        del self.thisptr
        self.thisptr = tree
        return self


    property index1:

        def __get__(self): return (self.thisptr.index1_[0], self.thisptr.index1_[1])

        def __set__(self, index1):
            self.thisptr.index1_[0] = index1[0]
            self.thisptr.index1_[1] = index1[1]


    property index2:

        def __get__(self): return (self.thisptr.index2_[0], self.thisptr.index2_[1])

        def __set__(self, index2):
            self.thisptr.index2_[0] = index2[0]
            self.thisptr.index2_[1] = index2[1]


    property pos:

        def __get__(self): return self.thisptr.pos_

        def __set__(self, pos): self.thisptr.pos_ = pos


    property left:

        def __get__(self):
            left = BinaryTree()
            left.clone(self.thisptr.left_)
            return left

        def __set__(self, BinaryTree left): self.thisptr.left_ = left.thisptr


    property right:

        def __get__(self):
            right = BinaryTree()
            right.clone(self.thisptr.right_)
            return right

        def __set__(self, BinaryTree right): self.thisptr.right_ = right.thisptr



node_separation_criterias = {"ls": LS, "var": VAR, "mean norm": MEAN_NORM, "normalized ls": NORMALIZED_LS, "normalized var": NORMALIZED_VAR}


cdef class ForestRegressorTraining:

    cdef _ForestRegressorTraining *thisptr
    cdef int sampled_random_features_number


    def __cinit__(self, int N, int D, criteria = None, sampled_random_features_number = None):
        initMatConversion()
        self.thisptr = new _ForestRegressorTraining(N, D)

        if sampled_random_features_number is not None:
            self.thisptr.setSampledRandomFeaturesNumber(sampled_random_features_number)
            self.sampled_random_features_number = sampled_random_features_number
        else:
            self.sampled_random_features_number = 500

        if criteria is not None:
            self.thisptr.setNodeSeparationCriteria(node_separation_criterias[criteria])


    def __dealloc__(self):
        del self.thisptr


    """
    def train(self, int l, shape, delta_s, image, double radius=25.0):
        cdef:
            vector[_Mat] _shape, _delta_s, _image
            vector[_BinaryTree*] _forest

        convertArrayToVector(shape, _shape)
        convertArrayToVector(delta_s, _delta_s)
        convertArrayToVector(image, _image)

        with nogil:
            _forest = self.thisptr.train(l, _shape, _delta_s, _image, radius)
            
        forest = []
        for i in range(_forest.size()):
            # FIXME cannot compile otherwise
            tree = BinaryTree()
            tree.clone(_forest[i])
            forest.append(tree)
        return forest
        
    def trainTree(self, l, mean_shape, shape, delta_s, image, radius=25.0):
        cdef:
            _Mat _mean_shape
            vector[_Mat] _shape, _delta_s, _image
            vector[int] _partition
            _BinaryTree* _tree
        
        for i in range(shape.shape[0]):
            _partition.push_back(i)

        createCMat(mean_shape, _mean_shape)
        convertArrayToVector(shape, _shape)
        convertArrayToVector(delta_s, _delta_s)
        convertArrayToVector(image, _image)

        self.thisptr.initTransformations(shape.shape[0], _mean_shape, _shape)
        _tree = self.thisptr.trainTree(l, radius, _partition, _shape, _delta_s, _image)
        tree = BinaryTree()
        tree.clone(_tree)
        return tree
    """

    """
    def computeTransformation(self, shape, mean_shape):
        cdef:
            _Mat _shape, _mean_shape
            double _transformation_matrix[4]
            double[:] transformation_matrix
        
        createCMat(shape, _shape)
        createCMat(mean_shape, _mean_shape)

        self.thisptr.computeTransformation(shape.shape[0], _shape, _mean_shape, _transformation_matrix)
        transformation_matrix = _transformation_matrix

        return np.array(transformation_matrix).reshape(2,2)
    """
    

    def generateRandomFeatures(self, radius=25.0):
        cdef vector[_Feature] _features = vector[_Feature](self.sampled_random_features_number)
        self.thisptr.generateRandomFeatures(radius, &_features[0])
        return [((_features[i].first[0], _features[i].first[1]), (_features[i].second[0], _features[i].second[1])) for i in range(_features.size())]


  
cdef class AlignmentMethod:

    cdef _AlignmentMethod *thisptr


    def __cinit__(self, string filename, method = "local_trees_regressors"):
        initMatConversion()
        if method == "lbf":
            self.thisptr = new _LocalBinaryFeatureAlignment(filename)
        else:
            self.thisptr = new _LocalRegressorAlignment(filename)


    def __dealloc__(self):
        del self.thisptr


    def align(self, bounding_box, img):
        cdef:
            _Rect _bounding_box
            _Mat _img
        createCRect(bounding_box, _bounding_box)
        createCMat(img, _img)
        return createPyMat(self.thisptr.align(_bounding_box, _img))

      

cdef class AlignmentMethodTraining:

    cdef _AlignmentMethodTraining *thisptr


    def __cinit__(self, R, T, N, D, method="local_trees_regressors", sampled_random_features_number=None):
        initMatConversion()
        if method == "lbf":
            self.thisptr = new _LocalBinaryFeatureTraining(R, T, N, D)
        else:
            self.thisptr = new _LocalRegressorAlignmentTraining(R, T, N, D)

        if sampled_random_features_number is not None:
            self.thisptr.getForestRegressorTraining().setSampledRandomFeaturesNumber(sampled_random_features_number)
        

    def __dealloc__(self):
        del self.thisptr


    def train(self, s0, s_star, bounding_boxes, imgs, eyes_indexes, model_filename = "model_filename.txt", node_separation_criteria=None):
        cdef:
            vector[_Mat] _s0, _s_star, _imgs
            _Rect _bounding_box
            vector[_Rect] _bounding_boxes
        
        convertArrayToVector(s0, _s0)
        convertArrayToVector(s_star, _s_star)
        convertArrayToVector(imgs, _imgs)

        for bounding_box in bounding_boxes:
            createCRect(bounding_box, _bounding_box)
            _bounding_boxes.push_back(_bounding_box)

        if node_separation_criteria is not None:
            self.thisptr.getForestRegressorTraining().setNodeSeparationCriteria(node_separation_criterias[node_separation_criteria])
        
        self.thisptr.train(_s0, _s_star, _bounding_boxes, _imgs, eyes_indexes[0], eyes_indexes[1], model_filename)
