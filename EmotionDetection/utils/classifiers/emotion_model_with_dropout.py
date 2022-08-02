from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *

class EmotionModelWithDropout(object):
    """
    The model has the following architecture:

    conv - relu - 2x2 max pool - conv - relu - max - flatten - dense - dropout - dense - dense

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    
    def __init__(
        self,
        input_dim=(3, 48, 48),
        num_filters=32,
        filter_size=7,
        hidden_dim=256,
        num_classes=7,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # Initialize weights and biases for the emotion model.              #
        # Weights should be initialized from a Gaussian centered at 0.0        #
        # with standard deviation equal to weight_scale; biases should be       #
        # initialized to zero. All weights and biases should be stored in the    #
        #  dictionary self.params.                                 #
        ############################################################################

        C, H, W = input_dim
        HP, HW = 1+(H-2)/2, 1+(W-2)/2
        
        HP = int(HP)
        HW = int(HW)
        
        # conv layer - 1
        num_filters = 64
        filter_size = 5
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        
        # conv layer - 2
        self.params['W2'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
        self.params['b2'] = np.zeros(num_filters)
        
        # affine relu - 1
        hidden_dim = 256
        self.params['W4'] = weight_scale * np.random.randn(6400, hidden_dim)
        self.params['b4'] = np.zeros(hidden_dim)
        
        # affine relu - 2
        hidden_dim = 128
        self.params['W5'] = weight_scale * np.random.randn(256, hidden_dim)
        self.params['b5'] = np.zeros(hidden_dim)
        
        # output affine layer
        self.params['W6'] = weight_scale * np.random.randn(128, num_classes)
        self.params['b6'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W4, b4 = self.params["W4"], self.params["b4"]
        W5, b5 = self.params["W5"], self.params["b5"]
        W6, b6 = self.params["W6"], self.params["b6"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}
        
        # dropout param
        dropout_param = {'p': 0.5, 'mode': 'train'}

        scores = None
        ############################################################################
        # Implement the forward pass for the emotion model,                 #
        # computing the class scores for X and storing them in the scores       #
        # variable.                                           #
        ############################################################################

        pool_out, cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param, "same")
        pool_out_2, cache_2 = conv_relu_pool_forward(pool_out, W2, b2, conv_param, pool_param, "valid")
        
        flatten_out = pool_out_2.reshape(pool_out_2.shape[0],pool_out_2.shape[1] * pool_out_2.shape[2] * pool_out_2.shape[3])
        
        X2, fc_cache = affine_relu_forward(flatten_out, W4, b4)  
        dropout_out, dropout_cache = dropout_forward(X2, dropout_param)
        X3, fc_cache_2 = affine_relu_forward(dropout_out, W5, b5)
        
        scores, fc_cache_3 = affine_forward(X3, W6, b6)

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # Implement the backward pass for the emotion model,                #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        ############################################################################

        loss, gradients = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(W1 * W1)
        loss += 0.5 * self.reg * np.sum(W2 * W2)
        loss += 0.5 * self.reg * np.sum(W4 * W4)
        loss += 0.5 * self.reg * np.sum(W5 * W5)
        loss += 0.5 * self.reg * np.sum(W6 * W6)
        
        dout, grads['W6'], grads['b6'] = affine_backward(gradients, fc_cache_3)
        dout, grads['W5'], grads['b5'] = affine_relu_backward(dout, fc_cache_2)
        dout = dropout_backward(dout, dropout_cache)
        dout, grads['W4'], grads['b4'] = affine_relu_backward(dout, fc_cache)
        dout = dout.reshape(pool_out_2.shape[0],pool_out_2.shape[1], pool_out_2.shape[2], pool_out_2.shape[3])
        dout, grads['W2'], grads['b2'] = conv_relu_pool_backward(dout, cache_2)
        dout, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout, cache)
        
        # L2 regulation
        grads['W6'] += self.reg * W6
        grads['W5'] += self.reg * W5
        grads['W4'] += self.reg * W4
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1
        
        return loss, grads
