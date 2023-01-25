
import os, sys
import numpy as onp
import jax.numpy as np
from jax import grad, jit, vmap
from jax.example_libraries.optimizers import adam
import jax.example_libraries.optimizers 
from treelib import Tree
import matplotlib.pyplot as plt

onp.random.seed(1030)

class Scone_GCN():
    def __init__(self, epochs, step_size, batch_size, weight_decay, weight_il,perturbation_level, verbose=True):
        """
        :param epochs: # of training epochs
        :param step_size: step size for use in training model
        :param batch_size: # of data points to train over in each gradient step
        :param verbose: whether to print training progress
        :param weight_decay: ridge regularization constant
        """

        self.random_targets = None

        self.trained = False
        self.model = None
        self.model_single = None
        self.shifts = None
        self.perturbed_shifts = None
        self.weights = None

        self.epochs = int(epochs)
        self.step_size = step_size
        self.batch_size = int(batch_size)
        self.weight_decay = weight_decay
        self.perturbation_level = perturbation_level
        self.weight_il = weight_il
        self.verbose = verbose


    def loss(self, weights, inputs, y, mask, l1_lower, l1_upper, k1, k2):
        """
        Computes cross-entropy loss per flow + penalty on filter weights leading to the integral lipschitz (il) property 
        We only do this il property guarantee for SCNN models
        """
        preds = self.model(weights, *self.shifts, *inputs)[0][mask==1] # dim: (200,13,1)
        # cross entropy + ridge regularization
        n_shifts = len(self.shifts)
        n_shifts += 1 # for identity layer
        
        # compute the eigenvalues 
        lam_g_min = onp.min(onp.real(onp.linalg.eigvals(l1_lower)))
        lam_g_max = onp.max(onp.real(onp.linalg.eigvals(l1_lower)))
        # create an uniform grid within the eigenvalue interval 
        lam_g = np.arange(lam_g_min, lam_g_max, 0.01)
        # build the filter functions, respectively gradient filter function f_g
        lam_g_k = np.array([lam_g**k for k in range(k1+1)]) 
        weights_g = np.array([np.squeeze(weights[k]) for k in range(k1+1)])
        # compute the gradient filter function for all filters/features, also the functions in the il condition: lambda * h'_g(lambda)
        f_g = lam_g_k.T@weights_g
        lam_g_k_lam = np.array([k*lam_g**k for k in range(k1+1)])
        f_g_il = lam_g_k_lam.T@weights_g

        # compute the eigenvalues 
        lam_c_min = onp.min(onp.linalg.eigvals(l1_upper))
        lam_c_max = onp.max(onp.linalg.eigvals(l1_upper))
        # create an uniform grid within the eigenvalue interval 
        lam_c = np.arange(lam_c_min, lam_c_max, 0.01)
        # build the filter functions, respectively gradient filter function f_g
        lam_c_k = np.array([lam_c**k for k in range(k2+1)]) 
        weights_c = np.array([np.squeeze(weights[k]) for k in iter(onp.append(0,k1+1+onp.arange(k2)))])
        #print(weights_c.shape)
        f_c = lam_c_k.T@weights_c
        lam_c_k_lam = np.array([k*lam_c**k for k in range(k2+1)])
        f_c_il = lam_c_k_lam.T@weights_c

        return -np.sum(preds * y[mask==1]) / np.sum(mask) + (self.weight_decay * (np.linalg.norm(weights[:n_shifts])**2 + np.linalg.norm(weights[n_shifts:-1])**2 + np.linalg.norm(weights[-1])**2)) + self.weight_il * (np.linalg.norm(f_g_il) + np.linalg.norm(f_c_il))

        
    def nn_output(self, shifts, inputs):
        nn_output = onp.array(self.model(self.weights, *shifts, *inputs)[1])
        return nn_output

    def accuracy(self, shifts, inputs, y, mask, n_nbrs):
        """
        Computes ratio of correct predictions
        """
        target_choice = np.argmax(y[mask==1], axis=1)
        preds = onp.array(self.model(self.weights, *shifts, *inputs)[0])
        # make best choice out of each node's neighbors
        for i in range(len(preds)):
            preds[i, n_nbrs[i]:] = -100

        pred_choice = np.argmax(preds[mask==1], axis=1)
        return np.mean(pred_choice == target_choice)

    def generate_weights(self, in_channels, hidden_layers, out_channels):
        """
        :param in_channels: # of channels in model inputs
        :param hidden_layers: see :function train:
        :param out_channels: # of channels in model outputs
        :param model_type:   what model this is (Bunch has slightly different weights)
        """
        weight_shapes = []
        if len(hidden_layers) > 0:
            weight_shapes += [(in_channels, hidden_layers[0][1])] * hidden_layers[0][0]

            for i in range(len(hidden_layers) - 1):
                for _ in range(hidden_layers[i+1][0]):
                    weight_shapes += [(hidden_layers[i][1], hidden_layers[i+1][1])]

            if self.model_type == 'bunch':
                weight_shapes += [(hidden_layers[-1][1], out_channels)] * hidden_layers[-1][0]
            else:
                weight_shapes += [(hidden_layers[-1][1], out_channels)]
            #print(weight_shapes)

            self.weights = []
            for s in weight_shapes:
                self.weights.append(0.01 * onp.random.randn(*s))

        else:
            self.weights = [(in_channels, out_channels)]

        print('# of parameters: {}'.format(onp.sum([onp.prod(w) for w in weight_shapes])))


    def setup(self, model, hidden_layers, shifts, inputs, y, in_axes, train_mask, model_type):
        """
        Set up model for training / calling
        in_axes: the keywords for vmap, for batching 
        """
        self.model_type = model_type
        n_train_samples = sum(train_mask)
        self.shifts = shifts # assign shift matrices
        # set up model for batching
        self.model = vmap(model, in_axes=in_axes)
        self.model_single = model
        # generate weights
        in_channels, out_channels = inputs[-1].shape[-1], y.shape[-1]
        # inputs[-1]=X, which is of #flows,#edges,1
        # in_channels = 1, out_channels=1
        self.generate_weights(in_channels, hidden_layers, out_channels)
        
    def setup_scnn(self, model, hidden_layers, k1, k2, shifts, perturbed_shifts, inputs, y, in_axes, train_mask, model_type):
        """
        Set up model for training / calling
        in_axes: the keywords for vmap, for batching 
        """
        self.model_type = model_type
        n_train_samples = sum(train_mask)
        self.shifts = shifts
        self.perturbed_shifts = perturbed_shifts
        # set up model for batching
        self.model = vmap(model, in_axes=in_axes)
        self.model_single = model
        
        # generate weights
        in_channels, out_channels = inputs[-1].shape[-1], y.shape[-1]
        # inputs[-1]=X, which is of #flows,#edges,1
        # in_channels = 1, out_channels=1
        # for scnn, we need to modify the hidden layers parameter
        # from e.g., 3_16_3_16 to 1+K_1+K_2,_16_1+K_1+K_2,_16
        hidden_layers = list((1+k1+k2,hidden_layers[i][1]) for i in range(len(hidden_layers)))
            
        self.generate_weights(in_channels, hidden_layers, out_channels)
        

    def train(self, inputs, y, train_mask, test_mask, n_nbrs, rev_inputs, rev_y, rev_n_nbrs, l1_lower, l1_upper, k1, k2):
        """
        Trains a batched SCoNe model to predict y using the given X and shift operators.
        Model can have any number of shifts and inputs.

        :param model: NN function
        :param hidden_layers: list of tuples (# weight matrices, # of channels) for each hidden layer
        :param inputs: inputs to model; X matrix must be last
        :param y: desired outputs
        :param in_axes: axes of model inputs to batch over
        :param test_ratio: ratio of data used as test data
        :param train_mask: 1-D binary array
        :param hops: number of steps to take before returning prediction todo implement
        """
        #orig_upper_weights = [self.weights[i*3 + 2] for i in range(3)]

        X = inputs[-1]
        N = X.shape[0]
        n_train_samples = sum(train_mask)
        n_test_samples = sum(test_mask)
        n_batches = n_train_samples // self.batch_size

        batch_mask = ''
        # @jit
        # def gradient_step(weights, inputs, y):
        #     grads = grad(self.loss)(weights, inputs, y, batch_mask)

        #     for i in range(len(weights)):
        #         weights[i] -= self.step_size * grads[i]

        #     return weights
        @jit
        def gradient_step(weights, inputs, y, l1_lower, l1_upper, k1, k2):
            grads = grad(self.loss)(weights, inputs, y, batch_mask, l1_lower, l1_upper, k1, k2)

            for i in range(len(weights)):
                weights[i] -= self.step_size * grads[i]

            return weights

        init_fun, update_fun, get_params = adam(self.step_size)

        # track gradients
        non_faces_all, non_faces = [], []
        faces_all, faces = [], []

        def adam_step(i, opt_state, inputs, y, l1_lower, l1_upper, k1, k2):
            g = grad(self.loss)(self.weights, inputs, y, batch_mask, l1_lower, l1_upper, k1, k2)
 
            return update_fun(i, g, opt_state)

        self.adam_state = init_fun(self.weights)
        unshuffled_batch_mask = onp.array([1] * self.batch_size + [0] * (N - self.batch_size))

        # train
        for i in range(self.epochs * n_batches):
            batch_mask = onp.array(unshuffled_batch_mask)
            onp.random.shuffle(batch_mask)

            batch_mask = onp.logical_and(batch_mask, train_mask)

            # self.weights = gradient_step(self.weights, batch_inputs, batch_y)
            self.adam_state = adam_step(i, self.adam_state, inputs, y, l1_lower, l1_upper, k1, k2)
            self.weights = get_params(self.adam_state)

            if i % n_batches == n_batches - 1:
                train_loss = self.loss(self.weights, inputs, y, train_mask, l1_lower, l1_upper, k1, k2)
                #print(self.shifts[0])
                train_acc = self.accuracy(self.shifts, inputs, y, train_mask, n_nbrs)
                test_loss = self.loss(self.weights, inputs, y, test_mask, l1_lower, l1_upper, k1, k2)
                #print(self.perturbed_shifts[0])
                test_acc = self.accuracy(self.shifts, inputs, y, test_mask, n_nbrs)
                test_acc_perturbed = self.accuracy(self.perturbed_shifts, inputs, y, test_mask, n_nbrs)
                #rev_test_loss = self.loss(self.weights, rev_inputs, rev_y, test_mask) 
                #rev_test_acc = self.accuracy(self.shifts, rev_inputs, rev_y, test_mask, rev_n_nbrs)
                output_true = self.nn_output(self.shifts, inputs)
                output_perturbed = self.nn_output(self.perturbed_shifts, inputs)
                dist = np.linalg.norm(output_true-output_perturbed)/np.linalg.norm(output_true)
                # write the distance between two operators
                dist_txt = open("dist_nn_" + self.model_type + "_" + str(len(self.weights[0][0])) + "fea" + "_" + str((len(self.weights) - 1)/(1+k1+k2)) + "layers" + "_"+ str(self.perturbation_level)+ "_"+ str(self.weight_il) +"_il.txt", "a")
                dist_txt.write(str(dist))
                dist_txt.write("\n")
                
                # write the accuracy distance between two operators
                dist_txt = open("test_acc_" + self.model_type + "_" + str(len(self.weights[0][0])) + "fea" + "_" + str((len(self.weights) - 1)/(1+k1+k2)) + "layers_" + str(self.perturbation_level)+ "_"+ str(self.weight_il)+ "_il.txt", "a")
                dist_txt.write(str(np.abs(test_acc_perturbed-test_acc)))
                dist_txt.write("\n")
                
                print('Epoch {} -- train loss: {:.6f} -- train acc {:.3f} \n-- test loss {:.6f} -- test acc {:.3f} -- test acc per {:.3f} \n-- dist {:.6f}'.format(i // n_batches, train_loss, train_acc, test_loss, test_acc, test_acc_perturbed, dist))

                non_faces_all.append(onp.mean(non_faces))
                faces_all.append(onp.mean(faces))

        print("Epochs: {}, learning rate: {}, batch size: {}, model: {}".format(
            self.epochs, self.step_size, self.batch_size, self.model.__name__)
        )

        return train_loss, train_acc, test_loss, test_acc

    def test(self, test_inputs, y, test_mask, n_nbrs, l1_lower, l1_upper, k1, k2):
        """
        Return the loss and accuracy for the given inputs
        """
        loss = self.loss(self.weights, test_inputs, y, test_mask, l1_lower, l1_upper, k1, k2)
        acc = self.accuracy(self.perturbed_shifts, test_inputs, y, test_mask, n_nbrs)

        if self.verbose:
            print("Test loss: {:.6f}, Test acc: {:.3f}".format(loss, acc))
        return loss, acc