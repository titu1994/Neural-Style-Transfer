from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


# Ported from https://pychao.com/2019/11/02/optimize-tensorflow-keras-models-with-l-bfgs-from-tensorflow-probability/
class AbstractTFPOptimizer(ABC):

    def __init__(self, trace_function=False):
        super(AbstractTFPOptimizer, self).__init__()
        self.trace_function = trace_function
        self.callback_list = None

    def _function_wrapper(self, loss_func, model):
        """A factory to create a function required by tfp.optimizer.lbfgs_minimize.

        Args:
            loss_func: a function with signature loss_value = loss(model).
            model: an instance of `tf.keras.Model` or its subclasses.

        Returns:
            A function that has a signature of:
                loss_value, gradients = f(model_parameters).
        """

        # obtain the shapes of all trainable parameters in the model
        shapes = tf.shape_n(model.trainable_variables)
        n_tensors = len(shapes)

        # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
        # prepare required information first
        count = 0
        idx = []  # stitch indices
        part = []  # partition indices

        for i, shape in enumerate(shapes):
            n = np.product(shape)
            idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
            part.extend([i] * n)
            count += n

        part = tf.constant(part)

        @tf.function
        def assign_new_model_parameters(params_1d):
            """A function updating the model's parameters with a 1D tf.Tensor.

            Args:
                params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
            """

            params = tf.dynamic_partition(params_1d, part, n_tensors)
            for i, (shape, param) in enumerate(zip(shapes, params)):
                model.trainable_variables[i].assign(tf.reshape(param, shape))

        # now create a function that will be returned by this factory
        def f(params_1d):
            """A function that can be used by tfp.optimizer.lbfgs_minimize.

            This function is created by function_factory.

            Args:
               params_1d [in]: a 1D tf.Tensor.

            Returns:
                A scalar loss and the gradients w.r.t. the `params_1d`.
            """

            # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
            with tf.GradientTape() as tape:
                # update the parameters in the model
                assign_new_model_parameters(params_1d)
                # calculate the loss
                loss_value = loss_func(model)

            # calculate gradients and convert to 1D tf.Tensor
            grads = tape.gradient(loss_value, model.trainable_variables)
            grads = tf.dynamic_stitch(idx, grads)

            # print out iteration & loss
            f.iter.assign_add(1)
            tf.print("Iter:", f.iter, "loss:", loss_value)

            if self.callback_list is not None:
                info_dict = {
                    'iter': f.iter,
                    'loss': loss_value,
                    'grad': grads,
                }

                for callback in self.callback_list:
                    callback(model, info_dict=info_dict)

            return loss_value, grads

        if self.trace_function:
            f = tf.function(f)

        # store these information as members so we can use them outside the scope
        f.iter = tf.Variable(0, trainable=False)
        f.idx = idx
        f.part = part
        f.shapes = shapes
        f.assign_new_model_parameters = assign_new_model_parameters

        return f

    def register_callback(self, callable):
        """
        Accepts a callable with signature `callback(model, info_dict=None)`.
        Callable should not return anything, it will not be dealt with.

        `info_dict` will contain the following information:
            - Optimizer iteration number (key = 'iter')
            - Loss value (key = 'loss')
            - Grad value (key = 'grad')

        Args:
            callable: A callable function with the signature `callable(model, info_dict=None)`.
            See above for what info_dict can contain.
        """

        if self.callback_list is None:
            self.callback_list = []

        self.callback_list.append(callable)

    @abstractmethod
    def minimize(self, loss_func, model):
        pass


class BFGSOptimizer(AbstractTFPOptimizer):

    def __init__(self, max_iterations=50, tolerance=1e-8, bfgs_kwargs=None, trace_function=False):
        super(BFGSOptimizer, self).__init__(trace_function=trace_function)

        self.max_iterations = max_iterations
        self.tolerance = tolerance

        bfgs_kwargs = bfgs_kwargs or {}

        if 'max_iterations' in bfgs_kwargs.keys():
            del bfgs_kwargs['max_iterations']

        if 'tolerance' in bfgs_kwargs.keys():
            keys = [key for key in bfgs_kwargs.keys()
                    if 'tolerance' in key]
            for key in keys:
                del bfgs_kwargs[key]

        self.bfgs_kwargs = bfgs_kwargs

    def minimize(self, loss_func, model):
        optim_func = self._function_wrapper(loss_func, model)

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(optim_func.idx, model.trainable_variables)

        # train the model with BFGS solver
        results = tfp.optimizer.bfgs_minimize(
            value_and_gradients_function=optim_func, initial_position=init_params,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            x_tolerance=self.tolerance,
            f_relative_tolerance=self.tolerance,
            **self.bfgs_kwargs)

        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        optim_func.assign_new_model_parameters(results.position)

        print("BFGS complete, and parameters updated !")
        return model


class LBFGSOptimizer(AbstractTFPOptimizer):

    def __init__(self, max_iterations=50, tolerance=1e-8, lbfgs_kwargs=None, trace_function=False):
        super(LBFGSOptimizer, self).__init__(trace_function=trace_function)

        self.max_iterations = max_iterations
        self.tolerance = tolerance

        lbfgs_kwargs = lbfgs_kwargs or {}

        if 'max_iterations' in lbfgs_kwargs.keys():
            del lbfgs_kwargs['max_iterations']

        if 'tolerance' in lbfgs_kwargs.keys():
            keys = [key for key in lbfgs_kwargs.keys()
                    if 'tolerance' in key]
            for key in keys:
                del lbfgs_kwargs[key]

        self.lbfgs_kwargs = lbfgs_kwargs

    def minimize(self, loss_func, model):
        optim_func = self._function_wrapper(loss_func, model)

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(optim_func.idx, model.trainable_variables)

        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=optim_func, initial_position=init_params,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            x_tolerance=self.tolerance,
            f_relative_tolerance=self.tolerance,
            **self.lbfgs_kwargs)

        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        optim_func.assign_new_model_parameters(results.position)

        print("L-BFGS complete, and parameters updated !")
        return model
