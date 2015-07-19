"""
Main routines for the descent package
"""

from toolz.curried import curry, juxt
from descent.utils import wrap, destruct, restruct


@curry
def optimize(algorithm, f_df, x0, callbacks=[], maxiter=1e3):
    """
    Main optimization loop

    Parameters
    ----------
    algorithm : function
        A function which returns a generator that yields new iterates
        in a descent sequence (for example, any of the other functions
        in this module)

    df : function
        A function which takes one parameter (a numpy.ndarray of parameters)
        and returns the gradient of the objective at that location

    x0 : array_like
        A numpy array consisting of the initial parameters

    callbacks : [function]
        A list of functions, each which takes one parameter (a dictionary
        containing metadata). These functions should have side effects, for
        example, they can log the parameters or compute the error in the
        objective and store it somewhere. Called at each iteration.

    maxiter : int
        The maximum number of iterations

    """

    # destruct the input into a numpy array
    x_init = destruct(x0)

    # get functions for the objective and gradient of the function
    obj, grad = wrap(f_df)

    # build the joint callback function
    callback = juxt(*callbacks)

    # run the optimizer
    for k, xk in enumerate(algorithm(grad, x_init, maxiter)):

        # get the objective and gradient and pass it to the callbacks
        callback({'obj': obj(xk), 'grad': grad(xk), 'params': xk, 'iter': k})

    # return the final parameters, reshaped in the original format
    return restruct(xk, x0)
