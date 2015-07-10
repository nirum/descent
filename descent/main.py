"""
Main routines for the descent package
"""

from toolz.curried import curry, juxt
from descent.utils import wrap


@curry
def loop(algorithm, f_df, x0, callbacks=[], maxiter=10000):
    """
    Main loop

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

    # get functions for the objective and gradient of the function
    obj, grad = wrap(f_df)

    # get the generator for the given algorithm
    opt = algorithm(grad, x0)

    # build the joint callback function
    callback = juxt(*callbacks)

    for k in range(int(maxiter)):

        # get the next iterate
        xk = next(opt)

        # get the objective and gradient and pass it to the callbacks
        callback({'obj': obj(xk), 'grad': grad(xk), 'iter': k})

    return xk
