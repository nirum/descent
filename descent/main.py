"""
Main routines for the descent package
"""

from toolz.curried import curry, juxt
from .utils import wrap, destruct, restruct, datum

__all__ = ['optimize']


@curry
def optimize(algorithm, f_df, xref, callbacks=[], maxiter=1e3):
    """
    Main optimization loop

    Parameters
    ----------
    algorithm : function
        A function which returns a generator that yields new iterates
        in a descent sequence (for example, any of the other functions
        in this module)

    f_df : function
        A function which takes one parameter (a numpy.ndarray of parameters)
        and returns the objective and gradient at that location

    x0 : array_like
        A numpy array consisting of the initial parameters

    callbacks : list, optional
        A list of functions, each which takes one parameter (a dictionary
        containing metadata). These functions should have side effects, for
        example, they can log the parameters or update a plot with the current
        objective value. Called at each iteration.

    maxiter : int, optional
        The maximum number of iterations (Default: 1000)

    minibatches : list, optional
        Used for minibatch optimization. An optional list of data (req)

    """

    # make sure the algorithm is valid
    valid = ['gdm', 'rmsprop', 'adam']
    assert algorithm.func_name in valid, \
        "Algorithm must be one of: " + ", ".join(valid)

    # get functions for the objective and gradient of the function
    obj, grad = wrap(f_df, xref)
    x0 = destruct(xref)

    # build the joint callback function
    callback = juxt(*callbacks)

    # run the optimizer
    for k, xk in enumerate(algorithm(grad, x0, maxiter)):

        try:

            # get the objective and gradient and pass it to the callbacks
            callback(datum(obj=obj(xk),
                        grad=restruct(grad(xk), xref),
                        params=restruct(xk, xref),
                        iteration=k))

        except KeyboardInterrupt:

            print('Stopping at iteration {}!'.format(k+1))
            return restruct(xk, xref)

    # return the final parameters, reshaped in the original format
    return restruct(xk, xref)
