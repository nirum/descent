==================
Proximal Operators
==================

The ``proximal_operators`` module contains functions to compute the following proximal operators:

- Proximal operator of the nuclear norm (``nucnorm``)
- Proximal operator of the l1 norm (``sparse``)
- Proximal operator corresponding to solving a linear system of equations (``linsys``)
- Proximal operator of the l2 norm, a squared error penalty (``squared_error``)
- Proximal operator for minimizing an arbitrary smooth function given an oracle that computes the function value and gradient (``lbfgs``)
- Pxorimal operator for the l2 penalty of the discrete different operator, to encourage smoothness (``smooth``)
- Projection onto the non-negative orthant (``nonneg``)
- Projection onto the semidefinite cone (``semidefinite_cone``)
