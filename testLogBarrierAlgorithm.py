from algorithms import log_barrier as logBar
import autograd as ag
import autograd.numpy as np
#==============================================================================
# title: solving convex optimization
# author: Farshad Noravesh
# implementation of interior point: log barrier approach
# objective: min f(x)=x[0, 0] * x[0, 0] + x[1, 0] * x[1, 0] + x[2, 0] * x[2, 0] * 4.
# equality constraint: Ax = b
# inequality constraint:  f_ineq(x) <= 0, where f_ineq is convex
# my_f_ineq = x-10<=10
#==============================================================================

A = np.array([[1., 1., 0.], [0., 1., 1.]])
b = np.array([[1.], [2.]])
x0 = 2. * np.ones((3, 1))

# objective
def my_f_obj(x):
    return x[0, 0] * x[0, 0] + x[1, 0] * x[1, 0] + x[2, 0] * x[2, 0] * 4.

# inequality constraint: Ix <= 10
def my_f_ineq(x):
    return x - 10.

ret=logBar.logBarrier().solve(my_f_obj, my_f_ineq, A, b, x0)

if ret is not None:
    (soln, dual, res, feas_err) = ret
    print("objective achieved: ", my_f_obj(soln))
    print("solution: ")
    print(soln)
    print("dual: ")
    print(dual)
    print("residual: ", res)
    print("equality constraint error: ", feas_err)
    assert(np.all(soln <= 10.))
    np.all(np.abs(np.dot(A, soln) - b) < 1e-7)
else:
    print("infeasible")


