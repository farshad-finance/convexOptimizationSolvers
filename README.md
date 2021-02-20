# Portfolio selection and convex optimization
It is possible to write many portfolio selection problems in investment theory as a convex optimization
to be solved by the current solver. An example is when
you write your objective function and constraints as a second order cone programming(SOCP) problem but here i did 
not mention about it.
## Interior point methods in optimization
log barrier is just one of interior point optimization methods.
## log barrier
This is a simple class to handle optimization subject
to both equality and inequality constraints.
## install the required libraries 
pip install -r requirements.txt
## results of the optimization
```
farshadRobotics:optimization macbook$ python testLogBarrierAlgorithm.py 
('objective achieved: ', 3.5000000000000004)
solution: 
[[-0.5]
 [ 1.5]
 [ 0.5]]
dual: 
[[  2.15197256e+09]
 [ -8.60789025e+09]]
('residual: ', 1.1368683772161603e-12)
('equality constraint error: ', 2.2204460492503131e-16)
farshadRobotics:optimization macbook$ 

```
