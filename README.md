## Project GitHub Pages

- [Leonard's Machine Learning Exercises](https://lnshi.github.io/ml-exercises/)

## Topics

- [Multiple linear regression](https://lnshi.github.io/ml-exercises/jupyter_notebooks_in_html/rdm001_multiple_linear_regression/multiple_linear_regression.html)

- [Gradient and gradient descent](https://lnshi.github.io/ml-exercises/jupyter_notebooks_in_html/rdm002_gradient_and_gradient_descent/gradient_and_gradient_descent.html)

## Accumulations / References

1. [如何理解最小二乘法？](https://mp.weixin.qq.com/s/4e9ZiiGIOWx_ZUGjzgavWw)

2. `np.array([0, 0])` vs `np.array([0., 0.])`

    ```
    >>> import numpy as np
    >>> t = np.array([0, 0])
    >>> t[0] = 0.97
    >>> t
    array([0, 0])
    >>> t[0] = 1.97
    >>> t
    array([1, 0])
    >>> t = np.array([0., 0.])
    >>> t[0] = 0.97
    >>> t
    array([0.97, 0.  ])
    >>> t = np.array([0, 0.])
    >>> t[0] = 0.97
    >>> t
    array([0.97, 0.  ])
    ```
