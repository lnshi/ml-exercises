## Project GitHub Pages

- [Leonard's Machine Learning Exercises](https://lnshi.github.io/ml-exercises/)

## Topics

### ml_basics

- [🐳 Multivariable linear regression(gradient descent)](https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm001_multivariable_linear_regression_gradient_descent/multivariable_linear_regression_gradient_descent.html)

- <details>
    <summary>
      <a href="https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm002_gradient_and_gradient_descent/gradient_and_gradient_descent.html">
        🐳 Gradient and gradient descent
      </a>
    </summary>
    <p>
      <ul>
        <li>Derivative</li>
        <li>Derivative and partial derivative</li>
        <li>Derivative and directional derivative</li>
        <li>Derivative and gradient</li>
        <li>Gradient descent algorithm</li>
      </ul>
    </p>
  </details>

- [🐳 Gradient descent learning rate chosen](https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm003_gradient_descent_learning_rate_chosen/gradient_descent_learning_rate_chosen.html)

- <details>
    <summary>
      <a href="https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm004_normal_equation/normal_equation.html">
        🐳 Normal equation
      </a>
    </summary>
    <p>
      <ul>
        <li>Vector addition and subtraction</li>
        <li>Vector dot product (scalar product, inner product)</li>
        <li>Vector cross product</li>
        <li>Normal equation</li>
      </ul>
    </p>
  </details>

- [🐶 PDF vs PMF vs CDF](https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm005_PDF_PMF_CDF/PDF_PMF_CDF.html)

- [🐶 Bayes’ Theorem and MLE MAP](https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm006_Bayes%E2%80%99%20Theorem_and_MLE_MAP/Bayes%E2%80%99%20Theorem_and_MLE_MAP.html)

## Questions

1. [In gradient descent, must there be a learning rate transition point(safety threshold) for all kinds of cost functions?](https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm003_gradient_descent_learning_rate_chosen/gradient_descent_learning_rate_chosen.html#Final-question)

2. [Question: how do we extend this to the cross product of a four dimensional vector or more higher, like the right part of the above graph?](http://localhost:8888/notebooks/ml_basics/rdm004_normal_equation/normal_equation.ipynb#Cross-product)

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
