## Project GitHub Pages

- [Leonard's Machine Learning Exercises](https://lnshi.github.io/ml-exercises/)

## Topics

### ml_basics

<details>
  <summary>
    <a href="https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm001_multivariable_linear_regression_gradient_descent/multivariable_linear_regression_gradient_descent.html">
      :whale: Multivariable linear regression(gradient descent)
    </a>
  </summary>

  - Gradient descent algorithm
</details>

<details>
  <summary>
    <a href="https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm002_gradient_and_gradient_descent/gradient_and_gradient_descent.html">
      :whale: Gradient and gradient descent
    </a>
  </summary>

  - Derivative
  - Derivative and partial derivative
  - Derivative and directional derivative
  - Derivative and gradient
  - Gradient descent algorithm
</details>

<details>
  <summary>
    <a href="https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm003_gradient_descent_learning_rate_chosen/gradient_descent_learning_rate_chosen.html">
      :whale: Gradient descent learning rate chosen
    </a>
  </summary>

  - Learning rate chosen
</details>

<details>
  <summary>
    <a href="https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm004_normal_equation/normal_equation.html">
      :whale: Normal equation
    </a>
  </summary>

  - Vector addition and subtraction
  - Vector dot product (scalar product, inner product)
  - Vector cross product
  - Normal equation
</details>

<details>
  <summary>
    <a href="https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm005_PDF_PMF_CDF/PDF_PMF_CDF.html">
      :dog: PDF vs PMF vs CDF
    </a>
  </summary>

  - PDF (probability density function)
  - PMF (probability mass function)
  - CDF (cumulative distribution function)
</details>

<details>
  <summary>
    <a href="https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm006_Bayes%E2%80%99%20Theorem_and_MLE_MAP/Bayes%E2%80%99%20Theorem_and_MLE_MAP.html">
      :dog: Bayes’ Theorem and MLE MAP
    </a>
  </summary>

  - Bayes' Theorem / Bayesian inference
  - Probability Function and Likelihood Function
  - MLE (Maximum Likelihood Estimation)
  - MAP (Maximum A Posteriori probability)
</details>

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
