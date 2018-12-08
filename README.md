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

- <details>
    <summary>
      <a href="https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm007_logistic_regression%28binomial_regression%29_and_regularization/logistic_regression%28binomial_regression%29_and_regularization.html">
        🐶 Logistic regression (binomial regression) and regularization
      </a>
    </summary>
    <p>
      <ul>
        <li>Experience scipy.optimize.fmin_tnc</li>
        <li>Regularization</li>
        <li>Norm of vector and matrix</li>
        <li>Dataset features expansion/extraction</li>
        <li>When a lower dimensional space NOT discriminable dataset is PROJECTED to a PROPER higher dimensional space it always will be discriminable, the boundary is a hyper plane or just a discrimination function.</li>
        <li>Model accuracy comparison between 10-dimensional and 6-dimensional</li>
        <li>'linear_model.LogisticRegression' with sklearn</li>
      </ul>
    </p>
  </details>

- <details>
    <summary>
      <a href="https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm008_GLM_and_exponential_family_distributions/GLM_and_exponential_family_distributions.html">
        🐶 GLM and exponential family distributions
      </a>
    </summary>
    <p>
      <ul>
        <li>Bernoulli distribution in GLM form</li>
        <li>Gaussian distribution (normal distribution) in GLM form</li>
        <li>Softmax regression (multinomial logistic regression) (categorical distribution (variant 3)) in GLM form</li>
        <li>GLM ⇒ linear regression</li>
        <li>GLM ⇒ logistic regression</li>
        <li>Why the PMF for categorical distribution(special form of multinomial distribution: k > 2 and n = 1) has no coefficient</li>
        <li>How to use the table here [Table of distributions](https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions) to build GLM quickly</li>
      </ul>
    </p>
  </details>

## Questions

1. [In gradient descent, must there be a learning rate transition point(safety threshold) for all kinds of cost functions?](https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm003_gradient_descent_learning_rate_chosen/gradient_descent_learning_rate_chosen.html#Final-question)

2. [How do we extend this to the cross product of a four dimensional vector or more higher, like the right part of the above graph?](https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm004_normal_equation/normal_equation.html#Cross-product)

3. [When a lower dimensional space NOT discriminable dataset is PROJECTED to a PROPER higher dimensional space it always will be discriminable, the boundary is a hyper plane or just a discrimination function, what are the differences of the 'a hyper plane' or 'a discrimination function' here?](https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm007_logistic_regression%28binomial_regression%29_and_regularization/logistic_regression%28binomial_regression%29_and_regularization.html#Question:-what-are-the-differences-of-the-'a-hyper-plane'-or-'a-discrimination-function'-here?)

4. [What are the best practices / skills / underlying theories for the features expansion/extraction?](https://lnshi.github.io/ml-exercises/ml_basics_in_html/rdm007_logistic_regression%28binomial_regression%29_and_regularization/logistic_regression%28binomial_regression%29_and_regularization.html#Question:-what-are-the-best-practices-/-skills-/-underlying-theories-for-the-features-expansion/extraction?)

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
