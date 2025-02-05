This assignment mainly use the proximal gradient method and the FISTA algorithm to solve the following sparse binary classification logistic regression model.

$$
  \min_{x} \ell(x) \triangleq \frac{1}{m} \sum_{i = 1}^{m} \ln(1 + \exp(- b_i a_i^T x)) + \lambda \|x\|_2^2 + \mu \|x\|_1
$$



The main code is "project.m", which implements the proximal gradient method and the FISTA algorithm. After running it, you can generate graphs showing the relationships between the condition number, function error, and the number of iterations for both algorithms.

The file "a9a.txt.mat" stores the original data.


The file "optional_point_prox.m" is used to solve for the optimal point. The core principle is to run the proximal gradient method 5000 times. After the operation, the data will be saved to x_optimal.mat. When "project.m" runs, it will read the optimal point data from x_optimal.mat.

The file "sparsity.m" is used to calculate the sparsity. The algorithm is exactly the same as that in project.m. The code for graphing has been removed, and code for printing information such as sparsity has been added. The value of Mu can be modified on line 25 of the program.
