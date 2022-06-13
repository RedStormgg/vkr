import numpy as np
from operator import itemgetter
import warnings
from scipy import optimize
from sklearn.metrics import mean_squared_error as MSE
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.gaussian_process import kernels
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
import math

def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)

def kmeans(X, k):
    """Performs k-means clustering for 1D input
    
    Arguments:
        X {ndarray} -- A Mx1 array of inputs
        k {int} -- Number of clusters
    
    Returns:
        ndarray -- A kx1 array of final cluster centers
    """
    # randomly select initial clusters from input data
    clusters = np.random.choice(np.squeeze(X), size=k)
    prevClusters = clusters.copy()
    stds = np.zeros(k)
    converged = False
    while not converged:
        """
        compute distances for each cluster center to each point 
        where (distances[i, j] represents the distance between the ith point and jth cluster)
        """
        distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
        # find the cluster that's closest to each point
        closestCluster = np.argmin(distances, axis=1)
        # update clusters by taking the mean of all of the points assigned to that cluster
        for i in range(k):
            pointsForCluster = X[closestCluster == i]
            if len(pointsForCluster) > 0:
                clusters[i] = np.mean(pointsForCluster, axis=0)
        # converge if clusters haven't moved
        converged = np.linalg.norm(clusters - prevClusters) < 1e-6
        prevClusters = clusters.copy()
    distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
    closestCluster = np.argmin(distances, axis=1)
    clustersWithNoPoints = []
    for i in range(k):
        pointsForCluster = X[closestCluster == i]
        if len(pointsForCluster) < 2:
            # keep track of clusters with no points or 1 point
            clustersWithNoPoints.append(i)
            continue
        else:
            stds[i] = np.std(X[closestCluster == i])
    # if there are clusters with 0 or 1 points, take the mean std of the other clusters
    if len(clustersWithNoPoints) > 0:
        pointsToAverage = []
        for i in range(k):
            if i not in clustersWithNoPoints:
                pointsToAverage.append(X[closestCluster == i])
        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))
    return clusters, stds

class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""
    def __init__(self, k=2, lr=0.01, epochs=100, rbf=rbf, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds
        self.w = np.random.randn(k)
        self.b = np.random.randn(1)

    def fit(self, X, y):
        if self.inferStds:
            # compute stds from data
            self.centers, self.stds = kmeans(X, self.k)
        else:
            # use a fixed std 
            self.centers, _ = kmeans(X, self.k)
            dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)
        # training
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # forward pass
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b
                loss = (y[i] - F).flatten() ** 2
                print('Loss: {0:.2f}'.format(loss[0]))
                # backward pass
                error = -(y[i] - F).flatten()
                # online update
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error
                
    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return np.array(y_pred)

class GRNN(BaseEstimator, RegressorMixin):
    """A General Regression Neural Network, based on the Nadaraya-Watson estimator.
    Parameters:
    ----------
    kernel : str, default="rbf"
        Kernel function to be casted in the regression.
        The radial basis function (rbf) is used by default: K(x,x')=exp(-|x-x'|^2/(2 sigma^2))
        
    sigma : float, array, default=None
        Bandwidth standard deviation parameter for the Kernel. All Kernels are casted from the 
        sklearn.metrics.pairwise library. Check its documentation for further 
        info and to find a list of all the built-in Kernels.
    
    n_splits : int, default 10
        Number of folds used in the K-Folds CV definition of the sigma. 
        Must be at least 2.
    
    calibration : str, default=None
        Type of calibration of the sigma. 
        'gradient_search' minimizes the loss function by applying the scipy.optimize.minimize function. 
        Gradient search can be used for isotropic Kernels (sigma = int, float),
        or on anisotropic Kernel (sigma = list).
        'warm_start' is used when sigma is a single scalar. Gradient search is applied 
        to find the best sigma value for an isotropic Kernel (all sigmas are the same). The optimized 
        parameter is then used as a starting point to search the optimal solution for an 
        anisotropic Kernel (having one sigma per feature).
    
    method: str, default=L-BFGS-B
        Type of solver for the gradient search (used to find the local minimum of the cost function). 
        The default solver used is the Nelder-Mead. 
        Other choises (such as the CG based on the Polak and 
        Ribiere algorithm) are discussed on the help of the scipy function.
    
    bounds : list, default=(0, None)
        (min, max) pairs for each element in x, defining the bounds on that parameter.
        Use None or +-inf for one of min or max when there is no bound in that direction.
    
    n_restarts_optimizer : int, default = 0
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the cost function. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from inital sigmas sampled log-uniform randomly
        from the space of allowed sigma-values. 
    
    seed : int, default = 42
        Random state used to initialize random generators.   
    
    
    Notes
    -----
    This Python code was developed and used for the following papers:
    F. Amato, F. Guignard, P. Jacquet, M. Kanveski. Exploration of data dependencies
    and feature selection using General Regression Neural Networks.
   
   References
    ----------
    F. Amato, F. Guignard, P. Jacquet, M. Kanveski. Exploration of data dependencies
    and feature selection using General Regression Neural Networks.
    D.F. Specht. A general regression neural network. IEEE transactions on neural 
    networks 2.6 (1991): 568-576.
    Examples
    --------
    import numpy as np
    from sklearn import datasets
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import  GridSearchCV
    from sklearn.metrics import mean_squared_error as MSE
    from pyGRNN import GRNN
    # Loading the diabetes dataset
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    # Splitting data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(preprocessing.minmax_scale(X),
                                                        preprocessing.minmax_scale(y.reshape((-1, 1))),
                                                        test_size=0.25)
    # Example 1: use Isotropic GRNN with a Grid Search Cross validation to select the optimal bandwidth
    IGRNN = GRNN()
    params_IGRNN = {'kernel':["RBF"],
                    'sigma' : list(np.arange(0.1, 4, 0.01)),
                    'calibration' : ['None']
                     }
    grid_IGRNN = GridSearchCV(estimator=IGRNN,
                              param_grid=params_IGRNN,
                              scoring='neg_mean_squared_error',
                              cv=5,
                              verbose=1
                              )
    grid_IGRNN.fit(X_train, y_train.ravel())
    best_model = grid_IGRNN.best_estimator_
    y_pred = best_model.predict(X_test)
    mse_IGRNN = MSE(y_test, y_pred)
    # Example 2: use Anisotropic GRNN with Limited-Memory BFGS algorithm to select the optimal bandwidths
    AGRNN = GRNN(calibration="gradient_search")
    AGRNN.fit(X_train, y_train.ravel())
    sigma=AGRNN.sigma 
    y_pred = AGRNN.predict(X_test)
    mse_AGRNN = MSE(y_test, y_pred)
    """

    def __init__(self, kernel='RBF', sigma=0.4, n_splits=5, calibration='warm_start', method='L-BFGS-B', bnds=(0, None), n_restarts_optimizer=0, seed = 42):
        self.kernel = kernel
        self.sigma = sigma
        self.n_splits = n_splits
        self.calibration = calibration
        self.method = method
        self.bnds = bnds
        self.n_restarts_optimizer = n_restarts_optimizer
        self.seed = seed
        
    def fit(self, X, y):
        """Fit the model.  
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples. Generally corresponds to the training features
        y : array-like, shape = [n_samples]
            The output or target values. Generally corresponds to the training targets
        Returns
        -------
        self : object
            Returns self.
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        self.X_ = X
        self.y_ = y
        bounds = self.bnds
        
        np.seterr(divide='ignore', invalid='ignore')
        
        def cost(sigma_):
            '''Cost function to be minimized. It computes the cross validation
                error for a given sigma vector.'''
            kf = KFold(n_splits= self.n_splits, random_state=self.seed, shuffle=True)
            kf.get_n_splits(self.X_)
            cv_err = []
            for train_index, validate_index in kf.split(self.X_):
                X_tr, X_val = self.X_[train_index], self.X_[validate_index]
                y_tr, y_val = self.y_[train_index], self.y_[validate_index]
                Kernel_def_= getattr(kernels, self.kernel)(length_scale=sigma_)
                K_ = Kernel_def_(X_tr, X_val)
                # If the distances are very high/low, zero-densities must be prevented:
                K_ = np.nan_to_num(K_)
                psum_ = K_.sum(axis=0).T # Cumulate denominator of the Nadaraya-Watson estimator
                psum_ = np.nan_to_num(psum_)
                y_pred_ = (np.dot(y_tr.T, K_) / psum_)
                y_pred_ = np.nan_to_num(y_pred_)
                cv_err.append(MSE(y_val, y_pred_.T))
            return np.mean(cv_err, axis=0) ## Mean error over the k splits                        
        
        def optimization(x0_):
            '''A function to find the optimal values of sigma (i.e. the values 
               minimizing the cost) given an inital guess x0.'''
            opt = optimize.minimize(cost, x0_, method=self.method, bounds=self.bnds)
            if opt['success'] is True:
                opt_sigma = opt['x']
                opt_cv_error = opt['fun']
            else:
                warnings.warn('Optimization may have failed. Consider changing optimization solver for the gradient search')
                opt_sigma = np.full(len(self.X_[0]), np.nan)
                opt_cv_error = np.inf
                pass
            return [opt_sigma, opt_cv_error]
        
        def calibrate_sigma(self):
            '''A function to find the values of sigma minimizing the CV-MSE. The 
            optimization is based on scipy.optimize.minimize.'''    
            x0 = np.asarray(self.sigma) # Starting guess (either user-defined or measured with warm start)
            if self.n_restarts_optimizer > 0:
                #First optimize starting from theta specified in kernel
                optima = [optimization(x0)] 
                # # Additional runs are performed from log-uniform chosen initial bandwidths
                r_s = np.random.RandomState(self.seed)
                for iteration in range(self.n_restarts_optimizer): 
                    x0_iter = np.full(len(self.X_[0]), np.around(r_s.uniform(0,1), decimals=3))
                    optima.append(optimization(x0_iter))             
            elif self.n_restarts_optimizer == 0:        
                optima = [optimization(x0)]            
            else:
                raise ValueError('n_restarts_optimizer must be a positive int!')
            
            # Select sigma from the run minimizing cost
            cost_values = list(map(itemgetter(1), optima))
            self.sigma = optima[np.argmin(cost_values)][0]
            self.cv_error = np.min(cost_values) 
            return self
        
        
        if self.calibration == 'warm_start':
            print('Executing warm start...')
            self.bnds = (bounds,)           
            x0 = np.asarray(self.sigma)
            optima = [optimization(x0)]            
            cost_values = list(map(itemgetter(1), optima))
            self.sigma = optima[np.argmin(cost_values)][0]
            print('Warm start concluded. The optimum isotropic sigma is ' + str(self.sigma))
            self.sigma = np.full(len(self.X_[0]), np.around(self.sigma, decimals=3))
            self.bnds = (bounds,)*len(self.X_[0])
            #print ('Executing gradient search...')
            calibrate_sigma(self)
            print('Gradient search concluded. The optimum sigma is ' + str(self.sigma))
        elif self.calibration == 'gradient_search':
            #print ('Executing gradient search...')
            self.sigma = np.full(len(self.X_[0]), self.sigma)
            self.bnds = (bounds,)*len(self.X_[0])
            calibrate_sigma(self)
            #print('Gradient search concluded. The optimum sigma is ' + str(self.sigma))
        else:
            pass
                   
        self.is_fitted_ = True
        # Return the regressor
        return self
     
    def predict(self, X):
        """Predict target values for X.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples. Generally corresponds to the testing features
        Returns
        -------
        y : array of shape = [n_samples]
            The predicted target value.
        """
        
         # Check if fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        
        # Input validation
        X = check_array(X)
        
        Kernel_def= getattr(kernels, self.kernel)(length_scale=self.sigma)
        K = Kernel_def(self.X_, X)
        # If the distances are very high/low, zero-densities must be prevented:
        K = np.nan_to_num(K)
        psum = K.sum(axis=0).T # Cumulate denominator of the Nadaraya-Watson estimator
        psum = np.nan_to_num(psum)
        return np.nan_to_num((np.dot(self.y_.T, K) / psum))

def ransac(points: np.ndarray,
          min_inliers: int = 4,
          max_distance: float = 0.15,
          outliers_fraction: float = 0.5,
          probability_of_success: float = 0.99):
   """
   RANdom SAmple Consensus метод нахождения наилучшей
   аппроксимирующей прямой.

   :param points: Входной массив точек формы [N, 2]
   :param min_inliers: Минимальное количество не-выбросов
   :param max_distance: максимальное расстояние до поддерживающей прямой,
                        чтобы точка считалась не-выбросом
   :param outliers_fraction: Ожидаемая доля выбросов
   :param probability_of_success: желаемая вероятность, что поддерживающая
                                  прямая не основана на точке-выбросе
   :param axis: Набор осей, на которых рисовать график
   :return: Numpy массив формы [N, 2] точек на прямой,
            None, если ответ не найден.
   """

   # Давайте вычислим необходимое количество итераций
   num_trials = int(math.log(1 - probability_of_success) /
                    math.log(1 - outliers_fraction**2))

   best_num_inliers = 0
   best_support = None
   for _ in range(num_trials):
       # В каждой итерации случайным образом выбираем две точки
       # из входного массива и называем их "суппорт"
       random_indices = np.random.choice(
           np.arange(0, len(points)), size=(2,), replace=False)
       assert random_indices[0] != random_indices[1]
       support = np.take(points, random_indices, axis=0)

       # Здесь мы считаем расстояния от всех точек до прямой
       # заданной суппортом. Для расчета расстояний от точки до
       # прямой подходит функция векторного произведения.
       # Особенность np.cross в том, что функция возвращает только
       # z координату векторного произведения, а она-то нам и нужна.
       cross_prod = np.cross(support[1, :] - support[0, :],
                             support[1, :] - points)
       support_length = np.linalg.norm(support[1, :] - support[0, :])
       # cross_prod содержит знаковое расстояние, поэтому нам нужно
       # взять модуль значений.
       distances = np.abs(cross_prod) / support_length

       # Не-выбросы - это все точки, которые ближе, чем max_distance
       # к нашей прямой-кандидату.
       num_inliers = np.sum(distances < max_distance)
       # Здесь мы обновляем лучший найденный суппорт
       if num_inliers >= min_inliers and num_inliers > best_num_inliers:
           best_num_inliers = num_inliers
           best_support = support

   # Если мы успешно нашли хотя бы один суппорт,
   # удовлетворяющий всем требованиям
   if best_support is not None:
       # Спроецируем точки из входного массива на найденную прямую
       support_start = best_support[0]
       support_vec = best_support[1] - best_support[0]
       # Для расчета проекций отлично подходит функция
       # скалярного произведения.
       offsets = np.dot(support_vec, (points - support_start).T)
       proj_vectors = np.outer(support_vec, offsets).T
       support_sq_len = np.inner(support_vec, support_vec)
       projected_vectors = proj_vectors / support_sq_len
       projected_points = support_start + projected_vectors

   return projected_points

def least_squares(points: np.ndarray, axis = None):
   """
   Функция для аппроксимации массива точек прямой, основанная на
   методе наименьших квадратов.

   :param points: Входной массив точек формы [N, 2]
   :return: Numpy массив формы [N, 2] точек на прямой
   """

   x = points[:, 0]
   y = points[:, 1]
   # Для метода наименьших квадратов нам нужно, чтобы X был матрицей,
   # в которой первый столбец - единицы, а второй - x координаты точек
   X = np.vstack((np.ones(x.shape[0]), x)).T
   normal_matrix = np.dot(X.T, X)
   moment_matrix = np.dot(X.T, y)
   # beta_hat это вектор [перехват, наклон], рассчитываем его в
   # в соответствии с формулой.
   beta_hat = np.dot(np.linalg.inv(normal_matrix), moment_matrix)
   intercept = beta_hat[0]
   slope = beta_hat[1]
   # Теперь, когда мы знаем параметры прямой, мы можем
   # легко вычислить y координаты точек на прямой.
   y_hat = intercept + slope * x
   # Соберем x и y в единую матрицу, которую мы собираемся вернуть
   # в качестве результата.
   points_hat = np.vstack((x, y_hat)).T

   return points_hat

def pca(points: np.ndarray, axis = None):
   """
   Метод главных компонент (PCA) оценки направления
   максимальной дисперсии облака точек.

   :param points: Входной массив точек формы [N, 2]
   :param axis: Набор осей, на которых рисовать график
   :return: Numpy массив формы [N, 2] точек на прямой
   """

   # Найдем главные компоненты.
   # В первую очередь нужно центрировать облако точек, вычтя среднее
   mean = np.mean(points, axis=0)
   centered = points - mean
   # Функция вычисления собственных значений и векторов np.linalg.eig
   # требует ковариационную матрицу в качестве аргумента.
   cov = np.cov(centered.T)
   # Теперь мы можем посчитать главные компоненты, заданные
   # собственными значениями и собственными векторами.
   eigenval, eigenvec = np.linalg.eig(cov)
   # Мы хотим параметризовать целевую прямую в координатной системе,
   # заданной собственным вектором, собственное значение которого
   # наиболее велико (направление наибольшей вариативности).
   argmax_eigen = np.argmax(eigenval)
   # Нам понадобятся проекции входных точек на наибольший собственный
   # вектор.
   loc_pca = np.dot(centered, eigenvec)
   loc_maxeigen = loc_pca[:, argmax_eigen]
   max_eigenval = eigenval[argmax_eigen]
   max_eigenvec = eigenvec[:, argmax_eigen]
   # Ре-параметризуем прямую, взяв за начало отрезка проекции
   # первой и последней точки на прямую.
   loc_start = mean + max_eigenvec * loc_maxeigen[0]
   loc_final = mean + max_eigenvec * loc_maxeigen[-1]
   linspace = np.linspace(0, 1, num=len(points))
   # Получаем позиции точек, которые идут с одинаковым интервалом,
   # таким образом удаляя шум измерений и вдоль траектории движения.
   positions = loc_start + np.outer(linspace, loc_final - loc_start)

   return positions

def getdeterminationcoef(y,y_pred):
    my = sum(y)/len(y)
    chisl = 0
    znam = 0
    for i in range(0,len(y)):
        chisl=chisl + (y[i] - y_pred[i])**2
        znam = znam + (y[i] - my)**2
    
    return 1- chisl/znam