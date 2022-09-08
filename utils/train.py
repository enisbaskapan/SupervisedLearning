from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

from utils.process import Format, Preprocess

class Train:
    
    def train_regression_model(self, data, algorithm, parameters = {}):
        """
        Train a regression model

        Parameters
        ----------
        data : tuple, default = None
            Independent and dependent variables inside a tuple. If input contains (X_test, y_test), function could be utilized for training purposes; if it 
            contains (X,y), model can be saved and used in deployment.


        algorithm : {"LR", "PLS", "RFR", "SVR", "PR"}
            algorithm abbreviation

            LR: LinearRegression
            PLS: PartialLeastSquaresRegressor
            RFR: RandomForestRegressor
            SVR: SupportVectorRegressor
            PR: PolinomialRegressor

            Note: Using "PR" would return a tuple including the regressor and the polynomial features object. Later, X_test should be transformed using 
            polinomial features object and predictions must be done on the X_test_polynomial object.

            X_test_polynomial = polynomial.transform(X_test)
            y_pred = regressor.predict(X_test_polynomial)

        parameters: dictionary
            Various parameters that could be defined in different choice of algorithms
        """
        if algorithm == 'PR':

            regressor = LinearRegression()

            polynomial = PolynomialFeatures(degree = parameters['degree'])
            X_polynomial = polynomial.fit_transform(data[0])

            regressor.fit(X_polynomial, data[1])

            return (regressor, polynomial)

        if algorithm == 'LR' : regressor = LinearRegression()

        if algorithm == 'PLS': regressor = PLSRegression(n_components = parameters['n_components'])

        if algorithm == 'RFR' : regressor = RandomForestRegressor(random_state = 0, 
                                                                  n_estimators = parameters['n_estimators'], 
                                                                  criterion = parameters['criterion'], )

        if algorithm == 'SVR' : regressor = SVR(kernel = parameters['kernel'])
        
        if algorithm == 'GBR' : regressor = GradientBoostingRegressor(random_state=0,
                                                                      learning_rate= parameters['learning_rate'],
                                                                      n_estimators = parameters['n_estimators'],
                                                                      loss = parameters['loss'],
                                                                      criterion = parameters['criterion'])

        regressor.fit(data[0], data[1])

        return regressor
    
class Build(Train, Format, Preprocess):
    
    def __init__(self, test_dict):
        
        self.test_dict = test_dict
        self.test_dict['predictions'] = {}
        self.test_dict['models'] = {}

    def build_regression_models(self, models_list, dependent_variable):
              
        for key, data in self.test_dict['data'].items():
            for model in models_list:
                
                X_train, X_test, y_train, y_test = self.preprocess_test_data(data, dependent_variable)
                
                model_name = model[0]
                algorithm = self.format_string(model_name)
                parameters = model[1]
                train_data = (X_train, y_train) 
                
                print(f'Training regression model {model_name} for {key}')
                regressor = self.train_regression_model(train_data , algorithm , parameters)
                print(f'Training done!')
                predictions = regressor.predict(X_test)
                print()
                
                self.test_dict['models'][key+model_name] = regressor
                self.test_dict['predictions'][key+model_name] = predictions
                
                
        
        self.test_dict['y_test'] = y_test
        self.test_dict['X_test'] = X_test
        
        
