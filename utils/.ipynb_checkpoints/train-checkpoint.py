from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split

from utils.process import Format, Preprocess

class Create:
    
    def create_regression_model(self, algorithm, parameters = {}):
        """
        Create a regression model

        Parameters
        ----------


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
        """
        if algorithm == 'PR':

            regressor = LinearRegression()

            polynomial = PolynomialFeatures(degree = parameters['degree'])
            X_polynomial = polynomial.fit_transform(data[0])

            regressor.fit(X_polynomial, data[1])

            return (regressor, polynomial)
         """

        if algorithm == 'LR' : regressor = LinearRegression(**parameters)

        if algorithm == 'PLS': regressor = PLSRegression(**parameters)

        if algorithm == 'RFR' : regressor = RandomForestRegressor(**parameters)

        if algorithm == 'SVR' : regressor = SVR(**parameters)
        
        if algorithm == 'GBR' : regressor = GradientBoostingRegressor(**parameters)


        return regressor
    
    def create_classification_model(self, algorithm, parameters = {}):
        """
        Create a classification model

        Parameters
        ----------
    
        algorithm : {"LR", "DTC", "RFC", "SVC", "KNN"}
            algorithm abbreviation

            LR: LogisticRegression
            DTC: DecisionTreeClassifier
            RFC: RandomForestClassifier
            SVC: SupportVectorClassifier
            KNN: KNeighborClassifier
            

        parameters: dictionary
            Various parameters that could be defined in different choice of algorithms
        """

        if algorithm == 'LR' : classifier = LogisticRegression(**parameters)

        if algorithm == 'DTC': classifier = DecisionTreeClassifier(**parameters)

        if algorithm == 'RFC' : classifier = RandomForestClassifier(**parameters)

        if algorithm == 'SVC' : classifier = SVC(**parameters)
        
        if algorithm == 'KNN' : classifier = KNeighborsClassifier(**parameters)

        return classifier
    
    def create_feature_selection_model(self, estimator_model, selector_algorithm, selector_parameters = {}):
        """
       Create an dimensionality reduct,on model

       Parameters
       ----------
        parameters: dictionary
                Various parameters that could be defined in different choice of algorithms

        algorithm:{'SFM','RFE','SKB','SFS'}
             algorithm abbreviation

        SFM: SelectFromModel
        RFE: RecursiveFeatureElimination
        SKB: SelectKBest
        SFS: SequentialFeatureSelection


        """
        
        if selector_algorithm == 'SFM' : feature_selector = SelectFromModel(estimator = estimator_model, **selector_parameters)

        if selector_algorithm == 'RFE': feature_selector = RFE(estimator = estimator_model, **selector_parameters)

        if selector_algorithm == 'SKB': feature_selector = SelectKBest(estimator = estimator_model, **selector_parameters)

        if selector_algorithm == 'SFS': feature_selector = SequentialFeatureSelector(estimator = estimator_model, **selector_parameters)
        
        return feature_selector
    
    def create_dimensionality_reduction_model(self,algorithm,parameters={}):
        """
        Create an dimensionality reduct,on model

        Parameters
        ----------
        parameters: dictionary
                Various parameters that could be defined in different choice of algorithms

        algorithm:{'PCA','SPCA','LDA','QDA','KPCA'}
             algorithm abbreviation

        PCA: PrincipalComponentAnalysis
        SPCA: SparsePCA
        LDA: LinearDisriminantAnalysis
        QDA: QuadriticDiscriminantAnalysis
        KPCA: KernelDiscriminantAnlaysis


        """

        if algorithm == 'PCA': model = PCA(**parameters)

        if algorithm == 'SPCA': model = SparcePCA(**parameters)

        if algorithm == 'LDA':model = LinearDiscriminantAnalysis(**parameters)

        if algorithm == 'QDA':model = QuariticDiscriminantAnalysis(**parameters)

        if algorithm == 'KPCA':model = KernelPCA(**parameters)

        return model
    
    
class Build(Create, Format, Preprocess):
    
    def __init__(self, test_dict):
        
        self.test_dict = test_dict
        self.test_dict['predictions'] = {}
        self.test_dict['models'] = {}
        self.test_dict['X_test'] = {}

    def build_regression_models(self, models_list, dependent_variable):
              
        for key, data in self.test_dict['data'].items():
            for model in models_list:
                
                X_train, X_test, y_train, y_test = self.preprocess_test_data(data, dependent_variable)
                
                model_name = model[0]
                algorithm = self.format_algorithm_string(model_name)
                parameters = model[1]
                
                print(f'Training regression model {model_name} for {key}')
                regressor = self.create_regression_model(algorithm , parameters)
                regressor.fit(X_train, y_train)
                print(f'Training done!')
                predictions = regressor.predict(X_test)
                print()
                
                self.test_dict['models'][key+model_name] = regressor
                self.test_dict['predictions'][key+model_name] = predictions
                self.test_dict['X_test'][key] = X_test
                
        
        self.test_dict['y_test'] = y_test
        
    def build_classification_models(self, models_list, dependent_variable):
              
        for key, data in self.test_dict['data'].items():
            for model in models_list:

                X_train, X_test, y_train, y_test = self.preprocess_test_data(data, dependent_variable)

                model_name = model[0]
                algorithm = self.format_algorithm_string(model_name)
                parameters = model[1]

                print(f'Training classification model {model_name} for {key}')
                classifier = self.create_classification_model(algorithm , parameters)
                classifier.fit(X_train, y_train)
                print(f'Training done!')
                predictions = classifier.predict(X_test)
                print()

                self.test_dict['models'][key+model_name] = classifier
                self.test_dict['predictions'][key+model_name] = predictions
                self.test_dict['X_test'][key] = X_test

                
        
        self.test_dict['y_test'] = y_test
    
    def build_feature_selected_regression_models(self, models_list, dependent_variable):
        
        for key, data in self.test_dict['data'].items():
     
            for model in models_list:
            
                model_name = model[0]
                parameters = model[1]
                algorithm = self.format_algorithm_string(model_name)
                selector_name = model[2]
                selector_algorithm = self.format_algorithm_string(selector_name)
                selector_parameters = model[3]
                
                X, y, _ = self.preprocess_data(data, dependent_variable)
                
                regressor = self.create_regression_model(algorithm , parameters)
                
                print(f'Selecting features with {selector_name} for {key}')
                feature_selector = self.create_feature_selection_model(regressor, selector_algorithm, selector_parameters)
                
                X_selected = feature_selector.fit_transform(X, y)
                
                X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=0)
                
                print(f'Training regression model {model_name} for {key}')
                regressor.fit(X_train, y_train)
                print(f'Training done!')
                predictions = regressor.predict(X_test)
                print()
                
                self.test_dict['models'][key+model_name+selector_name] = regressor
                self.test_dict['predictions'][key+model_name] = predictions
                self.test_dict['X_test'][key] = X_test
                
        self.test_dict['y_test'] = y_test
