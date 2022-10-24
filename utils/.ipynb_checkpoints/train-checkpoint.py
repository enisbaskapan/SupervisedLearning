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

from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential

from utils.process import Format, Preprocess

#import pmdarima as pm
#from pmdarima import model_selection
#from prophet import Prophet

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
    
        algorithm : {"LR", "DTC", "RFC", "SVC", "KNN", "LDA", "QDA"}
            algorithm abbreviation

            LR: LogisticRegression
            DTC: DecisionTreeClassifier
            RFC: RandomForestClassifier
            SVC: SupportVectorClassifier
            KNN: KNeighborClassifier
            LDA: LinearDisriminantAnalysis
            QDA: QuadriticDiscriminantAnalysis
            

        parameters: dictionary
            Various parameters that could be defined in different choice of algorithms
        """

        if algorithm == 'LR' : classifier = LogisticRegression(**parameters)

        if algorithm == 'DTC': classifier = DecisionTreeClassifier(**parameters)

        if algorithm == 'RFC' : classifier = RandomForestClassifier(**parameters)

        if algorithm == 'SVC' : classifier = SVC(**parameters)
        
        if algorithm == 'KNN' : classifier = KNeighborsClassifier(**parameters)
        
        if algorithm == 'LDA':model = LinearDiscriminantAnalysis(**parameters)

        if algorithm == 'QDA':model = QuadraticDiscriminantAnalysis(**parameters)


        return classifier
    
    def create_feature_selection_model(self, estimator_model, selector_algorithm, selector_parameters = {}):
        """
       Create a feature selection model

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
        Create a dimensionality reduction model

        Parameters
        ----------
        parameters: dictionary
                Various parameters that could be defined in different choice of algorithms

        algorithm:{'PCA','SPCA','LDA'}
             algorithm abbreviation

        PCA: PrincipalComponentAnalysis
        SPCA: SparsePCA
        KPCA: KernelDiscriminantAnlaysis


        """

        if algorithm == 'PCA': model = PCA(**parameters)

        if algorithm == 'SPCA': model = SparsePCA(**parameters)

        if algorithm == 'KPCA':model = KernelPCA(**parameters)
        
        if algorithm == 'AE': model = self.create_autoencoder_model(parameters['encoder_layers'], parameters['decoder_layers'], parameters['compile_parameters'])

        return model
    
    def create_deep_learning_model(self, layers, compile_parameters={}):
        """
        Create and compile a deep learning model
        
        Parameters
        ----------
        layers: list
            List of neural network layers
            
        compile_parameters: dictionary
            Parameters for compile method in dictionary format
        """
        network = Sequential(layers)
        network.compile(**compile_parameters)

        return network
    
    def create_autoencoder_model(self, encoder_layers, decoder_layers, compile_parameters={}):
        """
        Create and compile a autoencoder model. Returns Autoencoder and Encoder in tuple format
        
        Parameters
        ----------
        encoder_layers: list
            List of encoder layers
            
        decoder_layers: list
            List of decoder layers
            
        compile_parameters: dictionary
            Parameters for compile method in dictionary format
        """
        encoder = Sequential(encoder_layers)
        decoder = Sequential(decoder_layers)
        autoencoder = Sequential([encoder,decoder])
        autoencoder.compile(**compile_parameters)
        
        return (autoencoder, encoder)
    
    def create_time_series_model(self, algorithm, data, parameters={}):
        
        if algorithm == 'AAR' : ts_model = pm.auto_arima(data, **parameters)

        if algorithm == 'PRH': ts_model = Prophet(**parameters)
        
        return ts_model
    

class Include(Create, Format, Preprocess):

    def include_feature_selection(self, model, key, model_object, X_train, X_test, y_train):
        """
        Apply feature selection on train and test values
        """
        selector_model = model[1] 
        selector_name = selector_model['feature_selection'][0]
        selector_algorithm = self.format_algorithm_string(selector_name)
        selector_parameters = selector_model['feature_selection'][1]

        print(f'Selecting features with {selector_name} for {key}')
        feature_selector = self.create_feature_selection_model(model_object, selector_algorithm, selector_parameters)

        X_train_selected = feature_selector.fit_transform(X_train, y_train)
        X_test_selected = feature_selector.transform(X_test)
        
        return X_train_selected, X_test_selected, selector_name
    
    def include_dimensionality_reduction(self, model, key, X_train, X_test, y_train):
        """
        Apply dimensionality reduction on train and test values
        """
        dimensionality_reduction_model = model[1] 
        dimensionality_reduction_name = dimensionality_reduction_model['dimensionality_reduction'][0]
        dimensionality_reduction_algorithm = self.format_algorithm_string(dimensionality_reduction_name)
        dimensionality_reduction_parameters = dimensionality_reduction_model['dimensionality_reduction'][1]
        

        print(f'Reducing dimensions with {dimensionality_reduction_name} for {key}')
        
        dimensionality_reducer = self.create_dimensionality_reduction_model(dimensionality_reduction_algorithm, dimensionality_reduction_parameters)
        
        if dimensionality_reduction_algorithm == 'AE':
            autoencoder = dimensionality_reducer[0]
            encoder = dimensionality_reducer[1]
            autoencoder.fit(X_train, X_train, **dimensionality_reduction_parameters['fit_parameters'])
            X_train_reduced = encoder.predict(X_train)
            X_test_reduced = encoder.predict(X_test)
            
            return X_train_reduced, X_test_reduced, dimensionality_reduction_name
            
        X_train_reduced = dimensionality_reducer.fit_transform(X_train)
        X_test_reduced = dimensionality_reducer.transform(X_test)
        
        return X_train_reduced, X_test_reduced, dimensionality_reduction_name
    
    
class Build(Include):
    
    def __init__(self, test_dict, feature_selection=False, dimensionality_reduction=False):
        
        self.test_dict = test_dict
        self.test_dict['predictions'] = {}
        self.test_dict['models'] = {}
        self.test_dict['X_test'] = {}
        self.test_dict['y_test'] = {}
        self.test_dict['results'] = {}
        self.feature_selection = feature_selection
        self.dimensionality_reduction = dimensionality_reduction
        
    def build_regression_models(self, models_list, dependent_variable):
        """
        Build multiple regression models on multiple data.
        
        Parameters
        ----------
        models_list: list
            List of regression models and parameters.
            
        dependent_variable: string
            Name of the variable that model is going to predict
        """      
        for key, data in self.test_dict['data'].items():
            for model in models_list:
                
                X_train, X_test, y_train, y_test = self.preprocess_test_data(data, dependent_variable)
                
                regression_model = model[0]
                model_name = regression_model['model']
                algorithm = self.format_algorithm_string(model_name)
                parameters = regression_model['parameters']
                
                # Dict names
                label = key+model_name
                
                regressor = self.create_regression_model(algorithm , parameters)
                
                if self.feature_selection: 
                    X_train_selected, X_test_selected, selector_name = self.include_feature_selection(model, key, regressor, X_train, X_test, y_train)
                    X_train = X_train_selected
                    X_test = X_test_selected
                    label += selector_name

                if self.dimensionality_reduction:
                    X_train_reduced, X_test_reduced, reducer_name = self.include_dimensionality_reduction(model, key, X_train, X_test, y_train)
                    X_train = X_train_reduced
                    X_test = X_test_reduced
                    label += reducer_name

                print(f'Training regression model {model_name} for {key}')
                regressor.fit(X_train, y_train)
                print(f'Training done!')
                predictions = regressor.predict(X_test)
                print()
                
                self.test_dict['models'][label] = regressor
                self.test_dict['predictions'][label] = predictions
                self.test_dict['X_test'][key] = X_test
                self.test_dict['y_test'][key] = y_test
        
        
    def build_classification_models(self, models_list, dependent_variable):
        """
        Build multiple classification models on multiple data.
        
        Parameters
        ----------
        models_list: list
            List of classification models and parameters.
            
        dependent_variable: string
            Name of the variable that model is going to predict
        """       
        for key, data in self.test_dict['data'].items():
            for model in models_list:

                X_train, X_test, y_train, y_test = self.preprocess_test_data(data, dependent_variable)

                classification_model = model[0]
                model_name = classification_model['model']
                algorithm = self.format_algorithm_string(model_name)
                parameters = classification_model['parameters']
                
                # Dict names
                label = key+model_name
                
                classifier = self.create_classification_model(algorithm , parameters)
                
                if self.feature_selection: 
                    X_train_selected, X_test_selected, selector_name = self.include_feature_selection(model, key, classifier, X_train, X_test, y_train)
                    X_train = X_train_selected
                    X_test = X_test_selected
                    label += selector_name
                    
                if self.dimensionality_reduction:
                    X_train_reduced, X_test_reduced, reducer_name = self.include_dimensionality_reduction(model, key, X_train, X_test, y_train)
                    X_train = X_train_reduced
                    X_test = X_test_reduced
                    label += reducer_name

                print(f'Training classification model {model_name} for {key}')
                classifier.fit(X_train, y_train)
                print(f'Training done!')
                predictions = classifier.predict(X_test)
                print()

                self.test_dict['models'][label] = classifier
                self.test_dict['predictions'][label] = predictions
                self.test_dict['X_test'][key] = X_test
                self.test_dict['y_test'][key] = y_test
                
    def build_deep_learning_models(self, models_list, dependent_variable, compile_parameters={}, fit_parameters={}):
        """
        Build multiple regression modelson multiple data.
        
        Parameters
        ----------
        models_list: list
            List of deep learning models and parameters.
            
        dependent_variable: string
            Name of the variable that model is going to predict
        
        compile_parameters: dictionary
            Parameters that are going to be used for compiling the neural network.
        
        fit_parameters: dictionary
            Parameters that are going to be used for training the neural network
        """   
        for key, data in self.test_dict['data'].items():
            for model in models_list:

                X_train, X_test, y_train, y_test = self.preprocess_test_data(data, dependent_variable)

                network_model = model[0]
                model_name = network_model['model']
                kind = self.format_algorithm_string(model_name)
                layers = network_model['layers']
                compile_parameters = network_model['compile_parameters']
                fit_parameters = network_model['fit_parameters']

                # Dict names
                label = key+model_name
                
                #Create NN 
                nn = self.create_deep_learning_model(layers, compile_parameters)
                
                if self.feature_selection: 
                    X_train_selected, X_test_selected, selector_name = self.include_feature_selection(model, key, nn, X_train, X_test, y_train)
                    X_train = X_train_selected
                    X_test = X_test_selected
                    label += selector_name

                if self.dimensionality_reduction:
                    X_train_reduced, X_test_reduced, reducer_name = self.include_dimensionality_reduction(model, key, X_train, X_test, y_train)
                    X_train = X_train_reduced
                    X_test = X_test_reduced
                    label += reducer_name
                
                print(f'Training deeplearning model {model_name} for {key}')
                result = nn.fit(X_train, y_train, **fit_parameters)
                print(f'Training done!')
                predictions = nn.predict(X_test)
                print()

                self.test_dict['models'][label] = nn
                self.test_dict['predictions'][label] = predictions
                self.test_dict['X_test'][key] = X_test
                self.test_dict['y_test'][key] = y_test
                self.test_dict['results'][label] = result
             
    def build_time_series_models(self, models_list):

        for key, data in self.test_dict['data'].items():
            for model in models_list:
                
                time_series_model = model[0]
                model_name = time_series_model['model']
                algorithm = self.format_algorithm_string(model_name)
                parameters = time_series_model['parameters']
                test_size = time_series_model['test_size']
                
                #Change columns names if algorithm is Prophet
                if algorithm == 'PRH':
                    data = self.format_prophet_data(data)
                    
                train, test= self.preprocess_time_series_data(data, test_size)
    
                # Dict names
                label = key+model_name
                
                print(f'Creating time series model {model_name} for {key}')
                ts_model = self.create_time_series_model(algorithm, data , parameters)
                print('Model created!')
                ts_model.fit(train)
                
                if algorithm == 'PRH':
                    #Drop y values from test dataframe to make future dataframe for Prophet.
                    predictions = ts_model.predict(test.drop('y', axis=1))
                    predictions = self.format_prophet_predictions
                    
                    #Format the test and train tables so index is datetime column
                    train, test = self.format_prophet_index(train, test)
                
                else:
                    predictions = ts_model.predict(n_periods = test.shape[0])

                self.test_dict['models'][label] = ts_model
                self.test_dict['predictions'][label] = predictions
                #Because there is no X - y split in time series, train is labeled as X_test and test is labeled as y_test.
                self.test_dict['X_test'][key] = train
                self.test_dict['y_test'][key] = test

