import pandas as pd
import string
from math import ceil
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


from utils.test import Test

class Assemble(Test):
    ''' Create dataframse with test values '''
    
    def assemble_test_tables(self, regression_test_dict):
        ''' Create a dataframe with y, prediction and percentage error values and add it to dictionary'''
        
        test_tables_dict = {}
        for key, value in regression_test_dict['predictions'].items():

            error_df = pd.DataFrame()
            error_df['true_values'] = regression_test_dict['y_test']
            error_df['predictions'] = value

            # Retrieve percentage error values for each model
            error_df['error(%)'] = self.test_percentage_error(error_df)

            test_tables_dict[key] = error_df
        return test_tables_dict
        
    def assemble_error_values(self, regression_test_dict):
        ''' Returns a dataframe with error metrics for each model '''
        
        error_lists = []
        model_name_list = []
        for key, value in regression_test_dict['test_tables'].items():

            error_list = self.test_error_values(value)
            error_lists.append(error_list)
            model_name_list.append(key)

        error_values_df = pd.DataFrame(error_lists, index=model_name_list, columns = ['MEPE','MPE','MEAE','MAE','MSE','RMSE', 'STD'])

        return error_values_df.transpose()
    
    def assemble_variability_values(self, regression_test_dict):
        ''' Returns a dataframe with variability values for each model'''
        
        variability_values_lists = []   
        model_name_list = []
        for key, value in regression_test_dict['test_tables'].items():

            variability_values_list = self.test_variability_values(value, regression_test_dict['X_test'])  
            variability_values_lists.append(variability_values_list)
            model_name_list.append(key)

        variability_values_df = pd.DataFrame(variability_values_lists, index=model_name_list, columns = ['R2', 'R2^', 'EVS'])

        return variability_values_df.transpose()
    
class Categorize:
    
    def __init__(self):
        self.letters = string.ascii_letters
    
    def categorize_numerical_variable(self, df_series, numerical_ranges_list):
        ''' Returns a dictionary of categories with corresponding ranges 
        
            Parameters
            ----------
            df_series: Series
                    Numerical variable series
            numerical_ranges_list: List
                    List of ranges for numerical values
        '''
        
        letters = self.letters[:len(numerical_ranges_list)]
        range_dict = dict(zip(letters, numerical_ranges_list))
        category_dict = {}

        for key, value in range_dict.items():
            for data in  df_series.values:
                if int(data) in value:
                    category_dict[data] = key

        return category_dict
    
    def categorize_categorical_variable(self, grouped_data, categorical_ranges_list):
        ''' Returns a dictionary of categories with corresponding ranges 

            Parameters
            ----------
            grouped_data: Series
                    Series acquired after using groupby method
            categorical_ranges_list: List
                    List of ranges for categorical values
        '''

        letters = string.ascii_letters[:len(categorical_ranges_list)]
        range_dict = dict(zip(letters, categorical_ranges_list))
        category_dict = {}
        for name, data in enumerate(grouped_data):
            for key, value in range_dict.items():
                if int(data) in value:
                    category_dict[grouped_data.index[name]] = key

        return category_dict

    
class Transform:
    
    def transform_data(self, data, feature_range = (0,1)):
        
        scaler = MinMaxScaler(feature_range = feature_range)
        data_scaled = scaler.fit_transform(data)
        return data_scaled, scaler
        
    def transform_test_data(self, X_train, X_test):    
        
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test
    
class Format:

    def format_algorithm_string(self, string):
        
        digit_num = len([letter for letter in string if letter.isdigit()])
        algorithm = string[:-digit_num]
        
        return algorithm
    
class Preprocess(Transform):
    
    def preprocess_test_data(self, data, dependent_variable):
        

        X = data.drop(dependent_variable, axis=1)
        y = data[dependent_variable]
        
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        X_train, X_test = self.transform_test_data(X_train, X_test)


        return  X_train, X_test, y_train, y_test
        
    def preprocess_deployment_data(self, data, dependent_variable):
        

        X = data.drop(dependent_variable, axis=1)
        y = data[dependent_variable]
        
        X = pd.get_dummies(X, drop_first=True)
        
        X, scaler = self.transform_data(X)

        return X, y, scaler