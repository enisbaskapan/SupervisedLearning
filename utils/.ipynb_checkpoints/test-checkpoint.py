import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, explained_variance_score

class Test:
    ''' Calculates different metrics (percentage error, mean squared error, R2 ...) '''
    
    def test_percentage_error(self, error_df):
        ''' Returns a list of percentage error values '''
        
        # Create error list
        percentage_error_list = []
        
        # Iterate over rows to calculate PE
        for index, row in error_df.iterrows():
            percentage_error = (abs(row[error_df.columns[0]] - row[error_df.columns[1]]) / row[error_df.columns[0]]) * 100
            percentage_error_list.append(percentage_error)
            
        return percentage_error_list
    
    def test_error_values(self, error_df):
        ''' Return a list of error metrics '''
        
        mepe = round(error_df['error(%)'].median(), 3)
        mpe = round(error_df['error(%)'].mean(), 3)

        meae = round(median_absolute_error(error_df['true_values'], error_df['predictions']), 3)
        mae = round(mean_absolute_error(error_df['true_values'], error_df['predictions']), 3)
        mse = round(mean_squared_error(error_df['true_values'], error_df['predictions']), 3)
        rmse = round(np.sqrt(mse), 3)
        normalized_rmse = rmse/(error_df['true_values'].max()-error_df['true_values'].min())

        std = round(error_df['true_values'].std(), 3)

        error_list = [mepe, mpe, meae, mae, mse, rmse, normalized_rmse, std]

        return error_list
    
    def test_variability_values(self, error_df, X_test):
        ''' Returns a list of variability values '''
    
        # R2 Score
        r2 = r2_score(error_df['true_values'], error_df['predictions'])
        
        # Adj R2 ---> 1-(1-R2)*(n-1)/(n-p)
        adj_r2 = 1-((1-r2) * ((X_test.shape[0]-1) / (X_test.shape[0] - X_test.shape[1]-1)))
        
        # Explained Varience Score
        evs = explained_variance_score(error_df['true_values'], error_df['predictions'])

        variability_values_list = [r2, adj_r2, evs]

        return variability_values_list
    
    def test_classification_report(self, y_true, y_pred):
        ''' Returns a list of classification report values ''' 
        
        report = classification_report(y_true, y_pred, output_dict=True)
        accuracy = report['accuracy']
        report_list = []
        for target in dict(list(report.items())[:-3]):
            class_ = target
            precision = report[target]['precision']
            recall = report[target]['recall']
            f1score = report[target]['f1-score']
            support = report[target]['support']
            target_list = [class_, precision, recall, f1score, support, accuracy]
            report_list.append(target_list)
        return report_list