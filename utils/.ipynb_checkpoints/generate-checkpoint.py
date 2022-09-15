import string
from math import ceil

class Generate:

    def generate_numerical_ranges(self, numeric_variable, thresholds_list):
        ''' Create a list of ranges for numeric variable you want to categorize

            Parameters
            ----------
            numeric_variable: Series
                        Column of a DataFrame that contains numerical values that you want to categorize
            thresholds_list: List
                        List of the thresholds 
        '''
        numerical_ranges_list = []
        start = int(numeric_variable.min()-1)
        final = int(numeric_variable.max()*1.5)
        for threshold in thresholds_list:
            numerical_ranges_list.append(range(start, threshold))
            start = threshold

        numerical_ranges_list.append(range(threshold, final))

        return numerical_ranges_list

    def generate_categorical_ranges(self, interval,start,finish):
        ''' Create a list of ranges for categorcial variable you want to categorzie

            Parameters
            ----------
            interval: int
                Count of intervals to create
            start: int -- float
                minnimum value
            finish: int -- float
                maximum value
        '''
        categorical_ranges_list = []
        start = int(start)
        finish = int(finish)
        diff = int((finish - start) / interval)

        for i in range(interval):
            if interval-i != 1:
                categorical_ranges_list.append(range(start, start+ diff))
            else:
                categorical_ranges_list.append(range(start, ceil(1.5 * (start+diff))))

            start += diff

        return categorical_ranges_list
    