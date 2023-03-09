def explore_data(df):  
    ''' 
    Function for exploring the data
    '''
    print("Shape of the data:- ", df.shape) 
    print('======================================================')
    print("Non of Null values:- ", df.isnull().sum().sum()) 
    print('======================================================')
    print("Information about the data:- ", df.info())  
    print('======================================================')
    print("Describing the data:- ", df.describe())
    print('======================================================')

explore_data(data)