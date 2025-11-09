'''     load data with the etl module       '''

from etl import ProvenanceExtractor

# for now we set here the optimization parameters we want
#TODO: require these information as input
y_par = {
    'max': 'ACC_val',
    'min': 'cpu_usage'
}

def load_data(data_folder: str):
    '''use etl module to provide training data and parameters to optimize'''
    extractor = ProvenanceExtractor(data_folder, y_par)
    data, parameters = extractor.extract_all()
    #for now we need the sum of cpu_usage values:
    data['cpu_usage'] = data['cpu_usage'].sum()
    return data, parameters
