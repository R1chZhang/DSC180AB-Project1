import sys

sys.path.insert(0, 'test')
from testData import myTestData

sys.path.insert(0, 'src')
from model import model_build

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''
if 'test' in targets:
    data = myTestData().data

    build_model(GCN(data.num_features, 16, data.num_classes))
    build_model(GAT(hidden_channels=8, heads=8))

if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)