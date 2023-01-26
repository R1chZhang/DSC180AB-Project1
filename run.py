import sys

sys.path.insert(0, 'test')
from testData import myTestData

sys.path.insert(0, 'src')
from model import model_list, run

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''
    #Run Test
    if 'test' in targets:
        res = []
        
        model_list = model_list()
        for i in model_list:
            res.append(run(model,max_epoch=50))
            
    #Run Project
    if 'all' in targets:
        from torch_geometric.datasets import LRGBDataset
        
        #run first
        dataset = LRGBDataset.LRGBDataset(root='/tmp/lrgb', name='PascalVOC-SP')
        model_list = model_list()
        res=[]
        for i in model_list:
            res.append(run(model,max_epoch=500))
            
        #run second
        dataset = LRGBDataset.LRGBDataset(root='/tmp/lrgb', name='Peptides-func')
        
        #this import  overwrites some of the above classes
        import graph_level_model
        model_list2 = graph_level_model.models()
        res2=[]
        for i in model_list:
            res2.append(graph_level_model.run(model,max_epoch=500))
        
if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)
    
