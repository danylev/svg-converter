import xml.etree.ElementTree as ET 
from utils.custom_exceptions import NoContentException

class SvgTreeParser:
    
    def __init__(self, file=None, tree=None):
        '''
        Initilize parser via directly by file or by passing tree object of ElementTree,
        passing both will use only the file one
        '''
        if file:
            self.tree = ET.parse(file)
        else:
            self.tree = tree
        if self.tree is None:
            raise(NoContentException('No content suplied'))

if __name__ == '__main__':
    some_shit = SvgTreeParser(file='example.svg')
    import ipdb
    ipdb.set_trace()
    print('lul')