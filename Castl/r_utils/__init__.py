import os
import tempfile
from typing import Dict, List, Optional
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr, SignatureTranslatedAnonymousPackage

pandas2ri.activate()

class RUtils:
    def __init__(self):
        self.castl_r = self._load_castl_r()
    
    def _load_castl_r(self):
        r_dir = os.path.join(os.path.dirname(__file__), 'R')
        
        r_files = []
        for fname in os.listdir(r_dir):
            if fname.endswith('.R'):
                with open(os.path.join(r_dir, fname), 'r') as f:
                    r_files.append(f.read())
        
        return SignatureTranslatedAnonymousPackage('\n'.join(r_files), "castlRUtils")
    
    def calculate_qs(self, gene_list: pd.DataFrame, **kwargs):
        return self.castl_r.calculate_qs(gene_list, **kwargs)
    
    def perform_tea(self, gene_list: pd.DataFrame, **kwargs):
        return self.castl_r.perform_tea(gene_list, **kwargs)

# 创建单例实例
r_utils = RUtils()