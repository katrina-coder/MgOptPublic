import alloysHT
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, ExpSineSquared, ConstantKernel, DotProduct
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import warnings
from sklearn.ensemble import RandomForestRegressor
import copy
import time

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", 42)
pd.set_option("display.max_rows", 200)


bound_dict = {'Mg': (80.0, 95.0), 'Zn': (0.0, 14.3),'Al': (0.0, 11.0), 'Mn': (0.0, 2.0),
                           'Nd': (0.0, 8.05), 'Ce': (0.0, 3.92), 'La': (0.0, 6.0), 'Zr': (0.0, 3.0),
                           'Cu': (0.0, 0.5), 'Si': (0.0, 5.0), 'Y': (0.0, 19.0), 'Ca': (0.0, 10.0),
                           'Pr': (0.0, 1.76), 'Ni': (0.0, 1.0), 'Be': (0.0, 0.0), 'Fe': (0.0, 0.0),
                           'Li': (0.0, 3.0), 'Gd': (0.0, 10.0), 'Th': (0.0, 0.0), 'Sn': (0.0, 9.56),
                           'Sb': (0.0, 1.0001), 'Ag': (0.0, 0.5), 'Ga': (0, 1.0), 'Yb': (0, 3),
                           'Bi': (0.0, 0.5), 'Sc': (0.0, 0.5), 'Dy': (0.0, 0.0), 'Sr': (0.0, 2.45),
                           'Tb': (0.0, 1.0), 'Er': (0.0, 6.0), 'Ho': (0.0, 1.4),
                           'Extruded': (0,1), 'ECAP': (0,1), 'Cast_Slow': (0,1),
                           'Cast_Fast': (0,1), 'Cast_HT': (0,1),'Wrought': (0,1)}


bound_dict = dict(sorted(bound_dict.items(), key=lambda item: item[1][1], reverse=True))

CHT = list(bound_dict.keys())

data = alloysHT.data.drop_duplicates(subset = CHT, inplace=False)
X = data[CHT] # Comp + HT
UTS = data["UTS(MPa)"] #UTS
Ductility = data["Ductility"]

user_defined_range =  {'Mg': (80.0, 95.0), 'Zn': (0.0, 14.3),'Al': (0.0, 11.0), 'Mn': (0.0, 0),
                           'Nd': (0.0, 8.05), 'Ce': (0.0, 3.92), 'La': (0.0, 6.0), 'Zr': (0.0, 3.0),
                           'Cu': (0.0, 0), 'Si': (0.0, 0), 'Y': (0.0, 0.0), 'Ca': (0.0, 0.0),
                           'Pr': (0.0, 0), 'Ni': (0.0, 0), 'Be': (0.0, 0.0), 'Fe': (0.0, 0.0),
                           'Li': (0.0, 0), 'Gd': (0.0, 10.0), 'Th': (0.0, 0.0), 'Sn': (0.0, 0),
                           'Sb': (0.0, 0), 'Ag': (0.0, 0), 'Ga': (0, 0), 'Yb': (0, 0),
                           'Bi': (0.0, 0), 'Sc': (0.0, 0), 'Dy': (0.0, 0.0), 'Sr': (0.0, 2.45),
                           'Tb': (0.0, 0), 'Er': (0.0, 0), 'Ho': (0.0, 0),
                           'Extruded': (0,1), 'ECAP': (0,1), 'Cast_Slow': (0,1),
                           'Cast_Fast': (0,1), 'Cast_HT': (0,1),'Wrought': (0,1)}

class inverse_pred:

    def __init__(self,  x=X, y=UTS, desired_y=400, output_names = 'UTS' , num_elems= 6, sum_elems = 25, sample_size=100000,
                 iter_num=10, model_name='rf', user_defined_range = user_defined_range):
        self.x = x
        self.y = y
        if output_names.value=='Ductility':
            self.y=Ductility
        self.desired_y = desired_y
        self.output_names = [output_names.value]
        
        self.num_elems= num_elems
        self.sum_elems= sum_elems
        self.sample_size=sample_size
        self.iter_num = iter_num
        self.model_name = model_name
        
        self.bound_dict = user_defined_range
        self.elem_og_sorted = list(bound_dict.keys())
        self.samples_df = pd.DataFrame()
        
        if self.model_name=='rf':
            self.model = self.define_rf_model()
            self.model.fit(self.x,self.y)
            
    @staticmethod
    def define_rf_model():
        model = RandomForestRegressor(n_estimators = 800 , 
                                      min_samples_split = 5,
                                      min_samples_leaf = 1,
                                      max_features = 'sqrt',
                                      max_depth = 100, 
                                      bootstrap = False,
                                      random_state=0 )
        return model
                 
    def sampler(self):  # the sum of all random variables should be equal to one
        sample_size = self.sample_size
        num_elems = self.num_elems
        summ = self.sum_elems
        random_set = {}
        #bound_dict = dict(sorted(self.bound_dict.items(), key=lambda item: item[1][1], reverse=True))
        bound_dict = self.bound_dict
        keys = list(bound_dict.keys())
        
        # all the alloying elements except Fe and Be and Th
        ht_list = ['Extruded', 'ECAP','Cast_Slow', 'Cast_Fast', 'Cast_HT','Wrought']
        for i in ht_list:
            keys.remove(i)
        
        for key in keys:
            if bound_dict[key][1]==0.0:
                keys.remove(key)
        
        if 'Mg' in keys:
            keys.remove('Mg')
        else:
            keys = keys
            
        num_metals = np.random.randint(3,num_elems+1, size=sample_size) ### 3 means that minimum 3 alloying elements exist        
        final_samples = []
        
        s = 0
        while s <self.sample_size:
        #for s in range(sample_size):
            chosen_materials = np.random.choice(keys, num_metals[s], replace=False)
            for i, k in enumerate(bound_dict):
                if k in chosen_materials:
                    lower = max(summ-sum([v for _, v in random_set.items()])-bound_dict[k][1], 0) # this is upper bound of element k in chosen materials
                    # it means we are considering all chosen materials are sticking to their upper bounds.
                    if summ-sum([v for _, v in random_set.items()])-bound_dict[k][0] < 0:
                        upper = summ-sum([v for _, v in random_set.items()])
                    else:
                        upper = summ-sum([v for _, v in random_set.items()])-bound_dict[k][0]
                    search_bound = (lower,  upper)
                    r = np.random.uniform(search_bound[0],  search_bound[1])
                    random_set[k]= summ - r - sum([v for _, v in random_set.items()])
                else:
                    random_set[k]=0  
            random_set["Mg"]= 100 - sum([v for k, v in random_set.items()])            
            
            for l,j in enumerate(ht_list):
                temp_sample = copy.copy(random_set)
                temp_sample[ht_list[l]]= 1
                if temp_sample["Mg"]<bound_dict['Mg'][1] and bound_dict[ht_list[l]][1]==1: #### ignore samples with Mg content greater than upper bound of Mg (here 95%)
                    temp_sample
                    final_samples.append(temp_sample)
            s+=1                    
        
        sampler_df = pd.DataFrame (final_samples, columns= list(bound_dict.keys()))
        sampler_df = sampler_df[self.elem_og_sorted]


        #print(self.elem_og_sorted)
        #print(sampler_df.columns)

        return sampler_df             
    
    
    def prediction(self):
        ht_list = ['Extruded', 'ECAP','Cast_Slow', 'Cast_Fast', 'Cast_HT','Wrought']
        all_closest_pred = pd.DataFrame(columns = list(self.x.columns) + self.output_names)

        for i in range(self.iter_num):
            samples = self.sampler()
           
            yhat = self.model.predict(samples)
            closest_prediction_index = abs(yhat - self.desired_y).argmin()
            closest_prediction_y = yhat[closest_prediction_index]
            closest_prediction_cc = samples.iloc[closest_prediction_index, :]
            closest_prediction = pd.concat([closest_prediction_cc,pd.Series(closest_prediction_y) ], axis=0)
            #print(closest_prediction.values.reshape(1,-1))
            closest_prediction = pd.DataFrame(closest_prediction.values.reshape(1,-1), columns = all_closest_pred.columns)
            
            all_closest_pred = pd.concat([all_closest_pred,closest_prediction], axis=0)
            
            # ✅✅✅✅
            ##### Mangos' paper : mean of distribution will be changed in the next iteration ???????
            
            std_dv = 0.1 / ((i+1)/self.iter_num)
            for k,v in dict(closest_prediction_cc).items():
                
                if k not in ht_list:
                    if v != 0 :
                        self.bound_dict[k] = (max(0,v-2* std_dv), v+2* std_dv)
                    else:
                        self.bound_dict[k] = (0,0)
                else: ### k in ht_list
                    self.bound_dict[k] = (0,1)
            #print(self.bound_dict)
            
        return all_closest_pred
        

