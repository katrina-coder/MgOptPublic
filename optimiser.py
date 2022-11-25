from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import pickle
import ipywidgets as widgets
import random



from IPython import display as disp
if 'google.colab' in str(get_ipython()):
    from MgOptPublic.model_paths import models
    from MgOptPublic.BO import alloys_bayes_opt
    from MgOptPublic.BO_append import alloys_bayes_opt_append
    from MgOptPublic.inverse_prediction import inverse_pred
else:
    from model_paths import models
    from BO import alloys_bayes_opt
    from BO_append import alloys_bayes_opt_append
    from inverse_prediction import inverse_pred


class scanSettings:
    def __init__(self, mode):
        self.mode = mode
        if self.mode == 'Bayesian Optimization':
            self.sampling_size = 100000
            self.num_of_suggestions = 20
            self.num_elems = 6
            self.sum_elems = 20
            self.output_names = ['UTS', 'Ductility']
            self.normalize_target = 'True'
            self.append_suggestion = 'True'
            
            # todo: check arrange of bound dict
            self.range_based_inputs =  {
                'Mg': (80.0, 95.0), 'Y': (0.0, 19.0),
                'Zn': (0.0, 14.3), 'Al': (0.0, 11.0),
                'Ca': (0.0, 10.0), 'Gd': (0.0, 10.0),
                'Sn': (0.0, 9.56), 'Nd': (0.0, 8.05),
                'La': (0.0, 6.0), 'Er': (0.0, 6.0),
                'Si': (0.0, 5.0), 'Ce': (0.0, 3.92),
                'Zr': (0.0, 3.0), 'Li': (0.0, 3.0),
                'Yb': (0, 3), 'Sr': (0.0, 2.45),
                'Mn': (0.0, 2.0), 'Pr': (0.0, 1.76),
                'Ho': (0.0, 1.4), 'Sb': (0.0, 1.0001),
                'Ni': (0.0, 1.0), 'Ga': (0, 1.0),
                'Tb': (0.0, 1.0),
                'Extruded': (0, 1), 'ECAP': (0, 1), 'Cast_Slow': (0, 1),
                'Cast_Fast': (0, 1), 'Cast_HT': (0, 1), 'Wrought': (0, 1),
                'Cu': (0.0, 0.5), 'Ag': (0.0, 0.5),
                'Bi': (0.0, 0.5), 'Sc': (0.0, 0.5),
                'Be': (0.0, 0.0), 'Fe': (0.0, 0.0),
                'Th': (0.0, 0.0), 'Dy': (0.0, 0.0)}

### This is hard coded ***sorted original bound dict*** that is equal to the arrangment of training data (X).
### Models have been trained with this order of X, so we want to have the exact same order in sampler.df.

        elif self.mode == 'Inverse Prediction':
            self.sampling_size = 100
            self.num_of_suggestions = 2
            self.num_elems = 6
            self.sum_elems = 20
            self.categorical_default = "True"
            self.desired_y = 450
            self.output_name = 'UTS'
            
            # todo: check arrange of bound dict
            self.range_based_inputs =  {
                'Mg': (80.0, 95.0), 'Y': (0.0, 19.0),
                'Zn': (0.0, 14.3), 'Al': (0.0, 11.0),
                'Ca': (0.0, 10.0), 'Gd': (0.0, 10.0),
                'Sn': (0.0, 9.56), 'Nd': (0.0, 8.05),
                'La': (0.0, 6.0), 'Er': (0.0, 6.0),
                'Si': (0.0, 5.0), 'Ce': (0.0, 3.92),
                'Zr': (0.0, 3.0), 'Li': (0.0, 3.0),
                'Yb': (0, 3), 'Sr': (0.0, 2.45),
                'Mn': (0.0, 2.0), 'Pr': (0.0, 1.76),
                'Ho': (0.0, 1.4), 'Sb': (0.0, 1.0001),
                'Ni': (0.0, 1.0), 'Ga': (0, 1.0),
                'Tb': (0.0, 1.0),
                'Extruded': (0, 1), 'ECAP': (0, 1), 'Cast_Slow': (0, 1),
                'Cast_Fast': (0, 1), 'Cast_HT': (0, 1), 'Wrought': (0, 1),
                'Cu': (0.0, 0.5), 'Ag': (0.0, 0.5),
                'Bi': (0.0, 0.5), 'Sc': (0.0, 0.5),
                'Be': (0.0, 0.0), 'Fe': (0.0, 0.0),
                'Th': (0.0, 0.0), 'Dy': (0.0, 0.0)}
        
        else:
            self.loss_type = 'Percentage'
            self.max_steps = 1
            self.targets = {
                'elongation%': 6,
                'tensile strength(MPa)': 250
            }
            self.categorical_inputs = {
                'Heat Treatment': [1]
            }
            self.categorical_inputs_info = {
                'Heat Treatment': {'span': [1, 2, 3, 4, 5, 6], 'tag': ['Extruded', 'ECAP', 'Cast_Slow', 'Cast_Fast', 'Cast_HT', 'Wrought']}}
            
#             self.range_based_inputs = dict(zip(
#                 ['Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag', 'Ho', 'Mn',
#                  'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga', 'Be', 'Fe',
#                  'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'],
#                 [[0.827], [0.0026], [0], [0], [0], [0], [0.065], [0.0945], [0],
#                  [0], [0], [0], [0], [0], [0], [0.0076],
#                  [0], [0], [0], [0], [0], [0], [0], 
#                  [0], [0], [0], [0], [0], [0.0032], [0], [0]]))
            
            self.range_based_inputs = dict(zip(
                ['Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag', 'Ho', 'Mn',
                 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga', 'Be', 'Fe',
                 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'],
                [[100], [0] , [0] , [0] , [0] , [0] ,
                    [0] , [0] , [0] , [0] , [0],
                    [0] , [0] , [0] , [0] , [0], 
                    [0] , [0] , [0] , [0] , [0], 
                    [0] , [0] , [0] , [0] , [0], 
                    [0] , [0] , [0] , [0] , [0]]))
        
            self.range_based_inputs['Mg'] = [100 - sum(sum(row) for row in list(self.range_based_inputs.values())[1:-1])]
                

class optimiser:
    def __init__(self, settings): 
        self.mode = settings.mode
        if 'Bayesian' in self.mode:
            self.num_of_suggestions = settings.num_of_suggestions
            self.normalize_target = settings.normalize_target
            self.sampling_size = settings.sampling_size
            self.output_names = settings.output_names
            self.range_based_inputs = settings.range_based_inputs
            self.settings = settings
        elif 'Inverse' in self.mode:
            self.num_of_suggestions = settings.num_of_suggestions
            self.sampling_size = settings.sampling_size
            self.range_based_inputs = settings.range_based_inputs
            self.desired_y = settings.desired_y
            self.settings = settings
            self.output_name = settings.output_name
        else:
            self.step_batch_size = 100
            self.step_final_std = 0.01
            self.finetune_max_rounds = 3
            self.finetune_batch_size = 10
            self.mode = settings.mode
            self.loss_type = settings.loss_type
            self.targets = settings.targets
            self.max_steps = settings.max_steps
            self.categorical_inputs = settings.categorical_inputs
            self.range_based_inputs = settings.range_based_inputs
            self.settings = settings
        self.models = models
        self.run()

    def run(self):
        ############ here
        if 'Bayesian' in self.mode:
            print('========== Bayesian Optimization Started ==========')
            if self.normalize_target:
                if len(self.output_names)==1:
                    if 'UTS' in self.output_names:
                        gp_model_list = [self.models['gp_UTS_normalized']]
                    else:
                        gp_model_list = [self.models['gp_Ductility_normalized']]
                else:
                    gp_model_list = [self.models['gp_UTS_normalized'], self.models['gp_Ductility_normalized']]
            else:
                if len(self.output_names)==1:
                    if 'UTS' in self.output_names:
                        gp_model_list = [self.models['gp_UTS']]
                    else:
                        gp_model_list = [self.models['gp_Ductility']]
                else:
                    gp_model_list = [self.models['gp_UTS'], self.models['gp_Ductility']]
            if 'UTS' in self.output_names:
                if 'Ductility' in self.output_names:
                    rf_model_list = [self.models['rf_UTS'],self.models['rf_Ductility']]
                else:
                    rf_model_list = [self.models['rf_UTS']]
            else:
                rf_model_list = [self.models['rf_Ductility']]
            iter_num = self.num_of_suggestions
            bound_dict = self.range_based_inputs
            # categorical_dict = {'Extruded': (0,1), 'ECAP': (0,1), 'Cast_Slow': (0,1), 'Cast_Fast': (0,1), 'Cast_HT': (0,1),'Wrought': (0,1)}
            # bound_dict.update(categorical_dict)
            if self.settings.append_suggestion:
                opt = alloys_bayes_opt_append(output_names = self.output_names, 
                                num_elems=self.settings.num_elems, sum_elems = self.settings.sum_elems, 
                                sample_size=self.sampling_size, iter_num=iter_num,
                                append_suggestion=True, bound_dict = bound_dict, 
                                normalize_target=self.normalize_target)
                df = opt.get_suggestions()
            else:
                opt = alloys_bayes_opt(gp_model_list, rf_model_list, output_names = self.output_names, 
                                num_elems=self.settings.num_elems, sum_elems = self.settings.sum_elems, 
                                sample_size=self.sampling_size, iter_num=iter_num,
                                append_suggestion=False, bound_dict = bound_dict, 
                                normalize_target=self.normalize_target)
    
                df = opt.get_suggestions()
            pd.set_option("display.max_columns", 42)
            df.to_csv(str(random.random())+' suggestions.csv',index=False)
            disp.display(df)
            print('========== Bayesian Optimization Finished ==========')
            print()
        
        elif 'Inverse' in self.mode:
            print('========== Process Started ==========')
            #print(self.range_based_inputs)
            #print(self.desired_y)
            inversed_cc = inverse_pred(output_names=self.output_name, 
                                       desired_y=self.desired_y,
                                       sample_size=self.sampling_size, 
                                       iter_num=self.num_of_suggestions, 
                                       user_defined_range = self.range_based_inputs,
                                       num_elems=self.settings.num_elems,
                                       sum_elems=self.settings.sum_elems)
            all_pred = inversed_cc.prediction()
            disp.display(all_pred)
            print('========== Process Finished ==========')
            print()
        
        else:
            mg_balance = True
            print(self.settings.categorical_inputs)
            for key in self.settings.categorical_inputs:
                ht = self.settings.categorical_inputs[key]
            range_based_inputs = list(self.settings.range_based_inputs.values())
            range_based_inputs = [elem[0] for elem in range_based_inputs]
            if sum(range_based_inputs) != 100.0:
                mg_balance = False
            my_input = [100 - sum(range_based_inputs[1:])] + range_based_inputs[1:] + ht  
                    # [*self.range_based_inputs.values()] 
            alloy_array = np.reshape(my_input, (1, -1))
            final_alloy  = dict(zip(
                ['Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag', 'Ho', 'Mn',
                 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga', 'Be', 'Fe',
                 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi', 'Extruded', 'ECAP', 'Cast_Slow', 'Cast_Fast', 'Cast_HT', 'Wrought'],
                alloy_array.reshape(-1,)))
            print(final_alloy)
            print('\n')
            print('Predicted %f Elongation' % (self.models['elongation'].predict(alloy_array)[0]))
            print('Predicted %f Yield Strength' % (self.models['yield'].predict(alloy_array)[0]))
            print('Predicted %f Tensile Strength' % (self.models['tensile'].predict(alloy_array)[0]))
            
        

    def calculateStep(self, best_datapoint, step_number, target_var):
        if target_var == 'all':
            batch_size = self.step_batch_size
        else:
            batch_size = self.finetune_batch_size
        loss = [0] * batch_size
        datapoints = []
        std = self.step_final_std * (self.max_steps / float(step_number + 1))
        for i in range(batch_size):
            datapoints.append(deepcopy(best_datapoint))
            for key in self.categorical_inputs.keys():
                if target_var == key or target_var == 'all':
                    datapoints[i].categorical_inputs[key] = np.random.choice(self.categorical_inputs[key])
            for key in self.range_based_inputs.keys():
                if target_var == key or target_var == 'all':
                    if max(self.range_based_inputs[key]) != min(self.range_based_inputs[key]):
                        a = (min(self.range_based_inputs[key]) - np.mean(best_datapoint.range_based_inputs[key])) / std
                        b = (max(self.range_based_inputs[key]) - np.mean(best_datapoint.range_based_inputs[key])) / std
                        datapoints[i].range_based_inputs[key] = round(
                            float(truncnorm.rvs(a, b, loc=np.mean(best_datapoint.range_based_inputs[key]), scale=std)),
                            2)
                    else:
                        datapoints[i].range_based_inputs[key] = min(self.range_based_inputs[key])
            loss[i] = self.calculateLoss(datapoints[i])
        return min(loss), datapoints[loss.index(min(loss))]
