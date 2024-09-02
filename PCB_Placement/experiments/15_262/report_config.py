from datetime import datetime
import json

def save_report_config(filename, report_config):
    # Write default_hyperparameters dict to a json file
    with open(filename, 'w') as fp:
        json.dump(report_config, fp)
    
    fp.close()
        
def load_report_config(filename):
    fp = open(filename, 'r')
    report_config = json.load(fp)
    fp.close()
    return report_config    
        
charts = {
    'parameter_test_262': {'experiments': ['parameter_experiment_262'],
                                 'algorithms': ['TD3', 'SAC', 'PPO', 'DDPG', 'A2C'],
                                 'multi_agent': True,
                                 'mean_std_plot': True,
                                 'ylim_bot': -1000,
                                 'xscale': 'k',
                                 'window': 100,
                                 'title': "Parameter test w/ emphasis on EW",  # (W=2, H=6, O=2)
                                 'xlabel': "Timesteps (unit)",
                                 'ylabel': "Average return (unit)",
                                 'label': {
                                     # PARTIALLY SUPPORTED FOR MULTI AGENT
                                     'parameter_experiment_262:TD3': 'TD3',
                                     'parameter_experiment_262:SAC': 'SAC',
                                     'parameter_experiment_262:PPO': 'PPO',
                                     'parameter_experiment_262:DDPG': 'DDPG',
                                     'parameter_experiment_262:A2C': 'A2C',
                                 },
                                 },
         }
    
tables = {
         }

report_config = {'charts': charts,
                 'tables': tables,
                 'author': 'Luke Vassallo & YiFei Yu',
                 'timestamp': f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                }

save_report_config("./report_config.json", report_config)

#rc = load_report_config("./report_config.json")
#for key,value in rc.items():
    #print(f'{key} : {value}')
