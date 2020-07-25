import os
import sys
import argparse
import json
from datetime import datetime

import ConfigSpace as CS

from ray import tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="config.json", help="Directory for JSON config file")
parser.add_argument("--num_samples", type=int, default=30, help="Number of hyperparameter trials to run")
parser.add_argument("--log_dir", type=str, default="logs", help="Directory for trial metrics and output")
parser.add_argument("--mini_batching", type=bool, default=False, help="Whether to run with mini_batching")
parser.add_argument("--num_steps", type=int, default=20, help="Number of updates the model will take")
parser.add_argument("--num_reports", type=int, default=3, help="Number of times to intermittently report the verification accuracy")
args = parser.parse_args()
standard_flags = f'--tune=True --font_dict_path=../../fonts/multifont_mapping.pkl --log_dir={args.log_dir}'


def train_triplet_loss_modular(conf):
    global standard_flags
    flags = f'python3 ../../train_triplet_loss_modular.py {standard_flags}'
    if args.mini_batching:
      flags += f' --train_iterations={conf["batch_multiplier"]*args.num_steps} --reporting_interval={(conf["batch_multiplier"]*args.num_steps)//args.num_reports}'
    for i in conf:
      flags+=f' --{i}={str(conf[i])}'
    print(flags)
    os.system(flags)
    if os.path.exists("logs/metric.txt"):
      textfile = open("logs/metric.txt", 'r')
      metric = float(textfile.readline())
      textfile.close()
      tune.report(testing_acc=metric, done=True)
    else:
      # Most likely out of memory. We dont want to use this hyperparameter set.
      tune.report(testing_acc=0, done=True)

def triplet_loss_modular_hyperparameter_tuning():
  global standard_flags
  if not args.mini_batching:
    standard_flags += f' --train_iterations={args.num_steps} --reporting_interval={args.num_steps//args.num_reports} --mini_batching=False'
  else:
    standard_flags += f' --mini_batching=True'
  # Define logdir file, create it if does not exist
  _init_time = datetime.now()
  logdir = f"logs_hpo_{_init_time.astimezone().tzinfo.tzname(None)+_init_time.strftime('%Y%m%d_%H_%M_%S_%f')}"
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  
  # Load JSON config file
  with open(args.config_file) as config_file:
    data = json.load(config_file)
    # makes a copy of it in the session log
    with open(os.path.join(logdir, "config.json"), 'w') as copy:
      json.dump(data, copy)
  
  # Extract settings + hyperparameter config
  config = data['hyperparameter_config_space']
  
  # Add constant hyperparameters to flags
  for constant,value in config['constants'].items():
    standard_flags+=f' --{constant}={value}' 
    
  # Extract hyperparameters from JSON file and add to configuration space. Also account for any constraints.
  config_space = CS.ConfigurationSpace()
  constraints = {}
  for name,settings in config['search_space'].items():
    hp_type = settings['type']
    if hp_type == 'UF':
      hp = CS.UniformFloatHyperparameter(name, lower=float(settings['lower']), upper=float(settings['upper']))
    elif hp_type == 'UI':
      hp = CS.UniformIntegerHyperparameter(name, lower=int(settings['lower']), upper=int(settings['upper']))
    elif hp_type == 'C':
      hp = CS.CategoricalHyperparameter(name, choices=settings['options'].split(','))
    else:
      raise ValueError(f"Undefined Hyperparameter Type: {hp_type}")
    config_space.add_hyperparameter(hp)
  
  # Run hyperparameter optimization
  experiment_metrics = dict(metric="testing_acc", mode="min")
  bohb_hyperband = HyperBandForBOHB(time_attr="training_iteration",**experiment_metrics)
  bohb_search = TuneBOHB(config_space, **experiment_metrics)
  analysis = tune.run(train_triplet_loss_modular,
      name=logdir,
      scheduler=bohb_hyperband,
      search_alg=bohb_search,
      num_samples=args.num_samples, resources_per_trial={"gpu":1}, local_dir="./")
  print("Best config: ", analysis.get_best_config(metric="testing_acc", mode="max"))
  
  # saves relevant summary data to file under logdir
  df = analysis.dataframe()
  df.to_csv(os.path.join(logdir, "data.csv"))

if __name__ == '__main__':
  triplet_loss_modular_hyperparameter_tuning()
  