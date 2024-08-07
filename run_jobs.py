import itertools
import os
import subprocess
import argparse
import pickle

def hyperparam_combinations(hyperparams):
    keys, values = zip(*hyperparams.items())
    
    # Predefined hyperparameters
    predefined_hyperparams = {
        #'model': hyperparams['model'],
        'dataset': hyperparams['dataset'],
        'epochs': hyperparams['epochs'],
        'patience': hyperparams['patience'],
        'runs': hyperparams['runs']
    }
    
    # Generate combinations for variable hyperparameters
    variable_hyperparams = {key: value for key, value in hyperparams.items() if key not in predefined_hyperparams}
    variable_combinations = [dict(zip(variable_hyperparams, combination)) for combination in itertools.product(*variable_hyperparams.values())]
    
    # Add predefined hyperparameters to each combination
    combinations = [dict(predefined_hyperparams, **comb) for comb in variable_combinations]
    
    return combinations

def check_existing_results(combination):
   
    
    nhid_list = str([int(float(i)) for i in combination['nhid_list'].replace(' ', '').split(',')])
    #nhid_list = str([int(float(i)) for i in combination['nhid_list'].replace(' ', '').split(',') if i]) if combination['nhid_list'] else combination['nhid_list']
    
    file_name = f"/mnt/data-test/results/{combination['model']}_dataset={combination['dataset']}_nhid={nhid_list}_dropout={combination['dropout']}_epochs={combination['epochs']}_lr={combination['lr']}_wd={combination['wd']}_patience={combination['patience']}_runs={combination['runs']}.pkl"
    #file_name = f"results/{combination['model']}_dataset={combination['dataset']}_nhid={nhid_list}_dropout={combination['dropout']}_epochs=200_lr=0.05_wd=0.0005_patience=10_runs=3.pkl"
    #file_name = f"results/model_1_dataset=cora_nhid=[256, 512]_dropout=0.5_epochs=200_lr=0.05_wd=0.0005_patience=10_runs=3.pkl"
    
    return os.path.exists(file_name)

def run_experiment(combination):
   
    combination

    nhid_str = ' '.join(map(str, combination['nhid_list']))
    command = [
        'python', '/mnt/data-test/run_experiments.py',
        '--dataset', str(combination['dataset']),
        '--model', str(combination['model']),
        '--runs', str(combination['runs']),
        '--nhid_list', nhid_str,
        '--dropout', str(combination['dropout']),
        '--lr', str(combination['lr']),
        '--wd', str(combination['wd']),
        '--epochs', str(combination['epochs']),
        '--patience', str(combination['patience'])
    ]
    
    subprocess.run(command)

def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model_1', help='Which model to train.')
    parser.add_argument('--dataset', type=str, default='cora', help='dataset.')
    args = parser.parse_args()
    print(args.dataset)
    
    hyperparams = {
        'model': [args.model],
        'dataset': args.dataset,
        'epochs': 400,
        'patience':100,
        'runs': 10,
        'lr': [0.05, 0.01, 0.005, 0.001],
        'wd':[5e-4, 5e-3, 1e-4, 1e-3],
        'dropout':  [0.5, 0.3, 0.4, 0.2],
        'nhid_list': ['512,128,64,32','512,128,64','512,128','32,16','64,32','16,16','32,32','256','128','64','32','16']
    }
    
    combinations = hyperparam_combinations(hyperparams)
    print(combinations)
    
    for combination in combinations:
        if not check_existing_results(combination):
            run_experiment(combination)
        else:
            print(f"Skipping combination {combination} as results already exist.")

if __name__ == "__main__":
    main()
