import os
import pickle

def read_pickle_files(results_dir):
    results = {}
    for file_name in os.listdir(results_dir):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(results_dir, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                results[file_name] = data
    return results

def get_highest_accuracy(results, model_name, data_name):
    max_acc = 0
    test_acc = 0
    test_std = 0
    r_std = 0
    std = 0
    best_combination = None
    for file_name, data in results.items():
        if model_name in file_name and data_name in file_name:
            max_val_acc = data[0]['val_acc_mean']
            std = data[0]['val_acc_std']
            if max_val_acc > max_acc:
                max_acc = max_val_acc
                test_acc = data[0]['test_acc_mean']
                test_std = data[0]['test_acc_std']
                r_std = std
                best_combination = file_name
    return max_acc, r_std, best_combination, test_acc, test_std

def main():
    results_dir = '/mnt/data-test/results'  # Directory where the results pickle files are stored
    
    datasets = ['cora', 'citeseer', 'pubmed']  # List of datasets
    results = read_pickle_files(results_dir)
    print(type(results))
    print(results.keys())

    for model_name in ['model_13', 'model_14', 'model_15', 'model_16', 'model_17', 'model_18']:
        print(f"Results for {model_name}:")
        for data_name in datasets:
            max_acc, max_acc_std, best_combination, test_acc, test_std = get_highest_accuracy(results, model_name, data_name)
        
            if best_combination:
                print(f"  Dataset: {data_name}")
                print(f"    Highest validation accuracy: {max_acc*100:.2f}% with a standard deviation of {max_acc_std*100:.2f}%")
                print(f"    Test accuracy: {test_acc*100:.2f}% with a standard deviation of {test_std*100:.2f}%")
                print(f"    Best combination: {best_combination}")
            else:
                print(f"  Dataset: {data_name}")
                print(f"    No results found")
        
        print("\n")

if __name__ == "__main__":
    main()
