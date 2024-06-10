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

def get_highest_accuracy(results, model_name):
    max_acc = 0
    test_acc=0
    r_std=0
    std=0
    best_combination = None
    for file_name, data in results.items():
        if model_name in file_name:
            max_val_acc  = data[0]['val_acc_mean']
            std=data[0]['val_acc_std']
            if max_val_acc > max_acc:
                max_acc = max_val_acc
                test_acc=data[0]['test_acc_mean']
                test_std=data[0]['test_std_mean']
                r_std=std
                best_combination = file_name
    return max_acc,std, best_combination, test_acc,test_std

def main():
    results_dir = '/mnt/data-test/results'  # Directory where the results pickle files are stored
    
    results = read_pickle_files(results_dir)
    print(type(results))
    print(results.keys)
    for model_name in ['model_1','model_2','model_3','model_4','model_5','model_6','model_7','model_8','model_9','model_10']:
        max_acc, max_acc_std, best_combination,test_acc,test_std = get_highest_accuracy(results, model_name)
    
        if best_combination:
            print(f"The highest validation accuracy achieved by {model_name} is {max_acc*100:.2f}% with a standard deviation of {max_acc_std*100:.2f}%")
            print(f"Test acc is {test_acc*100:.2f}% with a standard deviation of {test_std*100:.2f}%")
            print(f"Best combination: {best_combination}")
        else:
            print(f"No results found for {model_name}")

        print("\n")
if __name__ == "__main__":
    main()
