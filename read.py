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
    r_std=0
    best_combination = None
    for file_name, data in results.items():
        if model_name in file_name:
           
            max_val_acc  = data[0]['val_acc_mean']
            std=data[0]['val_acc_std']
            if max_val_acc > max_acc:
                max_acc = max_val_acc
                r_std=std
                best_combination = file_name
    return max_acc,std, best_combination

def main():
    results_dir = 'results'  # Directory where the results pickle files are stored
    model_name = 'model_1'  # Specify the model name to search for
    
    results = read_pickle_files(results_dir)
    print(type(results))
    print(results.keys)
    max_acc,std, best_combination = get_highest_accuracy(results, model_name)
    
    if best_combination:
        print(f"The highest validation accuracy achieved by {model_name} is {max_acc*100:.2f}% with a standard deviation of {std*100:.2f}%")

        print(f"Best combination: {best_combination}")
    else:
        print(f"No results found for {model_name}")

if __name__ == "__main__":
    main()