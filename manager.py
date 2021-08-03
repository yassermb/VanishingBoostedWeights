from soa import do_expr
import sys
import argparse
import pandas as pd
import os
import multiprocessing




def main(argv):
    
    use_multiprocessing = True
    parser=argparse.ArgumentParser(
            description='''Vanishing Boosted Weights (VBW): A corrective fine-tuning procedure on decision stumps.''')
    parser.add_argument('-a', '--algorithm', type=str, nargs='+', default=['GBoost','CatB','GOSS','VBW','LightGBM','Averaged'], help='List of arguments (default: GBoost CatB GOSS VBW LightGBM Averaged])')
    parser.add_argument('-f', '--features', type=int, default=10, help='Number of features (default: 10)')
    parser.add_argument('-e', '--estimators', type=int, nargs='+', default=[1, 5, 10, 25, 50, 75, 100], help='List of number of estimators (default: 1 5 10 25 50 75 100)')
    parser.add_argument('-d', '--data', type=str, default='./Examples', help='Path to datasets (default: ./Examples)')
    parser.add_argument('-p', '--process', type=int, default=4, help='Number of processes (default: 4)')
    args=parser.parse_args()
    
    db_path = args.data
    algorithms = args.algorithm
    features = args.features
    estimators = args.estimators
    max_cpus = args.process
    
    if max_cpus == 1:
        use_multiprocessing = False

    db_cases = []
    for db_name in os.listdir(db_path):
        db_file = os.path.join(db_path, db_name)
        db_df = pd.read_table(db_file, sep = ' ', error_bad_lines=False, header = None).sample(frac=1)
        y_all = (db_df.iloc[:,-1].to_numpy() + 1) // 2
        X_all = db_df.drop(db_df.columns[-1],axis=1).to_numpy()    
        db_cases.append((X_all, y_all, db_name.replace('.txt',''), algorithms, features, estimators))
    
    if use_multiprocessing:
        manager = multiprocessing.Manager()
        report_dict = manager.dict()
        pool = multiprocessing.Pool(processes = min(max_cpus, multiprocessing.cpu_count()))
    else:
        report_dict = dict()
    
    for args in db_cases:
        args += (report_dict,)
        if use_multiprocessing:
            pool.apply_async(do_expr, args = args)
        else:
            do_expr(*args)
    
    if use_multiprocessing:
        pool.close()
        pool.join()
        
    print(dict(report_dict))

if __name__ == "__main__":
   main(sys.argv)
