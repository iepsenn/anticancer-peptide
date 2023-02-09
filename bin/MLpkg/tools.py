def save_model_joblib(model, root_dir, file_name):
    from joblib import dump
    dump(model, root_dir + file_name)

def load_model_joblib(model, root_dir, file_name):
    from joblib import load
    load(model, root_dir + file_name)

def save_model_pickle(model, root_dir, file_name):
    import pickle
    with open(root_dir + file_name, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model_pickle(model_file_path):
    import pickle
    model = None
    with open(model_file_path, 'rb') as handle:
        model = pickle.load(handle)
    return model
