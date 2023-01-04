from models_methods import ModelsComparison, read_df

for elem in ("disaster", "mental_health", "suspicious_communiaction"):
    print(elem)
    X_train, X_test, y_train, y_test = read_df(f"data/{elem}.json")
    compare = ModelsComparison((X_train, y_train), (X_test, y_test), elem)
    compare.train_all()
    compare.compare()
