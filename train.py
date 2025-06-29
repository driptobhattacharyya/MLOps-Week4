# train.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib  # For saving/loading the model

def load_data():
    data = pd.read_csv('data/iris.csv')
    train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
    X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
    y_train = train.species
    X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
    y_test = test.species
    return X_train, X_test, y_train, y_test

def train_model():
    X_train, X_test, y_train, y_test = load_data()
    # model = RandomForestClassifier(n_estimators=10)
    # model.fit(X_train, y_train)
    # # Save the model
    # joblib.dump(model, 'iris_model.pkl')
    # return model, X_test, y_test
    mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
    mod_dt.fit(X_train,y_train)
    prediction=mod_dt.predict(X_test)
    print('The accuracy of the Decision Tree is',"{:.3f}".format(accuracy_score(prediction,y_test)))
    joblib.dump(mod_dt, "artifacts/model.joblib")

if __name__ == "__main__":
    model, X_test, y_test = train_model()
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")