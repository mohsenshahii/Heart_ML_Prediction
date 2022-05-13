from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
def apply_model(x, y,model):
  '''
    This function receives the X and y dataset set, a model name, splits into train and test, 
    applies the model on the train data, make prediction after training, 
    and returns the accuracy and precision
  '''
  x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.3, random_state=1)
  model_ = model
  model_.fit(x_train, y_train)
  y_pred = model_.predict(x_test)

  accuracy = accuracy_score(y_test, y_pred)
 
  precision = precision_score(y_test, y_pred)

  return("Accuracy score:", accuracy,
          "Precision score:", precision
          ) 