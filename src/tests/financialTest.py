import numpy as np
import os
import sys

src_path = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
sys.path.insert(0, src_path)

from models.LinearRegression import LinearRegression
from models.DecisionTree import DecisionTree, prepare_data_for_decision_tree
from trade_stocks.trade import Trading

def financialTest(df,model=None,feature_columns=None,target_column=None):
    """
    Using economic criteria to determine a model is good or bad.
    This function will return all 4 economic features, so if you want to find another feature, u can modify the trading class.
    
    Economic features :
    - Total protfit
    - Total trade number
    - Win rate
    - Profit factor
    
    Args:
        df: (pd.DataFrame): The input data , can be in-sample or out-of-sample.
    
    """
    selected_features = []
    y_predict = None

    if feature_columns is None:
        selected_features = ['Net_advances','A/D','Schultz','McClellan_Oscillator']
    else:
        selected_features = feature_columns
    if target_column is None:
        target_feature = ['daily_returns']
    else:
        target_feature = target_column

    if model == 'LinearRegression':
        y_predict = []
        model = LinearRegression()
        ## Create new numpy array with the selected features    
        X_selected = df[selected_features].to_numpy()
        y_selected = df[target_feature].to_numpy()

        # Train the model
        len_X_selected = len(X_selected)
        # Predict the next number_of_days_to_predict days
        number_of_days_to_predict = 3
        y_pred_set = [[] for _ in range(len_X_selected+number_of_days_to_predict)]
        for i in range(len_X_selected-number_of_days_to_predict):
            if i < 30:
                continue
            X_train = X_selected[(i-30):i,]
            y_train = y_selected[(i-30):i,]
            model.fit(X_train,y_train)
            y_pred = model.predict(X_selected[(i+1):(i+1+number_of_days_to_predict),])
            for j in range(number_of_days_to_predict):
                if i+j+1 < len_X_selected:
                    y_pred_set[i+j+1].append(y_pred[j])

        # Calculate the mean of the predicted values
        y_pred_set = y_pred_set[:len_X_selected]
        for i in range(len_X_selected):
            if i < 30:
                y_predict.append(np.nan)
                continue
            y_predict.append(np.mean(y_pred_set[i]))

        # Convert the predicted values to a numpy array
        y_predict = np.array(y_predict).reshape(-1,1)
    if model == 'DecisionTree':
        model = DecisionTree(max_depth=10,min_sample_split=2)
        # Prepare data for decision tree training
        X, y = prepare_data_for_decision_tree(df,selected_features, target_column=target_feature)
        split_index = int(len(X) * 0.8)  # Calculate the 80% split index

        # Split the data
        X_train, X_test = X[:split_index], X[split_index:]
        y_train = y[:split_index]

        # Train the model
        model.fit(X_train, y_train)
        # Predict y
        y_test = model.predict(X_test)
        y_predict = np.pad(y_test, (len(y_train), 0),mode= 'constant', constant_values=0)
    
    if model == 'NeuralNetwork':
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        import tensorflow as tf
        from tensorflow import keras
        from keras import layers
        import numpy as np

        tf.get_logger().setLevel('ERROR')

        X,y = prepare_data_for_decision_tree(df, selected_features, target_feature,method_to_create_threshold='up_and_down')
        print(X.shape)
        print(y.shape)
        split_index = int(len(X) * 0.8)  # Calculate the 80% split index

        # Split the data
        X_train, X_test = X[:split_index], X[split_index:]
        y_train,y_test = y[:split_index], y[split_index:]        

        # Because sparse_categorical_crossentropy is used, the labels should be in the range of 0 to num_classes - 1
        # Convert y_train and y_test to positive integer labels
        y_train = y_train + 1
        y_test = y_test + 1

        # Convert the data to one-hot encoding
        y_train = tf.one_hot(y_train, depth=3)
        y_test = tf.one_hot(y_test, depth=3)

        # Define the model
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(16, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=50,verbose=0)    

        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = y_pred -1
        y_predict = np.pad(y_pred,(len(y_train),0),'constant',constant_values=0)    

    # Trading:
    trading = Trading()

    trade_signal = trading.generate_trade_signal(y_predict)
    finally_capital = trading.execute_trade(trade_signal,np.array(df['Close']))
    print("Finally capital: ", finally_capital)
    total_profit, total_trade_number, win_rate, profit_factor = trading.performance()
    
    # Evaluate the model
    if total_profit > 0:
        print("The model helps us to get profit")
    if total_trade_number > 3000:
        print("The total trade number is more than 3000, too high")
    if 0.5 < win_rate < 0.8:
        print("The win rate is higher than 0.5 and lower than 0.8")
    if profit_factor > 1:
        print("The profit factor is higher than 1, that's good")
    return total_profit, total_trade_number, win_rate, profit_factor