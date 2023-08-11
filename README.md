# deep-learning-challenge
Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
What variable(s) are the target(s) for your model?
What variable(s) are the feature(s) for your model?
Drop the EIN and NAME columns.

Determine the number of unique values for each column.

For columns that have more than 10 unique values, determine the number of data points for each unique value.

Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.

Use pd.get_dummies() to encode categorical variables.

Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

Compile, Train, and Evaluate the Model
 Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train_scaled[0])
hidden_nodes_layer1 = 10
hidden_nodes_layer2 = 5

nn_model = tf.keras.models.Sequential()

First hidden layer
nn_model.add(tf.keras.layers.Dense(units=hidden_nodes_layer1,
             input_dim=number_input_features, activation="tanh"))

Second hidden layer
nn_model.add(tf.keras.layers.Dense(
    units=hidden_nodes_layer2, activation="relu"))

Output layer
nn_model.add(tf.keras.layers.Dense(units=1, activation="relu"))


Check the structure of the model
nn_model.summary()


#Optimization:
Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
units_1 = 75
units_2 = 30
input_features = len(X_train_scaled[0])
nn_model1 = tf.keras.models.Sequential()

First hidden layer
nn_model1.add(tf.keras.layers.Dense(units=units_1, input_dim = input_features, activation = "relu"))

Second hidden layer
nn_model1.add(tf.keras.layers.Dense(units=units_2, activation = "relu"))

Output layer
nn_model1.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

Check the structure of the model
nn_model1.summary()

#RESULT:
268/268 - 0s - loss: 0.5196 - accuracy: 0.7885 - 263ms/epoch - 981us/step
Loss: 0.5195726752281189, Accuracy: 0.7884548306465149


