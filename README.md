# lithology-prediction
1.Built a machine learning model to automate rock type classification using well logs.

2.Engineered a Random Forest classifier on a datasets and achieved a 96.7% Accuracy on blind test data, successfully distinguishing between the various rock types.

3.To verify the model's performance, I generated a confusion matrix.The diagonal line represnts points where the model's prediction matched the actual prediction(1,1) means they are the same rock class.

4.I validated the model's geological accuracy by plotting the predicted lithology against the raw Gamma Ray(GR) curve.

5.Notice the correlation when the GR increases the predicted lithology is green(shale).This shows the model has learnt fundamental petrophysical relationship between radioactivity and lithology.

Note for Users:

1.This model is trained on the FORCE 2020 datasets

2.To run this on your own data ensure your CSV file contains the same feature columns  or modify the feature list in the script else the model will crash

3.Update the path variable in the script to your local file location
