How to run the program:

1. Install the libs used in this project.

2. Run 'data_sampling.py' to sample the original dataset.
    A csv file named 'processed_data.csv' will be generated.

3. Run 'preprocessing.py' to perform data cleaning, feature selection etc.

4. Run 'modeling.py' to train the processed dataset using Logistic Regression and Random Forest.



File description:

'US_Accidents_Dec20.csv' is the original dataset obtained from Kaggle.

'data_sampling.py' is the program that randomly samples 80,000 rows of the original dataset. 
 Its output is 'sampled_data.csv'.

'preprocessing.py' is the program that performs data cleaning, categorical data processing, feature selection, frequency encoding 
and one-hot encoding so that the output can be used to model training. 
Its output is 'processed_dataset.csv'.

'modeling.py' is the program that train the output from 'preprocessing.py'. It uses Logistic Regression and Random Forest to 
train the data and evaluated using Confusion matrix. Its output is the evaluation results.



Tips:

Some values obtained might be slightly different from the result shown in the final report due to the random sampling,
but it does not affect the analysis and conclusion.