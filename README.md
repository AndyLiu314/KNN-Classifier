# KNN CLASSIFIER
## Made By: Andy Liu

**Overview:**
A K-Nearest Neighbours learning algorithm created using K values of [ 3, 5, 7, 9, 11 ]. Learning model performance is assessed using 10-fold cross-validation and evaluated on Accuracy and F1 Score metrics. 

**Data Preprocessing:**
Cylinders: one-hot
  - Cylinders in a car engine is a categorical variable, not numerical. Therefore, representing 
    it with one-hot can prevent the model from treating it as a continuous value.
  - Negative trade-off is that this will lead to increased dimensionality due to the increased 
    number of columns
    
Displacement: standardize
  - Displacement values vary widely in the dataset and will therefore benefit from being 
    standardized. This will also prevent feature scaling issues.
    
Horsepower: standardize
  - Like displacement, horsepower values exist in a wide range and should be standardized to 
    avoid scaling issues.
    
Weight: standardize
  - Weight values in the dataset have the highest numerical value compared to any other 
    feature and risk overpowering the other features. Standardizing weight will remove this 
    issue.
    
Acceleration: standardize
  - Acceleration has the smallest numerical value compared to the other features that are to 
    be standardized. As such, it runs the risk of being overshadowed by features like weight. I
    will standardize acceleration to prevent this issue.
    
Origin: one-hot
  - This is another categorical variable where its numerical value does not matter. I will 
    therefore represent it using one-hot encoding.
    
Model Year: remove
  - The model year of a car, within the context of this dataset, is not a great indicator of its 
    fuel efficiency. As such, I will drop this feature.
  - Older cars designed with fuel efficiency in mind can often outperform newer cars in mpg.
  - The dataset tracks cars from the 1970s to 1982.
    - Since this time span is relatively short, there won’t be a major difference between the cars due to their model year

Car Name: remove
  - Car names have too many unique values and are not useful in predicting fuel efficiency.
    Like the model year feature, I will drop this one as well.

**How To Run:**
Download the repository as a zip file, then unzip the file. The Python file and the TSV file must be in the same directory. Once the folder is unzipped, you can run the Python file from the terminal. NOTE: You must have a working installation of Python set up on your computer. You must also have the numpy module for Python installed on your computer. To install it, run this command in your terminal: pip install numpy
