Depression Detection System

The goal of this project is to compile a random dataset of tweets about all aspects of mental health, both depressed and non-depressive. It compares various classifiers using a supervised machine learning methodology. The primary goal of the research was to determine each person's level of depression and to identify the signs of depression. As a result, it is changed to an unsupervised strategy. This study implements  K-means clustering, along with Natural Language Processing (NLP). The project uses K-means clustering to identify distinct clusters within the dataset. These clusters' study indicates trends and patterns related to depressive symptoms. For the Depression Level Detection system, we developed the user interface (UI), which is essential to giving users an engaging and easy-to-use experience when entering data and receiving analysis results. Users can enter two types of input into the user interface (UI) via a dropdown menu: either a username that is stored in the dataset or a text input expressing how they are feeling. 




Environment:

The IDE used for programming Source Code is Jupyter Notebook and Visual Studio Code. 

Libraries to be installed:
1.	Pandas

This library is used for data analysis and manipulation.

Code : 
pip install Pandas

2.	Nltk

This library is used for Natural Language Processing based text processing

Code:
Pip install nltk

3.	Scikit-Learn

This library is used for machine learning-based algorithms and pre-processing.

Code: 
pip install scikit-learn	

4.	Textblob

This library is used for processing text data.

Code: 
pip install textblob



Dataset:
The dataset has been included in the zip file with the name Dataset.csv.

The dataset is imported into the code using the following code:

Code:
df = pd.read_csv("Dataset.csv")



How to run the project:

In the “Depression Detection System using  Machine Learning “ Folder, run the manage.py python file inside myproject folder in the terminal using the following command:

Cd Depression Detection System using Machine Learning
cd myproject
python manage.py runserver







