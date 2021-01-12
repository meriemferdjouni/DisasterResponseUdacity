# Udacity Disaster Response Project 

This project includes a web application where an emergency worker can input a new message and get classification results under several messages categories. The web app will also display visualizations of the data.

# Getting Started
## Prerequisites
```
pip install pandas 
pip install Numpy
pip install Sci-kit Learn
pip install Flask 
pip install SQL Alchemy
pip install Plotly
pip install NLTK

```

## Dataset
We use/analyse data from [Figure Eightin](https://appen.com) this project is a labeld dataset contains disatser messages, each messagae is labeled by the category of the message. 

## Build With

* [Plotly](https://plotly.com) - For data visualaziation 
* [Bootstrap](https://getbootstrap.com) - Web framework, Front-End Library 
* [Flask](https://flask.palletsprojects.com/en/1.1.x/) - Web framework to build the Back-End. You should sign up for Flask if you don't have an account

# Folders and Files
This repository includes:
* **data** Where you the data is uploaded and processed.
* **app** running the web application.
* **models** where the machine learning model is trained/tested/saved.

# Running Instcutions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        ```
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```
    - To run ML pipeline that trains classifier and saves
        ```
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

2. Run the following command in the app's directory to run your web app.
    ```
    python run.py
    ```

3. Go to http://0.0.0.0:3001/

# Authors 
* **Meriem Ferdjouni**

# Acknowledgments

* [Udacity](https://www.udacity.com) (Data scientist Nanodegree Program)
