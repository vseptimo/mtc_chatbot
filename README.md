# MTC Chatbot
This is an MTC chatbot application built using Flask, TensorFlow, and NLTK. The chatbot can respond to various HR-related queries such as benefits, PTO, salary, and more.

# Prerequisites
Python 3.11.1
Visual Studio Code

# Setup Instructions

# Step 1: Set Up the Virtual Environment
1. Open the project in Visual Studio Code.
2. Open the terminal and run the following commands to create and activate a virtual environment:

python -m venv venv
venv\Scripts\activate

3. Install the required packages:

pip install flask numpy nltk tensorflow pickle-mixin

# Step 2: Train the Model
1. Run the train.py script to train the model and generate the necessary files:

python train.py
This will generate the following files:

chatbot_model.h5: The trained model.

intents.json: The intents file containing patterns and responses.
words.pkl: A pickle file containing the words used in the training.
classes.pkl: A pickle file containing the classes (intents) used in the training.

# Step 3: Run the Application
1. Run the app.py script to start the Flask application:

python app.py

2. Open your web browser and navigate to http://127.0.0.1:5000 to interact with the chatbot.

# Note
Make sure that your Python version is 3.11.1, as TensorFlow is compatible with versions less than 3.11.
