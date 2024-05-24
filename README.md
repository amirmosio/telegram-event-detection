# telegram-event-detection

Components of this project:
 - Telegram client to download messages from different groups
 - A script to select data for labeling manually or automatically with some preprocessing methods
 - A script to merge some messages related to each other based on some criteria to make the messages more informative and more related to the context of the conversation
 - A script to train the conversation model to detect changes of topic or detect new conversation
 - Telegram client to be up and running for users to interact with and request the service

 All of the components can be run from main.py

 Here is the link to the description of the project:
 
 [Google Slide](https://docs.google.com/presentation/d/1xOufGmwwDxyDUGt5lQMmtHxtgQivVYN68eHBDd4wogA/edit?usp=sharing)

# Running the Script
Before anything make sure you have the Python(version 3.10 and above) installed and working. Also for managing the Python library please make sure you have the Python env properly set up and ready to use.

Steps:
 - Create a file named ".env" similar to ".env_template" and fill in the required values such:
   - TELEGRAM_CLIENT_PHONE_NUMBER: It is the phone number of your telegram account which you want to run your client with.
   - TELEGRAM_CLIENT_API_ID, TELEGRAM_CLIENT_API_HASH: Read [detail](https://core.telegram.org/api/obtaining_api_id) for more information on how to create a telegram application to get these two values 
 - Activate your Python environment
 - In the directory of the project run this command:
   - python -m pip install -r requirements.txt
 - Then run the *main.py* file using:
     - python main.py
     - It would show you some options regarding the stages of training and running bot and ...
 - Enter option 6 to run the client
 - Now any user can start interacting with the telegram account used before, and set their preferences to get notified about their interested topics.
