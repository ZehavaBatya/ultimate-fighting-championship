# ML Zoomcamp Capstone 2 project: Stats Smackdown: ML Predictions in MMA
> Data Set: https://www.kaggle.com/datasets/asaniczka/ufc-fighters-statistics/data

> Technologies Used: Python, Docker, AWS, Flask, Scikit-learn, Numpy, Pandas, Matplotlib, Wordcloud, Jupyter, VS Code

## Table of Contents
1. [Problem Description](#problem-description)
3. [EDA](#eda)
4. [Model Training](#model-training)
5. [Exporting Notebook To Script](#notebook-to-script)
6. [Reproductibility](#reproductibility)
7. [Model Deployment](#model-deployment)
8. [Dependency and Envrionment Management](#dependency-and-environment-management)
9. [Containerization](#containerization)
10. [Cloud Deployment](#cloud-deployment)
11. [Project Lessons](#project-lessons)
12. [Project Challenges](#project-challenges)
13. [Future Opportunities](#future-opportunities)
14. [Conclusion](#conclusion)
15. [Social Learning In Public](#social-learning-in-public)

## Problem Description
### What is MMA and UFC?
Explore the dynamic world of Mixed Martial Arts (MMA), a widely followed combat sport where athletes showcase their skills in various disciplines. At the forefront of MMA stands the Ultimate Fighting Championship (UFC), recognized as the pinnacle organization in the realm of mixed martial arts.
### About the Dataset 
Delve into this comprehensive dataset, meticulously capturing the statistical insights of UFC fighters. From their victories and defeats to draws, delve into the intricacies of their physical attributes, adopted fighting styles, and notable career achievements. Gain a deeper understanding of the fighters' journeys within the UFC and uncover the nuanced details that contribute to their standing in this exhilarating sport.

The following key features are utilized:
* 'name': The fighter's full name.
* 'nickname': A commonly known alias or nickname associated with the fighter.
* 'wins': The total number of victories in the fighter's career.
* 'losses': The total number of losses in the fighter's career.
* 'draws': The total number of draws or ties in the fighter's career.
* 'height_cm': The fighter's height in centimeters.
* 'weight_in_kg': The fighter's weight in kilograms.
* 'reach_in_cm': The fighter's reach, measured in centimeters.
* 'stance': The preferred fighting stance of the fighter (orthodox, southpaw, etc.).
* 'date_of_birth': The fighter's date of birth.
* 'significant_strikes_landed_per_minute': The average number of significant strikes landed by the fighter per minute.
* 'significant_striking_accuracy': The percentage of significant strikes that the fighter lands successfully.
* 'significant_strikes_absorbed_per_minute': The average number of significant strikes absorbed by the fighter per minute.
* 'significant_strike_defence': The percentage of significant strikes that the fighter successfully defends against.
* 'average_takedowns_landed_per_15_minutes': The average number of successful takedowns by the fighter per 15 minutes.
* 'takedown_accuracy': The percentage of takedown attempts by the fighter that are successful.
* 'takedown_defense': The percentage of takedown attempts against the fighter that are defended successfully.
* 'average_submissions_attempted_per_15_minutes': The average number of submission attempts by the fighter per 15 minutes.

### The Problem 🧠
The challenge at hand is to address the lack of accurate and nuanced predictions for fight outcomes in mixed martial arts (MMA), particularly within the realm of UFC matches. The principal objective is to overcome the existing limitations by identifying and comprehending intricate patterns within the data, ultimately aiming to enhance the precision and reliability of predicting UFC fight outcomes. The problem statement centers on the need to design and implement machine learning models capable of effectively analyzing diverse fighter metrics, including win-loss records, physical attributes, fighting styles, and career achievements
### The Solution 🤓
The primary objective is to train predictive models capable of forecasting fight outcomes. Leveraging the extensive dataset containing comprehensive fighter statistics, the project aims to develop sophisticated models that can analyze factors such as a fighter's win-loss record, physical attributes, fighting style, and career achievements to accurately predict the likely result of a given match. This endeavor seeks to enhance our understanding of the multifaceted dynamics that influence fight results and pave the way for more informed analyses within the realm of combat sports.
### Project Inspiration 💡
Firstly, my active engagement in Krav Maga, Brazilian Jiu Jitsu, and kickboxing has fostered a profound appreciation for the intricate dynamics of MMA. Secondly, recognizing the paramount importance of self-defense as a vital life skill has served as a driving force. Lastly, acknowledging the holistic benefits of physical strength and exercise for both the mind and body, the project aims to explore the intersection of fitness, mental well-being, and predictive analytics in the context of UFC fights.

## EDA 
### Extensive EDA 🔎
Ranges of values, missing values, analysis of target variable, feature importance analysis

### Operations Performed:
1. Data Cleaning and Handling:
Removed columns with more than 50% missing values: 'reach_in_cm'.
Converted 'date_of_birth' to a datetime data type.
Dropped rows with missing values in key columns ('name', 'nickname', 'stance', 'date_of_birth').

2. Data Exploration:
Computed summary statistics for numeric columns.
Checked the distribution and statistics for each numeric column.
Explored the distribution of wins using a histogram.

3. Data Transformation:
Created a log-transformed wins column to handle the right-skewed distribution.
Normalized numeric columns using Min-Max scaling.
Calculated the correlation matrix between numeric features.
### Reasoning:
Log Transformation: Log transformation is applied to handle right-skewed data distributions, promoting better model performance when applicable.

Data Cleaning: Removing columns with a significant number of missing values ensures a cleaner dataset, and dropping rows with missing values in key columns maintains the integrity of critical information.

EDA Techniques: Summary statistics and visualizations (histograms) are chosen to understand the distribution, central tendency, and potential outliers in the numeric columns.

Normalization: Scaling numeric columns is essential for models that are sensitive to the scale of features, ensuring fair treatment to all variables.

Correlation Analysis: Understanding correlations helps identify potential multicollinearity issues and guides feature selection.

Word Cloud: Word clouds provide an intuitive and visual representation of the most frequent terms, which can be valuable in identifying patterns or trends, especially in text data like nicknames. It's a creative way to highlight popular aliases associated with fighters.

## Model Training
1. Logistic Regression Model
> Binary Outcome:

Logistic regression is particularly well-suited for binary classification problems, where the outcome variable has two possible classes. In MMA, one might be interested in predicting outcomes such as whether a fighter wins or loses a match. Logistic regression models the probability of the binary outcome, making it a natural choice for this type of prediction.
> Interpretability:

Logistic regression provides interpretable results. The coefficients in logistic regression represent the log-odds of the outcome. This makes it easy to understand the impact of each predictor variable on the probability of the event occurring. In the context of MMA, one can interpret how different fighter characteristics or match statistics contribute to the likelihood of winning a fight.
> Efficiency with Limited Data:

Logistic regression tends to perform well when the dataset is not extremely large. If one has a moderate-sized dataset, logistic regression can provide robust results without requiring a massive amount of data. This efficiency can be advantageous, especially if one is working with limited resources or collecting data on MMA matches, which may not be as abundant as data in some other domains.

2. Decision Tree Model 

> Interpretability:

Decision trees offer a highly interpretable representation of decision-making processes. The model is essentially a tree-like flowchart where each node represents a decision based on a specific feature. This makes it easy to understand how the model arrives at a particular prediction. In projects related to MMA, interpretability can be crucial for gaining insights into which features or factors contribute most to predicting outcomes.

> Handling Non-linearity and Interaction:

Decision trees are capable of capturing non-linear relationships and interactions between variables. In MMA, where the outcome of a match can depend on complex interactions between fighter attributes or match statistics, decision trees can be more flexible in capturing these patterns compared to linear models. They can identify intricate relationships that might be missed by simpler models.

> Feature Importance:

Decision trees provide a natural way to assess the importance of different features in making predictions. By evaluating the splits in the tree, one can identify which features are most influential in determining outcomes. For a project related to MMA, understanding the importance of various fighter characteristics or match statistics can offer valuable insights into what makes a difference in winning or losing a fight.

## Notebook To Script 
Use the command line with the following steps:

1. Open a terminal.

2. Navigate to the directory where your Jupyter Notebook file is located.

3. Run the following command:

`jupyter nbconvert --to script your_notebook.ipynb`

Replace your_notebook.ipynb with the name of your notebook file.

4. This will generate a Python script in the same directory with the same name as your notebook, but with a `.py` extension.

## Reproductibility
Downloading Project File Repository from GitHub:
1. Clone Repository:

Open a terminal or command prompt.

Navigate to the directory where you want to store your project.

Run the following command:

`git clone https://github.com/ZehavaBatya/ultimate-fighting-championship/`

2. Navigate to Project Directory:

Move into the project directory:

`cd your-project`

3. Download Kaggle set:
 
 `https://www.kaggle.com/datasets/asaniczka/ufc-fighters-statistics/data`


## Model Deployment

To run Flask with your project, you need to follow these steps. First, make sure you have Python and pip installed on your system.

Install Flask:
Open a terminal or command prompt and run the following command to install Flask using pip:

`pip install Flask`

Integrate with Your Project:
Copy your Jupyter Notebook (notebook.ipynb) into the my_flask_app directory.

To use your Jupyter Notebook in your Flask app, you might want to convert it into a Python script, as mentioned earlier.

Run the Flask App:
1. In the terminal, make sure you are in the the flask app directory.

2. Run the Flask app:

`python main.py`

This will start the development server. You should see output indicating that the server is running.

3. Open a web browser and go to `http://127.0.0.1:5000/`. You should see your Flask app's output.

![image](https://github.com/ZehavaBatya/ultimate-fighting-championship/assets/84485729/20881d32-fce2-4593-ac8a-6bfbc0a0ca91)

![image](https://github.com/ZehavaBatya/ultimate-fighting-championship/assets/84485729/0c545986-7ac9-439b-a3e6-fa852dc8e483)


## Dependency and Environment Management
1. Create a Virtual Environment.
Create a virtual environment in your project directory. This isolates dependencies for your project.

`python -m venv venv`

This command creates a virtual environment named venv.

2. Activate the Virtual Environment:

Activate the virtual environment. The activation command depends on your operating system:

For Windows:
`venv\Scripts\activate`

For Unix or MacOS:
`source venv/bin/activate`

![image](https://github.com/ZehavaBatya/ultimate-fighting-championship/assets/84485729/c83ec887-d046-420d-a697-8246de9b3234)



## Containerization
1. Download Docker

`pip install docker`

2. Install the Docker Extension for VS Code

3. Build and Run Docker Containers:

Create a Dockerfile in the root of your project. This file defines the configuration for building a Docker image.

`Example Dockerfile:

FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "./app.py"]`

Open the Command Palette (Ctrl+Shift+P) and run the command "Docker: Build Image" to build the Docker image based on your Dockerfile.

![image](https://github.com/ZehavaBatya/ultimate-fighting-championship/assets/84485729/272d3469-6894-47b0-8581-87af15736430)

Once the image is built, you can run a container based on that image. Use the "Docker: Run" command from the Docker extension.

## Cloud Deployment

Prerequisites:
>AWS account.
>Docker installed locally.
>AWS CLI installed locally.
>Python and pip installed locally.
>An ECS cluster set up on your AWS account.

Step 1: Dockerize Your Python Application
***See the above steps***

Step 2: Build and Push Docker Image to Amazon ECR
Assuming you have an ECR repository created, use the following commands:

# Login to ECR
`$(aws ecr get-login --no-include-email --region <your-region>)`

# Build Docker image
`docker build -t <your-repository-uri>:<tag> .`

# Tag the image
`docker tag <your-repository-uri>:<tag> <your-repository-uri>:latest`

# Push the image
docker push <your-repository-uri>:latest
Step 3: Create a Task Definition
Create a file task-definition.json:

`{
  "family": "your-app",
  "containerDefinitions": [
    {
      "name": "your-container",
      "image": "<your-repository-uri>:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 5000,
          "hostPort": 5000
        }
      ]
    }
  ]
}`

Step 4: Register the Task Definition

`aws ecs register-task-definition --cli-input-json file://task-definition.json`
Step 5: Create a Service
Create a file ecs-service.json:

`{
  "cluster": "your-cluster-name",
  "serviceName": "your-service",
  "taskDefinition": "your-app",
  "desiredCount": 1,
  "launchType": "EC2",
  "deploymentController": {
    "type": "ECS"
  }
}`
Step 6: Create/Update ECS Service

`aws ecs create-service --cli-input-json file://ecs-service.json`
# or
`aws ecs update-service --cli-input-json file://ecs-service.json`
Step 7: Access Your Application
Get the public IP address of your EC2 instance and access your application on port 5000.

This is a simplified guide. Adjustments may be needed based on your specific application and AWS setup.

## Project Lessons
1. How to execute Flask and Docker
2. The difference between Kubernetes versus AWS
3. Test. Test often. 

## Project Challenges
1. AWS Usage:

>Challenge: As a first-time user of AWS (Amazon Web Services), navigating through the numerous services, understanding their functionalities, and configuring them for your project can be daunting.
>Mitigation: Utilize AWS documentation, tutorials, and forums for guidance. Break down tasks into smaller, manageable steps. Experiment in a controlled environment to avoid unintended costs.

2. IPython Notebook Execution:

>Challenge: Your Jupyter (IPython) notebook didn't run through as expected, indicating a learning curve in using the tool effectively.
>Mitigation: Familiarize yourself with Jupyter Notebook functionalities, including code cells, markdown cells, and execution order. Ensure that dependencies are correctly installed and versions are compatible. Seek community forums or tutorials for specific issues.

3. Selecting Machine Learning Principles:

>Challenge: Identifying the appropriate machine learning principles and algorithms for your MMA (Mixed Martial Arts) project can be challenging, given the diverse range of available techniques.
>Mitigation: Start with a clear project objective. Understand the nature of your data and the type of predictions or insights you seek. Research different machine learning algorithms and their suitability for your data. Experiment with multiple models and assess their performance to make informed choices. Consider seeking expert advice or consulting with experienced practitioners in the field.

## Future Opportunities
1. Implement XGBoost and Random Forest models.
2. Add more commentary and conclusions after each finding.
3. Provide a pipfile dependecy and document the process.

## Conclusion 
### About DataTalksClub and ML ZoomCamp
DataTalksClub is a dynamic community focused on data science and machine learning discussions. Central to this community is the ML Zoomcamp, an interactive learning initiative. Tailored for all skill levels, ML Zoomcamp offers live Zoom sessions covering fundamental machine learning concepts, practical applications, and hands-on projects. Led by industry experts, participants gain valuable insights and skills for tackling data science challenges. Beyond education, ML Zoomcamp fosters networking and collaboration, creating a supportive environment for individuals eager to advance in the ever-evolving field of machine learning.

DataTalksClub: https://datatalks.club/

Machine Learning Zoomcamp: https://github.com/DataTalksClub/machine-learning-zoomcamp/

## Social Learning In Public
"Social learning in public" refers to the intentional sharing of your project progress, findings, and research across various social media and publishing platforms. Instead of keeping your work private, you choose to make it publicly accessible to a wider audience. This transparent and open dissemination of information aligns with the principles of knowledge sharing, fostering a collaborative and inclusive environment for discussions, insights, and shared learning experiences.





