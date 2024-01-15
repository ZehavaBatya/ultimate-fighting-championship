# ML Zoomcamp Capstone 2 project: Stats Smackdown: ML Predictions in MMA
> Data Set: https://www.kaggle.com/datasets/asaniczka/ufc-fighters-statistics/data

> Technologies Used: Python, Numpy, Pandas, Matplotlib, VS Code

## Table of Contents
1. [Problem Description](#problem-description)
3. [EDA](#eda)
4. [Model Training](#model-training)
5. [Exporting Notebook To Script](#notebook-to-script)
6. [Model Deplpyment](#model-deployment)
7. [Dependency and Envrionment Management](#dependency-and-environment-management)
8. [Containerization](#containerization)
9. [Cloud Deployment](#cloud-deployment)
10. [Project Lessons](#project-lessons)
11. [Project Challenges](#project-challenges)
12. [Future Opportunities](#future-opportunities)
13. [Conclusion](#conclusion)
14. [Social Learning In Public](#social-learning-in-public)

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

## Notebook To Script

## Model Deployment

## Dependency and Environment Management

## Containerization

## Cloud Deployment

## Project Lessons

## Project Challenges

## Future Opportunities

## Conclusion 
### About DataTalksClub and ML ZoomCamp
DataTalksClub is a dynamic community focused on data science and machine learning discussions. Central to this community is the ML Zoomcamp, an interactive learning initiative. Tailored for all skill levels, ML Zoomcamp offers live Zoom sessions covering fundamental machine learning concepts, practical applications, and hands-on projects. Led by industry experts, participants gain valuable insights and skills for tackling data science challenges. Beyond education, ML Zoomcamp fosters networking and collaboration, creating a supportive environment for individuals eager to advance in the ever-evolving field of machine learning.

DataTalksClub: https://datatalks.club/

Machine Learning Zoomcamp: https://github.com/DataTalksClub/machine-learning-zoomcamp/

## Social Learning In Public
"Social learning in public" refers to the intentional sharing of your project progress, findings, and research across various social media and publishing platforms. Instead of keeping your work private, you choose to make it publicly accessible to a wider audience. This transparent and open dissemination of information aligns with the principles of knowledge sharing, fostering a collaborative and inclusive environment for discussions, insights, and shared learning experiences.





