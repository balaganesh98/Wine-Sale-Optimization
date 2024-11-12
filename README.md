
Business Problem: 
The key business problem is to determine the best wines to stock in the newly acquired Fine Wines store to attract wine connoisseurs and maximize sales.

Analytics Problem: 

The analytics problem involves building a predictive model using historical wine review data to predict the best wines to stock based on features such as price, country, variety, and winery.

Approach
Using a Kaggle dataset containing extensive wine reviews, we conducted data cleaning, exploratory data analysis, and developed a predictive model using the CatBoost Regressor. The model aims to predict wine ratings based on features such as price, country, variety, and winery. The deployment plan involves cloud infrastructure and integration with the store's POS and inventory management systems. Continuous evaluation and retraining of the model will ensure its long-term effectiveness.

Key Findings:
•	The CatBoost Regressor model demonstrated promising performance with RMSE and R² metrics, indicating reliable predictions.
•	Riesling, Pinot Noir, and Syrah are among the top varieties with high ratings and moderate to high prices, making them attractive options for stocking.
•	Integration of influential wine tasters in marketing efforts can maximize outreach and strengthen the promotion of high-quality wines.

Recommendations:
We have identified 4 wineries who might hold a promising wine product to add to our catalogue. After testing their products with our trained model, we have finalized the 'Château du Taillan 2015' wine from Château du Taillan winery located in France. The model predicts a score of 89.33 for this wine. Which classifies as a Premium wine and performs better than approximately 70% of the highly rated wines.

About new winery identified:
Name: Château du Taillan
Country: France
Product: Château du Taillan 2015 du Taillan (Bordeaux SupÃ©rieur)
Variety: Bordeaux-style Red Blend
Price: $52
Château du Taillan has been a family property since 1896 but today it is above all a modern story, told by women! Five sisters now hold the reins of the Château located at the gateway to Bordeaux and the Médoc.
Their beautiful property was classified in February 2020 as a "Cru Bourgeois Exceptionnel” and invites everyone for a wine tasting whose key words are quality and sustainable agriculture.

CLASSIFICATION OF HIGHLY RATED WINES
Exotic Class: The top class. Expensive wines that provide the taster with great grape quality and an exquisite taste. Examples include:
•	Lanson NV Extra Age Brut - $105.00
•	Chambers Rosewood Vineyards NV Grand Muscat - $100.00
Premium Class: Wines with high points that are known for their balance of quality and price. They compete with exotic wines despite being significantly cheaper. Examples include:
•	Louis Roederer NV Brut Premier - $49.50
•	Roederer Estate NV Brut Rosé Sparkling - $28.67
Affordable Class: Wines that offer good quality and taste at a lower price. Examples include:
•	Scharffenberger NV Brut Excellence Sparkling - $20.00
•	Chambers Rosewood Vineyards NV Muscat - $15.67
In total, there are 26 Exotic, 53 Premium, and 25 Affordable wines. This classification helps in understanding the taste-based categorization of highly rated wines. The Exotic class includes the highest-scoring wines, known for their distinct and superior variety. We have identified 26 wines in this class.
                                            
San Jose
Santa Teresa and Castro have high median incomes ($148,750 and $169,459, respectively). Stocking more Exotic wines for our stores in these areas will cater to the affluent residents who prefer higher-priced wines.
North San Jose has a median income of $151,389, hence we will stock our store in this area with Premium wines, as the purchasing power of the residents here is high.

Los Angeles
In Brentwood, with a high median income of $143,000, we will stock more Exotic category wines. Affluent residents here are likely to purchase higher-priced wines.
In Cypress Park, with a median income of $80,386, we will be stocking more Affordable category wines. The lower income levels in this area and its surroundings make affordable wines more suitable for the local residents.
Deployment Plan:
•	Deploy the model on a cloud platform using a serverless architecture to handle prediction requests efficiently.
•	Integrate the model with the twitter reviews scraping and management system for real time updates and recommendations.



Model Lifecycle Management:
•	One of our analysts will monitor and work on the scraping, the model will then constantly reevaluate and provide the supply chain, marketing, business development departments with these iterations of wine selections. Each department will have to alter their strategies regularly based on the output provided by the model to maintain strong performance at each of these stores across California.
Our Model Life cycle can be categorized into these main points:
•	Data Collection – Use the cloud platform to seamless input twitter reviews data after web scrapping.
•	Data Preparation – Clean the data to be fed into the model
•	Governance – Make sure the model and recommendations are up to date with current laws and regulations in both California and the United States.
•	Code Development – Continually monitor the model and make changes to the code if and when needed.
•	Ongoing monitoring – an analyst will monitor the performance of the model regularly and provide feedback to the marketing, supply chain and business operations teams to help them adapt to changes.
•	Maintenance – Perform regular audits and make sure the model runs effectively.

Costs Involved 
•	Cloud platform fee
•	RDBS 
•	Tableau
•	Workforce salary

Through this we expect to achieve the following:
1. Provides the teams at our company with the most appropriate wine list to stock in our stores across California.
2. ⁠Help in identifying better wines and wineries to increase our wine catalogue.
3. ⁠Alerts involved stakeholders of any changes in trends and customer wine preferences.
Loyalty Program:
Customers can add their twitter handle to their profile with us. For every review on a product and tagging us they’ll get points which can be used to get discounts for next purchases. This will help in model Life cycle management- maintenance aspect. We will get more up to date information, maintain cash flow, and continuously train our model with current statistics. Win-Win!
Actionable Steps:
•	Target marketing efforts towards influential wine tasters and connoisseurs.
•	Stock high-quality and moderately priced varieties like Riesling, Pinot Noir, and Syrah to cater to diverse customer preferences.
•	Utilize the model's predictions to optimize inventory management and enhance customer satisfaction.

Conclusion
We have our model, deployment plan, model life cycle management strategies in place to be executed. We have trained our models well and cross verified our workings and plans again with all stakeholders involved to make sure the plan works out well. 

                                                                

Context: 
We are Good Times, a successful wine retailer in the United States with a major presence in the East Coast. The company is looking to expand into the huge West Coast market by acquiring a major historical wine store ‘Fine Wines’ in California. This store had a good reputation through the '70s but has failed to adjust to the newly emerging market trends, products, and customer preferences. We have a wonderful opportunity for market entry and to bring it back to life under our brand.

Objective: 
Understand the wine market in west coast (California, in specific) to identify the wines that are most likely to attract customers by the target population of wine connoisseurs, we do so in an attempt to improve customer satisfaction thereby increasing sales and profit and restore the shop to being the destination of choice for all wine lovers.

Business Problem: 
What are the best wines to stock in the store to attract wine connoisseurs and maximize sales?

Analytics Problem: 
How can we use historical wine review data from twitter, by building a predictive model to predict the best wines to stock in our store?

Problem Statement:
The purpose of this project is to utilize data analytics to identify the best wines to stock in a historically significant wine store in California. Our goal is to maximize sales and enhance customer satisfaction by recommending wines that meet current market demands and preferences. Specifically, we aim to:
1.	Identify the top 10% of wines in terms of quality and value for money.
2.	Predict the rating (points) of wines based on available features to guide stocking decisions.
3.	Integrate predictive analytics with market data to tailor recommendations for California's major cities and neighborhoods.

Description of Data:
The dataset used for this analysis is sourced from Kaggle and contains extensive wine reviews. It includes attributes such as country, description, designation, points, price, province, region, taster name, taster Twitter handle, title, variety, and winery. The dataset consists of approximately 130,000 records with 13 fields.
Overview of the Data:

Preliminary analysis revealed the following key points:
•	Missing Values: Significant missing values were found in designation, region_1, region_2, taster_name, and taster_twitter_handle. These columns were not critical for the initial analysis and were dropped.
•	Distribution Analysis: The points distribution is skewed towards higher ratings, with most wines scoring between 80 and 100 points. The price distribution is right-skewed, with most wines priced below $50.
Preliminary Analysis
After a preliminary analysis of the data set, we found the following:
•	Country: The country where the wine is produced.
•	Description: Textual review of the wine.
•	Designation: Vineyard designation of the wine.
•	Points: Wine rating score.
•	Price: Price of the wine.
•	Province: Province where the wine is produced.
•	Region: Specific region within the province.
•	Taster Name: Name of the wine taster.
•	Taster Twitter Handle: Twitter handle of the taster.
•	Title: Title of the wine.
•	Variety: Type of grape used.
•	Winery: Name of the winery.

The dataset consists of approximately 130,000 records with 13 fields, providing a comprehensive view of wine characteristics and reviews.
Summary of Missing Values
•	Country: 59 missing values
•	Description: No missing values
•	Designation: 34,646 missing values
•	Points: No missing values
•	Price: 8,466 missing values
•	Province: 59 missing values
•	Region_1: 19,682 missing values
•	Region_2: 73,769 missing values
•	Taster Name: 24,938 missing values
•	Taster Twitter Handle: 29,483 missing values
•	Title: No missing values
•	Variety: 1 missing value
•	Winery: No missing values
•	Wine Year: 4,304 missing values



Key Insights
•	The fields - Designation, Region_1, Region_2, Taster Name, and Taster Twitter Handle have a lot of missing values. These might not be critical for our predictive modeling and can be excluded if necessary. A lot of datasets have irregularities.
•	Country, Price, Province, and Variety have very fewer missing values. We will address these missing values along the course of the project as they can be important for our analysis.
•	Distribution Analysis: We will now analyze the distribution of key variables like points, price, variety, and region_1.
•	Exploratory Data Analysis (EDA): This includes visualizing the data to understand patterns and relationships.
 
•	USA and France have reliable quality and strong market presence while India and Austria have high-quality wines with fewer tasters. These wines could benefit from more marketing and reviews.
 
•	Roger Voss has the highest count of reviews (23,674), making him a highly influential taster with a significant reach on social media.
•	Paul Gregutt and Virginie Boone also have substantial influence, with 8,878 and 8,719 reviews, respectively.
•	By incorporating these tasters into marketing efforts, the company can maximize its outreach and strengthen the promotion of high-quality wines.
   
•	Riesling has the highest average points (89.49) with a moderate average price ($32.62), indicating a high-quality and reasonably priced option.
•	Pinot Noir and Syrah follow closely in quality, with average points of 89.43 and 89.31 respectively, and higher average prices ($47.86 for Pinot Noir and $39.14 for Syrah).
  
•	Nebbiolo has the highest average points (90.18) and is priced at $65.77, indicating its premium quality.
•	Champagne Blend also shows high quality with an average of 89.89 points and the highest average price ($74.35), highlighting its luxury status.
•	Pinot Noir and Syrah are notable for their high average points (89.66 and 89.75 respectively) and considerable average prices ($49.73 for Pinot Noir and $40.65 for Syrah), making them attractive premium options.


Features such as region_1, province, designation, region_2 and title have less significance based on the Catboost Feature importance Score. We plan to drop these features for Phase 3 as we believe there could be some collinearity and fine tune the model. 

Target Audience
•	Based on this data, we observe that the largest segment falls within the 0-60k income bracket, encompassing approximately 10 million people. The 60k-150k bracket includes around 5 million people, while the 150k and above bracket consists of approximately 2 million people. Given this distribution, we would like to target cheaper wines for lower-income families in the 0-60k bracket, ensuring affordability and accessibility. For the other two brackets, we aim to offer the best value and best-rated wines, catering to their higher disposable income and preference for quality. This approach will help us tailor our wine selections to meet the needs and expectations of our diverse customer base effectively. 
•	We will also be targeting wine connoisseurs in these locations. We have found our connoisseurs through our dataset. We have the twitter handle to find out the ids of the people and find out how many times they have reviewed any kind of wine. We have these connoisseurs, and we will link them to the famous wines. 


Stakeholder Analysis:
•	There are multiple stakeholders to be considered after the acquisition of the California Wine Retailer.
•	Firstly, the Branding team. They must develop a cohesive brand identity reflecting both retailers' strengths. This team needs to conduct brand audits for both companies to understand their strengths and areas for improvement. They need to collaborate with the marketing team to ensure that branding aligns with marketing strategies. Also develop new branding materials, including logos, packaging, signage, and digital assets.
•	Secondly, the Marketing team. The acquisition presents a wonderful opportunity to capitalize on the west coast market. As the company is looking to start strong, the marketing team needs to develop and execute strategies to attract and retain customers post-acquisition, like reaching out to the top wine connoisseurs and impressing with the best wine. They also need to leverage their experience/reviews and manage public relations and media outreach to ensure positive perception.
•	Thirdly, the Supple Chain/ Operations team. Post acquisition, based on our analysis, we will suggest a change in products, winery sources depending on location, population demography, etc. The team needs to optimize inventory management to prevent stockouts or overstock situations. They need to maintain relationships with suppliers and negotiate better terms if possible. This can be done by performing a comprehensive review of the current supply chain and implementing advanced supply chain management software to improve tracking and predictability.

Data Sources
•	The primary data source is the Kaggle dataset "Wine reviews.csv."
•	It may also be integrated with more information on the market trends and the preferences of customers to allow for further analysis.

Tools Used:
•	Python with libraries such as Pandas, Tableau, PowerPoint, Excel, and Word

Description of Data Transformation:
To handle missing values, we dropped rows with missing values in key columns such as country, price, province, and variety. 

Data Cleaning:
•	Dropped rows with missing values in key columns such as country, price, province, and variety.
•	Removed duplicate records to ensure data integrity.
•	Handled missing values in less critical columns by imputing or dropping them as necessary.

Tools Used:
•	Python with libraries such as Pandas for data manipulation and cleaning.

Data Visualization and Preliminary Analysis:
•	Variety Analysis: Top wine varieties included Pinot Noir, Chardonnay, and Cabernet Sauvignon.
•	Region Analysis: Leading regions were Napa Valley, Columbia Valley, and Russian River Valley.

Predictive Modeling:
1.	About the Model: We have used a self-adapting ML model, called CatBoost regressor to identify the most significant features from the dataset. The model we ran was a multivariate linear regression model. Instead of doing collinear analysis and hand-picking features for the model we proceeded with an automatic ML algorithm using the CatBoost Regressor. 
    With the help of this model, we can predict the score of a wine given the price, origin country, region/province, taster, wine title, grape variety, and winery. With more fine tuning and the help of this model we can scout out more wines which can be profitable and provide better performance for our new business venture on the west coast.
2.	Model Training: We started with a Linear Regression model to predict wine points based on the available features. 
    R^2 on training data: 0.65, 
    R^2 on test data: 0.59
    RMSE on training data: 1.83
    RMSE on test data: 2.04
3.	Model Evaluation: We used RMSE and R² as the evaluation metrics to assess model performance.

Analysis of Data
•	Visualized the distribution of key variables like points, price, variety, and region.
•	Identified patterns and relationships within the data, such as the correlation between price and points.

Predictive Modeling:
•	Developed a predictive model using the CatBoost Regressor to predict wine ratings based on features like price, country, variety, and winery.
•	Model Performance:
	R² on training data: 0.65
	R² on test data: 0.59
	RMSE on training data: 1.83
	RMSE on test data: 2.04

Tools Used:
•	Python with Scikit-learn and CatBoost for modeling and evaluation.

Conclusions

Key Learnings:
•	The initial Linear Regression model provided a baseline for predicting wine points, with the CatBoost Regressor showing improved performance.
•	The model's predictions are reliable, with high-quality varieties like Riesling, Pinot Noir, and Syrah emerging as top recommendations.
Leveraging influential wine tasters in marketing efforts can enhance the store's outreach and promotion.

Deployment Plan:
•	Deploy the model on a cloud platform using a serverless architecture to handle prediction requests efficiently.
•	 Integrate the model with the twitter reviews scraping and management system for real time updates and recommendations.

Model Lifecycle Management:
•	One of our analysts will monitor and work on the scraping, the model will then constantly reevaluate and provide the supply chain, marketing, business development departments with these iterations of wine selections. Each department will have to alter their strategies regularly based on the output provided by the model to maintain strong performance at each of these stores across California.

Actionable Steps:
•	Target marketing efforts towards influential wine tasters and connoisseurs.
•	Stock high-quality and moderately priced varieties like Riesling, Pinot Noir, and Syrah to cater to diverse customer preferences.
•	Utilize the model's predictions to optimize inventory management and enhance customer satisfaction.

Project Workflow
The attached image outlines the workflow of the project, which includes data retrieval, data preparation, modeling, model evaluation, and deployment.
 This comprehensive approach ensures that each step of the data analysis process is meticulously carried out to achieve optimal results.

