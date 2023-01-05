# Churn_TelcoCustomer

<img src="https://camo.githubusercontent.com/e33bd53e6b7096cd81bc40e157ef55ff90bf8b04505edfe7f9925e65e1507776/68747470733a2f2f626c6f672e616363657373646576656c6f706d656e742e636f6d2f68732d66732f68756266732f6d61676e6574253230637573746f6d6572732e6769663f77696474683d343633266e616d653d6d61676e6574253230637573746f6d6572732e676966" width=700px height=350px>

Customer churn is the most popular topic on every company that because finding a new customer cost is more than the customer retention cost for a company. Therefore, important to doing prediction the customers who may leave by looking at the characteristics of the customers who have left before. If the know the customer who will leave us, their dissatisfaction can change about our service, product etc.

Before predict the customer who will leave the company, it's important to do feature engineering. In the feature engineering stage, having a business knowledge is differs significantly. When you know how to read data correctly, you may find the proper explanation of variables.

## BUSINESS PROBLEM :

A telecom company want to know which customers who will leave their company. In our project, we will do feature engineering before modelling. 

## DATASET STORY :

Telco churn dataset gives third-quarter Demographic and Service information for 7043 customers of a California telecom company. It gives which customers left the company after using some services. Each row represents a customer. 

## VARIABLES :

* CustomerId - A unique ID that identifies each customer.
* Gender - The customer’s gender: Male, Female
* SeniorCitizen - Indicates if the customer is 65 or older: Yes, No
* Partner - Indicates if the customer is married: Yes, No
* Dependents - Indicates if the customer lives with any dependents: Yes, No. Dependents could be children, parents, grandparents, etc.
* tenure -Indicates the total amount of months that the customer has been with the company by the end of the quarter specified above.
* PhoneService - Indicates if the customer subscribes to home phone service with the company: Yes, No
* MultipleLines - Indicates if the customer subscribes to multiple telephone lines with the company: Yes, No, No Telephone service
* InternetService - Indicates if the customer subscribes to Internet service with the company: DSL, Fiber Optic, No.
* OnlineSecurity - Indicates if the customer subscribes to an additional online security service provided by the company: Yes, No
* Online Backup - Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No, No Telephone service
* DeviceProtection - Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: Yes, No, No Telephone service
* TechSupport - Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: Yes, No, No Telephone service
* StreamingTV - Indicates if the customer uses their Internet service to stream television programing from a third party provider: Yes, No, No Telephone service
* StreamingMovies - Indicates if the customer uses their Internet service to stream movies from a third party provider: Yes, No, No Telephone service
* Contract - Indicates the customer’s current contract type: Month-to-Month, One Year, Two Year.
* PaperlessBilling - Indicates if the customer has chosen paperless billing: Yes, No
* PaymentMethod - Indicates how the customer pays their bill: Bank Withdrawal, Credit Card, Postal Check, Electronic Check
* MonthlyCharges - Indicates the customer’s current total monthly charge for all their services from the company.
* TotalCharges - Indicates the customer’s total charges, calculated to the end of the quarter specified above.
* Churn - Yes = the customer left the company this quarter. No = the customer remained with the company. Directly related to Churn Value.


Orginal IBM churn dataset contains 5 table about Demographics, Location, Population, Services, Status. We used Demographics, some of the variables from Services and Churn variable from Status.

Dataset : https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113

