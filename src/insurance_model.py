import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def cleaning(FILENAME1, FILENAME2):
    claims = pd.read_csv(FILENAME1)
    policy = pd.read_csv(FILENAME2)

    policy['MonthlyPremium'] = policy['MonthlyPremium'].fillna(policy['MonthlyPremium'].mean())

    insurance = pd.merge(policy, claims, on='PolicyId', how='inner')
    insurance['EnrollDate'] = pd.to_datetime(insurance['EnrollDate'])
    insurance['CancelDate'] = pd.to_datetime(insurance['CancelDate'])
    insurance['ClaimDate'] = pd.to_datetime(insurance['ClaimDate'])
    insurance['EnrollYear'] = insurance['EnrollDate'].dt.year
    insurance['EnrollMonth'] = insurance['EnrollDate'].dt.month
    insurance['EnrollDay'] = insurance['EnrollDate'].dt.day
    insurance['ClaimYear'] = insurance['ClaimDate'].dt.year
    insurance['ClaimMonth'] = insurance['ClaimDate'].dt.month
    insurance['ClaimDay'] = insurance['ClaimDate'].dt.day
    insurance['CancelDate'] = insurance['CancelDate'].fillna(0)
    insurance['Churn'] = np.where(insurance['CancelDate'] == '1970-01-01', 0, 1)
    insurance['TotalNumberOfClaims'] = insurance.groupby(['PolicyId'])['MonthlyPremium'].transform('count')
    insurance['CustomerPaidAmount'] = insurance['ClaimedAmount'] - insurance['PaidAmount']
    insurance['DaysToClaim'] = insurance['ClaimDate'] - insurance['EnrollDate']
    insurance['DaysToClaim'] = insurance['DaysToClaim'].dt.days
    insurance['TotalPolicyHolderPaid'] = insurance.groupby(['PolicyId'])['CustomerPaidAmount'].transform('sum')

    return insurance



def main():
    FILENAME1 = 'ClaimLevel.csv'
    FILENAME2 = 'PolicyLevel.csv'
    insurance = cleaning(FILENAME1, FILENAME2)

    X = insurance[['MonthlyPremium',
                   'ClaimedAmount',
                   'PaidAmount',
                   'EnrollMonth',
                   'EnrollDay',
                   'ClaimMonth',
                   'ClaimDay',
                   'TotalNumberOfClaims',
                   'CustomerPaidAmount',
                   'DaysToClaim',
                   'TotalPolicyHolderPaid']]
    y = insurance['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X,y)

    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, max_features=1.0, min_samples_leaf=3)
    gbc.fit(X_train, y_train)

    insurance['PredictedProbabilityOfChurn'] = gbc.predict_proba(insurance[['MonthlyPremium',
                                                                       'ClaimedAmount',
                                                                       'PaidAmount',
                                                                       'EnrollMonth',
                                                                       'EnrollDay',
                                                                       'ClaimMonth',
                                                                       'ClaimDay',
                                                                       'TotalNumberOfClaims',
                                                                       'CustomerPaidAmount',
                                                                       'DaysToClaim',
                                                                       'TotalPolicyHolderPaid']])[:,1]
    insurance['PredictedProbabilityOfChurn'] = np.where(insurance['Churn'] == 1, 0, insurance['PredictedProbabilityOfChurn'])

    print insurance


if __name__ == '__main__':
    main()
