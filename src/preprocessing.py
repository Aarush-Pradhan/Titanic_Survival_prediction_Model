import pandas as pd
import numpy as np
import os

def load_data(path):         #fucntion to load data
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.join(script_dir, '..', 'data', 'titanic.csv')
    return pd.read_csv(absolute_path)


def clean_data(df):

    df['Age']=df['Age'].fillna(df['Age'].median())

    df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

    df.drop('Cabin',axis=1,inplace=True)

    df['Sex']=df['Sex'].map({"male":0 , "female":1})

    df.drop(['Name','Ticket','PassengerId'],axis=1,inplace=True)

    df=pd.get_dummies(df,columns=['Embarked'],drop_first=True)

    return df



