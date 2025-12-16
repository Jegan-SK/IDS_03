## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

#Feature Encoding

    #importing all the neccessary packages
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
    import seaborn as sns
    !pip install category_encoders # if category_encoders is not installed
    from category_encoders import BinaryEncoder
    data=pd.read_csv("data.csv")
    df=pd.DataFrame(data)
    df

<img width="751" height="366" alt="image" src="https://github.com/user-attachments/assets/90180171-ac1f-49e5-8b74-b2557e2968d7" />

    #Ordinal encoder without specifying order
    oe=OrdinalEncoder()
    df["OE_1]=oe.fit_transform(df[["Ord_1"]])
    df

<img width="817" height="363" alt="image" src="https://github.com/user-attachments/assets/7785bb4f-54fd-4d1a-98a2-5be49f6717bd" />

    #OrdinalEncoder with order specified
    oe=OrdinalEncoder(categories=[["Hot","Warm","Very Hot","Cold"]])
    df["OE1(1)"]=oe.fit_transform(df[["Ord_1"]])
    df

<img width="832" height="361" alt="image" src="https://github.com/user-attachments/assets/ac35b7cf-a9f4-4a65-9f97-4dd17930fc2e" />

    
    #LABEL ENCODER
    le=LabelEncoder()
    df["LE2"]=le.fit_transform(df["Ord_2"])
    df

<img width="902" height="356" alt="image" src="https://github.com/user-attachments/assets/9fb8270b-7e8b-4239-92a5-5a57bd4768da" />

    #ONE HOT ENCODER
    ohe=OneHotEncoder(sparse_output=False)
    enc=pd.DataFrame(ohe.fit_transform(df[["bin_1"]]))
    df=pd.concat([df,enc],axis=1)
    df

<img width="956" height="371" alt="image" src="https://github.com/user-attachments/assets/bf25d692-584b-4ad4-8959-e7e5192a118b" />

    get_dummies(df,columns=["bin_1"])

<img width="983" height="364" alt="image" src="https://github.com/user-attachments/assets/c2b1f4cb-4cb2-4ffd-8bb1-cfaae4aa3733" />

    # BINARY ENCODER
    be=BinaryEncoder()
    nd=be.fit_transform(df["bin_2"])
    df=pd.concat([df,nd],axis=1)
    df

<img width="1055" height="356" alt="image" src="https://github.com/user-attachments/assets/a8d3f3d1-2fdc-42f9-928d-5a7cd642c29f" />

#Feature Transformation

#1. Function Transformation

    #importing all neccessary packasges
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from sklearn.preprocessing import PowerTransformer
    data=pd.read_csv("Data_to_Transform.csv")
    df=pd.DataFrame(data)
    df

<img width="1085" height="416" alt="image" src="https://github.com/user-attachments/assets/d2d35d41-cd96-4b19-af45-0923702ab1fd" />

    df.skew()

<img width="695" height="123" alt="image" src="https://github.com/user-attachments/assets/af5b2daf-841d-49d7-8d36-16a2951f05f6" />

    #log transformation
    np.log(df["Highly Positive Skew"])

<img width="771" height="268" alt="image" src="https://github.com/user-attachments/assets/b348a680-7af0-43fa-a71c-f1b9bbe0fa4c" />

    #reciprocal transformation
    np.reciprocal(df["Highly Positive Skew"])

<img width="844" height="265" alt="image" src="https://github.com/user-attachments/assets/0a540ba8-54b7-4da7-8aa5-38338128b4a0" />

    #square root transformation
    np.sqrt(df["Highly Positive Skew"])

<img width="818" height="268" alt="image" src="https://github.com/user-attachments/assets/b7bce61f-44b2-4e95-923e-529d7c95d7bb" />

    #square transformation
    np.square(df["Highly Positive Skew"])

<img width="901" height="259" alt="image" src="https://github.com/user-attachments/assets/5a3883e6-870b-43f7-82fd-704379645a95" />

#2. Power Transformations

    #Yeo-Johnson Transformation
    pt_yj=PowerTransformer(method='yeo-johnson')
    df["YJ_skew"]=pt_yj.fit_transform(df[["Highly Negative Skew"]])
    df

<img width="1123" height="455" alt="image" src="https://github.com/user-attachments/assets/94cfda26-d8d2-47de-b745-9f8384524666" />

    #Box-cox Transformation
    pt_boxcox=PowerTransformer(method="box-cox")
    df["BC_skew"]=pt_boxcox.fit_transform(df[["Moderate Positive Skew"]])
    df

<img width="1068" height="446" alt="image" src="https://github.com/user-attachments/assets/bc58895d-19fa-404f-8460-af676844df75" />

    #to save the data to a new file
    df.to_csv("Transformed_data.csv",index=False)


# RESULT:
       Thus the rogarm o read the given data and perform Feature Encoding and Transformation process and save the data to a file is written and executed successfully.


       
