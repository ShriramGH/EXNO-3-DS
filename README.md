# EX NO:3-Feature Encoding and Transformation

## AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

## ALGORITHM:

### STEP 1:
Read the given Data.
### STEP 2:
Clean the Data Set using Data Cleaning Process.
### STEP 3:
Apply Feature Encoding for the feature in the data set.
### STEP 4:
Apply Feature Transformation for the feature in the data set.
### STEP 5:
Save the data to the file.

## FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

## Methods Used for Data Transformation:
  ### 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  ### 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

## CODING AND OUTPUT:
### Developed by : SHRIRAM S
### Reg No : 212222240098

```python

import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/7518dfdd-a869-4481-adf5-15b7d77a2299)


```py
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/d2b0a391-66ad-42eb-bd6a-989a53dc1463)

```py
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/f44e225e-8c22-4c00-ad38-961facb3caa7)

```py
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/08c9ad56-23ba-4549-ada7-185105d50f26)


```py
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/8ed1f84c-e7db-485d-958a-1751672b3955)

```py
df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/8b001581-9300-40bd-97dc-4a434c93307f)

```py
pd.get_dummies(df2,columns=["nom_0"])
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/5a7a6ad1-4703-4778-aacb-2efb0f2849cb)


```py
pip install --upgrade category_encoders
```
![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/090b221f-1cfa-4d83-9aca-d6fcc438cd5b)

```py
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/b76dd80d-bef7-4b62-8077-26fee40db525)

```py
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/52f84e42-82d1-43a0-a2ce-d4843c5ae781)

```py
dfb=pd.concat([df,nd],axis=1)
dfb
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/261246fd-6ab1-4942-a95f-f803b0a976a8)

```py
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/cca33cc1-c50b-4618-a6de-b9236dd26507)


```py
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/b44233ab-f18f-4222-b63f-29a74702838a)

```py
df.skew()
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/3429a0d5-1d3d-4660-b79b-46a2b12a8050)

```py
np.log(df["Highly Positive Skew"])
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/d4fbd14d-0f73-4810-9a37-7f90e09c7f0d)

```py
np.reciprocal(df["Moderate Positive Skew"])
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/62102da6-8d20-47c9-8aaa-f5543409c8c8)

```py
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/186d2b92-d488-41a6-a178-0df95e0c8364)

```py
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/5336a561-434d-4552-944a-97535ccbcc29)

```py
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/fd614327-c87a-46fa-883e-0dce5e136cce)

```py
df.skew()
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/805e86cf-f9f5-4e20-a2ea-6b7328bc6443)

```py
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/6ce34d13-c097-4706-b409-4cf80b55e39e)

```py
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/55399f4f-9c8c-4be0-b747-31da55de4387)

```py
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/54adc02b-d562-442c-be3a-466037a3c753)

```py
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/0388b0c8-b211-45ad-a901-321712a8c922)

```py
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/c82b4630-bfe9-4d2b-9250-bb22d423b2f4)

```py
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/ShriramGH/EXNO-3-DS/assets/117991122/e86811e2-d338-4d47-8285-9387cf87c191)


## RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
