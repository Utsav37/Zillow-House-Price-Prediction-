

```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
dftrain=pd.read_csv(("Data Science ZExercise_TRAINING_CONFIDENTIAL1.csv"))
dftest = pd.read_csv("Data Science ZExercise_TEST_CONFIDENTIAL2.csv")

```


```python
dftrain.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PropertyID</th>
      <th>SaleDollarCnt</th>
      <th>TransDate</th>
      <th>censusblockgroup</th>
      <th>ZoneCodeCounty</th>
      <th>Usecode</th>
      <th>BedroomCnt</th>
      <th>BathroomCnt</th>
      <th>FinishedSquareFeet</th>
      <th>GarageSquareFeet</th>
      <th>...</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>BGMedHomeValue</th>
      <th>BGMedRent</th>
      <th>BGMedYearBuilt</th>
      <th>BGPctOwn</th>
      <th>BGPctVacant</th>
      <th>BGMedIncome</th>
      <th>BGPctKids</th>
      <th>BGMedAge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48648941</td>
      <td>285000.0</td>
      <td>5/23/2015</td>
      <td>5.300000e+11</td>
      <td>R7</td>
      <td>9</td>
      <td>4.0</td>
      <td>2.00</td>
      <td>1900.0</td>
      <td>480.0</td>
      <td>...</td>
      <td>47321389</td>
      <td>-122213716</td>
      <td>107800.0</td>
      <td>844.0</td>
      <td>1975.0</td>
      <td>0.6685</td>
      <td>0.0780</td>
      <td>42854</td>
      <td>0.1924</td>
      <td>48.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48648982</td>
      <td>309950.0</td>
      <td>8/22/2015</td>
      <td>5.300000e+11</td>
      <td>R8P</td>
      <td>9</td>
      <td>3.0</td>
      <td>2.00</td>
      <td>2170.0</td>
      <td>320.0</td>
      <td>...</td>
      <td>47482082</td>
      <td>-122244269</td>
      <td>181500.0</td>
      <td>925.0</td>
      <td>1969.0</td>
      <td>0.5753</td>
      <td>0.0192</td>
      <td>54013</td>
      <td>0.3718</td>
      <td>42.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48649024</td>
      <td>476000.0</td>
      <td>8/27/2015</td>
      <td>5.300000e+11</td>
      <td>SF 7200</td>
      <td>9</td>
      <td>4.0</td>
      <td>1.00</td>
      <td>2150.0</td>
      <td>590.0</td>
      <td>...</td>
      <td>47561383</td>
      <td>-122308083</td>
      <td>344300.0</td>
      <td>733.0</td>
      <td>1946.0</td>
      <td>0.6331</td>
      <td>0.0000</td>
      <td>56782</td>
      <td>0.3207</td>
      <td>40.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48649040</td>
      <td>324950.0</td>
      <td>7/1/2015</td>
      <td>5.300000e+11</td>
      <td>R1</td>
      <td>9</td>
      <td>4.0</td>
      <td>2.25</td>
      <td>2560.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>47387929</td>
      <td>-122279389</td>
      <td>284200.0</td>
      <td>900.0</td>
      <td>1977.0</td>
      <td>0.5456</td>
      <td>0.0573</td>
      <td>44200</td>
      <td>0.3359</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>48649057</td>
      <td>325000.0</td>
      <td>6/20/2015</td>
      <td>5.300000e+11</td>
      <td>LDR</td>
      <td>9</td>
      <td>4.0</td>
      <td>1.75</td>
      <td>1720.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>47477068</td>
      <td>-122263852</td>
      <td>290100.0</td>
      <td>802.0</td>
      <td>1972.0</td>
      <td>0.4267</td>
      <td>0.0551</td>
      <td>65282</td>
      <td>0.1633</td>
      <td>44.4</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
dftest.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PropertyID</th>
      <th>SaleDollarCnt</th>
      <th>TransDate</th>
      <th>censusblockgroup</th>
      <th>ZoneCodeCounty</th>
      <th>Usecode</th>
      <th>BedroomCnt</th>
      <th>BathroomCnt</th>
      <th>FinishedSquareFeet</th>
      <th>GarageSquareFeet</th>
      <th>...</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>BGMedHomeValue</th>
      <th>BGMedRent</th>
      <th>BGMedYearBuilt</th>
      <th>BGPctOwn</th>
      <th>BGPctVacant</th>
      <th>BGMedIncome</th>
      <th>BGPctKids</th>
      <th>BGMedAge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48735321</td>
      <td>NaN</td>
      <td>10/31/2015</td>
      <td>5.300000e+11</td>
      <td>SF 9600</td>
      <td>9</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>5540</td>
      <td>NaN</td>
      <td>...</td>
      <td>47725642</td>
      <td>-122283771</td>
      <td>527700.0</td>
      <td>1750.0</td>
      <td>1956.0</td>
      <td>0.9134</td>
      <td>0.1061</td>
      <td>113450</td>
      <td>0.2524</td>
      <td>49.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48735471</td>
      <td>NaN</td>
      <td>11/6/2015</td>
      <td>5.300000e+11</td>
      <td>SF 9600</td>
      <td>9</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>2470</td>
      <td>510.0</td>
      <td>...</td>
      <td>47726993</td>
      <td>-122281969</td>
      <td>527700.0</td>
      <td>1750.0</td>
      <td>1956.0</td>
      <td>0.9134</td>
      <td>0.1061</td>
      <td>113450</td>
      <td>0.2524</td>
      <td>49.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49128764</td>
      <td>NaN</td>
      <td>10/17/2015</td>
      <td>5.300000e+11</td>
      <td>SF 7200</td>
      <td>9</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1680</td>
      <td>NaN</td>
      <td>...</td>
      <td>47731749</td>
      <td>-122289304</td>
      <td>527700.0</td>
      <td>1750.0</td>
      <td>1956.0</td>
      <td>0.9134</td>
      <td>0.1061</td>
      <td>113450</td>
      <td>0.2524</td>
      <td>49.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48897535</td>
      <td>NaN</td>
      <td>11/19/2015</td>
      <td>5.300000e+11</td>
      <td>SF 7200</td>
      <td>9</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>990</td>
      <td>260.0</td>
      <td>...</td>
      <td>47728810</td>
      <td>-122289224</td>
      <td>527700.0</td>
      <td>1750.0</td>
      <td>1956.0</td>
      <td>0.9134</td>
      <td>0.1061</td>
      <td>113450</td>
      <td>0.2524</td>
      <td>49.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49083957</td>
      <td>NaN</td>
      <td>12/15/2015</td>
      <td>5.300000e+11</td>
      <td>SF 9600</td>
      <td>9</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2960</td>
      <td>550.0</td>
      <td>...</td>
      <td>47731170</td>
      <td>-122282684</td>
      <td>527700.0</td>
      <td>1750.0</td>
      <td>1956.0</td>
      <td>0.9134</td>
      <td>0.1061</td>
      <td>113450</td>
      <td>0.2524</td>
      <td>49.6</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
dftest.isna().sum()
```




    PropertyID               0
    SaleDollarCnt         4402
    TransDate                0
    censusblockgroup         0
    ZoneCodeCounty           0
    Usecode                  0
    BedroomCnt               0
    BathroomCnt              0
    FinishedSquareFeet       0
    GarageSquareFeet      1138
    LotSizeSquareFeet        0
    StoryCnt                 0
    BuiltYear                0
    ViewType              3404
    Latitude                 0
    Longitude                0
    BGMedHomeValue           7
    BGMedRent              963
    BGMedYearBuilt          62
    BGPctOwn                 0
    BGPctVacant              0
    BGMedIncome              0
    BGPctKids                0
    BGMedAge                 0
    dtype: int64




```python
dftrain.shape
```




    (11588, 24)




```python
dftest.shape
```




    (4402, 24)




```python

# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
```


```python
missing_values_table(dftest)
```

    Your selected dataframe has 24 columns.
    There are 6 columns that have missing values.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing Values</th>
      <th>% of Total Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SaleDollarCnt</th>
      <td>4402</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>ViewType</th>
      <td>3404</td>
      <td>77.3</td>
    </tr>
    <tr>
      <th>GarageSquareFeet</th>
      <td>1138</td>
      <td>25.9</td>
    </tr>
    <tr>
      <th>BGMedRent</th>
      <td>963</td>
      <td>21.9</td>
    </tr>
    <tr>
      <th>BGMedYearBuilt</th>
      <td>62</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th>BGMedHomeValue</th>
      <td>7</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
missing_values_table(dftrain)
```

    Your selected dataframe has 24 columns.
    There are 5 columns that have missing values.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing Values</th>
      <th>% of Total Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ViewType</th>
      <td>8956</td>
      <td>77.3</td>
    </tr>
    <tr>
      <th>GarageSquareFeet</th>
      <td>2841</td>
      <td>24.5</td>
    </tr>
    <tr>
      <th>BGMedRent</th>
      <td>2631</td>
      <td>22.7</td>
    </tr>
    <tr>
      <th>BGMedYearBuilt</th>
      <td>247</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>BGMedHomeValue</th>
      <td>6</td>
      <td>0.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Shape of dftrain is : ",dftrain.shape)
dftrain_y=pd.DataFrame(dftrain['SaleDollarCnt'])
dftrain_X=dftrain.drop(['SaleDollarCnt'],axis=1)
print("Shape of X is : ",dftrain_X.shape)
```

    Shape of dftrain is :  (11588, 24)
    Shape of X is :  (11588, 23)
    


```python
print("Shape of dftest is : ",dftest.shape)
dftest_y=pd.DataFrame(dftest['SaleDollarCnt'])
dftest_X=dftest.drop(['SaleDollarCnt'],axis=1)
print("Shape of X is : ",dftest_X.shape)
print(dftest_y.shape)
```

    Shape of dftest is :  (4402, 24)
    Shape of X is :  (4402, 23)
    (4402, 1)
    


```python
dftest_X.dtypes=="object"
```




    PropertyID            False
    TransDate              True
    censusblockgroup      False
    ZoneCodeCounty         True
    Usecode               False
    BedroomCnt            False
    BathroomCnt           False
    FinishedSquareFeet    False
    GarageSquareFeet      False
    LotSizeSquareFeet     False
    StoryCnt              False
    BuiltYear             False
    ViewType              False
    Latitude              False
    Longitude             False
    BGMedHomeValue        False
    BGMedRent             False
    BGMedYearBuilt        False
    BGPctOwn              False
    BGPctVacant           False
    BGMedIncome           False
    BGPctKids             False
    BGMedAge              False
    dtype: bool




```python
dftrain_X["TransDate"]= dftrain_X.TransDate.str.slice(0,1)
dftrain_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PropertyID</th>
      <th>TransDate</th>
      <th>censusblockgroup</th>
      <th>ZoneCodeCounty</th>
      <th>Usecode</th>
      <th>BedroomCnt</th>
      <th>BathroomCnt</th>
      <th>FinishedSquareFeet</th>
      <th>GarageSquareFeet</th>
      <th>LotSizeSquareFeet</th>
      <th>...</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>BGMedHomeValue</th>
      <th>BGMedRent</th>
      <th>BGMedYearBuilt</th>
      <th>BGPctOwn</th>
      <th>BGPctVacant</th>
      <th>BGMedIncome</th>
      <th>BGPctKids</th>
      <th>BGMedAge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48648941</td>
      <td>5</td>
      <td>5.300000e+11</td>
      <td>R7</td>
      <td>9</td>
      <td>4.0</td>
      <td>2.00</td>
      <td>1900.0</td>
      <td>480.0</td>
      <td>7482</td>
      <td>...</td>
      <td>47321389</td>
      <td>-122213716</td>
      <td>107800.0</td>
      <td>844.0</td>
      <td>1975.0</td>
      <td>0.6685</td>
      <td>0.0780</td>
      <td>42854</td>
      <td>0.1924</td>
      <td>48.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48648982</td>
      <td>8</td>
      <td>5.300000e+11</td>
      <td>R8P</td>
      <td>9</td>
      <td>3.0</td>
      <td>2.00</td>
      <td>2170.0</td>
      <td>320.0</td>
      <td>14208</td>
      <td>...</td>
      <td>47482082</td>
      <td>-122244269</td>
      <td>181500.0</td>
      <td>925.0</td>
      <td>1969.0</td>
      <td>0.5753</td>
      <td>0.0192</td>
      <td>54013</td>
      <td>0.3718</td>
      <td>42.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48649024</td>
      <td>8</td>
      <td>5.300000e+11</td>
      <td>SF 7200</td>
      <td>9</td>
      <td>4.0</td>
      <td>1.00</td>
      <td>2150.0</td>
      <td>590.0</td>
      <td>6500</td>
      <td>...</td>
      <td>47561383</td>
      <td>-122308083</td>
      <td>344300.0</td>
      <td>733.0</td>
      <td>1946.0</td>
      <td>0.6331</td>
      <td>0.0000</td>
      <td>56782</td>
      <td>0.3207</td>
      <td>40.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48649040</td>
      <td>7</td>
      <td>5.300000e+11</td>
      <td>R1</td>
      <td>9</td>
      <td>4.0</td>
      <td>2.25</td>
      <td>2560.0</td>
      <td>NaN</td>
      <td>15767</td>
      <td>...</td>
      <td>47387929</td>
      <td>-122279389</td>
      <td>284200.0</td>
      <td>900.0</td>
      <td>1977.0</td>
      <td>0.5456</td>
      <td>0.0573</td>
      <td>44200</td>
      <td>0.3359</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>48649057</td>
      <td>6</td>
      <td>5.300000e+11</td>
      <td>LDR</td>
      <td>9</td>
      <td>4.0</td>
      <td>1.75</td>
      <td>1720.0</td>
      <td>NaN</td>
      <td>8620</td>
      <td>...</td>
      <td>47477068</td>
      <td>-122263852</td>
      <td>290100.0</td>
      <td>802.0</td>
      <td>1972.0</td>
      <td>0.4267</td>
      <td>0.0551</td>
      <td>65282</td>
      <td>0.1633</td>
      <td>44.4</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
dftrain_X.ZoneCodeCounty.value_counts()
```




    SF 5000    2243
    R6         1363
    R4         1120
    RA5         622
    R5          607
    SF 7200     465
    R8          416
    SR6         354
    RS7200      270
    R3.5        260
    RS7.2       252
    RSA 6       218
    R1          200
    RA2.5       182
    MU          178
    R6P         145
    R7          144
    RSX 7.2     121
    RS9.6       110
    URPSO        89
    UL7200       84
    R9.6         84
    RS 8.5       84
    LDR          83
    SR4.5        83
    R2           72
    R15          71
    UV           70
    LR1          65
    RS 7.2       55
               ... 
    NC365         1
    BO            1
    MUR35         1
    CBSO          1
    LR2 RC        1
    RSX 8.5       1
    AI1           1
    MRG           1
    NCC           1
    RMF           1
    OS2           1
    PLA 3C        1
    R6C           1
    RM18          1
    TC4           1
    T             1
    RSE           1
    RS 11         1
    RM24          1
    IG2 U65       1
    MSC 4         1
    R 40000       1
    IB U85        1
    SR30          1
    PRR           1
    MDR           1
    GDC           1
    RS35.0        1
    MC            1
    DC            1
    Name: ZoneCodeCounty, Length: 178, dtype: int64




```python
dftest_X.ZoneCodeCounty.value_counts()
```




    SF 5000         818
    R6              499
    R4              424
    RA5             222
    R5              215
    SF 7200         213
    SR6             153
    R8              147
    RS7200          130
    RS7.2           113
    R3.5             99
    RSA 6            90
    RA2.5            66
    R7               59
    R1               58
    MU               56
    R6P              49
    RS9.6            46
    RSX 7.2          45
    UL7200           41
    LDR              40
    RS 8.5           36
    URPSO            30
    R2               29
    LR1              28
    SR4.5            26
    RS 7.2           25
    R2.5             24
    R9.6             23
    R 9600           20
                   ... 
    PRR               1
    RS                1
    SR30              1
    NC130             1
    TC                1
    NC2P40            1
    SFR 10.0          1
    R7.5              1
    MRRC              1
    MRG               1
    C2                1
    RSA 1             1
    RIN SINGLE F      1
    CR                1
    PR                1
    MU12              1
    UHUCR             1
    TL 10A            1
    RA10PSO           1
    R12P              1
    RMF               1
    RM12              1
    SFD               1
    T                 1
    MR                1
    R18P              1
    RA2.5P            1
    NC240             1
    RM18              1
    DNTNMU            1
    Name: ZoneCodeCounty, Length: 143, dtype: int64




```python
zonedf2=pd.DataFrame(data=dftest,columns=['ZoneCodeCounty','SaleDollarCnt'])
zonedf2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ZoneCodeCounty</th>
      <th>SaleDollarCnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SF 9600</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SF 9600</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SF 7200</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SF 7200</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SF 9600</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
zonedf1=pd.DataFrame(data=dftrain,columns=['ZoneCodeCounty','SaleDollarCnt'])
zonedf1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ZoneCodeCounty</th>
      <th>SaleDollarCnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>R7</td>
      <td>285000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>R8P</td>
      <td>309950.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SF 7200</td>
      <td>476000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>R1</td>
      <td>324950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LDR</td>
      <td>325000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
zone1=zonedf1.groupby('ZoneCodeCounty').mean()
zone1["count"]=zonedf1.groupby('ZoneCodeCounty').count()
zone1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SaleDollarCnt</th>
      <th>count</th>
    </tr>
    <tr>
      <th>ZoneCodeCounty</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A10</th>
      <td>4.618250e+05</td>
      <td>16</td>
    </tr>
    <tr>
      <th>A35</th>
      <td>4.068126e+05</td>
      <td>13</td>
    </tr>
    <tr>
      <th>AI1</th>
      <td>2.825000e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>BO</th>
      <td>3.150000e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>C1</th>
      <td>2.055750e+05</td>
      <td>6</td>
    </tr>
    <tr>
      <th>C140</th>
      <td>3.967500e+05</td>
      <td>2</td>
    </tr>
    <tr>
      <th>CB</th>
      <td>3.214333e+05</td>
      <td>3</td>
    </tr>
    <tr>
      <th>CBSO</th>
      <td>2.499500e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CM2</th>
      <td>1.549000e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>DC</th>
      <td>2.500000e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>DCE</th>
      <td>1.900000e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>DUC</th>
      <td>2.200000e+05</td>
      <td>2</td>
    </tr>
    <tr>
      <th>F</th>
      <td>2.628250e+05</td>
      <td>8</td>
    </tr>
    <tr>
      <th>GDC</th>
      <td>6.645000e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>I</th>
      <td>3.584750e+05</td>
      <td>2</td>
    </tr>
    <tr>
      <th>IB U85</th>
      <td>2.750000e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>IG2 U65</th>
      <td>2.695000e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>LDR</th>
      <td>4.103939e+05</td>
      <td>83</td>
    </tr>
    <tr>
      <th>LR1</th>
      <td>6.167631e+05</td>
      <td>65</td>
    </tr>
    <tr>
      <th>LR2</th>
      <td>5.393516e+05</td>
      <td>38</td>
    </tr>
    <tr>
      <th>LR2 RC</th>
      <td>3.390000e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>LR3</th>
      <td>5.985191e+05</td>
      <td>38</td>
    </tr>
    <tr>
      <th>LR3 RC</th>
      <td>3.410000e+05</td>
      <td>2</td>
    </tr>
    <tr>
      <th>MC</th>
      <td>2.910000e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>MDR</th>
      <td>2.500000e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>MFM</th>
      <td>7.966693e+05</td>
      <td>3</td>
    </tr>
    <tr>
      <th>MHO</th>
      <td>2.645000e+05</td>
      <td>2</td>
    </tr>
    <tr>
      <th>MRD</th>
      <td>2.472667e+05</td>
      <td>9</td>
    </tr>
    <tr>
      <th>MRG</th>
      <td>3.238800e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>MRM</th>
      <td>2.550000e+05</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>RSX 35</th>
      <td>1.204000e+06</td>
      <td>4</td>
    </tr>
    <tr>
      <th>RSX 7.2</th>
      <td>6.920782e+05</td>
      <td>121</td>
    </tr>
    <tr>
      <th>RSX 8.5</th>
      <td>5.380000e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>SF 5000</th>
      <td>7.009578e+05</td>
      <td>2243</td>
    </tr>
    <tr>
      <th>SF 7200</th>
      <td>6.808870e+05</td>
      <td>465</td>
    </tr>
    <tr>
      <th>SF 9600</th>
      <td>1.293949e+06</td>
      <td>36</td>
    </tr>
    <tr>
      <th>SFD</th>
      <td>4.138125e+05</td>
      <td>8</td>
    </tr>
    <tr>
      <th>SFE</th>
      <td>7.235929e+05</td>
      <td>7</td>
    </tr>
    <tr>
      <th>SFR 10.0</th>
      <td>1.410000e+06</td>
      <td>1</td>
    </tr>
    <tr>
      <th>SFS</th>
      <td>7.225824e+05</td>
      <td>54</td>
    </tr>
    <tr>
      <th>SFSL</th>
      <td>8.182029e+05</td>
      <td>33</td>
    </tr>
    <tr>
      <th>SR1</th>
      <td>4.458000e+05</td>
      <td>6</td>
    </tr>
    <tr>
      <th>SR3</th>
      <td>4.245069e+05</td>
      <td>29</td>
    </tr>
    <tr>
      <th>SR30</th>
      <td>1.683000e+06</td>
      <td>1</td>
    </tr>
    <tr>
      <th>SR4.5</th>
      <td>3.733027e+05</td>
      <td>83</td>
    </tr>
    <tr>
      <th>SR6</th>
      <td>3.171909e+05</td>
      <td>354</td>
    </tr>
    <tr>
      <th>SR8</th>
      <td>3.361533e+05</td>
      <td>32</td>
    </tr>
    <tr>
      <th>SVV</th>
      <td>5.000000e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>T</th>
      <td>2.750000e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>TC</th>
      <td>3.836333e+05</td>
      <td>3</td>
    </tr>
    <tr>
      <th>TC4</th>
      <td>4.100000e+05</td>
      <td>1</td>
    </tr>
    <tr>
      <th>UL15000</th>
      <td>3.325000e+05</td>
      <td>3</td>
    </tr>
    <tr>
      <th>UL7200</th>
      <td>2.882157e+05</td>
      <td>84</td>
    </tr>
    <tr>
      <th>UL9600</th>
      <td>1.900000e+05</td>
      <td>2</td>
    </tr>
    <tr>
      <th>UM2400</th>
      <td>2.144750e+05</td>
      <td>2</td>
    </tr>
    <tr>
      <th>UR</th>
      <td>3.343276e+05</td>
      <td>45</td>
    </tr>
    <tr>
      <th>URPSO</th>
      <td>6.732950e+05</td>
      <td>89</td>
    </tr>
    <tr>
      <th>UV</th>
      <td>7.296507e+05</td>
      <td>70</td>
    </tr>
    <tr>
      <th>UVEV</th>
      <td>7.858723e+05</td>
      <td>40</td>
    </tr>
    <tr>
      <th>WD II</th>
      <td>2.525000e+06</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>178 rows × 2 columns</p>
</div>




```python
zone2=zonedf2.groupby('ZoneCodeCounty').mean()
zone2["count"]=zonedf2.groupby('ZoneCodeCounty').count()
zone2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SaleDollarCnt</th>
      <th>count</th>
    </tr>
    <tr>
      <th>ZoneCodeCounty</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A10</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>A35</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>C1</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>C2</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>CR</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>DC</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>DCE</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>DNTNMU</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>EP</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>F</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LDR</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LR1</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LR2</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LR3</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>MFM</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>MR</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>MRG</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>MRRC</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>MRT16</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>MU</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>MU12</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>MUR45</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>MUR70</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>NC130</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>NC240</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>NC2P40</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>O</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>PR</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>PRR</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>PUD</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>RSA 6</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>RSA 8</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>RSLTC</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>RSX 7.2</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>RSX 8.5</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SF 5000</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SF 7200</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SF 9600</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SFD</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SFE</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SFR 10.0</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SFS</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SFSL</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SR1</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SR3</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SR30</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SR4.5</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SR6</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SR8</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SVV</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>T</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>TC</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>TL 10A</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>UHUCR</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>UL7200</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>UR</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>URPSO</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>US R1</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>UV</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>UVEV</th>
      <td>NaN</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>143 rows × 2 columns</p>
</div>




```python
datedf1=pd.DataFrame(data=dftrain,columns=['TransDate','SaleDollarCnt'])
datedf1.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransDate</th>
      <th>SaleDollarCnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5/23/2015</td>
      <td>285000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8/22/2015</td>
      <td>309950.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8/27/2015</td>
      <td>476000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7/1/2015</td>
      <td>324950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6/20/2015</td>
      <td>325000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
datedf2=pd.DataFrame(data=dftest,columns=['TransDate','SaleDollarCnt'])
datedf2.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransDate</th>
      <th>SaleDollarCnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/31/2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11/6/2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10/17/2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11/19/2015</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12/15/2015</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python

import matplotlib.pyplot as plt
%matplotlib inline

datedf1['TransDate'] = pd.to_datetime(datedf1['TransDate'])

```


```python
datedf2['TransDate'] = pd.to_datetime(datedf2['TransDate'])

```


```python
datedf1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransDate</th>
      <th>SaleDollarCnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-05-23</td>
      <td>285000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-08-22</td>
      <td>309950.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-08-27</td>
      <td>476000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-07-01</td>
      <td>324950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-06-20</td>
      <td>325000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
datedf2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransDate</th>
      <th>SaleDollarCnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-10-31</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-11-06</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-10-17</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-11-19</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-12-15</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python

datedf1=datedf1.sort_values(by='TransDate')
datedf1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransDate</th>
      <th>SaleDollarCnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10314</th>
      <td>2015-04-01</td>
      <td>292500.0</td>
    </tr>
    <tr>
      <th>5907</th>
      <td>2015-04-01</td>
      <td>586000.0</td>
    </tr>
    <tr>
      <th>9526</th>
      <td>2015-04-01</td>
      <td>545000.0</td>
    </tr>
    <tr>
      <th>11152</th>
      <td>2015-04-01</td>
      <td>680000.0</td>
    </tr>
    <tr>
      <th>2718</th>
      <td>2015-04-01</td>
      <td>970000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

datedf2=datedf2.sort_values(by='TransDate')
datedf2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TransDate</th>
      <th>SaleDollarCnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3072</th>
      <td>2015-10-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3253</th>
      <td>2015-10-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3256</th>
      <td>2015-10-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>767</th>
      <td>2015-10-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3275</th>
      <td>2015-10-01</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python

datedf2.index=datedf2['TransDate']
date3=datedf2.resample('M').mean()
date3
date3['count']=datedf2.resample('M').count
date3['count']=datedf2.resample('M').count()
```


```python
date3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SaleDollarCnt</th>
      <th>count</th>
    </tr>
    <tr>
      <th>TransDate</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-10-31</th>
      <td>NaN</td>
      <td>1852</td>
    </tr>
    <tr>
      <th>2015-11-30</th>
      <td>NaN</td>
      <td>1034</td>
    </tr>
    <tr>
      <th>2015-12-31</th>
      <td>NaN</td>
      <td>1450</td>
    </tr>
    <tr>
      <th>2016-01-31</th>
      <td>NaN</td>
      <td>66</td>
    </tr>
  </tbody>
</table>
</div>




```python

datedf1.index=datedf1['TransDate']
date2=datedf1.resample('M').mean()
date2
date2['count']=datedf1.resample('M').count
date2['count']=datedf1.resample('M').count()
```


```python
date2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SaleDollarCnt</th>
      <th>count</th>
    </tr>
    <tr>
      <th>TransDate</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-04-30</th>
      <td>614949.954518</td>
      <td>1671</td>
    </tr>
    <tr>
      <th>2015-05-31</th>
      <td>614010.690454</td>
      <td>2116</td>
    </tr>
    <tr>
      <th>2015-06-30</th>
      <td>637301.737849</td>
      <td>1934</td>
    </tr>
    <tr>
      <th>2015-07-31</th>
      <td>601015.601758</td>
      <td>2275</td>
    </tr>
    <tr>
      <th>2015-08-31</th>
      <td>613819.837924</td>
      <td>1888</td>
    </tr>
    <tr>
      <th>2015-09-30</th>
      <td>602209.944249</td>
      <td>1704</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop both date columns 
print("X shape is : ",dftrain_X.shape)
dftrain_X.drop("TransDate",axis=1,inplace=True)
print("X shape is : ",dftrain_X.shape)

```

    X shape is :  (11588, 23)
    X shape is :  (11588, 22)
    


```python
# Drop both date columns 
print("X shape is : ",dftest_X.shape)
dftest_X.drop("TransDate",axis=1,inplace=True)
print("X shape is : ",dftest_X.shape)

```

    X shape is :  (4402, 23)
    X shape is :  (4402, 22)
    


```python
print("Shape of X is : ",dftrain_X.shape)
dftrain_X=pd.get_dummies(dftrain_X)
print("Shape of X is : ",dftrain_X.shape)
dftrain_X.head()
```

    Shape of X is :  (11588, 22)
    Shape of X is :  (11588, 199)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PropertyID</th>
      <th>censusblockgroup</th>
      <th>Usecode</th>
      <th>BedroomCnt</th>
      <th>BathroomCnt</th>
      <th>FinishedSquareFeet</th>
      <th>GarageSquareFeet</th>
      <th>LotSizeSquareFeet</th>
      <th>StoryCnt</th>
      <th>BuiltYear</th>
      <th>...</th>
      <th>ZoneCodeCounty_TC4</th>
      <th>ZoneCodeCounty_UL15000</th>
      <th>ZoneCodeCounty_UL7200</th>
      <th>ZoneCodeCounty_UL9600</th>
      <th>ZoneCodeCounty_UM2400</th>
      <th>ZoneCodeCounty_UR</th>
      <th>ZoneCodeCounty_URPSO</th>
      <th>ZoneCodeCounty_UV</th>
      <th>ZoneCodeCounty_UVEV</th>
      <th>ZoneCodeCounty_WD II</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48648941</td>
      <td>5.300000e+11</td>
      <td>9</td>
      <td>4.0</td>
      <td>2.00</td>
      <td>1900.0</td>
      <td>480.0</td>
      <td>7482</td>
      <td>1.0</td>
      <td>1965.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48648982</td>
      <td>5.300000e+11</td>
      <td>9</td>
      <td>3.0</td>
      <td>2.00</td>
      <td>2170.0</td>
      <td>320.0</td>
      <td>14208</td>
      <td>1.0</td>
      <td>1953.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48649024</td>
      <td>5.300000e+11</td>
      <td>9</td>
      <td>4.0</td>
      <td>1.00</td>
      <td>2150.0</td>
      <td>590.0</td>
      <td>6500</td>
      <td>1.0</td>
      <td>1955.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48649040</td>
      <td>5.300000e+11</td>
      <td>9</td>
      <td>4.0</td>
      <td>2.25</td>
      <td>2560.0</td>
      <td>NaN</td>
      <td>15767</td>
      <td>1.0</td>
      <td>1962.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>48649057</td>
      <td>5.300000e+11</td>
      <td>9</td>
      <td>4.0</td>
      <td>1.75</td>
      <td>1720.0</td>
      <td>NaN</td>
      <td>8620</td>
      <td>2.0</td>
      <td>1948.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 199 columns</p>
</div>




```python
print("Shape of X is : ",dftest_X.shape)
dftest_X=pd.get_dummies(dftest_X)
print("Shape of X is : ",dftest_X.shape)
dftest_X.head()
```

    Shape of X is :  (4402, 22)
    Shape of X is :  (4402, 164)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PropertyID</th>
      <th>censusblockgroup</th>
      <th>Usecode</th>
      <th>BedroomCnt</th>
      <th>BathroomCnt</th>
      <th>FinishedSquareFeet</th>
      <th>GarageSquareFeet</th>
      <th>LotSizeSquareFeet</th>
      <th>StoryCnt</th>
      <th>BuiltYear</th>
      <th>...</th>
      <th>ZoneCodeCounty_T</th>
      <th>ZoneCodeCounty_TC</th>
      <th>ZoneCodeCounty_TL 10A</th>
      <th>ZoneCodeCounty_UHUCR</th>
      <th>ZoneCodeCounty_UL7200</th>
      <th>ZoneCodeCounty_UR</th>
      <th>ZoneCodeCounty_URPSO</th>
      <th>ZoneCodeCounty_US R1</th>
      <th>ZoneCodeCounty_UV</th>
      <th>ZoneCodeCounty_UVEV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48735321</td>
      <td>5.300000e+11</td>
      <td>9</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>5540</td>
      <td>NaN</td>
      <td>25338</td>
      <td>1.0</td>
      <td>1940.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48735471</td>
      <td>5.300000e+11</td>
      <td>9</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>2470</td>
      <td>510.0</td>
      <td>26006</td>
      <td>1.0</td>
      <td>1966.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49128764</td>
      <td>5.300000e+11</td>
      <td>9</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1680</td>
      <td>NaN</td>
      <td>8743</td>
      <td>2.0</td>
      <td>1928.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48897535</td>
      <td>5.300000e+11</td>
      <td>9</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>990</td>
      <td>260.0</td>
      <td>12219</td>
      <td>1.0</td>
      <td>1940.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49083957</td>
      <td>5.300000e+11</td>
      <td>9</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2960</td>
      <td>550.0</td>
      <td>23568</td>
      <td>1.0</td>
      <td>1951.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 164 columns</p>
</div>




```python
dftrain_X.Usecode.value_counts()
```




    9    11588
    Name: Usecode, dtype: int64




```python
dftest_X.Usecode.value_counts()
```




    9    4402
    Name: Usecode, dtype: int64




```python
dftest_X.shape
```




    (4402, 164)




```python
# Drop both  columns 
print("X shape is : ",dftrain_X.shape)
dftrain_X.drop("Usecode",axis=1,inplace=True)
print("X shape is : ",dftrain_X.shape)



# Drop both columns 
print("X shape is : ",dftest_X.shape)
dftest_X.drop("Usecode",axis=1,inplace=True)
print("X shape is : ",dftest_X.shape)

```

    X shape is :  (11588, 199)
    X shape is :  (11588, 198)
    X shape is :  (4402, 164)
    X shape is :  (4402, 163)
    


```python
dftrain_X.censusblockgroup.value_counts()
```




    5.300000e+11    11588
    Name: censusblockgroup, dtype: int64




```python
dftest_X.censusblockgroup.value_counts()
```




    5.300000e+11    4402
    Name: censusblockgroup, dtype: int64




```python
# Drop both  columns 
print("X shape is : ",dftrain_X.shape)
dftrain_X.drop("censusblockgroup",axis=1,inplace=True)
print("X shape is : ",dftrain_X.shape)



# Drop both  columns 
print("X shape is : ",dftest_X.shape)
dftest_X.drop("censusblockgroup",axis=1,inplace=True)
print("X shape is : ",dftest_X.shape)

```

    X shape is :  (11588, 198)
    X shape is :  (11588, 197)
    X shape is :  (4402, 163)
    X shape is :  (4402, 162)
    


```python
print("X shape is : ",dftest_X.shape)
```

    X shape is :  (4402, 162)
    


```python
dftest_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PropertyID</th>
      <th>BedroomCnt</th>
      <th>BathroomCnt</th>
      <th>FinishedSquareFeet</th>
      <th>GarageSquareFeet</th>
      <th>LotSizeSquareFeet</th>
      <th>StoryCnt</th>
      <th>BuiltYear</th>
      <th>ViewType</th>
      <th>Latitude</th>
      <th>...</th>
      <th>ZoneCodeCounty_T</th>
      <th>ZoneCodeCounty_TC</th>
      <th>ZoneCodeCounty_TL 10A</th>
      <th>ZoneCodeCounty_UHUCR</th>
      <th>ZoneCodeCounty_UL7200</th>
      <th>ZoneCodeCounty_UR</th>
      <th>ZoneCodeCounty_URPSO</th>
      <th>ZoneCodeCounty_US R1</th>
      <th>ZoneCodeCounty_UV</th>
      <th>ZoneCodeCounty_UVEV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48735321</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>5540</td>
      <td>NaN</td>
      <td>25338</td>
      <td>1.0</td>
      <td>1940.0</td>
      <td>78.0</td>
      <td>47725642</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48735471</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>2470</td>
      <td>510.0</td>
      <td>26006</td>
      <td>1.0</td>
      <td>1966.0</td>
      <td>78.0</td>
      <td>47726993</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49128764</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1680</td>
      <td>NaN</td>
      <td>8743</td>
      <td>2.0</td>
      <td>1928.0</td>
      <td>NaN</td>
      <td>47731749</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48897535</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>990</td>
      <td>260.0</td>
      <td>12219</td>
      <td>1.0</td>
      <td>1940.0</td>
      <td>NaN</td>
      <td>47728810</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49083957</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2960</td>
      <td>550.0</td>
      <td>23568</td>
      <td>1.0</td>
      <td>1951.0</td>
      <td>82.0</td>
      <td>47731170</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 162 columns</p>
</div>




```python
dftrain_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PropertyID</th>
      <th>BedroomCnt</th>
      <th>BathroomCnt</th>
      <th>FinishedSquareFeet</th>
      <th>GarageSquareFeet</th>
      <th>LotSizeSquareFeet</th>
      <th>StoryCnt</th>
      <th>BuiltYear</th>
      <th>ViewType</th>
      <th>Latitude</th>
      <th>...</th>
      <th>ZoneCodeCounty_TC4</th>
      <th>ZoneCodeCounty_UL15000</th>
      <th>ZoneCodeCounty_UL7200</th>
      <th>ZoneCodeCounty_UL9600</th>
      <th>ZoneCodeCounty_UM2400</th>
      <th>ZoneCodeCounty_UR</th>
      <th>ZoneCodeCounty_URPSO</th>
      <th>ZoneCodeCounty_UV</th>
      <th>ZoneCodeCounty_UVEV</th>
      <th>ZoneCodeCounty_WD II</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48648941</td>
      <td>4.0</td>
      <td>2.00</td>
      <td>1900.0</td>
      <td>480.0</td>
      <td>7482</td>
      <td>1.0</td>
      <td>1965.0</td>
      <td>NaN</td>
      <td>47321389</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48648982</td>
      <td>3.0</td>
      <td>2.00</td>
      <td>2170.0</td>
      <td>320.0</td>
      <td>14208</td>
      <td>1.0</td>
      <td>1953.0</td>
      <td>79.0</td>
      <td>47482082</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48649024</td>
      <td>4.0</td>
      <td>1.00</td>
      <td>2150.0</td>
      <td>590.0</td>
      <td>6500</td>
      <td>1.0</td>
      <td>1955.0</td>
      <td>NaN</td>
      <td>47561383</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48649040</td>
      <td>4.0</td>
      <td>2.25</td>
      <td>2560.0</td>
      <td>NaN</td>
      <td>15767</td>
      <td>1.0</td>
      <td>1962.0</td>
      <td>79.0</td>
      <td>47387929</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>48649057</td>
      <td>4.0</td>
      <td>1.75</td>
      <td>1720.0</td>
      <td>NaN</td>
      <td>8620</td>
      <td>2.0</td>
      <td>1948.0</td>
      <td>78.0</td>
      <td>47477068</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 197 columns</p>
</div>




```python
dftrain_X.columns
```




    Index(['PropertyID', 'BedroomCnt', 'BathroomCnt', 'FinishedSquareFeet',
           'GarageSquareFeet', 'LotSizeSquareFeet', 'StoryCnt', 'BuiltYear',
           'ViewType', 'Latitude',
           ...
           'ZoneCodeCounty_TC4', 'ZoneCodeCounty_UL15000', 'ZoneCodeCounty_UL7200',
           'ZoneCodeCounty_UL9600', 'ZoneCodeCounty_UM2400', 'ZoneCodeCounty_UR',
           'ZoneCodeCounty_URPSO', 'ZoneCodeCounty_UV', 'ZoneCodeCounty_UVEV',
           'ZoneCodeCounty_WD II'],
          dtype='object', length=197)




```python

```


```python
cols = [col for col in dftest_X.columns if col in dftrain_X.columns]
dftrain_X = dftrain_X[cols]
```


```python
dftrain_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PropertyID</th>
      <th>BedroomCnt</th>
      <th>BathroomCnt</th>
      <th>FinishedSquareFeet</th>
      <th>GarageSquareFeet</th>
      <th>LotSizeSquareFeet</th>
      <th>StoryCnt</th>
      <th>BuiltYear</th>
      <th>ViewType</th>
      <th>Latitude</th>
      <th>...</th>
      <th>ZoneCodeCounty_SR6</th>
      <th>ZoneCodeCounty_SR8</th>
      <th>ZoneCodeCounty_SVV</th>
      <th>ZoneCodeCounty_T</th>
      <th>ZoneCodeCounty_TC</th>
      <th>ZoneCodeCounty_UL7200</th>
      <th>ZoneCodeCounty_UR</th>
      <th>ZoneCodeCounty_URPSO</th>
      <th>ZoneCodeCounty_UV</th>
      <th>ZoneCodeCounty_UVEV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48648941</td>
      <td>4.0</td>
      <td>2.00</td>
      <td>1900.0</td>
      <td>480.0</td>
      <td>7482</td>
      <td>1.0</td>
      <td>1965.0</td>
      <td>NaN</td>
      <td>47321389</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48648982</td>
      <td>3.0</td>
      <td>2.00</td>
      <td>2170.0</td>
      <td>320.0</td>
      <td>14208</td>
      <td>1.0</td>
      <td>1953.0</td>
      <td>79.0</td>
      <td>47482082</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48649024</td>
      <td>4.0</td>
      <td>1.00</td>
      <td>2150.0</td>
      <td>590.0</td>
      <td>6500</td>
      <td>1.0</td>
      <td>1955.0</td>
      <td>NaN</td>
      <td>47561383</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48649040</td>
      <td>4.0</td>
      <td>2.25</td>
      <td>2560.0</td>
      <td>NaN</td>
      <td>15767</td>
      <td>1.0</td>
      <td>1962.0</td>
      <td>79.0</td>
      <td>47387929</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>48649057</td>
      <td>4.0</td>
      <td>1.75</td>
      <td>1720.0</td>
      <td>NaN</td>
      <td>8620</td>
      <td>2.0</td>
      <td>1948.0</td>
      <td>78.0</td>
      <td>47477068</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 145 columns</p>
</div>




```python
colstoadd = [col for col in dftest_X.columns if col not in dftrain_X.columns]
colstoadd
```




    ['ZoneCodeCounty_C2',
     'ZoneCodeCounty_CR',
     'ZoneCodeCounty_DNTNMU',
     'ZoneCodeCounty_EP',
     'ZoneCodeCounty_MR',
     'ZoneCodeCounty_MRRC',
     'ZoneCodeCounty_NC2P40',
     'ZoneCodeCounty_PR',
     'ZoneCodeCounty_R1P',
     'ZoneCodeCounty_R30',
     'ZoneCodeCounty_RA10DPA',
     'ZoneCodeCounty_RM2400',
     'ZoneCodeCounty_RSA 1',
     'ZoneCodeCounty_RSLTC',
     'ZoneCodeCounty_TL 10A',
     'ZoneCodeCounty_UHUCR',
     'ZoneCodeCounty_US R1']




```python
dftest_X[colstoadd].sum()
```




    ZoneCodeCounty_C2         1
    ZoneCodeCounty_CR         1
    ZoneCodeCounty_DNTNMU     1
    ZoneCodeCounty_EP         2
    ZoneCodeCounty_MR         1
    ZoneCodeCounty_MRRC       1
    ZoneCodeCounty_NC2P40     1
    ZoneCodeCounty_PR         1
    ZoneCodeCounty_R1P        1
    ZoneCodeCounty_R30        2
    ZoneCodeCounty_RA10DPA    3
    ZoneCodeCounty_RM2400     2
    ZoneCodeCounty_RSA 1      1
    ZoneCodeCounty_RSLTC      1
    ZoneCodeCounty_TL 10A     1
    ZoneCodeCounty_UHUCR      1
    ZoneCodeCounty_US R1      2
    dtype: int64




```python
for col in colstoadd:
    dftrain_X[col] = 0

dftrain_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PropertyID</th>
      <th>BedroomCnt</th>
      <th>BathroomCnt</th>
      <th>FinishedSquareFeet</th>
      <th>GarageSquareFeet</th>
      <th>LotSizeSquareFeet</th>
      <th>StoryCnt</th>
      <th>BuiltYear</th>
      <th>ViewType</th>
      <th>Latitude</th>
      <th>...</th>
      <th>ZoneCodeCounty_PR</th>
      <th>ZoneCodeCounty_R1P</th>
      <th>ZoneCodeCounty_R30</th>
      <th>ZoneCodeCounty_RA10DPA</th>
      <th>ZoneCodeCounty_RM2400</th>
      <th>ZoneCodeCounty_RSA 1</th>
      <th>ZoneCodeCounty_RSLTC</th>
      <th>ZoneCodeCounty_TL 10A</th>
      <th>ZoneCodeCounty_UHUCR</th>
      <th>ZoneCodeCounty_US R1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48648941</td>
      <td>4.0</td>
      <td>2.00</td>
      <td>1900.0</td>
      <td>480.0</td>
      <td>7482</td>
      <td>1.0</td>
      <td>1965.0</td>
      <td>NaN</td>
      <td>47321389</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48648982</td>
      <td>3.0</td>
      <td>2.00</td>
      <td>2170.0</td>
      <td>320.0</td>
      <td>14208</td>
      <td>1.0</td>
      <td>1953.0</td>
      <td>79.0</td>
      <td>47482082</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48649024</td>
      <td>4.0</td>
      <td>1.00</td>
      <td>2150.0</td>
      <td>590.0</td>
      <td>6500</td>
      <td>1.0</td>
      <td>1955.0</td>
      <td>NaN</td>
      <td>47561383</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48649040</td>
      <td>4.0</td>
      <td>2.25</td>
      <td>2560.0</td>
      <td>NaN</td>
      <td>15767</td>
      <td>1.0</td>
      <td>1962.0</td>
      <td>79.0</td>
      <td>47387929</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>48649057</td>
      <td>4.0</td>
      <td>1.75</td>
      <td>1720.0</td>
      <td>NaN</td>
      <td>8620</td>
      <td>2.0</td>
      <td>1948.0</td>
      <td>78.0</td>
      <td>47477068</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 162 columns</p>
</div>




```python
dftrain_X[colstoadd].sum()
```




    ZoneCodeCounty_C2         0
    ZoneCodeCounty_CR         0
    ZoneCodeCounty_DNTNMU     0
    ZoneCodeCounty_EP         0
    ZoneCodeCounty_MR         0
    ZoneCodeCounty_MRRC       0
    ZoneCodeCounty_NC2P40     0
    ZoneCodeCounty_PR         0
    ZoneCodeCounty_R1P        0
    ZoneCodeCounty_R30        0
    ZoneCodeCounty_RA10DPA    0
    ZoneCodeCounty_RM2400     0
    ZoneCodeCounty_RSA 1      0
    ZoneCodeCounty_RSLTC      0
    ZoneCodeCounty_TL 10A     0
    ZoneCodeCounty_UHUCR      0
    ZoneCodeCounty_US R1      0
    dtype: int64




```python
traincolsthatwerenotadded=[col for col in dftrain.columns if col not in dftest_X.columns]
```


```python
traincolsthatwerenotadded
```




    ['SaleDollarCnt', 'TransDate', 'censusblockgroup', 'ZoneCodeCounty', 'Usecode']




```python
pqr=pd.DataFrame(data=dftrain)
pqr.drop(['SaleDollarCnt', 'TransDate', 'censusblockgroup', 'Usecode'],inplace=True,axis=1)
```


```python
pqr.shape
```




    (11588, 20)




```python
xyz=pd.get_dummies(pqr)
xyz.shape
```




    (11588, 197)




```python
dftrain.shape
```




    (11588, 24)




```python
xyz.shape
```




    (11588, 197)




```python
traincolsthatwerenotadded=[col for col in xyz.columns if col not in dftest_X.columns]
```


```python
xyz[traincolsthatwerenotadded].sum()
```




    ZoneCodeCounty_AI1             1
    ZoneCodeCounty_BO              1
    ZoneCodeCounty_C140            2
    ZoneCodeCounty_CB              3
    ZoneCodeCounty_CBSO            1
    ZoneCodeCounty_CM2             1
    ZoneCodeCounty_DUC             2
    ZoneCodeCounty_GDC             1
    ZoneCodeCounty_I               2
    ZoneCodeCounty_IB U85          1
    ZoneCodeCounty_IG2 U65         1
    ZoneCodeCounty_LR2 RC          1
    ZoneCodeCounty_LR3 RC          2
    ZoneCodeCounty_MC              1
    ZoneCodeCounty_MDR             1
    ZoneCodeCounty_MHO             2
    ZoneCodeCounty_MRD             9
    ZoneCodeCounty_MRM             3
    ZoneCodeCounty_MSC 4           1
    ZoneCodeCounty_MUR             3
    ZoneCodeCounty_MUR35           1
    ZoneCodeCounty_NC365           1
    ZoneCodeCounty_NCC             1
    ZoneCodeCounty_OS2             1
    ZoneCodeCounty_PLA 17          1
    ZoneCodeCounty_PLA 3C          1
    ZoneCodeCounty_PLA 6D          2
    ZoneCodeCounty_PLA 6E          1
    ZoneCodeCounty_R               2
    ZoneCodeCounty_R 2800, OP      1
    ZoneCodeCounty_R 40000         1
    ZoneCodeCounty_R 5400A, OP     2
    ZoneCodeCounty_R1SO            2
    ZoneCodeCounty_R4C             1
    ZoneCodeCounty_R4P            13
    ZoneCodeCounty_R6C             1
    ZoneCodeCounty_RA3600          1
    ZoneCodeCounty_RB              1
    ZoneCodeCounty_RCC             1
    ZoneCodeCounty_RM1800          3
    ZoneCodeCounty_RM3600          2
    ZoneCodeCounty_RO              3
    ZoneCodeCounty_RS 11           1
    ZoneCodeCounty_RS 6.3          2
    ZoneCodeCounty_RS35.0          1
    ZoneCodeCounty_RSE             1
    ZoneCodeCounty_RSX 35          4
    ZoneCodeCounty_TC4             1
    ZoneCodeCounty_UL15000         3
    ZoneCodeCounty_UL9600          2
    ZoneCodeCounty_UM2400          2
    ZoneCodeCounty_WD II           2
    dtype: int64




```python
dftrain_X.shape
```




    (11588, 162)




```python
dftest_X.shape
```




    (4402, 162)




```python
check1=[col for col in dftrain_X.columns if col not in dftest_X.columns]
```


```python
check1
```




    []




```python
check2=[col for col in dftrain_X.columns if col in dftest_X.columns]
len(check2)
```




    162




```python
dftrain_X['Missing ViewType']=(np.isfinite(dftrain_X['ViewType'])==False)
dftrain_X['Missing ViewType']= dftrain_X['Missing ViewType'].astype(int)
dftrain_X['Missing GarageSquareFeet']=(np.isfinite(dftrain_X['GarageSquareFeet'])==False)
dftrain_X['Missing GarageSquareFeet']= dftrain_X['Missing GarageSquareFeet'].astype(int)
dftrain_X['Missing BGMedYearBuilt']=(np.isfinite(dftrain_X['BGMedYearBuilt'])==False)
dftrain_X['Missing BGMedYearBuilt']= dftrain_X['Missing BGMedYearBuilt'].astype(int)
dftrain_X['Missing BGMedRent']=(np.isfinite(dftrain_X['BGMedRent'])==False)
dftrain_X['Missing BGMedRent']= dftrain_X['Missing BGMedRent'].astype(int)
dftrain_X['Missing BGMedHomeValue']=(np.isfinite(dftrain_X['BGMedHomeValue'])==False)
dftrain_X['Missing BGMedHomeValue']= dftrain_X['Missing BGMedHomeValue'].astype(int)
print("Shape of train is : ",dftrain_X.shape)
dftrain_X.head()

```

    Shape of train is :  (11588, 167)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PropertyID</th>
      <th>BedroomCnt</th>
      <th>BathroomCnt</th>
      <th>FinishedSquareFeet</th>
      <th>GarageSquareFeet</th>
      <th>LotSizeSquareFeet</th>
      <th>StoryCnt</th>
      <th>BuiltYear</th>
      <th>ViewType</th>
      <th>Latitude</th>
      <th>...</th>
      <th>ZoneCodeCounty_RSA 1</th>
      <th>ZoneCodeCounty_RSLTC</th>
      <th>ZoneCodeCounty_TL 10A</th>
      <th>ZoneCodeCounty_UHUCR</th>
      <th>ZoneCodeCounty_US R1</th>
      <th>Missing ViewType</th>
      <th>Missing GarageSquareFeet</th>
      <th>Missing BGMedYearBuilt</th>
      <th>Missing BGMedRent</th>
      <th>Missing BGMedHomeValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48648941</td>
      <td>4.0</td>
      <td>2.00</td>
      <td>1900.0</td>
      <td>480.0</td>
      <td>7482</td>
      <td>1.0</td>
      <td>1965.0</td>
      <td>NaN</td>
      <td>47321389</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48648982</td>
      <td>3.0</td>
      <td>2.00</td>
      <td>2170.0</td>
      <td>320.0</td>
      <td>14208</td>
      <td>1.0</td>
      <td>1953.0</td>
      <td>79.0</td>
      <td>47482082</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48649024</td>
      <td>4.0</td>
      <td>1.00</td>
      <td>2150.0</td>
      <td>590.0</td>
      <td>6500</td>
      <td>1.0</td>
      <td>1955.0</td>
      <td>NaN</td>
      <td>47561383</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48649040</td>
      <td>4.0</td>
      <td>2.25</td>
      <td>2560.0</td>
      <td>NaN</td>
      <td>15767</td>
      <td>1.0</td>
      <td>1962.0</td>
      <td>79.0</td>
      <td>47387929</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>48649057</td>
      <td>4.0</td>
      <td>1.75</td>
      <td>1720.0</td>
      <td>NaN</td>
      <td>8620</td>
      <td>2.0</td>
      <td>1948.0</td>
      <td>78.0</td>
      <td>47477068</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 167 columns</p>
</div>




```python
dftest_X['Missing ViewType']=(np.isfinite(dftest_X['ViewType'])==False)
dftest_X['Missing ViewType']= dftest_X['Missing ViewType'].astype(int)
dftest_X['Missing GarageSquareFeet']=(np.isfinite(dftest_X['GarageSquareFeet'])==False)
dftest_X['Missing GarageSquareFeet']= dftest_X['Missing GarageSquareFeet'].astype(int)
dftest_X['Missing BGMedYearBuilt']=(np.isfinite(dftest_X['BGMedYearBuilt'])==False)
dftest_X['Missing BGMedYearBuilt']= dftest_X['Missing BGMedYearBuilt'].astype(int)
dftest_X['Missing BGMedRent']=(np.isfinite(dftest_X['BGMedRent'])==False)
dftest_X['Missing BGMedRent']= dftest_X['Missing BGMedRent'].astype(int)
dftest_X['Missing BGMedHomeValue']=(np.isfinite(dftest_X['BGMedHomeValue'])==False)
dftest_X['Missing BGMedHomeValue']= dftest_X['Missing BGMedHomeValue'].astype(int)
print("Shape of test is : ",dftest_X.shape)
dftest_X.head()
```

    Shape of test is :  (4402, 167)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PropertyID</th>
      <th>BedroomCnt</th>
      <th>BathroomCnt</th>
      <th>FinishedSquareFeet</th>
      <th>GarageSquareFeet</th>
      <th>LotSizeSquareFeet</th>
      <th>StoryCnt</th>
      <th>BuiltYear</th>
      <th>ViewType</th>
      <th>Latitude</th>
      <th>...</th>
      <th>ZoneCodeCounty_UR</th>
      <th>ZoneCodeCounty_URPSO</th>
      <th>ZoneCodeCounty_US R1</th>
      <th>ZoneCodeCounty_UV</th>
      <th>ZoneCodeCounty_UVEV</th>
      <th>Missing ViewType</th>
      <th>Missing GarageSquareFeet</th>
      <th>Missing BGMedYearBuilt</th>
      <th>Missing BGMedRent</th>
      <th>Missing BGMedHomeValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48735321</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>5540</td>
      <td>NaN</td>
      <td>25338</td>
      <td>1.0</td>
      <td>1940.0</td>
      <td>78.0</td>
      <td>47725642</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48735471</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>2470</td>
      <td>510.0</td>
      <td>26006</td>
      <td>1.0</td>
      <td>1966.0</td>
      <td>78.0</td>
      <td>47726993</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49128764</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1680</td>
      <td>NaN</td>
      <td>8743</td>
      <td>2.0</td>
      <td>1928.0</td>
      <td>NaN</td>
      <td>47731749</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48897535</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>990</td>
      <td>260.0</td>
      <td>12219</td>
      <td>1.0</td>
      <td>1940.0</td>
      <td>NaN</td>
      <td>47728810</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49083957</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>2960</td>
      <td>550.0</td>
      <td>23568</td>
      <td>1.0</td>
      <td>1951.0</td>
      <td>82.0</td>
      <td>47731170</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 167 columns</p>
</div>




```python
dftrain_X['BedroomCnt'].value_counts()
```




    3.000000    5107
    4.000000    4177
    2.000000    1102
    5.000000     959
    6.000000     114
    1.000000     103
    7.000000      15
    9.000000       5
    8.000000       3
    3.615385       1
    3.384615       1
    3.461538       1
    Name: BedroomCnt, dtype: int64




```python
dftest_X['BedroomCnt'].value_counts()
```




    3.000000     1942
    4.000000     1570
    2.000000      424
    5.000000      363
    6.000000       49
    1.000000       39
    8.000000        5
    7.000000        4
    3.076923        2
    12.000000       2
    3.538462        1
    3.307692        1
    Name: BedroomCnt, dtype: int64




```python
dftrain_X = dftrain_X.reindex(sorted(dftrain_X.columns), axis=1)
```


```python
dftest_X = dftest_X.reindex(sorted(dftest_X.columns), axis=1)
```


```python
dftrain_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BGMedAge</th>
      <th>BGMedHomeValue</th>
      <th>BGMedIncome</th>
      <th>BGMedRent</th>
      <th>BGMedYearBuilt</th>
      <th>BGPctKids</th>
      <th>BGPctOwn</th>
      <th>BGPctVacant</th>
      <th>BathroomCnt</th>
      <th>BedroomCnt</th>
      <th>...</th>
      <th>ZoneCodeCounty_T</th>
      <th>ZoneCodeCounty_TC</th>
      <th>ZoneCodeCounty_TL 10A</th>
      <th>ZoneCodeCounty_UHUCR</th>
      <th>ZoneCodeCounty_UL7200</th>
      <th>ZoneCodeCounty_UR</th>
      <th>ZoneCodeCounty_URPSO</th>
      <th>ZoneCodeCounty_US R1</th>
      <th>ZoneCodeCounty_UV</th>
      <th>ZoneCodeCounty_UVEV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48.6</td>
      <td>107800.0</td>
      <td>42854</td>
      <td>844.0</td>
      <td>1975.0</td>
      <td>0.1924</td>
      <td>0.6685</td>
      <td>0.0780</td>
      <td>2.00</td>
      <td>4.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42.6</td>
      <td>181500.0</td>
      <td>54013</td>
      <td>925.0</td>
      <td>1969.0</td>
      <td>0.3718</td>
      <td>0.5753</td>
      <td>0.0192</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40.7</td>
      <td>344300.0</td>
      <td>56782</td>
      <td>733.0</td>
      <td>1946.0</td>
      <td>0.3207</td>
      <td>0.6331</td>
      <td>0.0000</td>
      <td>1.00</td>
      <td>4.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40.0</td>
      <td>284200.0</td>
      <td>44200</td>
      <td>900.0</td>
      <td>1977.0</td>
      <td>0.3359</td>
      <td>0.5456</td>
      <td>0.0573</td>
      <td>2.25</td>
      <td>4.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>44.4</td>
      <td>290100.0</td>
      <td>65282</td>
      <td>802.0</td>
      <td>1972.0</td>
      <td>0.1633</td>
      <td>0.4267</td>
      <td>0.0551</td>
      <td>1.75</td>
      <td>4.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 167 columns</p>
</div>




```python
dftest_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BGMedAge</th>
      <th>BGMedHomeValue</th>
      <th>BGMedIncome</th>
      <th>BGMedRent</th>
      <th>BGMedYearBuilt</th>
      <th>BGPctKids</th>
      <th>BGPctOwn</th>
      <th>BGPctVacant</th>
      <th>BathroomCnt</th>
      <th>BedroomCnt</th>
      <th>...</th>
      <th>ZoneCodeCounty_T</th>
      <th>ZoneCodeCounty_TC</th>
      <th>ZoneCodeCounty_TL 10A</th>
      <th>ZoneCodeCounty_UHUCR</th>
      <th>ZoneCodeCounty_UL7200</th>
      <th>ZoneCodeCounty_UR</th>
      <th>ZoneCodeCounty_URPSO</th>
      <th>ZoneCodeCounty_US R1</th>
      <th>ZoneCodeCounty_UV</th>
      <th>ZoneCodeCounty_UVEV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49.6</td>
      <td>527700.0</td>
      <td>113450</td>
      <td>1750.0</td>
      <td>1956.0</td>
      <td>0.2524</td>
      <td>0.9134</td>
      <td>0.1061</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49.6</td>
      <td>527700.0</td>
      <td>113450</td>
      <td>1750.0</td>
      <td>1956.0</td>
      <td>0.2524</td>
      <td>0.9134</td>
      <td>0.1061</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49.6</td>
      <td>527700.0</td>
      <td>113450</td>
      <td>1750.0</td>
      <td>1956.0</td>
      <td>0.2524</td>
      <td>0.9134</td>
      <td>0.1061</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>49.6</td>
      <td>527700.0</td>
      <td>113450</td>
      <td>1750.0</td>
      <td>1956.0</td>
      <td>0.2524</td>
      <td>0.9134</td>
      <td>0.1061</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49.6</td>
      <td>527700.0</td>
      <td>113450</td>
      <td>1750.0</td>
      <td>1956.0</td>
      <td>0.2524</td>
      <td>0.9134</td>
      <td>0.1061</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 167 columns</p>
</div>




```python
submissiondf=pd.DataFrame(data=dftest_X['PropertyID'])
submissiondf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PropertyID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48735321</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48735471</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49128764</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48897535</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49083957</td>
    </tr>
  </tbody>
</table>
</div>




```python
submissiondf.shape
```




    (4402, 1)




```python
dftrain_X=pd.read_csv("dftrain_X_imputed.csv")
dftest_X = pd.read_csv("dftest_X_imputed.csv")
    
```


```python
dftrain_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>BGMedAge</th>
      <th>BGMedHomeValue</th>
      <th>BGMedIncome</th>
      <th>BGMedRent</th>
      <th>BGMedYearBuilt</th>
      <th>BGPctKids</th>
      <th>BGPctOwn</th>
      <th>BGPctVacant</th>
      <th>BathroomCnt</th>
      <th>...</th>
      <th>ZoneCodeCounty_T</th>
      <th>ZoneCodeCounty_TC</th>
      <th>ZoneCodeCounty_TL 10A</th>
      <th>ZoneCodeCounty_UHUCR</th>
      <th>ZoneCodeCounty_UL7200</th>
      <th>ZoneCodeCounty_UR</th>
      <th>ZoneCodeCounty_URPSO</th>
      <th>ZoneCodeCounty_US R1</th>
      <th>ZoneCodeCounty_UV</th>
      <th>ZoneCodeCounty_UVEV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.312359</td>
      <td>-1.829768</td>
      <td>-1.433279</td>
      <td>-0.992881</td>
      <td>0.092411</td>
      <td>-1.193404</td>
      <td>-0.403853</td>
      <td>0.462342</td>
      <td>-0.375478</td>
      <td>...</td>
      <td>-0.00929</td>
      <td>-0.016092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.085451</td>
      <td>-0.062438</td>
      <td>-0.087976</td>
      <td>0.0</td>
      <td>-0.077958</td>
      <td>-0.058854</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.420317</td>
      <td>-1.416003</td>
      <td>-1.125734</td>
      <td>-0.787479</td>
      <td>-0.244938</td>
      <td>0.083579</td>
      <td>-0.878712</td>
      <td>-0.539812</td>
      <td>-0.375478</td>
      <td>...</td>
      <td>-0.00929</td>
      <td>-0.016092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.085451</td>
      <td>-0.062438</td>
      <td>-0.087976</td>
      <td>0.0</td>
      <td>-0.077958</td>
      <td>-0.058854</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.137837</td>
      <td>-0.502016</td>
      <td>-1.049419</td>
      <td>-1.274357</td>
      <td>-1.538110</td>
      <td>-0.280155</td>
      <td>-0.584218</td>
      <td>-0.867046</td>
      <td>-1.521526</td>
      <td>...</td>
      <td>-0.00929</td>
      <td>-0.016092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.085451</td>
      <td>-0.062438</td>
      <td>-0.087976</td>
      <td>0.0</td>
      <td>-0.077958</td>
      <td>-0.058854</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.033766</td>
      <td>-0.839428</td>
      <td>-1.396183</td>
      <td>-0.850874</td>
      <td>0.204861</td>
      <td>-0.171960</td>
      <td>-1.030035</td>
      <td>0.109543</td>
      <td>-0.088966</td>
      <td>...</td>
      <td>-0.00929</td>
      <td>-0.016092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.085451</td>
      <td>-0.062438</td>
      <td>-0.087976</td>
      <td>0.0</td>
      <td>-0.077958</td>
      <td>-0.058854</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.687930</td>
      <td>-0.806304</td>
      <td>-0.815157</td>
      <td>-1.099385</td>
      <td>-0.076264</td>
      <td>-1.400540</td>
      <td>-1.635837</td>
      <td>0.072048</td>
      <td>-0.661990</td>
      <td>...</td>
      <td>-0.00929</td>
      <td>-0.016092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.085451</td>
      <td>-0.062438</td>
      <td>-0.087976</td>
      <td>0.0</td>
      <td>-0.077958</td>
      <td>-0.058854</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 168 columns</p>
</div>




```python
dftest_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>BGMedAge</th>
      <th>BGMedHomeValue</th>
      <th>BGMedIncome</th>
      <th>BGMedRent</th>
      <th>BGMedYearBuilt</th>
      <th>BGPctKids</th>
      <th>BGPctOwn</th>
      <th>BGPctVacant</th>
      <th>BathroomCnt</th>
      <th>...</th>
      <th>ZoneCodeCounty_T</th>
      <th>ZoneCodeCounty_TC</th>
      <th>ZoneCodeCounty_TL 10A</th>
      <th>ZoneCodeCounty_UHUCR</th>
      <th>ZoneCodeCounty_UL7200</th>
      <th>ZoneCodeCounty_UR</th>
      <th>ZoneCodeCounty_URPSO</th>
      <th>ZoneCodeCounty_US R1</th>
      <th>ZoneCodeCounty_UV</th>
      <th>ZoneCodeCounty_UVEV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.442128</td>
      <td>0.568365</td>
      <td>0.557939</td>
      <td>1.356873</td>
      <td>-0.940332</td>
      <td>-0.739685</td>
      <td>0.861138</td>
      <td>0.908763</td>
      <td>1.586067</td>
      <td>...</td>
      <td>-0.015074</td>
      <td>-0.015074</td>
      <td>-0.015074</td>
      <td>-0.015074</td>
      <td>-0.096961</td>
      <td>-0.064077</td>
      <td>-0.082836</td>
      <td>-0.02132</td>
      <td>-0.058474</td>
      <td>-0.039909</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1.442128</td>
      <td>0.568365</td>
      <td>0.557939</td>
      <td>1.356873</td>
      <td>-0.940332</td>
      <td>-0.739685</td>
      <td>0.861138</td>
      <td>0.908763</td>
      <td>0.564978</td>
      <td>...</td>
      <td>-0.015074</td>
      <td>-0.015074</td>
      <td>-0.015074</td>
      <td>-0.015074</td>
      <td>-0.096961</td>
      <td>-0.064077</td>
      <td>-0.082836</td>
      <td>-0.02132</td>
      <td>-0.058474</td>
      <td>-0.039909</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1.442128</td>
      <td>0.568365</td>
      <td>0.557939</td>
      <td>1.356873</td>
      <td>-0.940332</td>
      <td>-0.739685</td>
      <td>0.861138</td>
      <td>0.908763</td>
      <td>-0.456111</td>
      <td>...</td>
      <td>-0.015074</td>
      <td>-0.015074</td>
      <td>-0.015074</td>
      <td>-0.015074</td>
      <td>-0.096961</td>
      <td>-0.064077</td>
      <td>-0.082836</td>
      <td>-0.02132</td>
      <td>-0.058474</td>
      <td>-0.039909</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1.442128</td>
      <td>0.568365</td>
      <td>0.557939</td>
      <td>1.356873</td>
      <td>-0.940332</td>
      <td>-0.739685</td>
      <td>0.861138</td>
      <td>0.908763</td>
      <td>-1.477200</td>
      <td>...</td>
      <td>-0.015074</td>
      <td>-0.015074</td>
      <td>-0.015074</td>
      <td>-0.015074</td>
      <td>-0.096961</td>
      <td>-0.064077</td>
      <td>-0.082836</td>
      <td>-0.02132</td>
      <td>-0.058474</td>
      <td>-0.039909</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1.442128</td>
      <td>0.568365</td>
      <td>0.557939</td>
      <td>1.356873</td>
      <td>-0.940332</td>
      <td>-0.739685</td>
      <td>0.861138</td>
      <td>0.908763</td>
      <td>0.564978</td>
      <td>...</td>
      <td>-0.015074</td>
      <td>-0.015074</td>
      <td>-0.015074</td>
      <td>-0.015074</td>
      <td>-0.096961</td>
      <td>-0.064077</td>
      <td>-0.082836</td>
      <td>-0.02132</td>
      <td>-0.058474</td>
      <td>-0.039909</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 168 columns</p>
</div>




```python
dftrain_X.drop(["Unnamed: 0"],axis=1,inplace=True)
```


```python

```


```python
dftest_X.drop(["Unnamed: 0"],axis=1,inplace=True)
```


```python
dftrain_X = dftrain_X.reindex(sorted(dftrain_X.columns), axis=1)
dftest_X = dftest_X.reindex(sorted(dftest_X.columns), axis=1)
```


```python
dftrain_X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BGMedAge</th>
      <th>BGMedHomeValue</th>
      <th>BGMedIncome</th>
      <th>BGMedRent</th>
      <th>BGMedYearBuilt</th>
      <th>BGPctKids</th>
      <th>BGPctOwn</th>
      <th>BGPctVacant</th>
      <th>BathroomCnt</th>
      <th>BedroomCnt</th>
      <th>...</th>
      <th>ZoneCodeCounty_T</th>
      <th>ZoneCodeCounty_TC</th>
      <th>ZoneCodeCounty_TL 10A</th>
      <th>ZoneCodeCounty_UHUCR</th>
      <th>ZoneCodeCounty_UL7200</th>
      <th>ZoneCodeCounty_UR</th>
      <th>ZoneCodeCounty_URPSO</th>
      <th>ZoneCodeCounty_US R1</th>
      <th>ZoneCodeCounty_UV</th>
      <th>ZoneCodeCounty_UVEV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.312359</td>
      <td>-1.829768</td>
      <td>-1.433279</td>
      <td>-0.992881</td>
      <td>0.092411</td>
      <td>-1.193404</td>
      <td>-0.403853</td>
      <td>0.462342</td>
      <td>-0.375478</td>
      <td>0.633285</td>
      <td>...</td>
      <td>-0.00929</td>
      <td>-0.016092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.085451</td>
      <td>-0.062438</td>
      <td>-0.087976</td>
      <td>0.0</td>
      <td>-0.077958</td>
      <td>-0.058854</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.420317</td>
      <td>-1.416003</td>
      <td>-1.125734</td>
      <td>-0.787479</td>
      <td>-0.244938</td>
      <td>0.083579</td>
      <td>-0.878712</td>
      <td>-0.539812</td>
      <td>-0.375478</td>
      <td>-0.521924</td>
      <td>...</td>
      <td>-0.00929</td>
      <td>-0.016092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.085451</td>
      <td>-0.062438</td>
      <td>-0.087976</td>
      <td>0.0</td>
      <td>-0.077958</td>
      <td>-0.058854</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.137837</td>
      <td>-0.502016</td>
      <td>-1.049419</td>
      <td>-1.274357</td>
      <td>-1.538110</td>
      <td>-0.280155</td>
      <td>-0.584218</td>
      <td>-0.867046</td>
      <td>-1.521526</td>
      <td>0.633285</td>
      <td>...</td>
      <td>-0.00929</td>
      <td>-0.016092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.085451</td>
      <td>-0.062438</td>
      <td>-0.087976</td>
      <td>0.0</td>
      <td>-0.077958</td>
      <td>-0.058854</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.033766</td>
      <td>-0.839428</td>
      <td>-1.396183</td>
      <td>-0.850874</td>
      <td>0.204861</td>
      <td>-0.171960</td>
      <td>-1.030035</td>
      <td>0.109543</td>
      <td>-0.088966</td>
      <td>0.633285</td>
      <td>...</td>
      <td>-0.00929</td>
      <td>-0.016092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.085451</td>
      <td>-0.062438</td>
      <td>-0.087976</td>
      <td>0.0</td>
      <td>-0.077958</td>
      <td>-0.058854</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.687930</td>
      <td>-0.806304</td>
      <td>-0.815157</td>
      <td>-1.099385</td>
      <td>-0.076264</td>
      <td>-1.400540</td>
      <td>-1.635837</td>
      <td>0.072048</td>
      <td>-0.661990</td>
      <td>0.633285</td>
      <td>...</td>
      <td>-0.00929</td>
      <td>-0.016092</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.085451</td>
      <td>-0.062438</td>
      <td>-0.087976</td>
      <td>0.0</td>
      <td>-0.077958</td>
      <td>-0.058854</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 167 columns</p>
</div>




```python
result = pd.concat([dftrain_y, dftrain_X], axis=1, sort=False)
result.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SaleDollarCnt</th>
      <th>BGMedAge</th>
      <th>BGMedHomeValue</th>
      <th>BGMedIncome</th>
      <th>BGMedRent</th>
      <th>BGMedYearBuilt</th>
      <th>BGPctKids</th>
      <th>BGPctOwn</th>
      <th>BGPctVacant</th>
      <th>BathroomCnt</th>
      <th>...</th>
      <th>ZoneCodeCounty_T</th>
      <th>ZoneCodeCounty_TC</th>
      <th>ZoneCodeCounty_TL 10A</th>
      <th>ZoneCodeCounty_UHUCR</th>
      <th>ZoneCodeCounty_UL7200</th>
      <th>ZoneCodeCounty_UR</th>
      <th>ZoneCodeCounty_URPSO</th>
      <th>ZoneCodeCounty_US R1</th>
      <th>ZoneCodeCounty_UV</th>
      <th>ZoneCodeCounty_UVEV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SaleDollarCnt</th>
      <td>1.000000</td>
      <td>0.173956</td>
      <td>0.681915</td>
      <td>0.427578</td>
      <td>0.285612</td>
      <td>-0.116480</td>
      <td>-0.028768</td>
      <td>0.094028</td>
      <td>0.010543</td>
      <td>0.506672</td>
      <td>...</td>
      <td>-0.006874</td>
      <td>-0.008089</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.060764</td>
      <td>-0.038110</td>
      <td>0.011451</td>
      <td>NaN</td>
      <td>0.019745</td>
      <td>0.022135</td>
    </tr>
    <tr>
      <th>BGMedAge</th>
      <td>0.173956</td>
      <td>1.000000</td>
      <td>0.262665</td>
      <td>0.159478</td>
      <td>0.104708</td>
      <td>-0.153470</td>
      <td>-0.473199</td>
      <td>0.357654</td>
      <td>0.039717</td>
      <td>0.060423</td>
      <td>...</td>
      <td>-0.000239</td>
      <td>0.012506</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.016201</td>
      <td>0.003449</td>
      <td>0.124730</td>
      <td>NaN</td>
      <td>-0.076844</td>
      <td>-0.046138</td>
    </tr>
    <tr>
      <th>BGMedHomeValue</th>
      <td>0.681915</td>
      <td>0.262665</td>
      <td>1.000000</td>
      <td>0.684423</td>
      <td>0.458181</td>
      <td>-0.121641</td>
      <td>0.052913</td>
      <td>0.254099</td>
      <td>-0.046364</td>
      <td>0.316697</td>
      <td>...</td>
      <td>-0.007518</td>
      <td>-0.014412</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.095693</td>
      <td>-0.035577</td>
      <td>0.032169</td>
      <td>NaN</td>
      <td>0.035798</td>
      <td>0.057218</td>
    </tr>
    <tr>
      <th>BGMedIncome</th>
      <td>0.427578</td>
      <td>0.159478</td>
      <td>0.684423</td>
      <td>1.000000</td>
      <td>0.621630</td>
      <td>0.193459</td>
      <td>0.367387</td>
      <td>0.602389</td>
      <td>-0.096249</td>
      <td>0.352881</td>
      <td>...</td>
      <td>-0.011348</td>
      <td>-0.019103</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.097016</td>
      <td>-0.000763</td>
      <td>0.019685</td>
      <td>NaN</td>
      <td>0.062884</td>
      <td>0.052581</td>
    </tr>
    <tr>
      <th>BGMedRent</th>
      <td>0.285612</td>
      <td>0.104708</td>
      <td>0.458181</td>
      <td>0.621630</td>
      <td>1.000000</td>
      <td>0.233537</td>
      <td>0.273633</td>
      <td>0.459024</td>
      <td>-0.094842</td>
      <td>0.246985</td>
      <td>...</td>
      <td>-0.011522</td>
      <td>-0.017006</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.054025</td>
      <td>0.023170</td>
      <td>0.119470</td>
      <td>NaN</td>
      <td>0.085470</td>
      <td>0.025389</td>
    </tr>
    <tr>
      <th>BGMedYearBuilt</th>
      <td>-0.116480</td>
      <td>-0.153470</td>
      <td>-0.121641</td>
      <td>0.193459</td>
      <td>0.233537</td>
      <td>1.000000</td>
      <td>0.446102</td>
      <td>0.155998</td>
      <td>-0.020721</td>
      <td>0.290310</td>
      <td>...</td>
      <td>-0.004573</td>
      <td>0.006381</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.048170</td>
      <td>0.017669</td>
      <td>0.146308</td>
      <td>NaN</td>
      <td>0.130523</td>
      <td>0.017966</td>
    </tr>
    <tr>
      <th>BGPctKids</th>
      <td>-0.028768</td>
      <td>-0.473199</td>
      <td>0.052913</td>
      <td>0.367387</td>
      <td>0.273633</td>
      <td>0.446102</td>
      <td>1.000000</td>
      <td>0.288577</td>
      <td>-0.081650</td>
      <td>0.181858</td>
      <td>...</td>
      <td>0.000016</td>
      <td>-0.008132</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.038564</td>
      <td>0.006530</td>
      <td>-0.020327</td>
      <td>NaN</td>
      <td>0.088087</td>
      <td>-0.013765</td>
    </tr>
    <tr>
      <th>BGPctOwn</th>
      <td>0.094028</td>
      <td>0.357654</td>
      <td>0.254099</td>
      <td>0.602389</td>
      <td>0.459024</td>
      <td>0.155998</td>
      <td>0.288577</td>
      <td>1.000000</td>
      <td>-0.103676</td>
      <td>0.201332</td>
      <td>...</td>
      <td>-0.019618</td>
      <td>-0.001006</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.042348</td>
      <td>0.037383</td>
      <td>0.042641</td>
      <td>NaN</td>
      <td>-0.003606</td>
      <td>-0.043589</td>
    </tr>
    <tr>
      <th>BGPctVacant</th>
      <td>0.010543</td>
      <td>0.039717</td>
      <td>-0.046364</td>
      <td>-0.096249</td>
      <td>-0.094842</td>
      <td>-0.020721</td>
      <td>-0.081650</td>
      <td>-0.103676</td>
      <td>1.000000</td>
      <td>-0.012473</td>
      <td>...</td>
      <td>0.002110</td>
      <td>0.002101</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000830</td>
      <td>0.029866</td>
      <td>0.013982</td>
      <td>NaN</td>
      <td>0.035740</td>
      <td>-0.038190</td>
    </tr>
    <tr>
      <th>BathroomCnt</th>
      <td>0.506672</td>
      <td>0.060423</td>
      <td>0.316697</td>
      <td>0.352881</td>
      <td>0.246985</td>
      <td>0.290310</td>
      <td>0.181858</td>
      <td>0.201332</td>
      <td>-0.012473</td>
      <td>1.000000</td>
      <td>...</td>
      <td>-0.003488</td>
      <td>-0.009116</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.057733</td>
      <td>-0.038550</td>
      <td>0.026726</td>
      <td>NaN</td>
      <td>0.043161</td>
      <td>0.063900</td>
    </tr>
    <tr>
      <th>BedroomCnt</th>
      <td>0.310897</td>
      <td>0.022350</td>
      <td>0.185010</td>
      <td>0.203819</td>
      <td>0.160628</td>
      <td>0.142163</td>
      <td>0.122363</td>
      <td>0.126537</td>
      <td>-0.037825</td>
      <td>0.547931</td>
      <td>...</td>
      <td>0.005883</td>
      <td>0.016387</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.025796</td>
      <td>-0.032588</td>
      <td>-0.061904</td>
      <td>NaN</td>
      <td>0.005627</td>
      <td>0.027073</td>
    </tr>
    <tr>
      <th>BuiltYear</th>
      <td>0.139941</td>
      <td>-0.053678</td>
      <td>-0.002238</td>
      <td>0.213867</td>
      <td>0.209572</td>
      <td>0.563977</td>
      <td>0.320432</td>
      <td>0.234920</td>
      <td>-0.010501</td>
      <td>0.562734</td>
      <td>...</td>
      <td>-0.003452</td>
      <td>-0.006170</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.039002</td>
      <td>-0.004107</td>
      <td>0.095293</td>
      <td>NaN</td>
      <td>0.083544</td>
      <td>0.069290</td>
    </tr>
    <tr>
      <th>FinishedSquareFeet</th>
      <td>0.678446</td>
      <td>0.126701</td>
      <td>0.451460</td>
      <td>0.421043</td>
      <td>0.286116</td>
      <td>0.210854</td>
      <td>0.150903</td>
      <td>0.214079</td>
      <td>-0.015851</td>
      <td>0.770779</td>
      <td>...</td>
      <td>-0.000396</td>
      <td>-0.004808</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.050917</td>
      <td>-0.043065</td>
      <td>0.011758</td>
      <td>NaN</td>
      <td>0.024925</td>
      <td>0.043110</td>
    </tr>
    <tr>
      <th>GarageSquareFeet</th>
      <td>0.282781</td>
      <td>0.102894</td>
      <td>0.177867</td>
      <td>0.287018</td>
      <td>0.221304</td>
      <td>0.384254</td>
      <td>0.205274</td>
      <td>0.256248</td>
      <td>0.007768</td>
      <td>0.437512</td>
      <td>...</td>
      <td>0.000037</td>
      <td>-0.001112</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.037866</td>
      <td>0.032329</td>
      <td>0.010505</td>
      <td>NaN</td>
      <td>0.007687</td>
      <td>0.013301</td>
    </tr>
    <tr>
      <th>Latitude</th>
      <td>0.317772</td>
      <td>0.093957</td>
      <td>0.438911</td>
      <td>0.239937</td>
      <td>0.178521</td>
      <td>-0.251774</td>
      <td>-0.147922</td>
      <td>-0.041684</td>
      <td>-0.052146</td>
      <td>0.023533</td>
      <td>...</td>
      <td>-0.005493</td>
      <td>-0.010283</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.061017</td>
      <td>-0.036703</td>
      <td>0.091889</td>
      <td>NaN</td>
      <td>-0.001870</td>
      <td>-0.006866</td>
    </tr>
    <tr>
      <th>Longitude</th>
      <td>-0.020657</td>
      <td>-0.080426</td>
      <td>0.042980</td>
      <td>0.337984</td>
      <td>0.270668</td>
      <td>0.626320</td>
      <td>0.422751</td>
      <td>0.307125</td>
      <td>0.012217</td>
      <td>0.232815</td>
      <td>...</td>
      <td>-0.006157</td>
      <td>0.009355</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.054061</td>
      <td>0.169401</td>
      <td>0.107156</td>
      <td>NaN</td>
      <td>0.109749</td>
      <td>0.052847</td>
    </tr>
    <tr>
      <th>LotSizeSquareFeet</th>
      <td>0.067874</td>
      <td>0.128735</td>
      <td>0.042843</td>
      <td>0.054269</td>
      <td>-0.036392</td>
      <td>0.110082</td>
      <td>0.003284</td>
      <td>0.125225</td>
      <td>0.013974</td>
      <td>0.107850</td>
      <td>...</td>
      <td>-0.001682</td>
      <td>-0.001983</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.012561</td>
      <td>0.018047</td>
      <td>-0.020723</td>
      <td>NaN</td>
      <td>-0.020910</td>
      <td>-0.015896</td>
    </tr>
    <tr>
      <th>Missing BGMedHomeValue</th>
      <td>-0.009463</td>
      <td>-0.049708</td>
      <td>-0.019036</td>
      <td>-0.023734</td>
      <td>-0.015104</td>
      <td>0.017034</td>
      <td>0.006536</td>
      <td>-0.085022</td>
      <td>0.083497</td>
      <td>-0.004199</td>
      <td>...</td>
      <td>-0.000211</td>
      <td>-0.000366</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.001945</td>
      <td>-0.001421</td>
      <td>-0.002002</td>
      <td>NaN</td>
      <td>-0.001774</td>
      <td>-0.001340</td>
    </tr>
    <tr>
      <th>Missing BGMedRent</th>
      <td>0.050122</td>
      <td>0.221339</td>
      <td>0.162539</td>
      <td>0.370692</td>
      <td>0.212484</td>
      <td>0.150722</td>
      <td>0.159108</td>
      <td>0.524399</td>
      <td>-0.021814</td>
      <td>0.138428</td>
      <td>...</td>
      <td>-0.005035</td>
      <td>-0.008722</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.041455</td>
      <td>0.022466</td>
      <td>-0.047681</td>
      <td>NaN</td>
      <td>-0.042251</td>
      <td>-0.031897</td>
    </tr>
    <tr>
      <th>Missing BGMedYearBuilt</th>
      <td>0.015801</td>
      <td>-0.104180</td>
      <td>0.064138</td>
      <td>0.160738</td>
      <td>0.165488</td>
      <td>0.190160</td>
      <td>0.179951</td>
      <td>0.022063</td>
      <td>-0.010892</td>
      <td>0.101908</td>
      <td>...</td>
      <td>-0.001371</td>
      <td>-0.002375</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.012611</td>
      <td>-0.009214</td>
      <td>-0.012983</td>
      <td>NaN</td>
      <td>0.374033</td>
      <td>0.398799</td>
    </tr>
    <tr>
      <th>Missing GarageSquareFeet</th>
      <td>-0.081234</td>
      <td>0.020496</td>
      <td>-0.025990</td>
      <td>-0.189072</td>
      <td>-0.202730</td>
      <td>-0.367624</td>
      <td>-0.229116</td>
      <td>-0.208572</td>
      <td>0.045624</td>
      <td>-0.373407</td>
      <td>...</td>
      <td>-0.005294</td>
      <td>0.028236</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.036431</td>
      <td>0.006346</td>
      <td>-0.031756</td>
      <td>NaN</td>
      <td>-0.028896</td>
      <td>-0.030121</td>
    </tr>
    <tr>
      <th>Missing ViewType</th>
      <td>-0.265936</td>
      <td>-0.178623</td>
      <td>-0.218149</td>
      <td>-0.089974</td>
      <td>-0.017195</td>
      <td>0.147170</td>
      <td>0.135395</td>
      <td>-0.009566</td>
      <td>-0.045902</td>
      <td>-0.109534</td>
      <td>...</td>
      <td>0.005036</td>
      <td>-0.004079</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.019616</td>
      <td>-0.035696</td>
      <td>0.028818</td>
      <td>NaN</td>
      <td>0.015681</td>
      <td>-0.006724</td>
    </tr>
    <tr>
      <th>PropertyID</th>
      <td>0.024807</td>
      <td>-0.121188</td>
      <td>-0.070863</td>
      <td>0.023011</td>
      <td>0.019197</td>
      <td>0.235369</td>
      <td>0.139690</td>
      <td>0.028997</td>
      <td>0.013874</td>
      <td>0.252908</td>
      <td>...</td>
      <td>-0.003599</td>
      <td>-0.006238</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.023284</td>
      <td>-0.016237</td>
      <td>0.097707</td>
      <td>NaN</td>
      <td>0.071418</td>
      <td>0.093619</td>
    </tr>
    <tr>
      <th>StoryCnt</th>
      <td>0.267300</td>
      <td>-0.082155</td>
      <td>0.181544</td>
      <td>0.212164</td>
      <td>0.118088</td>
      <td>0.196404</td>
      <td>0.172151</td>
      <td>0.062497</td>
      <td>0.002787</td>
      <td>0.475484</td>
      <td>...</td>
      <td>-0.009410</td>
      <td>-0.006021</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.049514</td>
      <td>-0.041972</td>
      <td>0.005602</td>
      <td>NaN</td>
      <td>0.078964</td>
      <td>0.053169</td>
    </tr>
    <tr>
      <th>ViewType</th>
      <td>0.030121</td>
      <td>0.005822</td>
      <td>0.065888</td>
      <td>0.173572</td>
      <td>0.164393</td>
      <td>0.321736</td>
      <td>0.195612</td>
      <td>0.159080</td>
      <td>-0.026782</td>
      <td>0.167468</td>
      <td>...</td>
      <td>-0.004819</td>
      <td>-0.008233</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.070200</td>
      <td>-0.005693</td>
      <td>0.276114</td>
      <td>NaN</td>
      <td>-0.041532</td>
      <td>0.019277</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_A10</th>
      <td>-0.012339</td>
      <td>0.036878</td>
      <td>-0.010913</td>
      <td>-0.006664</td>
      <td>0.001983</td>
      <td>0.006870</td>
      <td>-0.017555</td>
      <td>0.012782</td>
      <td>-0.012396</td>
      <td>-0.001976</td>
      <td>...</td>
      <td>-0.000345</td>
      <td>-0.000598</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.003177</td>
      <td>-0.002322</td>
      <td>-0.003271</td>
      <td>NaN</td>
      <td>-0.002899</td>
      <td>-0.002188</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_A35</th>
      <td>-0.015148</td>
      <td>0.051685</td>
      <td>-0.013500</td>
      <td>-0.016842</td>
      <td>-0.013628</td>
      <td>0.007561</td>
      <td>-0.031597</td>
      <td>0.015529</td>
      <td>-0.015389</td>
      <td>-0.015538</td>
      <td>...</td>
      <td>-0.000311</td>
      <td>-0.000539</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.002864</td>
      <td>-0.002092</td>
      <td>-0.002948</td>
      <td>NaN</td>
      <td>-0.002613</td>
      <td>-0.001972</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_C1</th>
      <td>-0.020294</td>
      <td>-0.018238</td>
      <td>-0.028867</td>
      <td>-0.027676</td>
      <td>-0.019935</td>
      <td>0.000386</td>
      <td>0.006271</td>
      <td>-0.007574</td>
      <td>0.021320</td>
      <td>-0.018328</td>
      <td>...</td>
      <td>-0.000211</td>
      <td>-0.000366</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.001945</td>
      <td>-0.001421</td>
      <td>-0.002002</td>
      <td>NaN</td>
      <td>-0.001774</td>
      <td>-0.001340</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_C2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_CR</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_RSA 6</th>
      <td>-0.017218</td>
      <td>-0.031341</td>
      <td>-0.026786</td>
      <td>-0.019406</td>
      <td>0.043847</td>
      <td>0.017085</td>
      <td>0.005510</td>
      <td>0.027547</td>
      <td>-0.052987</td>
      <td>-0.011241</td>
      <td>...</td>
      <td>-0.001286</td>
      <td>-0.002228</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.011832</td>
      <td>-0.008646</td>
      <td>-0.012182</td>
      <td>NaN</td>
      <td>-0.010795</td>
      <td>-0.008149</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_RSA 8</th>
      <td>0.004212</td>
      <td>-0.012531</td>
      <td>-0.003531</td>
      <td>-0.001087</td>
      <td>0.018307</td>
      <td>0.010873</td>
      <td>-0.004209</td>
      <td>-0.000392</td>
      <td>-0.008340</td>
      <td>0.018665</td>
      <td>...</td>
      <td>-0.000345</td>
      <td>-0.000598</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.003177</td>
      <td>-0.002322</td>
      <td>-0.003271</td>
      <td>NaN</td>
      <td>-0.002899</td>
      <td>-0.002188</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_RSLTC</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_RSX 7.2</th>
      <td>0.017586</td>
      <td>-0.048495</td>
      <td>-0.002511</td>
      <td>-0.012116</td>
      <td>0.034779</td>
      <td>0.042292</td>
      <td>-0.016296</td>
      <td>-0.020259</td>
      <td>0.021272</td>
      <td>0.005942</td>
      <td>...</td>
      <td>-0.000954</td>
      <td>-0.001653</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.008778</td>
      <td>-0.006414</td>
      <td>-0.009037</td>
      <td>NaN</td>
      <td>-0.008008</td>
      <td>-0.006046</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_RSX 8.5</th>
      <td>-0.001537</td>
      <td>-0.005211</td>
      <td>-0.011561</td>
      <td>-0.006611</td>
      <td>-0.000894</td>
      <td>0.010393</td>
      <td>-0.010928</td>
      <td>-0.019192</td>
      <td>-0.002387</td>
      <td>-0.000826</td>
      <td>...</td>
      <td>-0.000086</td>
      <td>-0.000149</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.000794</td>
      <td>-0.000580</td>
      <td>-0.000817</td>
      <td>NaN</td>
      <td>-0.000724</td>
      <td>-0.000547</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_SF 5000</th>
      <td>0.093375</td>
      <td>-0.039478</td>
      <td>0.146622</td>
      <td>-0.113508</td>
      <td>-0.195087</td>
      <td>-0.609872</td>
      <td>-0.287926</td>
      <td>-0.247538</td>
      <td>0.003279</td>
      <td>-0.200475</td>
      <td>...</td>
      <td>-0.004551</td>
      <td>-0.007884</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.041864</td>
      <td>-0.030589</td>
      <td>-0.043101</td>
      <td>NaN</td>
      <td>-0.038193</td>
      <td>-0.028834</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_SF 7200</th>
      <td>0.030004</td>
      <td>0.129687</td>
      <td>0.031195</td>
      <td>-0.070525</td>
      <td>-0.055253</td>
      <td>-0.163114</td>
      <td>-0.143069</td>
      <td>-0.044690</td>
      <td>0.028277</td>
      <td>-0.060016</td>
      <td>...</td>
      <td>-0.001899</td>
      <td>-0.003290</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.017472</td>
      <td>-0.012766</td>
      <td>-0.017988</td>
      <td>NaN</td>
      <td>-0.015940</td>
      <td>-0.012034</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_SF 9600</th>
      <td>0.082959</td>
      <td>0.057907</td>
      <td>0.063362</td>
      <td>0.048893</td>
      <td>0.042489</td>
      <td>-0.056168</td>
      <td>-0.025672</td>
      <td>0.033671</td>
      <td>0.021666</td>
      <td>0.029244</td>
      <td>...</td>
      <td>-0.000519</td>
      <td>-0.000898</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.004770</td>
      <td>-0.003486</td>
      <td>-0.004911</td>
      <td>NaN</td>
      <td>-0.004352</td>
      <td>-0.003285</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_SFD</th>
      <td>-0.011479</td>
      <td>0.004991</td>
      <td>-0.013741</td>
      <td>-0.027947</td>
      <td>-0.025582</td>
      <td>0.009328</td>
      <td>-0.009164</td>
      <td>-0.024720</td>
      <td>0.044417</td>
      <td>-0.034344</td>
      <td>...</td>
      <td>-0.000244</td>
      <td>-0.000423</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.002246</td>
      <td>-0.001641</td>
      <td>-0.002312</td>
      <td>NaN</td>
      <td>-0.002049</td>
      <td>-0.001547</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_SFE</th>
      <td>0.005902</td>
      <td>-0.017707</td>
      <td>0.032653</td>
      <td>-0.007318</td>
      <td>-0.001894</td>
      <td>0.032773</td>
      <td>-0.010363</td>
      <td>-0.036651</td>
      <td>0.019310</td>
      <td>0.006869</td>
      <td>...</td>
      <td>-0.000228</td>
      <td>-0.000396</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.002101</td>
      <td>-0.001535</td>
      <td>-0.002163</td>
      <td>NaN</td>
      <td>-0.001917</td>
      <td>-0.001447</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_SFR 10.0</th>
      <td>0.016161</td>
      <td>0.005976</td>
      <td>0.013637</td>
      <td>0.015782</td>
      <td>0.011973</td>
      <td>-0.008702</td>
      <td>0.005240</td>
      <td>0.006756</td>
      <td>0.004263</td>
      <td>0.007159</td>
      <td>...</td>
      <td>-0.000086</td>
      <td>-0.000149</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.000794</td>
      <td>-0.000580</td>
      <td>-0.000817</td>
      <td>NaN</td>
      <td>-0.000724</td>
      <td>-0.000547</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_SFS</th>
      <td>0.016274</td>
      <td>0.050876</td>
      <td>0.036276</td>
      <td>0.022181</td>
      <td>-0.016802</td>
      <td>0.049448</td>
      <td>-0.030857</td>
      <td>-0.006965</td>
      <td>-0.009263</td>
      <td>0.025860</td>
      <td>...</td>
      <td>-0.000636</td>
      <td>-0.001101</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.005847</td>
      <td>-0.004272</td>
      <td>-0.006020</td>
      <td>NaN</td>
      <td>-0.005334</td>
      <td>-0.004027</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_SFSL</th>
      <td>0.023874</td>
      <td>-0.007128</td>
      <td>0.029884</td>
      <td>0.057119</td>
      <td>-0.030391</td>
      <td>0.016604</td>
      <td>0.009651</td>
      <td>0.023631</td>
      <td>0.010427</td>
      <td>0.012413</td>
      <td>...</td>
      <td>-0.000496</td>
      <td>-0.000860</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.004567</td>
      <td>-0.003337</td>
      <td>-0.004702</td>
      <td>NaN</td>
      <td>-0.004166</td>
      <td>-0.003145</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_SR1</th>
      <td>-0.008349</td>
      <td>-0.001657</td>
      <td>-0.012559</td>
      <td>-0.005732</td>
      <td>0.000145</td>
      <td>0.012398</td>
      <td>0.003360</td>
      <td>0.010051</td>
      <td>0.001304</td>
      <td>0.001236</td>
      <td>...</td>
      <td>-0.000211</td>
      <td>-0.000366</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.001945</td>
      <td>-0.001421</td>
      <td>-0.002002</td>
      <td>NaN</td>
      <td>-0.001774</td>
      <td>-0.001340</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_SR3</th>
      <td>-0.020704</td>
      <td>-0.022395</td>
      <td>-0.028158</td>
      <td>-0.031804</td>
      <td>0.011278</td>
      <td>0.068222</td>
      <td>0.029284</td>
      <td>0.050314</td>
      <td>-0.001496</td>
      <td>0.025730</td>
      <td>...</td>
      <td>-0.000465</td>
      <td>-0.000806</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.004280</td>
      <td>-0.003127</td>
      <td>-0.004407</td>
      <td>NaN</td>
      <td>-0.003905</td>
      <td>-0.002948</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_SR30</th>
      <td>0.021701</td>
      <td>0.006253</td>
      <td>0.029541</td>
      <td>0.010661</td>
      <td>0.016969</td>
      <td>-0.003541</td>
      <td>0.004367</td>
      <td>0.005772</td>
      <td>0.016851</td>
      <td>-0.000826</td>
      <td>...</td>
      <td>-0.000086</td>
      <td>-0.000149</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.000794</td>
      <td>-0.000580</td>
      <td>-0.000817</td>
      <td>NaN</td>
      <td>-0.000724</td>
      <td>-0.000547</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_SR4.5</th>
      <td>-0.044610</td>
      <td>0.019056</td>
      <td>-0.055431</td>
      <td>-0.000128</td>
      <td>0.002170</td>
      <td>0.042324</td>
      <td>-0.017150</td>
      <td>0.030315</td>
      <td>-0.042163</td>
      <td>0.011208</td>
      <td>...</td>
      <td>-0.000789</td>
      <td>-0.001367</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.007258</td>
      <td>-0.005303</td>
      <td>-0.007472</td>
      <td>NaN</td>
      <td>-0.006621</td>
      <td>-0.004999</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_SR6</th>
      <td>-0.114994</td>
      <td>-0.105985</td>
      <td>-0.156554</td>
      <td>-0.097459</td>
      <td>-0.017317</td>
      <td>0.091901</td>
      <td>0.100661</td>
      <td>-0.020570</td>
      <td>-0.015967</td>
      <td>-0.025131</td>
      <td>...</td>
      <td>-0.001649</td>
      <td>-0.002857</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.015169</td>
      <td>-0.011084</td>
      <td>-0.015617</td>
      <td>NaN</td>
      <td>-0.013839</td>
      <td>-0.010447</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_SR8</th>
      <td>-0.031909</td>
      <td>-0.002428</td>
      <td>-0.049825</td>
      <td>-0.041919</td>
      <td>-0.010297</td>
      <td>0.042245</td>
      <td>-0.010263</td>
      <td>-0.025561</td>
      <td>-0.033527</td>
      <td>0.010395</td>
      <td>...</td>
      <td>-0.000489</td>
      <td>-0.000847</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.004497</td>
      <td>-0.003286</td>
      <td>-0.004630</td>
      <td>NaN</td>
      <td>-0.004102</td>
      <td>-0.003097</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_SVV</th>
      <td>-0.002308</td>
      <td>-0.006592</td>
      <td>-0.005740</td>
      <td>-0.010876</td>
      <td>-0.008341</td>
      <td>0.003684</td>
      <td>-0.004977</td>
      <td>-0.015026</td>
      <td>-0.008055</td>
      <td>-0.003488</td>
      <td>...</td>
      <td>-0.000086</td>
      <td>-0.000149</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.000794</td>
      <td>-0.000580</td>
      <td>-0.000817</td>
      <td>NaN</td>
      <td>-0.000724</td>
      <td>-0.000547</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_T</th>
      <td>-0.006874</td>
      <td>-0.000239</td>
      <td>-0.007518</td>
      <td>-0.011348</td>
      <td>-0.011522</td>
      <td>-0.004573</td>
      <td>0.000016</td>
      <td>-0.019618</td>
      <td>0.002110</td>
      <td>-0.003488</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-0.000149</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.000794</td>
      <td>-0.000580</td>
      <td>-0.000817</td>
      <td>NaN</td>
      <td>-0.000724</td>
      <td>-0.000547</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_TC</th>
      <td>-0.008089</td>
      <td>0.012506</td>
      <td>-0.014412</td>
      <td>-0.019103</td>
      <td>-0.017006</td>
      <td>0.006381</td>
      <td>-0.008132</td>
      <td>-0.001006</td>
      <td>0.002101</td>
      <td>-0.009116</td>
      <td>...</td>
      <td>-0.000149</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.001375</td>
      <td>-0.001005</td>
      <td>-0.001416</td>
      <td>NaN</td>
      <td>-0.001255</td>
      <td>-0.000947</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_TL 10A</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_UHUCR</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_UL7200</th>
      <td>-0.060764</td>
      <td>-0.016201</td>
      <td>-0.095693</td>
      <td>-0.097016</td>
      <td>-0.054025</td>
      <td>-0.048170</td>
      <td>-0.038564</td>
      <td>-0.042348</td>
      <td>0.000830</td>
      <td>-0.057733</td>
      <td>...</td>
      <td>-0.000794</td>
      <td>-0.001375</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>-0.005335</td>
      <td>-0.007518</td>
      <td>NaN</td>
      <td>-0.006662</td>
      <td>-0.005029</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_UR</th>
      <td>-0.038110</td>
      <td>0.003449</td>
      <td>-0.035577</td>
      <td>-0.000763</td>
      <td>0.023170</td>
      <td>0.017669</td>
      <td>0.006530</td>
      <td>0.037383</td>
      <td>0.029866</td>
      <td>-0.038550</td>
      <td>...</td>
      <td>-0.000580</td>
      <td>-0.001005</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.005335</td>
      <td>1.000000</td>
      <td>-0.005493</td>
      <td>NaN</td>
      <td>-0.004868</td>
      <td>-0.003675</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_URPSO</th>
      <td>0.011451</td>
      <td>0.124730</td>
      <td>0.032169</td>
      <td>0.019685</td>
      <td>0.119470</td>
      <td>0.146308</td>
      <td>-0.020327</td>
      <td>0.042641</td>
      <td>0.013982</td>
      <td>0.026726</td>
      <td>...</td>
      <td>-0.000817</td>
      <td>-0.001416</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.007518</td>
      <td>-0.005493</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>-0.006858</td>
      <td>-0.005178</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_US R1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_UV</th>
      <td>0.019745</td>
      <td>-0.076844</td>
      <td>0.035798</td>
      <td>0.062884</td>
      <td>0.085470</td>
      <td>0.130523</td>
      <td>0.088087</td>
      <td>-0.003606</td>
      <td>0.035740</td>
      <td>0.043161</td>
      <td>...</td>
      <td>-0.000724</td>
      <td>-0.001255</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.006662</td>
      <td>-0.004868</td>
      <td>-0.006858</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>-0.004588</td>
    </tr>
    <tr>
      <th>ZoneCodeCounty_UVEV</th>
      <td>0.022135</td>
      <td>-0.046138</td>
      <td>0.057218</td>
      <td>0.052581</td>
      <td>0.025389</td>
      <td>0.017966</td>
      <td>-0.013765</td>
      <td>-0.043589</td>
      <td>-0.038190</td>
      <td>0.063900</td>
      <td>...</td>
      <td>-0.000547</td>
      <td>-0.000947</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.005029</td>
      <td>-0.003675</td>
      <td>-0.005178</td>
      <td>NaN</td>
      <td>-0.004588</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>168 rows × 168 columns</p>
</div>




```python
dftrain_X.isna().sum().sum()
```




    0



# All Data preprocessing complete 


# Machine Learning Modeling Started


```python
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBRegressor
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
def cross_validation_function4(X,y,models,fold,w):
    size=int(len(X)/fold)
    l=0
    u=size
    scores=[]
    for f in range(fold):
#         print("l =",l,"u =",u,"")
        dfxtest=X.iloc[l:u,:]
        dfytest=y.iloc[l:u,:]
        dfxtrain=X.drop(X.iloc[l:u].index,axis=0)
        dfytrain=y.drop(X.iloc[l:u].index,axis=0)
        
        y_true=pd.Series(dfytest.iloc[:,0])
        
        model1.fit(dfxtrain,dfytrain.values.ravel())
        model2.fit(dfxtrain,dfytrain.values.ravel())
        model3.fit(dfxtrain,dfytrain.values.ravel())
        model4.fit(dfxtrain,dfytrain.values.ravel())
        model5.fit(dfxtrain,dfytrain.values.ravel())
        model6.fit(dfxtrain,dfytrain.values.ravel())
        model7.fit(dfxtrain,dfytrain.values.ravel())
        model8.fit(dfxtrain,dfytrain.values.ravel())
        model9.fit(dfxtrain,dfytrain.values.ravel())
        model10.fit(dfxtrain,dfytrain.values.ravel())
        model11.fit(dfxtrain,dfytrain.values.ravel())

        y_pred1=model1.predict(dfxtest)
        y_pred2=model2.predict(dfxtest)
        y_pred3=model3.predict(dfxtest)
        y_pred4=model4.predict(dfxtest)
        y_pred5=model5.predict(dfxtest)
        y_pred6=model6.predict(dfxtest)
        y_pred7=model7.predict(dfxtest)
        y_pred8=model8.predict(dfxtest)
        y_pred9=model9.predict(dfxtest)
        y_pred10=model10.predict(dfxtest)
        y_pred11=model11.predict(dfxtest)

            
        p=[0]*12
        for i in range(0,len(w)):
            p[i+1]=w[i]
        
        y_pred=(p[1]*y_pred1 + p[2]*y_pred2 + p[3]*y_pred3 + p[4]*y_pred4 + p[5]*y_pred5 + p[6]*y_pred6 + p[7]*y_pred7 + p[8]*y_pred8 + p[9]*y_pred9 + p[10]*y_pred10 + p[11]*y_pred11)
        sc=np.mean(np.abs((y_true - y_pred) / y_true))
        scores.append(sc)
        print("        FOLD  :",f)
        print("        Score :", sc)
        l=l+size
        u=u+size
    print("Final score is: ",np.mean(scores))
    return np.mean(scores)
      
        
```


```python

```


```python
w=[1/11]*11


model1 = LGBMRegressor(boosting_type='dart', max_depth=8, learning_rate=0.13, n_estimators=600,objective='regression',metric='mape',tree_learner='serial',random_state=30, n_jobs=-1 )
model2= LGBMRegressor(boosting_type='dart', max_depth=9, learning_rate=0.19, n_estimators=400,objective='regression',metric='mape',tree_learner='serial',random_state=30, n_jobs=-1 )
model3 = LGBMRegressor(boosting_type='dart', max_depth=8, learning_rate=0.24, n_estimators=350,objective='regression',metric='mape',tree_learner='serial',random_state=30, n_jobs=-1 )
model4 = LGBMRegressor(boosting_type='dart',num_iterations=900 ,max_depth=7, learning_rate=0.09, n_estimators=1000,objective='regression',metric='mape',tree_learner='serial',random_state=30, n_jobs=-1 )
model5 = LGBMRegressor(boosting_type='dart', num_leaves=29,min_data_in_leaf=15 ,max_depth=7, learning_rate=0.09, n_estimators=1000,objective='regression',metric='mape',tree_learner='serial',random_state=30, n_jobs=-1 )
model6 = LGBMRegressor(boosting_type='dart', num_leaves=25, max_depth=7, learning_rate=0.13, n_estimators=1000,objective='regression',metric='mape',tree_learner='serial',random_state=30, n_jobs=-1 )
model7 = ensemble.GradientBoostingRegressor(n_estimators= 300, max_depth=7,learning_rate=0.054 , loss= 'ls',random_state=30)
model8 = XGBRegressor(max_depth=8,learning_rate=0.04, n_estimators=500,booster='gbtree',random_state=30)
model9 = XGBRegressor(max_depth=8,learning_rate=0.081, n_estimators=100,booster='gbtree',random_state=30)
model10 = RandomForestRegressor(max_depth=18, random_state=30,n_estimators=91)
model11= KNeighborsRegressor(n_neighbors=6,weights="distance")



model1.fit(dftrain_X,dftrain_y.values.ravel())
model2.fit(dftrain_X,dftrain_y.values.ravel())
model3.fit(dftrain_X,dftrain_y.values.ravel())
model4.fit(dftrain_X,dftrain_y.values.ravel())
model5.fit(dftrain_X,dftrain_y.values.ravel())
model6.fit(dftrain_X,dftrain_y.values.ravel())
model7.fit(dftrain_X,dftrain_y.values.ravel())
model8.fit(dftrain_X,dftrain_y.values.ravel())
model9.fit(dftrain_X,dftrain_y.values.ravel())
model10.fit(dftrain_X,dftrain_y.values.ravel())
model11.fit(dftrain_X,dftrain_y.values.ravel())

y_pred1=model1.predict(dftrain_X)
y_pred2=model2.predict(dftrain_X)
y_pred3=model3.predict(dftrain_X)
y_pred4=model4.predict(dftrain_X)
y_pred5=model5.predict(dftrain_X)
y_pred6=model6.predict(dftrain_X)
y_pred7=model7.predict(dftrain_X)
y_pred8=model8.predict(dftrain_X)
y_pred9=model9.predict(dftrain_X)
y_pred10=model10.predict(dftrain_X)
y_pred11=model11.predict(dftrain_X)

y_true=pd.Series(dftrain_y.iloc[:,0])
p=[0]*12
for i in range(0,len(w)):
    p[i+1]=w[i]
y_pred=(p[1]*y_pred1 + p[2]*y_pred2 + p[3]*y_pred3 + p[4]*y_pred4 + p[5]*y_pred5 + p[6]*y_pred6 + p[7]*y_pred7 + p[8]*y_pred8 + p[9]*y_pred9 + p[10]*y_pred10 + p[11]*y_pred11 )

print("Training Error : ", np.mean(np.abs((y_true - y_pred) / y_true)))

model1 = LGBMRegressor(boosting_type='dart', max_depth=8, learning_rate=0.13, n_estimators=600,objective='regression',metric='mape',tree_learner='serial',random_state=30, n_jobs=-1 )
model2= LGBMRegressor(boosting_type='dart', max_depth=9, learning_rate=0.19, n_estimators=400,objective='regression',metric='mape',tree_learner='serial',random_state=30, n_jobs=-1 )
model3 = LGBMRegressor(boosting_type='dart', max_depth=8, learning_rate=0.24, n_estimators=350,objective='regression',metric='mape',tree_learner='serial',random_state=30, n_jobs=-1 )
model4 = LGBMRegressor(boosting_type='dart',num_iterations=900 ,max_depth=7, learning_rate=0.09, n_estimators=1000,objective='regression',metric='mape',tree_learner='serial',random_state=30, n_jobs=-1 )
model5 = LGBMRegressor(boosting_type='dart', num_leaves=29,min_data_in_leaf=15 ,max_depth=7, learning_rate=0.09, n_estimators=1000,objective='regression',metric='mape',tree_learner='serial',random_state=30, n_jobs=-1 )
model6 = LGBMRegressor(boosting_type='dart', num_leaves=25, max_depth=7, learning_rate=0.13, n_estimators=1000,objective='regression',metric='mape',tree_learner='serial',random_state=30, n_jobs=-1 )
model7 = ensemble.GradientBoostingRegressor(n_estimators= 300, max_depth=7,learning_rate=0.054 , loss= 'ls',random_state=30)
model8 = XGBRegressor(max_depth=8,learning_rate=0.04, n_estimators=500,booster='gbtree',random_state=30)
model9 = XGBRegressor(max_depth=8,learning_rate=0.081, n_estimators=100,booster='gbtree',random_state=30)
model10 = RandomForestRegressor(max_depth=18, random_state=30,n_estimators=91)
model11= KNeighborsRegressor(n_neighbors=6,weights="distance")

listofmodels=[model1,model2,model3,model4,model5,model6,model7,model8,model9,model10,model11]


newscore=cross_validation_function4(dftrain_X,dftrain_y,listofmodels,5,w)
print("Testing Cross Validation Error: ",newscore)
```

    C:\ProgramData\Anaconda3\lib\site-packages\lightgbm\engine.py:116: UserWarning: Found `num_iterations` in params. Will use it instead of argument
      warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
    

    Training Error :  0.07195523378492123
    

    C:\ProgramData\Anaconda3\lib\site-packages\lightgbm\engine.py:116: UserWarning: Found `num_iterations` in params. Will use it instead of argument
      warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
    

            FOLD  : 0
            Score : 0.1275520988360718
    

    C:\ProgramData\Anaconda3\lib\site-packages\lightgbm\engine.py:116: UserWarning: Found `num_iterations` in params. Will use it instead of argument
      warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
    

            FOLD  : 1
            Score : 0.1313034818099944
    

    C:\ProgramData\Anaconda3\lib\site-packages\lightgbm\engine.py:116: UserWarning: Found `num_iterations` in params. Will use it instead of argument
      warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
    

            FOLD  : 2
            Score : 0.12364673538951014
    

    C:\ProgramData\Anaconda3\lib\site-packages\lightgbm\engine.py:116: UserWarning: Found `num_iterations` in params. Will use it instead of argument
      warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
    

            FOLD  : 3
            Score : 0.1308716112652768
    

    C:\ProgramData\Anaconda3\lib\site-packages\lightgbm\engine.py:116: UserWarning: Found `num_iterations` in params. Will use it instead of argument
      warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
    

            FOLD  : 4
            Score : 0.12847116418944649
    Final score is:  0.1283690182980599
    Testing Cross Validation Error:  0.1283690182980599
    


```python
w=[1/11]*11

```

#  May be Final Model


```python
w=[1/11]*11


model1 = LGBMRegressor(boosting_type='dart', max_depth=8, learning_rate=0.13, n_estimators=600,objective='regression',metric='mape',tree_learner='serial',random_state=30, n_jobs=-1 )
model2= LGBMRegressor(boosting_type='dart', max_depth=9, learning_rate=0.19, n_estimators=400,objective='regression',metric='mape',tree_learner='serial',random_state=30, n_jobs=-1 )
model3 = LGBMRegressor(boosting_type='dart', max_depth=8, learning_rate=0.24, n_estimators=350,objective='regression',metric='mape',tree_learner='serial',random_state=30, n_jobs=-1 )
model4 = LGBMRegressor(boosting_type='dart',num_iterations=900 ,max_depth=7, learning_rate=0.09, n_estimators=1000,objective='regression',metric='mape',tree_learner='serial',random_state=30, n_jobs=-1 )
model5 = LGBMRegressor(boosting_type='dart', num_leaves=29,min_data_in_leaf=15 ,max_depth=7, learning_rate=0.09, n_estimators=1000,objective='regression',metric='mape',tree_learner='serial',random_state=30, n_jobs=-1 )
model6 = LGBMRegressor(boosting_type='dart', num_leaves=25, max_depth=7, learning_rate=0.13, n_estimators=1000,objective='regression',metric='mape',tree_learner='serial',random_state=30, n_jobs=-1 )
model7 = ensemble.GradientBoostingRegressor(n_estimators= 300, max_depth=7,learning_rate=0.054 , loss= 'ls',random_state=30)
model8 = XGBRegressor(max_depth=8,learning_rate=0.04, n_estimators=500,booster='gbtree',random_state=30)
model9 = XGBRegressor(max_depth=8,learning_rate=0.081, n_estimators=100,booster='gbtree',random_state=30)
model10 = RandomForestRegressor(max_depth=18, random_state=30,n_estimators=91)
model11= KNeighborsRegressor(n_neighbors=6,weights="distance")



model1.fit(dftrain_X,dftrain_y.values.ravel())
model2.fit(dftrain_X,dftrain_y.values.ravel())
model3.fit(dftrain_X,dftrain_y.values.ravel())
model4.fit(dftrain_X,dftrain_y.values.ravel())
model5.fit(dftrain_X,dftrain_y.values.ravel())
model6.fit(dftrain_X,dftrain_y.values.ravel())
model7.fit(dftrain_X,dftrain_y.values.ravel())
model8.fit(dftrain_X,dftrain_y.values.ravel())
model9.fit(dftrain_X,dftrain_y.values.ravel())
model10.fit(dftrain_X,dftrain_y.values.ravel())
model11.fit(dftrain_X,dftrain_y.values.ravel())

y_pred1=model1.predict(dftest_X)
y_pred2=model2.predict(dftest_X)
y_pred3=model3.predict(dftest_X)
y_pred4=model4.predict(dftest_X)
y_pred5=model5.predict(dftest_X)
y_pred6=model6.predict(dftest_X)
y_pred7=model7.predict(dftest_X)
y_pred8=model8.predict(dftest_X)
y_pred9=model9.predict(dftest_X)
y_pred10=model10.predict(dftest_X)
y_pred11=model11.predict(dftest_X)

# y_true=pd.Series(dftest_y.iloc[:,0])
p=[0]*12
print("p is :",p)
for i in range(0,len(w)):
    p[i+1]=w[i]
y_pred=(p[1]*y_pred1 + p[2]*y_pred2 + p[3]*y_pred3 + p[4]*y_pred4 + p[5]*y_pred5 + p[6]*y_pred6 + p[7]*y_pred7 + p[8]*y_pred8 + p[9]*y_pred9 + p[10]*y_pred10 + p[11]*y_pred11 )

# print("Training Error : ", np.mean(np.abs((y_true - y_pred) / y_true)))

```

    C:\ProgramData\Anaconda3\lib\site-packages\lightgbm\engine.py:116: UserWarning: Found `num_iterations` in params. Will use it instead of argument
      warnings.warn("Found `{}` in params. Will use it instead of argument".format(alias))
    

    p is : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    


```python
submissiondf['SaleDollarCnt']=y_pred
```


```python
submissiondf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PropertyID</th>
      <th>SaleDollarCnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48735321</td>
      <td>2.019671e+06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48735471</td>
      <td>1.033734e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49128764</td>
      <td>5.454984e+05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48897535</td>
      <td>4.536903e+05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49083957</td>
      <td>1.172719e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
submissiondf.isna().sum()
```




    PropertyID       0
    SaleDollarCnt    0
    dtype: int64




```python

```


```python
submissiondf.to_csv("Zillow_submission.csv",index=False)
```


```python
temp=pd.read_csv("Zillow_submission.csv")
```


```python
temp.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PropertyID</th>
      <th>SaleDollarCnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.402000e+03</td>
      <td>4.402000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.348500e+07</td>
      <td>6.085027e+05</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.363566e+07</td>
      <td>4.094884e+05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.864910e+07</td>
      <td>1.734947e+05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.880002e+07</td>
      <td>3.611596e+05</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.894242e+07</td>
      <td>5.115664e+05</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.909140e+07</td>
      <td>7.191109e+05</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.244396e+08</td>
      <td>5.709375e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
dftrain.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PropertyID</th>
      <th>SaleDollarCnt</th>
      <th>censusblockgroup</th>
      <th>Usecode</th>
      <th>BedroomCnt</th>
      <th>BathroomCnt</th>
      <th>FinishedSquareFeet</th>
      <th>GarageSquareFeet</th>
      <th>LotSizeSquareFeet</th>
      <th>StoryCnt</th>
      <th>...</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>BGMedHomeValue</th>
      <th>BGMedRent</th>
      <th>BGMedYearBuilt</th>
      <th>BGPctOwn</th>
      <th>BGPctVacant</th>
      <th>BGMedIncome</th>
      <th>BGPctKids</th>
      <th>BGMedAge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.158800e+04</td>
      <td>1.158800e+04</td>
      <td>1.158800e+04</td>
      <td>11588.0</td>
      <td>11588.000000</td>
      <td>11588.000000</td>
      <td>11588.000000</td>
      <td>8747.000000</td>
      <td>1.158800e+04</td>
      <td>11588.000000</td>
      <td>...</td>
      <td>1.158800e+04</td>
      <td>1.158800e+04</td>
      <td>1.158200e+04</td>
      <td>8957.000000</td>
      <td>11341.000000</td>
      <td>11588.000000</td>
      <td>11588.000000</td>
      <td>11588.000000</td>
      <td>11588.000000</td>
      <td>11588.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.502866e+07</td>
      <td>6.137157e+05</td>
      <td>5.300000e+11</td>
      <td>9.0</td>
      <td>3.451800</td>
      <td>2.327628</td>
      <td>2199.899249</td>
      <td>490.981022</td>
      <td>1.601437e+04</td>
      <td>1.528571</td>
      <td>...</td>
      <td>4.755070e+07</td>
      <td>-1.221995e+08</td>
      <td>4.337194e+05</td>
      <td>1235.541699</td>
      <td>1973.356406</td>
      <td>0.747764</td>
      <td>0.050873</td>
      <td>94859.222817</td>
      <td>0.360058</td>
      <td>39.772886</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.605832e+07</td>
      <td>4.577593e+05</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.865682</td>
      <td>0.872601</td>
      <td>936.877939</td>
      <td>220.434500</td>
      <td>4.324446e+04</td>
      <td>0.521864</td>
      <td>...</td>
      <td>1.424218e+05</td>
      <td>1.417068e+05</td>
      <td>1.781283e+05</td>
      <td>394.371247</td>
      <td>17.786514</td>
      <td>0.196277</td>
      <td>0.058676</td>
      <td>36285.661949</td>
      <td>0.140494</td>
      <td>6.726432</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.864894e+07</td>
      <td>2.000000e+04</td>
      <td>5.300000e+11</td>
      <td>9.0</td>
      <td>1.000000</td>
      <td>0.750000</td>
      <td>270.000000</td>
      <td>10.000000</td>
      <td>1.034000e+03</td>
      <td>1.000000</td>
      <td>...</td>
      <td>4.716120e+07</td>
      <td>-1.225150e+08</td>
      <td>1.480000e+04</td>
      <td>185.000000</td>
      <td>1939.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>17500.000000</td>
      <td>0.000000</td>
      <td>18.200000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.880374e+07</td>
      <td>3.550000e+05</td>
      <td>5.300000e+11</td>
      <td>9.0</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1530.000000</td>
      <td>380.000000</td>
      <td>5.683750e+03</td>
      <td>1.000000</td>
      <td>...</td>
      <td>4.744830e+07</td>
      <td>-1.223147e+08</td>
      <td>3.020000e+05</td>
      <td>933.000000</td>
      <td>1960.000000</td>
      <td>0.618700</td>
      <td>0.000000</td>
      <td>69167.000000</td>
      <td>0.260700</td>
      <td>35.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.895489e+07</td>
      <td>5.050000e+05</td>
      <td>5.300000e+11</td>
      <td>9.0</td>
      <td>3.000000</td>
      <td>2.500000</td>
      <td>2060.000000</td>
      <td>476.000000</td>
      <td>7.886500e+03</td>
      <td>2.000000</td>
      <td>...</td>
      <td>4.756348e+07</td>
      <td>-1.222056e+08</td>
      <td>3.969000e+05</td>
      <td>1173.000000</td>
      <td>1975.000000</td>
      <td>0.802200</td>
      <td>0.038900</td>
      <td>90455.000000</td>
      <td>0.352600</td>
      <td>39.400000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.910697e+07</td>
      <td>7.150000e+05</td>
      <td>5.300000e+11</td>
      <td>9.0</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>2700.000000</td>
      <td>600.000000</td>
      <td>1.111100e+04</td>
      <td>2.000000</td>
      <td>...</td>
      <td>4.767496e+07</td>
      <td>-1.221112e+08</td>
      <td>5.256000e+05</td>
      <td>1508.000000</td>
      <td>1987.000000</td>
      <td>0.903800</td>
      <td>0.080800</td>
      <td>114306.000000</td>
      <td>0.444200</td>
      <td>43.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.244354e+08</td>
      <td>7.880000e+06</td>
      <td>5.300000e+11</td>
      <td>9.0</td>
      <td>9.000000</td>
      <td>9.500000</td>
      <td>12130.000000</td>
      <td>7504.000000</td>
      <td>1.157824e+06</td>
      <td>3.000000</td>
      <td>...</td>
      <td>4.785848e+07</td>
      <td>-1.211670e+08</td>
      <td>1.000001e+06</td>
      <td>2001.000000</td>
      <td>2005.000000</td>
      <td>1.000000</td>
      <td>0.638400</td>
      <td>250001.000000</td>
      <td>0.934100</td>
      <td>70.200000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 22 columns</p>
</div>




```python

```
