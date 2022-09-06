from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import boxcox


def under_sampling(dat, col, n):
    return pd.concat([dat[dat[col] == 1].sample(n, replace=True),
                      dat[dat[col] == 0].sample(n)], axis=0)


df = pd.read_csv('./sampled_data.csv').drop(['Unnamed: 0'], axis=1)
df1 = pd.read_csv('./sampled_data.csv').drop(['Unnamed: 0'], axis=1)

# View data source
source = df.groupby(['Source', 'Severity']).count()
df = df.loc[df['Source'] == "MapQuest",]
df = df.drop(['Source'], axis=1)

# Missing value handling
missing = pd.DataFrame(df.isnull().sum()).reset_index()
missing.columns = ['Feature', 'Missing_Percent(%)']
missing['Missing_Percent(%)'] = missing['Missing_Percent(%)'].apply(lambda x: x / df.shape[0] * 100)
missing = missing.loc[missing['Missing_Percent(%)'] > 0, :]

# missing.to_csv('missing_percentage.csv')
drop_set1 = ['End_Lat', 'End_Lng', 'Number', 'Wind_Chill(F)']
df = df.drop(drop_set1, axis=1)
df['Precipitation_NA'] = 0
df.loc[df['Precipitation(in)'].isnull(), 'Precipitation_NA'] = 1
df['Precipitation(in)'] = df['Precipitation(in)'].fillna(df['Precipitation(in)'].median())
df = df.dropna()

# Check unique values
cat_attr_all = df.columns.values
cat_attr = ['Side', 'Country', 'Timezone', 'Amenity', 'Bump', 'Crossing', 'Wind_Direction'
                                                                          'Give_Way', 'Junction', 'No_Exit', 'Railway',
            'Roundabout', 'Station', 'Weather_Condition'
                                     'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 'Sunrise_Sunset',
            'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']
for i in cat_attr_all:
    print(i, df[i].unique().size)
df = df.drop(['Country', 'Turning_Loop'], axis=1)
print("Wind direction before precessing: ", df['Wind_Direction'].unique())

# Wind dir
df.loc[(df['Wind_Direction'] == 'North') | (df['Wind_Direction'] == 'NNW') | (
        df['Wind_Direction'] == 'NNE'), 'Wind_Direction'] = 'N'
df.loc[(df['Wind_Direction'] == 'South') | (df['Wind_Direction'] == 'SSW') | (
        df['Wind_Direction'] == 'SSE'), 'Wind_Direction'] = 'S'
df.loc[(df['Wind_Direction'] == 'East') | (df['Wind_Direction'] == 'ESE') | (
        df['Wind_Direction'] == 'ENE'), 'Wind_Direction'] = 'E'
df.loc[(df['Wind_Direction'] == 'West') | (df['Wind_Direction'] == 'WSW') | (
        df['Wind_Direction'] == 'WNW'), 'Wind_Direction'] = 'W'
df.loc[df['Wind_Direction'] == 'Variable', 'Wind_Direction'] = 'VAR'
df.loc[df['Wind_Direction'] == 'Calm', 'Wind_Direction'] = 'CALM'
print("Wind Directions after processing: ", df['Wind_Direction'].unique())

# Weather condition
print("Weather conditions before processing: ", df['Weather_Condition'].unique())
df['Clear'] = np.where(df['Weather_Condition'].str.contains('Clear', case=False, na=False), True, False)
df['Cloud'] = np.where(df['Weather_Condition'].str.contains('Cloud|Overcast', case=False, na=False), True, False)
df['Rain'] = np.where(df['Weather_Condition'].str.contains('Rain|storm', case=False, na=False), True, False)
df['Heavy_Rain'] = np.where(
    df['Weather_Condition'].str.contains('Heavy Rain|Rain Shower|Heavy T-Storm|Heavy Thunderstorms',
                                         case=False, na=False), True, False)
df['Snow'] = np.where(df['Weather_Condition'].str.contains('Snow|Sleet|Ice', case=False, na=False), True, False)
df['Heavy_Snow'] = np.where(
    df['Weather_Condition'].str.contains('Heavy Snow|Heavy Sleet|Heavy Ice Pellets|Snow Showers|Squalls',
                                         case=False, na=False), True, False)
df['Fog'] = np.where(df['Weather_Condition'].str.contains('Fog', case=False, na=False), True, False)
weathers = ['Clear', 'Cloud', 'Rain', 'Heavy_Rain', 'Snow', 'Heavy_Snow', 'Fog']
df.loc[:, ['Weather_Condition'] + weathers]
df = df.drop(['Weather_Condition'], axis=1)

# Drop useless attributes
df = df.drop(['ID', 'TMC', 'End_Time', 'Distance(mi)', 'Description'], axis=1)

# Severity class
df['Severity4'] = 0
df.loc[df['Severity'] == 4, 'Severity4'] = 1
df = df.drop(['Severity'], axis=1)
print("Number of Severity4: ", df.Severity4.value_counts())

# create a list of top 40 most common words of street names
st_type = ' '.join(df['Street'].unique().tolist())  # flat the array of street name
st_type = re.split(" |-", st_type)  # split the long string by space and hyphen
st_type = [x[0] for x in Counter(st_type).most_common(40)]  # select the 40 most common words
print('the 40 most common words')
print(*st_type, sep=", ")

# Remove irrelevant words and name processing
st_type = [' Rd', ' St', ' Dr', ' Ave', ' Blvd', ' Ln', ' Highway', ' Pkwy', ' Hwy',
           ' Way', ' Ct', 'Pl', ' Road', 'US-', 'Creek', ' Cir', 'Route',
           'I-', 'Trl', 'Pike', ' Fwy']
print(*st_type, sep=", ")

# Create sparse matrix for street
for i in st_type:
    df[i.strip()] = np.where(df['Street'].str.contains(i, case=True, na=False), True, False)
df.loc[df['Road'] == 1, 'Rd'] = True
df.loc[df['Highway'] == 1, 'Hwy'] = True

# Undersample for plot
df_bl = under_sampling(df, 'Severity4', 20000)

# Plot correlation of attributes
df_bl['Severity4'] = df_bl['Severity4'].astype(int)
street_corr = df_bl.loc[:, ['Severity4'] + [x.strip() for x in st_type]].corr()

# Time format processing
df = df.drop(["Weather_Timestamp"], axis=1)
start_time_datetimg = pd.to_datetime(df['Start_Time'], format='%Y-%m-%d')
df['Year'] = start_time_datetimg.dt.year
nmonth = start_time_datetimg.dt.month
df['Month'] = nmonth
df['Weekday'] = start_time_datetimg.dt.weekday

days_each_month = np.cumsum(np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]))
nday = [days_each_month[arg - 1] for arg in nmonth.values]
nday = nday + start_time_datetimg.dt.day.values
df['Day'] = nday
df['Hour'] = start_time_datetimg.dt.hour
df['Minute'] = df['Hour'] * 60.0 + start_time_datetimg.dt.minute
time_mapping = df.loc[:4, ['Start_Time', 'Year', 'Month', 'Weekday', 'Day', 'Hour', 'Minute']]
df = df.drop(['Start_Time'], axis=1)

# Frequency encoding and log-transform
df['Minute_Freq'] = df.groupby(['Minute'])['Minute'].transform('count')
df['Minute_Freq'] = df['Minute_Freq'] / df.shape[0] * 24 * 60
df['Minute_Freq'] = df['Minute_Freq'].apply(lambda x: np.log(x + 1))

fre_list = ['Street', 'City', 'County', 'Zipcode', 'Airport_Code', 'State']
for i in fre_list:
    newname = i + '_Freq'
    df[newname] = df.groupby([i])[i].transform('count')
    df[newname] = df[newname] / df.shape[0] * df[i].unique().size
    df[newname] = df[newname].apply(lambda x: np.log(x + 1))
df = df.drop(fre_list, axis=1)

# Weather features normalization
df['Pressure_bc'] = boxcox(df['Pressure(in)'].apply(lambda x: x + 1), lmbda=6)
df['Visibility_bc'] = boxcox(df['Visibility(mi)'].apply(lambda x: x + 1), lmbda=0.1)
df['Wind_Speed_bc'] = boxcox(df['Wind_Speed(mph)'].apply(lambda x: x + 1), lmbda=-0.2)
df = df.drop(['Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)'], axis=1)

# Wind_direction deletion
plt.figure(figsize=(10, 5))
chart = sns.countplot(x='Wind_Direction', hue='Severity4', data=df_bl, palette="Set2")
plt.title("Count of Accidents in Wind Direction (resampled data)", size=15, y=1.05)
plt.show()
df = df.drop(['Wind_Direction'], axis=1)

# POI features checking
POI_features = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
                'Stop', 'Traffic_Calming', 'Traffic_Signal']

fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(15, 10))  # Check distribution
plt.subplots_adjust(hspace=0.5, wspace=0.5)
for i, feature in enumerate(POI_features, 1):
    plt.subplot(3, 4, i)
    sns.countplot(x=feature, hue='Severity4', data=df_bl, palette="Set2")
    plt.xlabel('{}'.format(feature), size=12, labelpad=3)
    plt.ylabel('Accident Count', size=12, labelpad=3)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.legend(['0', '1'], loc='upper right', prop={'size': 10})
    plt.title('Count of Severity in {}'.format(feature), size=14, y=1.05)
fig.suptitle('Count of Accidents in POI Features (resampled data)', y=1.02, fontsize=16)
plt.show()

df = df.drop(['Amenity', 'Bump', 'Give_Way', 'No_Exit', 'Roundabout', 'Traffic_Calming'], axis=1)

# one-hot encoding
period_features = ['Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']
df[period_features] = df[period_features].astype('category')
df = pd.get_dummies(df, columns=period_features, drop_first=True)
# Undersampling and check correlation again
df_bl = under_sampling(df, 'Severity4', 20000)
# plot correlation
correlation = df_bl.corr()
# Check dataset size
print(df.shape)

# Drop features according to correlation
df = df.drop(
    ['Temperature(F)', 'Humidity(%)', 'Precipitation(in)', 'Precipitation_NA', 'Visibility_bc', 'Wind_Speed_bc',
     'Clear', 'Cloud', 'Snow', 'Crossing', 'Junction', 'Railway', 'Month',
     'Hour', 'Day', 'Minute', 'City_Freq', 'County_Freq', 'Airport_Code_Freq', 'Zipcode_Freq',
     'Sunrise_Sunset_Night', 'Civil_Twilight_Night', 'Nautical_Twilight_Night'], axis=1)

df = df.replace([True, False], [1, 0])
cat = ['Side', 'Timezone', 'Weekday']
df[cat] = df[cat].astype('category')
df = pd.get_dummies(df, columns=cat, drop_first=True)

print(df.shape)

# Down cast the data types for model training
df_int = df.select_dtypes(include=['int64']).apply(pd.to_numeric, downcast='unsigned')
df_float = df.select_dtypes(include=['float64']).apply(pd.to_numeric, downcast='float')
df = pd.concat([df.select_dtypes(include=['uint8', 'uint16']), df_int, df_float], axis=1)
df.info()

# Store the processed data
# df.to_csv('./processed_dataset.csv')
