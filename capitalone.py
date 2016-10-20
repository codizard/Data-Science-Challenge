import urllib2
import csv
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
	import matplotlib.pyplot as plt
except Exception, e:
	print "Installing matplotlib module.."
	os.system("python -m pip install -U matplotlib")
	import matplotlib.pyplot as plt

try:
	import pandas as pd
except:
	print "Installing Pandas Module.."		
	os.system("python -m pip install pandas")
	import pandas as pd


# Question 1 
# Programmatically download and load into your favorite analytical tool the trip data for September 2015.
# url = 'https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2015-09.csv'
url = 'C:\Users\Shashaank\Downloads\green_tripdata_2015-09.csv'
print "Downloading data from "+url+".."
html = pd.read_csv(url)
print "Data loaded successfully!"

# Report how many rows and columns of data you have loaded.
print "Calculating number of rows and columns.."
nrows = len(html)
df = pd.DataFrame(html)
ncols = len(df.columns)	
print "Number of rows loaded (Excluding the header): ",nrows
print "Number of columns loaded: ",ncols

# Question 2
# Plot a histogram of the number of the trip distance ("Trip Distance").
df1 = df[(df['Trip_distance'] >= 0) & (df['Trip_distance'] <= 2 )]
df2 = df[(df['Trip_distance'] > 2) & (df['Trip_distance'] <= 5 )]
df3 = df[(df['Trip_distance'] > 5) & (df['Trip_distance'] <= 10 )]
df4 = df[(df['Trip_distance'] > 10) & (df['Trip_distance'] <= max(df['Trip_distance']) )]
to_plot = np.array([len(df1),len(df2),len(df3),len(df4)])


x = np.arange(0,4)
my_xticks = np.array(['0','1','2','3'])
new_ticks = ['0-2','2-5','5-10','>10']

plt.xlabel('Miles',fontsize=15, color = 'red')
plt.ylabel('Frequency',fontsize=15, color = 'red')
plt.title('Histogram of Trip Distance',fontsize=20, color = 'black')
plt.axis([0,5,10000,900000])
plt.xticks(x,new_ticks)
plt.bar(my_xticks,to_plot)
plt.grid(True)
#plt.show()

# Report any structure you find and any hypotheses you have about that structure.

# Question 3
# Report mean and median trip distance grouped by hour of day.
timestamp= np.array(df['lpep_pickup_datetime'].str.replace(r'[0-9]{4}-[0-9]{2}-[0-9]{2} ','').str.split(':'))
a = zip(*timestamp)
myHour = a[0]
asd = np.array(myHour)
new_df = pd.DataFrame({'x':asd, 'y':df['Trip_distance']})

new_myHour = list(set(myHour))
new_myHour.sort()


# for i in range(0,len(new_myHour)):	
# 	temp_mean = np.mean(new_df.loc[new_df['x']==str(new_myHour[i])])
# 	x = np.array(new_df.loc[new_df['x']==str(new_myHour[i])])
# 	temp_median = np.median(x[:,1])
# 	print ("HOUR:"+str(new_myHour[i])+"       MEAN:"+str(temp_mean[1])+"        MEDIAN:" +str(temp_median))

# We'd like to get a rough sense of identifying trips that originate or terminate at one of the NYC area airports. 
# Can you provide a count of how many transactions fit this criteria, the average fair, and any other interesting characteristics of these trips.

# The latitute and longitude for the JFK airport has been considered to be
#  40.6400 < Latitude < 40.6450 and -73.7762 < Longitude < -73.7822
df_JFK_pickup = df.loc[(df['Pickup_latitude'] > 40.6400) & (df['Pickup_latitude'] < 40.6450) & (df['Pickup_longitude'] < -73.7762) & (df['Pickup_longitude'] > -73.7822)]
df_JFK_dropoff = df.loc[(df['Dropoff_latitude'] > 40.6400) & (df['Dropoff_latitude'] < 40.6450) & (df['Dropoff_longitude'] < -73.7762) & (df['Dropoff_longitude'] > -73.7822)]


average_fair = (sum(df_JFK_dropoff['Fare_amount']) + sum(df_JFK_pickup['Fare_amount'])) / (len(df_JFK_pickup['Fare_amount']) + len (df_JFK_dropoff['Fare_amount'])) 
no_of_transactions = (len(df_JFK_pickup['Pickup_latitude']) + len(df_JFK_dropoff['Dropoff_latitude']))

print "Number of transactions originate or terminate at JFK: ", no_of_transactions
print "Average fair: ", average_fair

# Question 4
# Build a derived variable for tip as a percentage of the total fare.
new_tip = (df['Tip_amount']*df['Total_amount'])/100

# Build a predictive model for tip as a percentage of the total fare. Use as much of the data as you like (or all of it). We will validate a sample.
x_train = new_tip
y_train = df['Tip_amount']
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
