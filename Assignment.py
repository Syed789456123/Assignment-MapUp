#!/usr/bin/env python
# coding: utf-8

# In[1]:


##importing important libraries
import pandas as pd
import numpy as np


# # Python Task 1

# In[2]:


##reading data
df=pd.read_csv('dataset-1.csv')
df.head()


# In[68]:


#question1
###creating pivot table using value as car
def generate_car_matrix(data):
    pivot_df = pd.pivot(data, index='id_1', columns='id_2', values='car')
    pivot_df = pivot_df.fillna(0) ###filling the nanvalues with zero
    return pivot_df
data = r'C:\Users\SyedMohammad\Downloads\assignment\dataset-1.csv'
df = pd.read_csv(data)
result = generate_car_matrix(df)
print(result)


# In[4]:


#question 2
def get_type_count(data): ###defining function
    conditions = [
        (data['car'] <= 15),
        (data['car'] > 15) & (data['car'] <= 25),
        (data['car'] > 25)
    ]
    choices = ['low', 'medium', 'high']
    data['car_type'] = pd.Series(np.select(conditions, choices), dtype='category')
    type_counts = data['car_type'].value_counts().to_dict()
    sorted_type_counts = dict(sorted(type_counts.items()))
    return sorted_type_counts
dataset_path = r'C:\Users\SyedMohammad\Downloads\assignment\dataset-1.csv'
df = pd.read_csv(dataset_path)
result = get_type_count(df)
print(result)



# In[5]:


#question 3
import pandas as pd
import numpy as np

def get_bus_indexes(data):
   
    mean = np.mean(data['bus'])###calculating the mean
    indices = data[data['bus'] > 2 * mean].index.tolist()# Identifying indices where 'bus' values are greater than twice the mean
    indices.sort()##sorting
    return indices
dataset_path = r'C:\Users\SyedMohammad\Downloads\assignment\dataset-1.csv'
df = pd.read_csv(dataset_path)
result = get_bus_indexes(df)
print(result)


# In[6]:


#question 4
import pandas as pd
import numpy as np

def filter_routes(data):
    
    truck_values = data.loc[data['truck'] > 7]# Filtering rows where 'truck' values are greater than 7
    
    
    route_values = truck_values.groupby(by='route')['truck'].mean().reset_index()###grouping the route and calculating the 
    
    
    route_values = route_values.sort_values(by='truck', ascending=True)# Sort the DataFrame by mean 'truck' values
    route_values = list(route_values['route'])### into list
    route_values.sort()##sorting
    return route_values
    

dataset_path = r'C:\Users\SyedMohammad\Downloads\assignment\dataset-1.csv'
df = pd.read_csv(dataset_path)
result = filter_routes(df)
print(result)


# In[7]:


#question 5
import pandas as pd
import numpy as np
def multiply_matrix(df):
    pivot_df = pd.pivot(df, index='id_1', columns='id_2', values=['moto', 'car', 'rv', 'bus', 'truck'])### create the pivot using values 'moto', 'car', 'rv', 'bus', 'truck'  
    pivot_df = pivot_df.fillna(0)
    ###multiply the value with 0.75 which is greater then 20  else with 1.25
    modified_df = pivot_df.applymap(lambda x: round(x * 0.75,1) if x > 20 else round(x * 1.25,1))
    return modified_df
dataset_path = r'C:\Users\SyedMohammad\Downloads\assignment\dataset-1.csv'
df = pd.read_csv(dataset_path)
result = multiply_matrix(df)
print(result)


# In[48]:


import pandas as pd
import numpy as np

def time_boolean(data):
    # Convert 'startTime' and 'endTime' to datetime
    data['startTime'] = pd.to_datetime(data['startTime'])
    data['endTime'] = pd.to_datetime(data['endTime'])

    # Assuming 'startDay' and 'endDay' contain weekdays as strings
    # If they are already datetime, you can skip this step
    data['startDay'] = pd.Categorical(data['startDay'], categories=['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], ordered=True)
    data['endDay'] = pd.Categorical(data['endDay'], categories=['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], ordered=True)

    # Map weekdays to numeric values
    weekday_to_numeric = {'Saturday': 0, 'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4, 'Thursday': 5, 'Friday': 6}
    data['start_day_numeric'] = data['startDay'].map(weekday_to_numeric)
    data['end_day_numeric'] = data['endDay'].map(weekday_to_numeric)

    # Convert categorical columns to numeric
    data['no_of_weeks'] = data['end_day_numeric'].astype(int) - data['start_day_numeric'].astype(int)
    data['hrs'] = round((data['endTime'] - data['startTime']).dt.total_seconds() / 3600)  # Convert timedelta to hours

    # Modify the condition to check if timestamps cover a full 24-hour period and span all 7 days
    data['boolean_time'] = data.apply(lambda x: True if (x['no_of_weeks'] >= 1 and x['hrs'] >= 24.0) else False, axis=1)

    return data

data = pd.read_csv(r'C:\Users\SyedMohammad\Downloads\assignment\dataset-2.csv')
result = time_boolean(data)
print(result)


# # Python task 2

# In[75]:


import pandas as pd

def calculate_distance_matrix(data):
    # Create a DataFrame with unique IDs
    unique_ids = sorted(set(data['id_start'].unique()) | set(data['id_end'].unique()))
    distance_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids)
    
    # Initialize the matrix with zeros
    distance_matrix = distance_matrix.fillna(0)
    
    # Fill the matrix with cumulative distances
    for index, row in data.iterrows():
        start_id = row['id_start']
        end_id = row['id_end']
        distance = row['distance']
        
        # Update the matrix with the cumulative distances
        distance_matrix.loc[start_id, end_id] += distance
        distance_matrix.loc[end_id, start_id] += distance  # Ensure the matrix is symmetric
    
    return distance_matrix

data = pd.read_csv(r'C:\Users\SyedMohammad\Downloads\assignment\dataset-3.csv')
result = calculate_distance_matrix(data)
print(result)


# In[52]:


##question2
import pandas as pd

def unroll_distance_matrix(distance_matrix):
    # Reset the index to ensure we can access it by index
    distance_matrix_reset = distance_matrix.reset_index(drop=True)

    # Initialize an empty list to store unrolled data
    unrolled_data = []

    # Iterate over the DataFrame rows
    for i in range(len(distance_matrix_reset)):
        id_start = distance_matrix_reset.loc[i, 'id_start']
        distance_start = distance_matrix_reset.loc[i, 'distance']

        for j in range(len(distance_matrix_reset)):
            id_end = distance_matrix_reset.loc[j, 'id_end']
            distance_end = distance_matrix_reset.loc[j, 'distance']

            # Exclude combinations where id_start equals id_end
            if id_start != id_end:
                # Append the combination and cumulative distance to the list
                unrolled_data.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance_start + distance_end
                })

    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df


data = pd.read_csv(r'C:\Users\SyedMohammad\Downloads\assignment\dataset-3.csv')
result = unroll_distance_matrix(data)
print(result)


# In[53]:


#questions 3
import pandas as pd

def find_ids_within_ten_percentage_threshold(df, reference_value):
    # Filter rows with the given reference value in the id_start column
    reference_rows = df[df['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    reference_avg_distance = reference_rows['distance'].mean()

    # Calculate the threshold range within 10% of the average
    threshold_range = (0.9 * reference_avg_distance, 1.1 * reference_avg_distance)

    # Filter rows within the threshold range
    within_threshold = df[(df['distance'] >= threshold_range[0]) & (df['distance'] <= threshold_range[1])]

    # Get unique values from the id_start column and sort them
    result_ids = sorted(within_threshold['id_start'].unique())

    return result_ids


# Assuming 'result' is the DataFrame created in Question 2
reference_value = 1001402  ###reference value
result_ids_within_threshold = find_ids_within_ten_percentage_threshold(result, reference_value)

print(result_ids_within_threshold)


# In[54]:


##questions 4
import pandas as pd

def calculate_toll_rate(data):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        column_name = f'{vehicle_type}_toll'
        df[column_name] = df['distance'] * rate_coefficient

    return data

data = pd.read_csv(r'C:\Users\SyedMohammad\Downloads\assignment\dataset-3.csv')
result = unroll_distance_matrix(data)
print(result)


# In[63]:


import pandas as pd

def calculate_toll_rate(input_df):
    # Copy the input DataFrame to avoid modifying the original data
    result_df = input_df.copy()

    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate in rate_coefficients.items():
        # Create a new column for each vehicle type with the calculated toll rates
        result_df[vehicle_type] = result_df['distance'] * rate

    return result_df


data = pd.read_csv(r'C:\Users\SyedMohammad\Downloads\assignment\dataset-3.csv')
df = pd.DataFrame(data)
result = calculate_toll_rate(df)
print(result)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




