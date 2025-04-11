import numpy as np
import pandas as pd

def change_column_names(data: pd.DataFrame):
    return (
        data.rename(str.lower,axis=1)
        .rename({
            "delivery_person_id" : "rider_id",
            "delivery_person_age": "age",
            "delivery_person_ratings": "ratings",
            "delivery_location_latitude": "delivery_latitude",
            "delivery_location_longitude": "delivery_longitude",
            "time_orderd": "order_time",
            "time_order_picked": "order_picked_time",
            "weatherconditions": "weather",
            "road_traffic_density": "traffic",
            "city": "city_type",
            "time_taken(min)": "time_taken"},axis=1)
    )


# Building an function to clean the data. 
def data_cleaning_1(data: pd.DataFrame):
    
    # Fetching some inappropriate column values. 
    minors_data = data.loc[data['age'].astype('float') < 18]
    minor_index = minors_data.index.tolist()
    six_star_data = data.loc[data['ratings'] == "6"]
    six_star_index = six_star_data.index.tolist()

    # Drop unncessary columns and rows. 
    data = data.drop(columns='id')
    data = data.drop(index=minor_index)
    data = data.drop(index=six_star_index)
    
    # Replace "NaN" string with actual np.NaN
    data = data.replace("NaN ", np.NaN)
    
    # Extract City name from rider_id 
    data['city_name'] =  data['rider_id'].astype(str).str.split("RES").str[0]

    # Convert to appropriate types. 
    data["age"] = data["age"].astype(float)
    data["ratings"] = data["ratings"].astype(float)
    data["multiple_deliveries"] = data["multiple_deliveries"].astype(float)

    # Absolute values for coordinates. 
    data["restaurant_latitude"] = data["restaurant_latitude"].abs()
    data["restaurant_longitude"] = data["restaurant_longitude"].abs()
    data["delivery_latitude"] = data["delivery_latitude"].abs()
    data["delivery_longitude"] = data["delivery_longitude"].abs()

    # Convert order date to datetime and extract features
    data["order_date"] = pd.to_datetime(data["order_date"], dayfirst=True)
    data["order_day"] = data["order_date"].dt.day
    data["order_month"] = data["order_date"].dt.month
    data["order_day_of_week"] = data["order_date"].dt.day_name().str.lower()
    data["is_weekend"] = data["order_date"].dt.day_name().isin(["Saturday", "Sunday"]).astype(int)

    # Convert time columns to datetime
    data["order_time"] = pd.to_datetime(data["order_time"])
    data["order_picked_time"] = pd.to_datetime(data["order_picked_time"])

    # Calculate pickup time in minutes
    data["pickup_time_minutes"] = (data["order_picked_time"] - data["order_time"]).dt.total_seconds() / 60

    # Extract hour and time of day. 
    data["order_time_hour"] = data["order_time"].dt.hour
    # data["order_time_of_delay"] = data["order_time_hour"].apply(time_of_day)
    data["order_time_of_delay"] = time_of_day(data["order_time_hour"])

    # Clean categorical columns
    data["weather"] = data["weather"].astype(str).str.replace("conditions ", "", regex=False).str.lower().replace("nan", np.NaN)
    data["traffic"] = data["traffic"].astype(str).str.rstrip().str.lower()
    data["type_of_order"] = data["type_of_order"].astype(str).str.rstrip().str.lower()
    data["type_of_vehicle"] = data["type_of_vehicle"].astype(str).str.rstrip().str.lower()
    data["festival"] = data["festival"].astype(str).str.rstrip().str.lower()
    data["city_type"] = data["city_type"].astype(str).str.rstrip().str.lower()

    # Clean target column
    data['time_taken'] = df['time_taken'].astype(str).str.replace(r"\(min\)\s*", "", regex=True).astype(int)

    # Drop original time columns after extracting what we need.
    data = data.drop(columns=["order_time", "order_picked_time"])
    
    return data

def clean_lat_long(data: pd.DataFrame, threshold=1):
    
    # Get the list of location-related column names. 
    location_columns = location_subset.columns.tolist()

    # For each location column, replace values below threshold with NaN. 
    for col in location_columns:
        data[col] = np.where(data[col] < threshold, np.NaN, data[col])

    return data


# Extract day, day name, month and year. 
def extract_datetime_features(date_col):

    # Convert the input series to datetime format. 
    date_col = pd.to_datetime(date_col, dayfirst=True)

    # Create a DataFrame with extracted features.
    result = pd.DataFrame()
    result["day"] = date_col.dt.day
    result["month"] = date_col.dt.month
    result["year"] = date_col.dt.year
    result["day_of_week"] = date_col.dt.day_name()
    result["is_weekend"] = date_col.dt.day_name().isin(["Saturday", "Sunday"]).astype(int)

    return result
    
def time_of_day(ser):

    return(
        pd.cut(ser,bins=[0,6,12,17,20,24],right=True,
               labels=["after_midnight","morning","afternoon","evening","night"])
    )

def calculate_haversine_distance(df):
    location_columns = ['restaurant_latitude',
                        'restaurant_longitude',
                        'delivery_latitude',
                        'delivery_longitude']
    
    lat1 = df[location_columns[0]]
    lon1 = df[location_columns[1]]
    lat2 = df[location_columns[2]]
    lon2 = df[location_columns[3]]

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(
        dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c

    return (
        df.assign(
            distance = distance)
    )

def create_distance_type(data: pd.DataFrame):
    return(
        data
        .assign(
                distance_type = pd.cut(data["distance"],bins=[0,5,10,15,25],
                                        right=False,labels=["short","medium","long","very_long"])
    ))


def perform_data_cleaning(data: pd.DataFrame, saved_data_path="swiggy_cleaned.csv"):
    
    cleaned_data = (
        data
        .pipe(change_column_names)
        .pipe(data_cleaning)
        .pipe(clean_lat_long)
        .pipe(calculate_haversine_distance)
        .pipe(create_distance_type)
    )
    
    # save the data
    cleaned_data.to_csv(saved_data_path,index=False)
    
    

if __name__ == "__main__":
    # data path for data
    DATA_PATH = "swiggy.csv"
    
    # read the data from path
    df = pd.read_csv(DATA_PATH)
    print('swiggy data loaded successfuly')
    
    perform_data_cleaning(df)