import requests
import pandas as pd
from prophet import Prophet

class AirQuality:



    def __init__(self, location):

        self.location = location
        self.raw_data = []
        self.model = None


    def pull_data(self, data_points):

        """Function queries data from openAQ api and returns dataframe

            Input: data_points
                int, contains max datapoints to return from API
            Returns: 
                None, updates internal data structure
        """

        # Defining url for location and requesting data
        converted = []
        try:
            url = "https://api.openaq.org/v2/measurements?date_from=2024-05-30T00%3A00%3A00Z&date_to=2024-06-06T20%3A45%3A00Z&limit={}&page=1&offset=0&sort=desc&parameter_id=2&radius=1000&location_id={}&order_by=datetime".format(data_points, self.location)
            headers = {"accept": "application/json"}
            response = requests.get(url, headers=headers)
            converted = response.json()['results']
        except:
            print("Invalid query")

        # Defining lists for data parsing
        values, date, location, parameter, latitude, longitude = [], [], [], [], [], []
        
        # Iterating through dict and appending values
        for entry in converted:
            values.append(entry['value'])
            date.append(entry['date']['utc'])
            location.append(entry['locationId'])
            parameter.append(entry['parameter'])
            latitude.append(entry['coordinates']['latitude'])
            longitude.append(entry['coordinates']['longitude'])

        df = pd.DataFrame.from_dict({"Date": date,
                                "Value": values,
                                "location": location,
                                "parameter": parameter,
                                'longitude': longitude,
                                "latitude": latitude})
        df['Date'] =  df['Date'].astype(str)
        
        # Converting datetime
        df['Data_Converted'] = df['Date'].str.slice(start = 0, stop = 10) + " " + df['Date'].str.slice(start = 11)
        df['Data_Converted'] = pd.to_datetime(df['Data_Converted'], format='mixed') 
        self.raw_data = df

    def train_model(self, periods):

        train = self.raw_data[['Data_Converted', 'Value']]
        train = train.rename(columns = {'Data_Converted': 'ds', 'Value': 'y'})
        train['ds'] = train['ds'].dt.tz_localize(None)

        # Fitting model
        model = Prophet()
        model.fit(train)

        # Making prediction
        future = model.make_future_dataframe(periods=periods)

        forecast = model.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

        # Plotting forecast
        fig1 = model.plot(forecast)

        # Saving model
        self.model = model

