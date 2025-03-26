import pandas as pd

def check_csv_properties(file_path):
    try:
        data = pd.read_csv(file_path)
        
        num_rows = data.shape[0]
        num_cols = data.shape[1]
        
        column_names = data.columns.tolist()

        data_types = data.dtypes
        
        print(f"Number of Rows: {num_rows}")
        print(f"Number of Columns: {num_cols}")
        print(f"Column Names: {column_names}")
        print(f"Data Types:\n{data_types}")
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        print(f"File is empty: {file_path}")
    except pd.errors.ParserError:
        print(f"Error parsing file: {file_path}")

file_path = '/home/adelechinda/home/projects/powerup/data/noaa_data.csv'
check_csv_properties(file_path)

#* Eaglei_outages
"""
Number of Rows: 161563385
Number of Columns: 5
Column Names: ['fips_code', 'county', 'state', 'sum', 'run_start_time']
Data Types:
fips_code           int64
county             object
state              object
sum               float64
run_start_time     object
dtype: object
"""

#* VTEC
"""
Number of Rows: 1253997
Number of Columns: 25
Column Names: ['WFO', 'ISSUED', 'EXPIRED', 'INIT_ISS', 'INIT_EXP', 'PHENOM', 'GTYPE', 'SIG', 'ETN', 'STATUS', 'NWS_UGC', 'AREA_KM2', 'UPDATED', 'HVTEC_NWSLI', 'HVTEC_SEVERITY', 'HVTEC_CAUSE', 'HVTEC_RECORD', 'IS_EMERGENCY', 'POLYBEGIN', 'POLYEND', 'WINDTAG', 'HAILTAG', 'TORNADOTAG', 'DAMAGETAG', 'PRODUCT_ID']
Data Types:
WFO                object
ISSUED             object
EXPIRED            object
INIT_ISS           object
INIT_EXP           object
PHENOM             object
GTYPE              object
SIG                object
ETN                 int64
STATUS             object
NWS_UGC            object
AREA_KM2          float64
UPDATED            object
HVTEC_NWSLI        object
HVTEC_SEVERITY     object
HVTEC_CAUSE        object
HVTEC_RECORD       object
IS_EMERGENCY         bool
POLYBEGIN          object
POLYEND            object
WINDTAG           float64
HAILTAG           float64
TORNADOTAG         object
DAMAGETAG          object
PRODUCT_ID         object
dtype: object
"""




#* NOAA
"""
Number of Rows: <dask_expr.expr.Scalar: expr=(RenameFrame(frame=MapPartitions(lambda), columns={'BGN_TIME': 'BEGIN_TIME', 'BGN_DATE': 'BEGIN_DATE', 'EVTYPE': 'EVENT_TYPE', 'PROPDMG': 'DAMAGE_PROPERTY', 'PROPDMGEXP': 'DAMAGE_PROPERTY_EXP', 'CROPDMG': 'DAMAGE_CROP', 'CROPDMGEXP': 'DAMAGE_CROP_EXP', 'BGN_RANGE': 'BEGIN_RANGE', 'BGN_LOCATI': 'BEGIN_LOCATION', 'BGN_AZI': 'BEGIN_AZIMUTH', 'END_AZI': 'END_AZIMUTH', 'END_LOCATI': 'END_LOCATION', 'LATITUDE': 'BEGIN_LAT', 'LONGITUDE': 'BEGIN_LON', 'LATITUDE_E': 'END_LAT', 'LONGITUDE_': 'END_LON', 'REMARKS': 'EPISODE_NARRATIVE', 'TIME_ZONE': 'CZ_TIMEZONE'})).size() // 38, dtype=int64>
Number of Columns: 38
Column Names: ['STATE__', 'BEGIN_DATE', 'BEGIN_TIME', 'CZ_TIMEZONE', 'COUNTY', 'COUNTYNAME', 'STATE', 'EVENT_TYPE', 'BEGIN_RANGE', 'BEGIN_AZIMUTH', 'BEGIN_LOCATION', 'END_DATE', 'END_TIME', 'COUNTY_END', 'COUNTYENDN', 'END_RANGE', 'END_AZIMUTH', 'END_LOCATION', 'LENGTH', 'WIDTH', 'F', 'MAG', 'FATALITIES', 'INJURIES', 'DAMAGE_PROPERTY', 'DAMAGE_PROPERTY_EXP', 'DAMAGE_CROP', 'DAMAGE_CROP_EXP', 'WFO', 'STATEOFFIC', 'ZONENAMES', 'BEGIN_LAT', 'BEGIN_LON', 'END_LAT', 'END_LON', 'EPISODE_NARRATIVE', 'REFNUM', 'BEGIN_DATE']
Data Types:
STATE__                        float64
BEGIN_DATE             string[pyarrow]
BEGIN_TIME                       int64
CZ_TIMEZONE            string[pyarrow]
COUNTY                         float64
COUNTYNAME             string[pyarrow]
STATE                  string[pyarrow]
EVENT_TYPE             string[pyarrow]
BEGIN_RANGE                    float64
BEGIN_AZIMUTH                  float64
BEGIN_LOCATION                 float64
END_DATE                datetime64[ns]
END_TIME                       float64
COUNTY_END                     float64
COUNTYENDN                     float64
END_RANGE                      float64
END_AZIMUTH                    float64
END_LOCATION                   float64
LENGTH                         float64
...
EPISODE_NARRATIVE              float64
REFNUM                         float64
BEGIN_DATE              datetime64[ns]
dtype: object


"""
