import pandas as pd
from pymongo import MongoClient

# Set up the MongoDB client and database
client = MongoClient("mongodb://localhost:27017/")
db = client["lastfm"]
collection = db["tracks"]

# Read the TSV file using pandas
tracks = pd.read_csv('data/userid-timestamp-artid-artname-traid-traname.tsv',
                     sep='\t',
                     names=['user_id', 'timestamp', 'artid', 'artname', 'trackid', 'trackname'],
                     skiprows=[2120260-1, 2446318-1, 11141081-1, 11152099-1, 11152402-1, 11882087-1, 12902539-1, 12935044-1, 17589539-1])

tracks = tracks.drop_duplicates().dropna()
tracks.timestamp = pd.to_datetime(tracks.timestamp, infer_datetime_format=True)

# Convert the DataFrame to a list of dictionaries
tracks_dict = tracks.to_dict(orient="records")

# Insert the list of dictionaries into the "tracks" collection
collection.insert_many(tracks_dict)

print("Upload complete!")
