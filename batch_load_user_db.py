import csv
from datetime import datetime
from pymongo import MongoClient

# Set up the MongoDB client and database
client = MongoClient("mongodb://localhost:27017/")
db = client["lastfm"]
collection = db["users"]

# Open the CSV file and read the lastfm-dataset-1k
with open("data/userid-profile.tsv", "r") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        # Rename the #id column to user_id
        row["user_id"] = row.pop("#id")
        # Insert the row as a document in the "users" collection
        collection.insert_one(row)

print("Upload complete!")

# Add a new "password" field to each document in the collection
collection.update_many({}, {"$set": {"password": "12345"}})

print("Password update complete!")
