from typing import List
from pymongo import MongoClient
from datetime import datetime
from bson.objectid import ObjectId
from models.user import UserCreate, UserInDB


class UserService:
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["lastfm"]
        self.collection = self.db["users"]

    def create_user(self, user: UserCreate):
        user_dict = user.dict()
        largest_user_id = 0

        for user in self.collection.find():
            user_id = int(user["user_id"].split("_")[1])
            if user_id > largest_user_id:
                largest_user_id = user_id

        # Increment the user_id and format it as a string with leading zeros
        user_dict['user_id'] = f"user_{int(largest_user_id + 1):06d}"

        result = self.collection.insert_one(user_dict)
        user_created = self.collection.find_one({"_id": ObjectId(result.inserted_id)})
        return UserInDB(**user_created)

    def get_user(self, user_id: str):
        user_found = self.collection.find_one({"#id": user_id})
        if user_found:
            return UserInDB(**user_found)

    def get_all_users(self) -> List[UserInDB]:
        users = []
        for user in self.collection.find():
            users.append(UserInDB(**user))
        return users

    def update_user(self, user_id: str, user: UserCreate):
        user_dict = user.dict(exclude_unset=True)
        if 'registered' in user_dict:
            user_dict['registered'] = datetime.strptime(user_dict['registered'], '%b %d, %Y')
        result = self.collection.update_one({"#id": user_id}, {"$set": user_dict})
        if result.modified_count > 0:
            updated_user = self.collection.find_one({"#id": user_id})
            return UserInDB(**updated_user)

    def delete_user(self, user_id: str):
        result = self.collection.delete_one({"#id": user_id})
        if result.deleted_count > 0:
            return True

    def get_user_by_id(self, user_id: str):
        user = self.collection.find_one({"user_id": user_id})
        if user:
            return UserInDB(**user)
        else:
            return None

