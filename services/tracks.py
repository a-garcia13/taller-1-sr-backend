from typing import List
from pymongo import MongoClient
from bson.objectid import ObjectId
from models.tracks import TrackCreate, TrackInDB


class TrackService:
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["lastfm"]
        self.collection = self.db["tracks"]

    def create_track(self, track: TrackCreate):
        track_dict = track.dict()
        result = self.collection.insert_one(track_dict)
        track_dict['_id'] = str(result.inserted_id)
        return TrackInDB(**track_dict)

    def get_track_by_id(self, track_id: str) -> TrackInDB:
        track_dict = self.collection.find_one({"_id": ObjectId(track_id)})
        if track_dict:
            track_dict['_id'] = str(track_dict['_id'])
            return TrackInDB(**track_dict)

    def get_tracks_by_user(self, user_id: str) -> List[TrackInDB]:
        track_dicts = self.collection.find({"user_id": user_id})
        tracks = []
        for track_dict in track_dicts:
            track_dict['_id'] = str(track_dict['_id'])
            tracks.append(TrackInDB(**track_dict))
        return tracks

    def update_track(self, track_id: str, track: TrackCreate) -> TrackInDB:
        track_dict = track.dict()
        result = self.collection.update_one(
            {"_id": ObjectId(track_id)},
            {"$set": track_dict}
        )
        if result.modified_count == 1:
            track_dict['_id'] = track_id
            return TrackInDB(**track_dict)

    def delete_track(self, track_id: str) -> bool:
        result = self.collection.delete_one({"_id": ObjectId(track_id)})
        return result.deleted_count == 1

    def get_distinct_artists(self) -> List[str]:
        return self.collection.distinct("artname")

    def get_tracks_by_artist(self, artist_name: str) -> List[TrackInDB]:
        track_dicts = self.collection.find({"artname": artist_name})
        tracks = []
        for track_dict in track_dicts:
            track_dict['_id'] = str(track_dict['_id'])
            tracks.append(TrackInDB(**track_dict))
        return tracks

    def get_tracks_by_user_id(self, user_id: str) -> List[TrackInDB]:
        track_dicts = self.collection.find({"user_id": user_id})
        tracks = []
        for track_dict in track_dicts:
            track_dict['_id'] = str(track_dict['_id'])
            tracks.append(TrackInDB(**track_dict))
        return tracks
