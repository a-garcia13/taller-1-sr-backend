from fastapi import FastAPI
from typing import List

from services.user import UserService
from services.tracks import TrackService
from services.recomendation import RecomendationService
from models.user import UserCreate, UserOut
from models.tracks import TrackCreate, TrackInDB, TrackOut

app = FastAPI()
user_service = UserService()
track_service = TrackService()
recomendation_service = RecomendationService()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/users")
def get_users():
    users = user_service.get_all_users()
    return users


@app.post("/users")
def create_user(user: UserCreate):
    created_user = user_service.create_user(user)
    return UserOut.from_orm(created_user)


@app.get("/users/{user_id}")
def get_user(user_id: str):
    user = user_service.get_user_by_id(user_id)
    return UserOut.from_orm(user)


@app.put("/users/{user_id}")
def update_user(user_id: str, user: UserCreate):
    updated_user = user_service.update_user(user_id, user)
    return UserOut.from_orm(updated_user)


@app.delete("/users/{user_id}")
def delete_user(user_id: str):
    deleted_user = user_service.delete_user(user_id)
    return UserOut.from_orm(deleted_user)


@app.post("/tracks")
def create_track(track: TrackCreate):
    created_track = track_service.create_track(track)
    return TrackOut.from_orm(created_track)


@app.get("/tracks/{track_id}")
def get_track(track_id: str):
    track = track_service.get_track_by_id(track_id)
    return TrackOut.from_orm(track)


@app.put("/tracks/{track_id}")
def update_track(track_id: str, track: TrackCreate):
    updated_track = track_service.update_track(track_id, track)
    return TrackOut.from_orm(updated_track)


@app.delete("/tracks/{track_id}")
def delete_track(track_id: str):
    deleted_track = track_service.delete_track(track_id)
    return TrackOut.from_orm(deleted_track)


@app.get("/users/{user_id}/tracks")
def get_user_tracks(user_id: str):
    user_tracks = track_service.get_tracks_by_user_id(user_id)
    return [TrackOut.from_orm(track) for track in user_tracks]


@app.get("/tracks/artists")
def get_distinct_artists():
    artists = track_service.get_distinct_artists()
    return artists


@app.get("/tracks/artists/{artist}")
def get_tracks_by_artist(artist: str):
    artist_tracks = track_service.get_tracks_by_artist(artist)
    return [TrackOut.from_orm(track) for track in artist_tracks]


@app.get("/users/{user_id}/tracks")
def get_tracks_by_user_id(user_id: str) -> List[TrackInDB]:
    tracks = track_service.get_tracks_by_user(user_id)
    return [TrackInDB.from_orm(track) for track in tracks]


@app.get("/recomendation/all_tracks/{user_id}")
async def get_all_recomended_tracks(user_id: str):
    recomendation = recomendation_service.make_prediction(user_id)
    return recomendation.to_dict('records')