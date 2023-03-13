from pydantic import BaseModel


class UserBase(BaseModel):
    # Define the fields that are shared by input and output models
    gender: str = None
    age: str = None
    country: str = None
    registered: str = None


class UserCreate(UserBase):
    # Define the fields that are required for input only
    password: str


class UserOut(UserBase):
    user_id: str

    class Config:
        orm_mode = True


class UserInDB(UserBase):
    user_id: str

    class Config:
        orm_mode = True
