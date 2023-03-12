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
    pass


class UserInDB(UserBase):
    pass

    class Config:
        orm_mode = True
