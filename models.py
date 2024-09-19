from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Content(Base):
    __tablename__ = 'content'
    
    id = Column(Integer, primary_key=True)
    views = Column(Integer)
    comments = Column(Integer)
    shares = Column(Integer)
    likes = Column(Integer)
    bookmark = Column(Integer)
    duration = Column(Float)
    link_to_tiktok = Column(String)
    caption = Column(String)
    transcripts = Column(String)
    hashtags = Column(String)
    cover_image = Column(String)
    audio = Column(String)
    date_posted = Column(String)  # Adjust type as necessary

    