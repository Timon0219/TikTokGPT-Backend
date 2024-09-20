from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from models import Content  # Adjust import based on your module structure
import pandas as pd
import os

DATABASE_URL = 'sqlite:///tiktok.db'

# Remove the existing database file (optional)
if os.path.exists('tiktok.db'):
    os.remove('tiktok.db')

Base = declarative_base()

engine = create_engine(DATABASE_URL)

Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

df = pd.read_csv('data.csv', encoding='latin1')

# Add data to the database
for _, row in df.iterrows():
    content = Content(
        views=row.get('Views'),
        comments=row.get('Comments'),
        shares=row.get('Shares'),
        likes=row.get('Likes'),
        bookmark=row.get('Bookmark'),
        duration=row.get('Duration'),
        link_to_tiktok=row.get('Link to TikTok'),
        caption=row.get('Caption'),
        transcripts=row.get('Transcripts'),
        hashtags=row.get('Hashtags'),
        cover_image=row.get('Cover Image'),
        audio=row.get('Audio'),
        date_posted=row.get('Date Posted')
    )
    session.add(content)

session.commit()