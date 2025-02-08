#%%
import requests
import psycopg2
from psycopg2 import pool
from datetime import datetime
from contextlib import contextmanager

#%%
class DatabasePool:
    _instance = None
    _pool = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabasePool, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._pool is None:
            self.db_params = {
                "dbname": "glance_db",
                "user": "postgres",
                "password": "postgres",
                "host": "localhost",
                "port": "7777"
            }
            self._create_pool()
    
    def _create_pool(self, minconn=1, maxconn=10):
        """Create a connection pool with specified minimum and maximum connections"""
        try:
            self._pool = psycopg2.pool.SimpleConnectionPool(
                minconn,
                maxconn,
                **self.db_params
            )
            print("Connection pool created successfully")
        except psycopg2.Error as e:
            print(f"Error creating connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool with context management"""
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        finally:
            if conn:
                self._pool.putconn(conn)
    
    def close_pool(self):
        """Close all connections in the pool"""
        if self._pool:
            self._pool.closeall()
            print("Connection pool closed")

class ProfileManager:
    def __init__(self):
        self.db_pool = DatabasePool()
    
    def insert_profile(self, data):
        """Insert or update profile data using a connection from the pool"""
        try:
            with self.db_pool.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Extract bio from nested structure
                    bio = data.get('lead', {}).get('profile', {}).get('bio', {}).get('text', '')
                    
                    # Convert Unix timestamp to datetime
                    created_at = datetime.fromtimestamp(data.get('created_at', 0))
                    
                    # SQL query
                    insert_query = """
                        INSERT INTO profiles (id, name, bio, follower_count, description, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE 
                        SET name = EXCLUDED.name,
                            bio = EXCLUDED.bio,
                            follower_count = EXCLUDED.follower_count,
                            description = EXCLUDED.description,
                            updated_at = CURRENT_TIMESTAMP
                        RETURNING id;
                    """
                    
                    # Values to insert
                    values = (
                        data.get('id', ''),
                        data.get('name', ''),
                        bio,
                        data.get('follower_count', 0),
                        data.get('description', ''),
                        created_at
                    )
                    
                    cursor.execute(insert_query, values)
                    inserted_id = cursor.fetchone()[0]
                    conn.commit()
                    
                    return {
                        "status": "success",
                        "message": f"Profile {inserted_id} inserted/updated successfully"
                    }
                    
        except psycopg2.Error as e:
            conn.rollback() if 'conn' in locals() else None
            return {
                "status": "error",
                "message": f"Database error: {str(e)}"
            }
        except Exception as e:
            conn.rollback() if 'conn' in locals() else None
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }
# Create profile manager instance
profile_manager = ProfileManager()
#%%


#%%
url = "https://api.pinata.cloud/v3/farcaster/channels"

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySW5mb3JtYXRpb24iOnsiaWQiOiJmNTBjNGEyYS0wNmM3LTQxNTQtYTc1My1iMWQyNGU5NDIzM2QiLCJlbWFpbCI6InRpdGlwYWtvcm4ucEBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwicGluX3BvbGljeSI6eyJyZWdpb25zIjpbeyJkZXNpcmVkUmVwbGljYXRpb25Db3VudCI6MSwiaWQiOiJGUkExIn0seyJkZXNpcmVkUmVwbGljYXRpb25Db3VudCI6MSwiaWQiOiJOWUMxIn1dLCJ2ZXJzaW9uIjoxfSwibWZhX2VuYWJsZWQiOmZhbHNlLCJzdGF0dXMiOiJBQ1RJVkUifSwiYXV0aGVudGljYXRpb25UeXBlIjoic2NvcGVkS2V5Iiwic2NvcGVkS2V5S2V5IjoiMDNmMzg1OTE3ZDFhMDJhMDhmNzQiLCJzY29wZWRLZXlTZWNyZXQiOiJlZTQ4MDgyMTA2NWZkMmUyMDc5OTgyMDBkN2Y2MjQ3NDZlYjEzNTcyNzdmYmY1NjE0MzRlNmIwNDZiZTRhYjQ5IiwiZXhwIjoxNzcwNTMxMzc4fQ.-uViBP8DJPeCl57yoO-x26BPpRJy9MhwY0O-ulPfKu0"

headers = {"Authorization": f"Bearer {token}"}

response = requests.request("GET", url, headers=headers)

channels = response.json()['channels']

#%%
for _ in range(100):
    for c in channels:
        profile_manager.insert_profile(c)
    next_page_token = response.json()['next']['cursor']

    params = {
        "pageSize": 200,
        "pageToken": next_page_token
    }
    response = requests.request("GET", url, headers=headers, params=params)

    channels = response.json()['channels']
#%%

params = {
    "pageSize": 200,
    "pageToken": next_page_token
}
response = requests.request("GET", url, headers=headers, params=params)

channels = response.json()['channels']