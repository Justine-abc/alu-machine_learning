#!/usr/bin/env python3
""" Provides stats about Nginx logs in MongoDB """
from pymongo import MongoClient

if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    collection = client.logs.nginx
    
    # Total logs
    print("{} logs".format(collection.count_documents({})))
    
    # Methods
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        count = collection.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, count))
        
    # Status check
    status_check = collection.count_documents({"method": "GET", "path": "/status"})
    print("{} status check".format(status_check))
