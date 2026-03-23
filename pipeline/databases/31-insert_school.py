#!/usr/bin/env python3
""" Inserts a document in a collection """

def insert_school(mongo_collection, **kwargs):
    """ returns the new _id """
    return mongo_collection.insert_one(kwargs).inserted_id
