#!/usr/bin/env python3
""" Changes all topics of a school document based on name """

def update_topics(mongo_collection, name, topics):
    """ Updates topics attribute """
    mongo_collection.update_many({"name": name}, {"$set": {"topics": topics}})
