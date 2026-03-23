#!/usr/bin/env python3
""" Lists all documents in a collection """

def list_all(mongo_collection):
    """ returns list of all documents """
    return list(mongo_collection.find())
