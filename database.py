import os
from pymongo import MongoClient
from bson.objectid import ObjectId
import pandas as pd
from urllib.parse import quote_plus

import streamlit as st

client = None
db = None


def init_db():
    global client, db

    # Use st.secrets for the connection string when deployed on Streamlit Cloud
    # For local development, you can use environment variables
    import streamlit as st
    MONGO_URI = st.secrets["MONGO_URI"] if "MONGO_URI" in st.secrets else os.environ.get("MONGO_URI")

    if not MONGO_URI:
        st.error(
            "MongoDB connection string not found. Please set the MONGO_URI in your Streamlit secrets or environment variables.")
        st.stop()

    try:
        # Parse and reconstruct the connection string with properly encoded username and password
        prefix, rest = MONGO_URI.split("://", 1)
        credentials, hosts_and_options = rest.split("@", 1)
        username, password = credentials.split(":", 1)

        # Encode username and password
        encoded_username = quote_plus(username)
        encoded_password = quote_plus(password)

        # Reconstruct the connection string
        encoded_uri = f"{prefix}://{encoded_username}:{encoded_password}@{hosts_and_options}"

        client = MongoClient(encoded_uri)
        db = client.options_pricing_db

    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        st.stop()


def save_inputs(S, K, T, r, sigma, model):
    try:
        inputs = {
            "stock_price": S,
            "strike_price": K,
            "time_to_expiry": T,
            "risk_free_rate": r,
            "volatility": sigma,
            "model": model
        }
        result = db.inputs.insert_one(inputs)
        return result.inserted_id
    except Exception as e:
        st.error(f"Failed to save inputs: {str(e)}")
        return None


def save_outputs(input_id, volatility_shock, stock_price_shock, option_price, is_call, pnl):
    try:
        output = {
            "input_id": input_id,
            "volatility_shock": volatility_shock,
            "stock_price_shock": stock_price_shock,
            "option_price": option_price,
            "is_call": is_call,
            "pnl": pnl
        }
        db.outputs.insert_one(output)
    except Exception as e:
        st.error(f"Failed to save outputs: {str(e)}")


def get_past_calculations():
    try:
        pipeline = [
            {
                '$lookup': {
                    'from': 'outputs',
                    'localField': '_id',
                    'foreignField': 'input_id',
                    'as': 'outputs'
                }
            },
            {
                '$unwind': '$outputs'
            },
            {
                '$project': {
                    'stock_price': 1,
                    'strike_price': 1,
                    'time_to_expiry': 1,
                    'risk_free_rate': 1,
                    'volatility': 1,
                    'model': 1,
                    'option_price': '$outputs.option_price',
                    'is_call': '$outputs.is_call',
                    'pnl': '$outputs.pnl'
                }
            },
            {
                '$sort': {'_id': -1}
            },
            {
                '$limit': 1000
            }
        ]

        cursor = db.inputs.aggregate(pipeline)
        return pd.DataFrame(list(cursor))
    except Exception as e:
        st.error(f"Failed to retrieve past calculations: {str(e)}")
        return pd.DataFrame()

def get_latest_calculation_id():
    try:
        latest = db.inputs.find_one(sort=[("_id", -1)])
        return latest["_id"] if latest else None
    except Exception as e:
        st.error(f"Failed to retrieve latest calculation ID: {str(e)}")
        return None