from openai import OpenAI
import datetime
import os

import pandas as pd
import numpy as np

from pgvector.sqlalchemy import Vector
import sshtunnel
import sqlalchemy
from sqlalchemy import Integer, String, TIMESTAMP, Float
from sqlalchemy.orm import DeclarativeBase, mapped_column, Session
from sentence_transformers import SentenceTransformer
client = OpenAI(
    api_key=""
)
with open("openai-prompt", "r") as f:
    recsys_llm_prompt = f.read()
def get_openai_context(prompt: str, chat_history: str) -> str:
    """Get context from OpenAI model."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": chat_history}
        ],
        temperature=1,
    )
    return response.choices[0].message.content



get_openai_context(recsys_llm_prompt, "pick shirts formal wear")

get_openai_context(recsys_llm_prompt, "What should I wear for my brother's wedding?")

get_openai_context(recsys_llm_prompt, "What is the price of item no. 3?")



ec2_url = ""

dbhost = ""
dbport = 1111
dbuser = ""
dbpass = ""

server = sshtunnel.SSHTunnelForwarder(
    ssh_address_or_host=(ec2_url, 22),
    ssh_username="",
    ssh_pkey="",
    remote_bind_address=(dbhost, dbport),
    local_bind_address=("localhost", 8008),
)

EMBEDDINGS_LENGTH = 768


class Base(DeclarativeBase):
    pass


class Products(Base):
    __tablename__ = "products"
    __table_args__ = {'extend_existing': True}

    pid = mapped_column(Integer, primary_key=True)
    pname = mapped_column(String)
    brand = mapped_column(String)
    gender = mapped_column(String)
    price = mapped_column(Float)
    n_images = mapped_column(Integer)
    description = mapped_column(String)
    color = mapped_column(String)
    embeddings = mapped_column(Vector(EMBEDDINGS_LENGTH))
    added_timestamp = mapped_column(TIMESTAMP)


model = SentenceTransformer("./embedding_model")
def generate_query_embeddings(user_message: str, embedding_model):
    """Generate user message embeddings."""
    openai_context = get_openai_context(recsys_llm_prompt, user_message)

    query_emb = embedding_model.encode(user_message + " " + openai_context)

    return query_emb

def query_product_names_from_embeddings(query_emb, engine, Table, top_k):
    """Search ANN products using embeddings."""
    with Session(engine) as session:
        stmt = sqlalchemy.select(
            Table.pid, Table.pname, Table.brand, Table.gender, Table.gender, Table.price, Table.description, Table.color
        ).order_by(Table.embeddings.l2_distance(query_emb)).limit(top_k)
        stmt_response = session.execute(stmt).mappings().all()

    return stmt_response

def get_recommendations(user_message: str, embedding_model, engine, Table, top_k=5):
    """Get recommendations."""
    embeddings = generate_query_embeddings(user_message, embedding_model)

    p_names = query_product_names_from_embeddings(embeddings, engine, Table, top_k)

    return p_names


server.start()

engine = sqlalchemy.create_engine(
    f"""postgresql+psycopg2://{dbuser}:"""
    f"""{dbpass}@{server.local_bind_host}:"""
    f"""{server.local_bind_port}/feature_store""",
    echo=False,
)
response = get_recommendations("pink strip shirts for men", model, engine, Products)
response
second_llm_prompt = (
    """
    You can recommendation engine chatbot agent for an Indian apparel brand.
    You are provided with users questions and some apparel recommendations from the brand's database.
    Your job is to present the most relevant items from the data give to you.
    If user is asking a clarifying question about one of the recommended item, like what is it's price or brand, then answer that question from its description.
    Do not answer anything else apart from apparel recommendation from the company's database.
    """
)

def get_openai_context(prompt: str, chat_history: str) -> str:
    """Get context from OpenAI model."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": chat_history}
        ],
        temperature=1,
    )
    return response.choices[0].message.content

user_query = "pink strip shirts for men"

recommendations = [
    {'pid': 10237350, 'pname': 'I AM FOR YOU Women Red & Off-White Checked Shirt Style Top', 'brand': 'I AM FOR YOU',
     'gender': 'Women', 'gender_1': 'Women', 'price': 662.0,
     'description': 'Red, off-white and teal blue checked woven shirt style top, has a shirt collar, long sleeves, short button placket, one pocket, curved hem',
     'color': ' Red'},
    {'pid': 10198533, 'pname': 'The Pink Moon Plus Size Women White  Blue Regular Fit Striped Casual Shirt',
     'brand': 'The Pink Moon', 'gender': 'Women', 'gender_1': 'Women', 'price': 1999.0,
     'description': 'White and blue striped casual shirt, has a spread collar, long sleeves, button placket, and curved hem, 2 insert pockets',
     'color': 'Blue'},
    {'pid': 10268637, 'pname': 'ONLY Women Red & White Regular Fit Printed Casual Shirt', 'brand': 'ONLY',
     'gender': 'Women', 'gender_1': 'Women', 'price': 919.0,
     'description': 'Red and white printed casual shirt, has a spread collar, long sleeves, button placket, curved hem,1 patch pocket',
     'color': ' Red'},
    {'pid': 10262399, 'pname': 'ONLY Women Navy Blue & Mustard Brown Regular Fit Striped Casual Shirt', 'brand': 'ONLY',
     'gender': 'Women', 'gender_1': 'Women', 'price': 1119.0,
     'description': 'Navy blue and mustard brown striped casual shirt, has a spread collar, long sleeves, concealed button placket, curved hem, and 2 flap pockets',
     'color': 'Blue'},
    {'pid': 10019431, 'pname': 'Raymond Men Pink Regular Fit Striped Formal Shirt', 'brand': 'Raymond', 'gender': 'Men',
     'gender_1': 'Men', 'price': 1649.0,
     'description': 'Pink striped formal shirt, has a spread collar, long sleeves, button placket, straight hem, and 1 patch pocket',
     'color': ' Pink'}]

get_openai_context(second_llm_prompt, f"User question = '{user_query}', our recommendations = {recommendations}")

server.stop()
engine.dispose()
E:\LLM\recsys-llm-chatbot\rec-llms.py
