
import datetime
import os

import pandas as pd
import numpy as np
# import psycopg2
import sshtunnel
import sqlalchemy
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer

DATA_DIR = "./data"
FILE_NAME = "data.csv"

raw_df = pd.read_csv(os.path.join(DATA_DIR, FILE_NAME))

raw_df.info()

raw_df.sample(3)

MODEL_PATH = "./embedding_model"

embedding_model = SentenceTransformer(MODEL_PATH)  # "all-mpnet-base-v2"

# embedding_model.save(MODEL_PATH)


product_name_embeddings = embedding_model.encode(raw_df["ProductName"].values)

EMBEDDING_FILE_NAME = "product_name_embeddings.pk"

import pickle

# with open("./product_name_embeddings.pk", "wb") as f:
#     pickle.dump(product_name_embeddings, f)

with open(os.path.join(DATA_DIR, EMBEDDING_FILE_NAME), "rb") as f:
    product_name_embeddings = pickle.load(f)

product_name_embeddings.shape

product_name_embeddings_dict = [
    {"ProductID": raw_df["ProductID"].values[i], "embeddings": product_name_embeddings[i]} for i in
    range(len(product_name_embeddings))
]

### Uploading data to feature store

df = raw_df.merge(
    pd.DataFrame(product_name_embeddings_dict),
    on="ProductID",
    how="left"
)

df.shape


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
    local_bind_address=("localhost", 5433),
)
# ** Table( or model) **

from pgvector.sqlalchemy import Vector
import sqlalchemy
from sqlalchemy.orm import DeclarativeBase, mapped_column
from sqlalchemy import Integer, String, TIMESTAMP, Float

EMBEDDINGS_LENGTH = df["embeddings"].iloc[0].shape[0]


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



server.start()

engine = sqlalchemy.create_engine(
    f"""postgresql+psycopg2://{dbuser}:"""
    f"""{dbpass}@{server.local_bind_host}:"""
    f"""{server.local_bind_port}/feature_store""",
    echo=False,
)

Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

#** Insert  data in table **

df.columns.tolist()

df_dict = [{"pid": rec["ProductID"], "pname": rec["ProductName"], "brand": rec["ProductBrand"], "gender": rec["Gender"],
            "price": rec["Price (INR)"], "n_images": rec["NumImages"], "description": rec["Description"],
            "color": rec["PrimaryColor"], "embeddings": rec["embeddings"], "added_timestamp": datetime.datetime.now()}
           for rec in df.to_dict(orient="records")]

assert (len(df_dict) == len(df))
len(df_dict)

with Session(engine) as session:
    session.execute(sqlalchemy.insert(Products), df_dict)
    session.commit()

with Session(engine) as session:
    stmt = sqlalchemy.text("SELECT count(*) from products;")
    stmt_response = session.scalars(stmt).first()

stmt_response

server.start()

engine = sqlalchemy.create_engine(
    f"""postgresql+psycopg2://{dbuser}:"""
    f"""{dbpass}@{server.local_bind_host}:"""
    f"""{server.local_bind_port}/feature_store""",
    echo=False,
)

query_emb = embedding_model.encode("dress for office party for women")

with Session(engine) as session:
    stmt = sqlalchemy.select(Products.pid).order_by(Products.embeddings.l2_distance(query_emb)).limit(10)
    stmt_response = session.scalars(stmt).all()

stmt_response

raw_df.loc[
    raw_df["ProductID"].isin(stmt_response),
    "ProductName"
].values

server.stop()
engine.dispose()
