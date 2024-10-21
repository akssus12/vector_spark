from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, DoubleType, StringType
from pyspark.sql import functions as F
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import chromadb
from chromadb.config import Settings
import datetime as dt
import time

spark_version = '3.5.1'
#os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.apache.kafka:kafka-clients:2.5.0:{}'.format(spark_version)

# consumer = KafkaConsumer('chroma_stream', bootstrap_servers=['163.239.199.209:9092'])
kafka_servers = '163.:9092'
topic = 'chroma'

spark = SparkSession \
        .builder \
        .appName("PysparkKafka") \
        .getOrCreate()
spark.sparkContext.setLogLevel('WARN')

df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_servers) \
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss","False") \
        .option("subscribe", topic) \
        .load()

#print(df.show())
sentence_df = df.selectExpr(
        "CAST(value AS STRING) as sentence",
        "timestamp"
        )
# sentence_df.printSchema()
@pandas_udf(ArrayType(StringType()))
def query_chromadb_udf(sentences_series: pd.Series, timestamps_series: pd.Series) -> pd.Series:

    client = chromadb.HttpClient(host='163.', port=0)
    collection = client.get_collection(name="wikipedia_huge1")
    existing_count = collection.count()
    # data = {'timestamp' : ts}
    # df = pd.DataFrame(data)
    # print("collection count : " + str(existing_count))

    results = []
    for sentence, ts in zip(sentences_series.tolist(), timestamps_series.tolist()): # sentence is normally given
        #print("parsed sentence : " + sentence)
        search_start=time.time()
        query_result = collection.query(
            query_texts=[sentence],
            n_results=5
        )
        search_end=time.time()
        search_time=search_end-search_start
        # print("query results : " + str(query_result))
        similar_sentences = query_result['documents'][0] if query_result['documents'] else []
        print(f"Searching Latency: {search_time:.4f} seconds")
        print(f"Similar_sentences: {similar_sentences}")
        print(f"Timestamp: {ts}")
        results.append(similar_sentences)
    return pd.Series(results)

results_df = sentence_df.withColumn("similar_sentences", query_chromadb_udf(col("sentence"),col("timestamp")))
# results_df   .select("sentence", "similar_sentences")
query = results_df \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination()
