#!/bin/bash

$SPARK_HOME/bin/spark-submit --master spark://163.:77 \
	--conf spark.executor.memory=36g \
	--conf spark.executor.cores=8 \
	--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1, \
	pyspark_chromadb.py
