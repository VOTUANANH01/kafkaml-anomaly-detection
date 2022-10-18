from ctypes.wintypes import PLARGE_INTEGER
import json
import os
from joblib import load
import logging
from multiprocessing import Process

import numpy as np

import sys
sys.path.append(r'F:\tai_lieu\Nam_4\semester 7th\Parallel and distributed computing\assignment\slide 5-6\Tutorial 10\kafkaml-anomaly-detection' )
from settings import TRANSACTIONS_TOPIC, TRANSACTIONS_CONSUMER_GROUP, ANOMALIES_TOPIC, NUM_PARTITIONS

from streaming.utils import create_producer, create_consumer

model_path = r'F:\tai_lieu\Nam_4\semester 7th\Parallel and distributed computing\assignment\slide 5-6\Tutorial 10\kafkaml-anomaly-detection\model\isolation_forest.joblib'


def detect():
    consumer = create_consumer(topic=TRANSACTIONS_TOPIC, group_id=TRANSACTIONS_CONSUMER_GROUP)

    producer = create_producer()

    clf = load(model_path)

    while True:
        message = consumer.poll(timeout=50)
        if message is None:
            continue
        if message.error():
            logging.error("Consumer error: {}".format(message.error()))
            continue

        # Message that came from producer
        record = json.loads(message.value().decode('utf-8'))
        data = record["data"]
        print(data)
        prediction = clf.predict(data)
        if prediction[0]==1:
            print("Normal")
        # If an anomaly comes in, send it to anomalies topic
        if prediction[0] == -1:
            print("Abonormal")
            score = clf.score_samples(data)
            record["score"] = np.round(score, 3).tolist()

            _id = str(record["id"])
            record = json.dumps(record).encode("utf-8")

            producer.produce(topic=ANOMALIES_TOPIC,
                             value=record)
            producer.flush()
            print(record)
            print("Alert sent!")

        # consumer.commit() # Uncomment to process all messages, not just new ones

    consumer.close()


# One consumer per partition
for _ in range(NUM_PARTITIONS):
    p = Process(target=detect)
    p.start()
if __name__ == "__main__":
    detect() 