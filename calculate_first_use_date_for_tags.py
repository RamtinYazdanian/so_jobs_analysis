from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrame
from utilities.common_utils import get_field, make_sure_path_exists
from utilities.text_utils import get_tags
from utilities.constants import DT_FORMAT
from datetime import datetime
import pickle
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--posts', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    conf = SparkConf().set("spark.driver.maxResultSize", "10G"). \
        set("spark.hadoop.validateOutputSpecs", "false"). \
        set('spark.default.parallelism', '400')

    spark = SparkSession.builder.\
        appName("SO Tag first usage date").\
        config(conf=conf).\
        getOrCreate()

    sc = spark.sparkContext

    in_rdd = sc.textFile(args.posts).filter(lambda x: get_field(x, 'Id') is not None).\
                                map(lambda x: (int(get_field(x, 'Id')), x))

    in_rdd = in_rdd.filter(lambda x: get_field(x[1], 'Tags') is not None).\
                    map(lambda x: (datetime.strptime(get_field(x[1], 'CreationDate'), DT_FORMAT),
                                   get_tags(get_field(x[1], 'Tags')))).\
                    flatMap(lambda x: [(x[0], y) for y in x[1]])

    tag_date_df = in_rdd.toDF(['CreationDate', 'Tag'])
    tag_first_appearance = tag_date_df.groupBy('Tag').agg({'CreationDate': 'min'})
    tag_first_appearance = tag_first_appearance.toPandas()

    make_sure_path_exists(args.output_dir)
    with open(os.path.join(args.output_dir, 'tag_earliest_appearance.pkl'), 'wb') as f:
        pickle.dump(tag_first_appearance, f)

if __name__ == '__main__':
    main()