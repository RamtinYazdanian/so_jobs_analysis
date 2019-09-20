from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col
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
    parser.add_argument('--num_top', type=int, default=1)
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

    in_rdd = in_rdd.filter(lambda x: get_field(x[1], 'Tags') is not None and get_field(x[1], 'CreationDate') is not None).\
                    map(lambda x: (datetime.strptime(get_field(x[1], 'CreationDate').decode('utf-8'), DT_FORMAT),
                                   get_tags(get_field(x[1], 'Tags').decode('utf-8')))).\
                    flatMap(lambda x: [(x[0], y) for y in x[1]])

    tag_date_df = in_rdd.toDF(['CreationDate', 'Tag'])
    window = Window.partitionBy(tag_date_df['Tag']).orderBy(tag_date_df['CreationDate'].asc())
    #tag_first_appearances = tag_date_df.groupBy('Tag').agg({'CreationDate': 'min'})
    tag_first_appearances = tag_date_df.select('*', rank().over(window).alias('rank')).\
                        filter(col('rank') <= args.num_top)
    tag_first_appearances_pd = tag_first_appearances.toPandas().drop(columns=['rank'])

    make_sure_path_exists(args.output_dir)
    with open(os.path.join(args.output_dir, 'tag_'+str(args.num_top)+'_earliest_appearance.csv'), 'w') as f:
        tag_first_appearances_pd.to_csv(f)

if __name__ == '__main__':
    main()