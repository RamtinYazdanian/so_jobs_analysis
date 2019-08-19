from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import Word2Vec
from utilities.common_utils import get_field, make_sure_path_exists
from utilities.text_utils import tokenise_stem_punkt_and_stopword
from utilities.constants import STOPWORDS, PUNKT
import pickle
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--posts', type=str, required=True)
    parser.add_argument('--dim', type=int, default=200)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    conf = SparkConf().set("spark.driver.maxResultSize", "10G"). \
        set("spark.hadoop.validateOutputSpecs", "false"). \
        set('spark.default.parallelism', '400')

    spark = SparkSession.builder.\
        appName("SO Word2Vec").\
        config(conf=conf).\
        getOrCreate()

    sc = spark.sparkContext

    in_rdd = sc.textFile(args.posts).filter(lambda x: get_field(x, 'Id') is not None).\
                                map(lambda x: (int(get_field(x, 'Id')), x))
    reduced_punkt = ''.join([x for x in PUNKT if x != '.'])
    in_rdd = in_rdd.filter(lambda x: get_field(x[1], 'Title') is not None and get_field(x[1], 'Body') is not None).\
        map(lambda x: (x[0],
        tokenise_stem_punkt_and_stopword(get_field(x[1], 'Title').decode('utf-8'), remove_numbers=True, remove_code=False,
                                 punkt_to_remove=reduced_punkt, remove_periods=True, stopword_set=STOPWORDS),
        tokenise_stem_punkt_and_stopword(get_field(x[1], 'Body').decode('utf-8'), remove_numbers=True, remove_code=True,
                                 punkt_to_remove=reduced_punkt, remove_periods=True, stopword_set=STOPWORDS)))
    in_rdd = in_rdd.map(lambda x: (x[0], x[1]+x[2]))

    sent_dataframe = in_rdd.toDF(['Id', 'Text'])
    w2vmodel = Word2Vec(vectorSize=args.dim, seed=42, inputCol="Text", outputCol="Model")
    model = w2vmodel.fit(sent_dataframe)

    vectors_df = model.getVectors().toPandas()
    print(vectors_df.head(20))

    make_sure_path_exists(args.output_dir)
    with open(os.path.join(args.output_dir, 'word_to_vec_df.pkl'), 'wb') as f:
        pickle.dump(vectors_df, f)

if __name__ == '__main__':
    main()


