#!/usr/bin/env python
"""
"""

# __author__ = 'rca'
# __maintainer__ = 'rca'
# __copyright__ = 'Copyright 2015'
# __version__ = '1.0'
# __email__ = 'ads@cfa.harvard.edu'
# __status__ = 'Production'
# __credit__ = ['J. Elliott']
# __license__ = 'MIT'

import os
import csv
# import sys
# import time
# import json
import argparse
# import logging
# import traceback
# import warnings
# from urllib3 import exceptions
# warnings.simplefilter('ignore', exceptions.InsecurePlatformWarning)

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# from adsputils import get_date
# from adsmsg import OrcidClaims
from ClassifierPipeline import classifier, tasks
# from ADSOrcid import updater, tasks
# from ADSOrcid.models import ClaimsLog, KeyValue, Records, AuthorInfo
from run import score_record, classify_record_from_scores, prepare_records
from harvest_solr import harvest_solr

# # ============================= INITIALIZATION ==================================== #

from adsputils import setup_logging, load_config
proj_home = os.path.realpath(os.path.dirname(__file__))
global config
config = load_config(proj_home=proj_home)
logger = setup_logging('run.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', False))

# app = tasks.app

# =============================== FUNCTIONS ======================================= #



# =============================== MAIN ======================================= #


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process user input.')


    parser.add_argument('-r',
                        '--records',
                        dest='records',
                        action='store',
                        help='Path to comma delimited list of new records' +
                             'to process: columns: bibcode, title, abstract')

    parser.add_argument('-s',
                        '--sort',
                        dest='sort',
                        action='store_true',
                        help='Sorts return list in ascending order according to' +
                             'the minimum collection score. Default is False.')

    args = parser.parse_args()


    if args.records:
        records_path = args.records
        out_path = records_path.replace('.csv', '_classified.csv')
        print(f'Reading in {records_path} and writing to {out_path}')
    else:
        print("Please provide a path to a .csv file with records to process.")
        exit()
        # Open .csv file and read in records
        # Convert records to send to classifier

    records = pd.read_csv(records_path)

    # Start with just bibcodes
    bibcodes = records['bibcode'].tolist()


    # Harvest Title and Abstract from Solr
    records = harvest_solr(bibcodes, start_index=0, fields='bibcode, title, abstract')

    # import pdb;pdb.set_trace()
    # Prepare records for classification
    # Check if just bibcodes or full records
    # if just bibcodes then get records from solr

    # Loop through records and classify
    for index, record in enumerate(records):
        record = score_record(record)
        record = classify_record_from_scores(record)

        # Get scores for elements in collections
        collection_scores = []
        for collection in record['collections']:

                # meet_threshold[categories.index('Earth Science')] = True
            collection_scores.append(record['scores'][record['categories'].index(collection)]) 

        # Update records
        record['collection_scores'] = collection_scores
        record['minimum_collection_score'] = min(collection_scores)
        records[index] = record

        
    # Convert records to a dataframe
    records = pd.DataFrame(records)

    # Sort if needed
    if args.sort is True:
        records = records.sort_values(by=['minimum_collection_score'], ascending=True)

    records.to_csv(out_path, index=False)



    print("Done")
    print(f"Results save to {out_path}")
