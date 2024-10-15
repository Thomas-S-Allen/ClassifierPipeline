import os
import json
import pickle
import zlib
import csv

from adsputils import get_date, ADSCelery, u2asc
from adsputils import load_config, setup_logging


proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))

config = load_config(proj_home=proj_home)
logger = setup_logging('utilities.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', True))


def prepare_output_file(output_path,tsv_output=True):
    """
    Prepares an output file
    """
    logger.info('Preparing output file - utilities.py')

    # header = 'bibcode,title,abstract,run_id,categories,scores,collections,collection_scores,earth_science_adjustment,override\n'
    header = ['bibcode','title','abstract','run_id','categories','scores','collections','collection_scores','earth_science_adjustment','override']

    with open(output_path, 'w', newline='') as file:
        if tsv_output:
            writer = csv.writer(file, delimiter='\t')
        else:
            writer = csv.writer(file)
        writer.writerow(header)
        # file.write('\t'.join(header) + '\n')
        logger.info(f'Prepared {output_path} for writing.')


def check_is_allowed_category(categories_list):
    """
    Check if provided categories are in list of approved categories

    Parameters
    ----------
    categories_list : list (required) list of categories to check if allowed

    Returns
    ----------
    True if all categories in approved
    """
    allowed = config.get('ALLOWED_CATEGORIES')

    categories_list = [s.lower() for s in categories_list]

    result = [element in allowed for element in categories_list]

    # Only return True if all True
    # May want ot revisit logig
    if sum(result) == len(result):
        return True
    else:
        return False

def return_fake_data(record):
# def score_record(self, record,fake_data=False):
    """
    Provide classification scores for a record using the following
        categories:
            0 - Astronomy
            1 - HelioPhysics
            2 - Planetary Science
            3 - Earth Science
            4 - Biological and Physical Sciences
            5 - Other Physics
            6 - Other
            7 - Garbage

    Parameters
    ----------
    records_path : str (required) (default=None) Path to a .csv file of records

    Returns
    -------
    records : dictionary with the following keys: bibcode, text,
                categories, scores, and model information
    """

    
    logger.info('Retruning Fake data')

    record['categories'] = ["Astronomy", "Heliophysics", "Planetary Science", "Earth Science", "NASA-funded Biophysics", "Other Physics", "Other", "Text Garbage"]
    record['scores'] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    record['model'] = {'model' : config['CLASSIFICATION_PRETRAINED_MODEL'],
                       'revision' : config['CLASSIFICATION_PRETRAINED_MODEL_REVISION'],
                       'tokenizer' : config['CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER']}


    record['postprocessing'] = {'ADDITIONAL_EARTH_SCIENCE_PROCESSING' : True,
                                'ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD' : 0.015,
                                'CLASSIFICATION_THRESHOLDS' : [0.06, 0.03, 0.04, 0.02, 0.99, 0.02, 0.02, 0.99]}


    return record


def extract_records_from_message(message):
    """
    Extract records from a message.

    Parameters
    ----------
    message - protobuff defined serialized message

    Returns
    -------
    record or list of records
    """
    parsed_message = json.loads(message)

    record = parsed_message['classifyRequests'][0]

    return record, parsed_message.copy()

def package_records_to_message(record_list, out_message=None):
    """
    Package records in a message.

    Parameters
    ----------
    list of records - can be single element list

    Returns
    -------
    message - protobuff defined serialized message
    """
    if not_out_message:
        # handle here
        pass
  
    out_message['classifyRequests'] = [record]
    return json.dumps(out_message)

    # if not delay_message:
