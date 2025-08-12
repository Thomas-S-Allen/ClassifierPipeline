import os
import json
import pickle
import zlib
import csv
import re

from google.protobuf.json_format import Parse, MessageToDict, ParseDict
from adsmsg import ClassifyRequestRecord, ClassifyRequestRecordList, ClassifyResponseRecord, ClassifyResponseRecordList

from adsputils import get_date, ADSCelery, u2asc
from adsputils import load_config, setup_logging


proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))

config = load_config(proj_home=proj_home)
logger = setup_logging('utilities.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', True))


def classify_record_from_scores(record):
    """
    Classify a scored record into one or more collections based on thresholds.

    Parameters
    ----------
    record : dict
        Dictionary containing at least the keys:
        ``categories`` (list[str]), ``scores`` (list[float]).
        Optionally includes metadata such as ``bibcode``, ``text``, ``validate``,
        and model information. Scores must be aligned with categories by index.

    Returns
    -------
    dict
        The input ``record`` augmented with:
        ``collections`` (list[str]) and ``collection_scores`` (list[float, rounded to 2 dp]).

    Notes
    -----
    - Thresholds are read from ``config['CLASSIFICATION_THRESHOLDS']`` and zipped
      against ``scores``.
    - If ``config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'] == 'active'`` and the
      "Other" category meets its threshold, the Earth Science score is compared to
      ``config['ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD']``; if exceeded, the
      "Other" selection is removed and "Earth Science" is added.
    """

    logger.info('Classify Record From Scores')
    logger.info('RECORD: {}'.format(record))
    # Fetch thresholds from config file
    thresholds = config['CLASSIFICATION_THRESHOLDS']
    logger.info(f'Classification Thresholds: {thresholds}')


    scores = record['scores']
    categories = record['categories']

    meet_threshold = [score > threshold for score, threshold in zip(scores, thresholds)]

    # Extra step to check for "Earth Science" articles miscategorized as "Other"
    # This is expected to be less neccessary with improved training data
    if config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'] == 'active':
        logger.info('Additional Earth Science Processing')
        if meet_threshold[categories.index('Other')] is True:
            # If Earth Science score above additional threshold
            if scores[categories.index('Earth Science')] \
                    > config['ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD']:
                meet_threshold[categories.index('Other')] = False
                meet_threshold[categories.index('Earth Science')] = True

    record['collections'] = [category for category, threshold in zip(categories, meet_threshold) if threshold is True]
    record['collection_scores'] = [score for score, threshold in zip(scores, meet_threshold) if threshold is True]
    record['collection_scores'] = [round(score, 2) for score in record['collection_scores']]

    return record


def prepare_output_file(output_path):
    """
    Create/overwrite a tab-delimited results file with a fixed header.

    Parameters
    ----------
    output_path : str or os.PathLike
        Target file path. Existing file is truncated.

    Side Effects
    ------------
    Writes a single header row with columns:
    ``['bibcode','scix_id','title','abstract','run_id','categories','scores',
    'collections','collection_scores','earth_science_adjustment','override']``.
    """ 

    logger.info('Preparing output file - utilities.py')

    header = ['bibcode','scix_id','title','abstract','run_id','categories','scores','collections','collection_scores','earth_science_adjustment','override']

    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(header)
    logger.info(f'Prepared {output_path} for writing.')


def check_is_allowed_category(categories_list):
    """
    Check that all requested categories are present in the configured allowlist.

    Parameters
    ----------
    categories_list : list[str]
        Categories to validate (case-insensitive).

    Returns
    -------
    bool
        ``True`` iff every element of ``categories_list`` appears in
        ``config['ALLOWED_CATEGORIES']`` (case-insensitive).
    """

    logger.info(f"Cheking allowed categories for {categories_list}")
    allowed = config.get('ALLOWED_CATEGORIES')
    allowed = [s.lower() for s in allowed]

    categories_list = [s.lower() for s in categories_list]

    result = [element in allowed for element in categories_list]

    logger.info(f"Cheking allowed categories for (after lowercase) {categories_list}")
    # Only return True if all True
    if sum(result) == len(result):
        return True
    else:
        return False

def return_fake_data(record):
    """
    Populate a record with deterministic fake categories/scores and model metadata.

    Parameters
    ----------
    record : dict
        Mutable dictionary to augment.

    Returns
    -------
    dict
        The same dict with keys ``categories``, ``scores``, ``model``, and
        ``postprocessing`` added based on the current configuration.
    """

    
    logger.info('Retruning Fake data')

    record['categories'] = ["Astronomy", "Heliophysics", "Planetary Science", "Earth Science", "NASA-funded Biophysics", "Other Physics", "Other", "Text Garbage"]
    record['scores'] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    record['model'] = {'model' : config['CLASSIFICATION_PRETRAINED_MODEL'],
                       'revision' : config['CLASSIFICATION_PRETRAINED_MODEL_REVISION'],
                       'tokenizer' : config['CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER']}


    record['postprocessing'] = {'ADDITIONAL_EARTH_SCIENCE_PROCESSING' : config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'],
                                'ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD' : config['ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD'],
                                'CLASSIFICATION_THRESHOLDS' : config['CLASSIFICATION_THRESHOLDS']}


    return record


def filter_allowed_fields(input_dict, allowed_fields=None,response=False):
    """
    Filter keys to those supported by the protobuf schema helpers.

    Parameters
    ----------
    input_dict : dict
        Dictionary to filter.
    allowed_fields : set[str], optional
        Explicit allowlist; if omitted, a default set is chosen based on ``response``.
    response : bool, default False
        If ``True``, the response-oriented field set is used.

    Returns
    -------
    dict
        A new dict containing only allowed keys.
    """

    if allowed_fields is None:
        if response is False:
            allowed_fields = {'bibcode', 'scix_id', 'status', 'title', 'abstract', 
                            'operation_step', 'run_id', 'override', 'output_path', 
                            'scores', 'collections', 'collection_scores'}
        else:
            allowed_fields = {'bibcode', 'scix_id', 'status', 'collections'}

    return {key: value for key, value in input_dict.items() if key in allowed_fields}


def dict_to_ClassifyRequestRecord(input_dict):
    """
    Convert a Python dict into a ``ClassifyRequestRecord`` protobuf message.

    The input is first filtered via :func:`filter_allowed_fields`.

    Parameters
    ----------
    input_dict : dict
        Request fields.

    Returns
    -------
    ClassifyRequestRecord
        Protobuf message instance.
    """

    input_dict = filter_allowed_fields(input_dict)

    request_message = ClassifyRequestRecord()
    message = ParseDict(input_dict, request_message)
    return message


def list_to_ClassifyRequestRecordList(input_list):
    """
    Convert a list of request dicts into a ``ClassifyRequestRecordList`` message.

    Parameters
    ----------
    input_list : list[dict]
        Each element is filtered via :func:`filter_allowed_fields`.

    Returns
    -------
    ClassifyRequestRecordList
        Protobuf list message with ``classify_requests`` populated.
    """

    input_list = list(map(lambda d: filter_allowed_fields(d), input_list))

    logger.info(f'Created ClassifyResponseRecord message from list: {input_list}')

    request_list_dict = {
            'classify_requests' : input_list,
            # 'status' : 99
            }

    request_message = ClassifyRequestRecordList()
    message = ParseDict(request_list_dict, request_message)
    return message


def dict_to_ClassifyResponseRecord(input_dict):
    """
    Convert a response dict into a ``ClassifyResponseRecord`` protobuf message.

    Parameters
    ----------
    input_dict : dict
        Response fields (filtered to response-allowed keys).

    Returns
    -------
    ClassifyResponseRecord
        Protobuf message instance.
    """

    input_dict = filter_allowed_fields(input_dict, response=True)

    request_message = ClassifyResponseRecord()
    message = ParseDict(input_dict, request_message)
    return message


def list_to_ClassifyResponseRecordList(input_list):
    """
    Convert a list of response dicts into a ``ClassifyResponseRecordList`` message.

    Parameters
    ----------
    input_list : list[dict]
        Each element is filtered to response-allowed keys.

    Returns
    -------
    ClassifyResponseRecordList
        Protobuf list message with ``classifyResponses`` populated.
    """

    input_list = list(map(lambda d: filter_allowed_fields(d, response=True), input_list))

    response_list_dict = {
            'classifyResponses' : input_list,
            # 'status' : 99
            }

    logger.info(f"Dictionary for Response Message {response_list_dict}")
    response_message = ClassifyResponseRecordList()
    message = ParseDict(response_list_dict, response_message)
    return message


def list_to_output_message(input_list):
    """
    Convert a list of dictionaries into a ``ClassifyResponseRecordList`` for the
    master pipeline.

    Parameters
    ----------
    input_list : list[dict]
        Each item may include ``bibcode``, ``status``, and ``collections``.

    Returns
    -------
    ClassifyResponseRecordList
        Message where each element of ``classify_requests`` is populated.  
        **Caution:** The current implementation assigns ``collections`` to
        ``entry.status`` (overwriting ``status``), which is likely a bug.
    """

    message = ClassifyResponseRecordList()

    for item in input_list:
        entry = message.classify_requests.add()
        try:
            entry.bibcode = item.get('bibcode')
        except:
            entry.bibcode = None
        # try:
        #     entry.scix_id = item.get('scix_id')
        # except:
        #     entry.scix_id = None
        try:
            entry.status = item.get('status')
        except:
            entry.status = None
        try:
            entry.status = item.get('collections')
        except:
            entry.status = None

    return message
     

def classifyRequestRecordList_to_list(message):
    """
    Convert ``ClassifyRequestRecordList`` into a list of Python dicts.

    Parameters
    ----------
    message : ClassifyRequestRecordList
        Source message with ``classify_requests``.

    Returns
    -------
    list[dict]
        One dict per request, via ``MessageToDict(..., preserving_proto_field_name=True)``.
    """

    logger.info(f'Converting message to list: {message}')
    output_list = []
    request_list = message.classify_requests
    for request in request_list:
        logger.info(f'Unpacking request: {request}')
        output_list.append(MessageToDict(request,preserving_proto_field_name=True))

    logger.info(f'Output list from message: {output_list}')

    return output_list


def check_identifier(identifier):
    """
    Determine whether an identifier is a SciX ID (``scix:`` form) or a bibcode.

    Parameters
    ----------
    identifier : str
        Candidate identifier. Must be length 19 to be considered.

    Returns
    -------
    {"scix_id", "bibcode"} or None
        ``"scix_id"`` if it matches ``scix:XXXX-XXXX-XXXX`` (alphanumeric groups),
        ``"bibcode"`` if length is 19 but not a SciX pattern, else ``None``.
    """

    identifier = str(identifier)

    if len(identifier) != 19:
        return None
    scix_match_pattern = r'^scix:[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}$' 
    if re.match(scix_match_pattern, identifier) is not None:
        return 'scix_id'
    else:
        return 'bibcode'



