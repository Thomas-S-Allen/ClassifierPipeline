import os
import sys
import json
import requests
from urllib.parse import urlencode, quote_plus
from time import perf_counter, time
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import fbeta_score, precision_score, recall_score, precision_recall_fscore_support

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from batch_assign_SciX_categories import batch_assign_SciX_categories
# from astrobert.finetuning_2_seq_classification import batch_assign_SciX_categories
# from astrobert.finetuning_2_seq_classification import article_assign_SciX_categories
from adsputils import setup_logging, load_config

# import pdb;pdb.set_trace()

config_dict = load_config(proj_home=os.path.realpath(os.path.join(os.path.dirname(__file__))))#, '.')
 
# def parse_inputs():
#     '''Parse and error check input for nearest_buildings function'''
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config_file', required=True,\
#                         help='Path to config file')
#     args = parser.parse_args()
#     args_dict = vars(args)

    # Check inputs

    # Check Tables
    # try:
    #     config_dict = load_config(args_dict['config_file'])
    # except:
    #     print('Loading default config file')
    #     config_dict = load_config(proj_home=os.path.realpath(os.path.join(os.path.dirname(__file__))))#, '.')

    # return args_dict


# args_dict = parse_inputs()

# Now save the config dict to the following
def save_config_dict(config_dict):
    """Save config dict to json file"""

    config_dict['API_TOKEN'] = ''
    config_filename = f'config.{time()}.json'

    output_file = os.path.join(config_dict['CONFIG_DIR'], config_filename) 

    with open(output_file, 'w') as f:
        json.dump(config_dict, f, indent=4)

save_config_dict(config_dict)
# import pdb;pdb.set_trace()

def classify_sample(return_df=False):
    """Classify a sample of text for comparison with ground truth"""

    # Previously generated and classified data
    # df_full = pd.read_csv(config_dict['DATA_FULL_SAMPLE'])
    # df_truth = pd.read_csv(config_dict['DATA_GROUND_TRUTH'])
    # df_truth = pd.read_csv(config_dict['DATA_GROUND_TRUTH_ALL'])
    df_in = pd.read_csv(config_dict['DATA_GROUND_TRUTH_ALL'])
    # remove the 'title' and 'abstract' columns from df_truth
    # df_truth = df_truth.drop(columns=['title', 'abstract'])

    # Open pickle file
    # df_full = pd.read_pickle(config_dict['DATA_FULL_SAMPLE'])
    # df_in = pd.read_pickle(config_dict['DATA_GROUND_TRUTH_ALL_PICKLE'])

    # Open JSON file
    # df_in = pd.read_json(config_dict['DATA_GROUND_TRUTH_ALL_JSON'])

    # Read JSON file into a dictionary
    # with open(config_dict['DATA_GROUND_TRUTH_ALL_JSON']) as f:
    #     data = json.load(f)

    # Now convert the dictionary to a dataframe
    # df_in = pd.DataFrame.from_dict(data)
    
    # import pdb;pdb.set_trace()
    # df_in = df_truth.merge(df_full, on='bibcode', how='left')
    # 3 for intital test
    # nn = 3
    # test_bibs = list(df_in['bibcode'].values[:nn])
    # test_text = list(df_in['abstract'].values[:nn])
    # test_title = list(df_in['title'].values[:nn])
    test_bibs = list(df_in['bibcode'].values[:])
    test_abstract = list(df_in['abstract'].values[:])
    test_title = list(df_in['title'].values[:])
    # test_title = [title for title in test_title]
    # import pdb;pdb.set_trace()

    # Ordered list of Categories
    cat_list = ['Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science', 'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage']
    

    # Initialize lists
    list_of_categories = []
    list_of_scores = []
    list_of_AST = []
    list_of_Helio = []
    list_of_Planet = []
    list_of_Earth = []
    list_of_Bio = []
    list_of_Phys = []
    list_of_Other = []
    list_of_Garbage = []

    # Want to load the tokenizer and model once
    # tokenizer = AutoTokenizer.from_pretrained(config_dict['CLASSIFICATION_PRETRAINED_MODEL'])
    # model = AutoModelForSequenceClassification.from_pretrained(config_dict['CLASSIFICATION_PRETRAINED_MODEL'])

    
    # tokenize the text
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config_dict['CLASSIFICATION_PRETRAINED_MODEL'],
                                              revision=config_dict['CLASSIFICATION_PRETRAINED_MODEL_REVISION'],
                                              do_lower_case=False)
    
    # load model
    labels = ['Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science', 'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage']
    id2label = {i:c for i,c in enumerate(labels) }
    label2id = {v:k for k,v in id2label.items()}
    if config_dict['PUBLISHED_MODEL'] is True:
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=config_dict['CLASSIFICATION_PRETRAINED_MODEL'],
                                        revision=config_dict['CLASSIFICATION_PRETRAINED_MODEL_REVISION'],
                                        num_labels=len(labels),
                                        problem_type='multi_label_classification',
                                        id2label=id2label,
                                        label2id=label2id
                                        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=config_dict['CLASSIFICATION_PRETRAINED_MODEL_UNPUBLISHED'],
                                        revision=config_dict['CLASSIFICATION_PRETRAINED_MODEL_REVISION'],
                                        num_labels=len(labels),
                                        problem_type='multi_label_classification',
                                        id2label=id2label,
                                        label2id=label2id
                                        )
    

    print('Tokenizer and model loaded')
    print(tokenizer)
    print(model)

    # EnumeratedlLoop through each sample and assign categories
    # for bib, text in zip(test_bibs, test_text):
    for index, (bib, title, abstract) in enumerate(zip(test_bibs, test_title, test_abstract)):
        print()
        print("Text number: ", index)
        print(bib)
        print(f'Title: {title}')
        print(f'Abstract: {abstract}')
        # print(text)
        # import pdb;pdb.set_trace()

        #check if abstract is nan
        if pd.isnull(abstract):
            abstract = ''

        # Check CLASSIFICATION_INPUT_TEXT for input text
        if config_dict['TEST_LABELS']:
            if config_dict['CLASSIFICATION_INPUT_TEXT'] == 'title':
                text = f"Title: {str(title)}"
            elif config_dict['CLASSIFICATION_INPUT_TEXT'] == 'abstract':
                text = f"Abstract: {str(abstract)}"
            elif config_dict['CLASSIFICATION_INPUT_TEXT'] == 'title abstract':
                text = f"Title: {str(title)} \nAbstract: {str(abstract)}"
        else:
            if config_dict['CLASSIFICATION_INPUT_TEXT'] == 'title':
                text = f"{str(title)}"
            elif config_dict['CLASSIFICATION_INPUT_TEXT'] == 'abstract':
                text = f"{str(abstract)}"
            elif config_dict['CLASSIFICATION_INPUT_TEXT'] == 'title abstract':
                text = f"{str(title)} {str(abstract)}"

        print(f'Text: {text}')

        # import pdb;pdb.set_trace()

        # CLASSIFICATION_PRETRAINED_MODEL = 'adsabs/astroBERT'
        # CLASSIFICATION_PRETRAINED_MODEL_REVISION = 'SciX-Categorizer'
        # Assign categories
        tmp_categories, tmp_scores = batch_assign_SciX_categories([text],tokenizer,model,labels,id2label,label2id)

        # import pdb;pdb.set_trace()
        tmp_categories = tmp_categories[0]
        tmp_scores = tmp_scores[0]
        print(tmp_categories)
        print(tmp_scores)
        # import pdb;pdb.set_trace()

        # Append to lists
        list_of_categories.append(tmp_categories)
        list_of_scores.append(tmp_scores)
        list_of_AST.append(tmp_scores[0])
        list_of_Helio.append(tmp_scores[1])
        list_of_Planet.append(tmp_scores[2])
        list_of_Earth.append(tmp_scores[3])
        list_of_Bio.append(tmp_scores[4])
        list_of_Phys.append(tmp_scores[5])
        list_of_Other.append(tmp_scores[6])
        list_of_Garbage.append(tmp_scores[7])


    score_dict = {'bibcode': test_bibs,
                'category': list_of_categories,
                'score': list_of_scores,
                'new score AST': list_of_AST,
                'new score Helio': list_of_Helio,
                'new score Planetary': list_of_Planet,
                'new score Earth': list_of_Earth,
                'new score BPS': list_of_Bio,
                'new score Other PHY': list_of_Phys,
                'new score Other': list_of_Other,
                'new score Garbage': list_of_Garbage
                }

    score_df = pd.DataFrame(score_dict)

    
    # Join df_in and score_df on bibcode
    # import pdb;pdb.set_trace()
    df_out = df_in.merge(score_df, on='bibcode', how='left')
    df_out.to_csv(config_dict['DATA_SAMPLE_CLASSIFIED_NEW'], index=False)

    if return_df:
        return df_out


def relabel_categorical_categories(df, column='primaryClass'):
    """Rename categories in selected column of dataframe"""

    # import pdb;pdb.set_trace()
    mapping = {'Biology': 'NASA-funded Biophysics', 'FALSE': 'Other'}#, 'False': 'Other'}
    # mapping = {'Biology': 'BPS', 'FALSE': 'Other', 'False': 'Other'}

    # if there are NaNs in the column, replace them with 'FALSE'j                                       
    df[column] = df[column].fillna('False')
    # import pdb;pdb.set_trace()

    df[column] = df[column].astype('category')
    df[column] = df[column].cat.rename_categories(mapping)

    # import pdb;pdb.set_trace()

    return df

def short2cononical(df, column='primaryClass'):
    """Rename categories in selected column of dataframe"""

    mapping = {'BPS':'', 'FALSE': 'Other', 'False': 'Other'}

def plot_boxplot_category(df, cat, categories, short_categories, index,new_scores=True, column='primaryClass',show_plot=False):
    """Plot boxplot of scores for each category"""

    # categories = ['Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science', 'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage']
    # short_categories = ['AST', 'Helio', 'Planetary', 'Earth', 'BPS', 'Other PHY', 'Other', 'Text Garbage']
    print()
    print("Beginning category: ", cat)
    # import pdb;pdb.set_trace()

    print(f"Score Data from {config_dict['DATA_SAMPLE_CLASSIFIED_NEW']}")

    # Use a list comprehension to create a list of the new score categories from the short categories
    new_score_categories = [f'new score {cat}' for cat in short_categories]
    # Now do the same for the score categories
    score_categories = [f'score {cat}' for cat in short_categories]
    # keep_categories = ['bibcode'] + score_categories + new_score_categories
    if new_scores:
        keep_categories = ['bibcode'] + new_score_categories
    else:
        keep_categories = ['bibcode'] + score_categories

    short_cat = short_categories[index] 
    # import pdb;pdb.set_trace()
    df_cat = df[df[column] == cat]
    df_cat = df_cat[keep_categories]

    # if there are no papers in the category, skip it
    if len(df_cat) == 0:
        print(f'No papers in category {cat}')
        return None

    # df_cat = df_cat[keep_categories]
    # import pdb;pdb.set_trace()
    # Transform the current dataframe into a long form, whith one column for the score, one column for the new score, and one column for the category
    # df_cat_long = pd.melt(df_cat, id_vars=['primaryClass'], value_vars=[f'score {short_cat}', f'new score {short_cat}'])
    df_cat_long = pd.melt(df_cat, id_vars=['bibcode'], value_vars=keep_categories)
    # stack the two dataframes
    # df_cat_long = pd.concat([df_cat_long1, df_cat_long2])
    # dfns = df_cat_long[df_cat_long['variable'] == f'new score {short_cat}']
    # import pdb;pdb.set_trace()

    pretrained_model = config_dict['CLASSIFICATION_PRETRAINED_MODEL']
    pretrained_model = pretrained_model.replace('/', '_')

    xs = 14
    ys = 8

    fig, ax = plt.subplots(figsize=(xs, ys))
    plot_box_score = sns.boxplot(x='variable', y='value', data=df_cat_long,ax=ax)
    plot_box_score.set(title=f"Boxplot of scores for articles classified as {cat}\n{config_dict['CLASSIFICATION_INPUT_TEXT']}\n{config_dict['CLASSIFICATION_PRETRAINED_MODEL']} - {config_dict['CLASSIFICATION_PRETRAINED_MODEL_REVISION']}\n{column}")
    plot_filepath = f'{config_dict["BOXPLOT_SAVE_DIR"]}/{column}/boxplot_scores_{cat}_{column}_{config_dict["CLASSIFICATION_INPUT_TEXT"]}_{str(config_dict["TEST_THRESHOLDS"])}_{config_dict["TEST_THRESHOLDS_METHOD"]}_{config_dict["TEST_LABELS"]}_{pretrained_model}_{config_dict["CLASSIFICATION_PRETRAINED_MODEL_REVISION"]}.png'
    # plot_filepath = f'{config_dict["BOXPLOT_SAVE_DIR"]}boxplot_scores_{cat}_{column}_{config_dict["CLASSIFICATION_INPUT_TEXT"]}_{pretrained_model}_{config_dict["CLASSIFICATION_PRETRAINED_MODEL_REVISION"]}.png'
    # import pdb;pdb.set_trace()
    plt.savefig(plot_filepath)
    if show_plot:
        plt.show()
    plt.close()
    plt.clf()

# CLASSIFICATION_INPUT_TEXT = 'abstract' # 'title', 'abstract', 'body'
# CLASSIFICATION_PRETRAINED_MODEL = 'adsabs/astroBERT'
# CLASSIFICATION_PRETRAINED_MODEL_REVISION = 'SciX-Categorizer'
# Plots
    # fig, ax = plt.subplots(figsize=(xs, ys))
    # plot_box_newscore = sns.boxplot(x='variable', y='value', data=df_cat_long2,ax=ax)
    # plot_box_newscore.set(title=f'Boxplot of new scores for articles classified as {short_cat}')
    # plt.show()
    # plt.close()
    # plt.clf()

    print("Finished with category: ", cat)

def calculate_precision(ground_truth, model_predictions):
    """Calculate the precision for a given category

    Parameters
    ----------
    ground_truth : list

    model_predictions : list

    Returns
    -------
    precision : float
    """

    pass

def calculate_recall(ground_truth, model_predictions):
    """Calculate the recall for a given category

    Parameters
    ----------
    ground_truth : list

    model_predictions : list

    Returns
    -------
    recall : float
    """

    pass

def calculate_f_beta_score(ground_truth, model_predictions, beta=1):
    """Calculate the f_beta score for a given category

    Parameters
    ----------
    ground_truth : list

    model_predictions : list

    beta : float

    Returns
    -------
    f_beta_score : float
    """

    pass


def display_mtrics(df, truth_column = 'primaryClass', model_column='primaryCategory',multiple=False):
    """Display the precision, recall, and f_beta score for each category"""

    # let beta range from 0 to 1 in 0.1 increments
    # beta_range = np.arange(0.5, 2.2, 0.25)
    beta_range = np.array([0.5, 1.0, 2.0])

    average_mode = None

    for index, beta in enumerate(beta_range):

        if multiple is False:
            metrics_beta = fbeta_score(df[truth_column],
                                       df[model_column],
                                       beta=beta,
                                       average=average_mode,
                                       labels=categories,
                                       zero_division=np.nan)
            metrics_recall = recall_score(df[truth_column],
                                       df[model_column],
                                       average=average_mode,
                                       labels=categories,
                                       zero_division=np.nan)
            metrics_precision = precision_score(df[truth_column],
                                       df[model_column],
                                       average=average_mode,
                                       labels=categories,
                                       zero_division=np.nan)
            metrics_f1 = fbeta_score(df[truth_column],
                                       df[model_column],
                                       beta=1,
                                       average=average_mode,
                                       labels=categories,
                                       zero_division=np.nan)
        elif multiple is True:
            metrics_beta = fbeta_score(df[truth_column],
                                       df[model_column],
                                       beta=beta,
                                       average=average_mode,
                                       # labels=categories,
                                       zero_division=np.nan)
            metrics_recall = recall_score(df[truth_column],
                                       df[model_column],
                                       average=average_mode,
                                       # labels=categories,
                                       zero_division=np.nan)
            metrics_precision = precision_score(df[truth_column],
                                       df[model_column],
                                       average=average_mode,
                                       # labels=categories,
                                       zero_division=np.nan)
            metrics_f1 = fbeta_score(df[truth_column],
                                       df[model_column],
                                       beta=1,
                                       average=average_mode,
                                       # labels=categories,
                                       zero_division=np.nan)
        print()
        print(f'beta: {beta}')
        print(metrics_beta)
        # import pdb;pdb.set_trace()
        for i, cat in enumerate(categories):
            print(f'{cat} fbeta: {metrics_beta[i]:.2f}')
        for i, cat in enumerate(categories):
            print(f'{cat} recall: {metrics_recall[i]:.2f}')
        for i, cat in enumerate(categories):
            print(f'{cat} precision: {metrics_precision[i]:.2f}')

    # combine categories and metrics into a dataframe
    metrics_dict = {'category': categories,
                    'fbeta': metrics_beta,
                    'f1': metrics_f1,
                    'recall': metrics_recall,
                    'precision': metrics_precision}

    metrics_df = pd.DataFrame(metrics_dict)

    print('Dataframe')
    print(metrics_df)

    return metrics_df


def count_multiple_classes(df,class_columns=['primaryClass','secondaryClass'],model_columns=['primaryCategory']):

    # Select coluumns to join with long table after melt
    keep_cols = ['bibcode','title','abstract','new score AST', 'new score Helio',
                        'new score Planetary', 'new score Earth', 'new score BPS',
                        'new score Other PHY', 'new score Other', 'new score Text Garbage',
                        'primaryCategory', 'revised_category']

    df_to_merge = df[keep_cols]


    # loop through each row in the dataframe
    for index, row in df.iterrows():

        classes = [row['primaryClass']]
        if row['secondaryClass'] != "FALSE":
            classes.append(row['secondaryClass'])

        # print()
        # print(index)
        # print(classes)

    # make a long dataframe with one column that combines the primary and secondary classes
    df_long = pd.melt(df, id_vars=['bibcode'], value_vars=['primaryClass', 'secondaryClass'])

    df_long = df_long[df_long['value'] != 'FALSE']

    df_long = df_long.merge(df_to_merge, on='bibcode', how='left')

    return df_long

def threshold_scores(df,score_cols=['category','score'],ground_truth_cols=['primaryClass','secondaryClass'],threshold_deltas=None):

    print(df)

    keep_cols = ['bibcode'] + score_cols

    df_score = df[keep_cols]

    max_category = []
    max_score = []
    all_categories = []
    all_scores = []

    out_list = []

    general_threshold = config_dict['GENERAL_THRESHOLD']
    earth_science_tweak_threshold = config_dict['EARTH_SCIENCE_TWEAK_THRESHOLD']
    indices=None
    minor_class=None
    select_minor_class=None
    select_minor_score=None

    # if thresholds is None:
    thresholds = [general_threshold] * len(eval(df_score['score'].values[0]))

    astronomy_threshold_delta = config_dict['ASTRONOMY_THRESHOLD_DELTA']
    heliophysics_threshold_delta = config_dict['HELIOPHYSICS_THRESHOLD_DELTA']
    planetary_science_threshold_delta = config_dict['PLANETARY_SCIENCE_THRESHOLD_DELTA']
    earth_science_threshold_delta = config_dict['EARTH_SCIENCE_THRESHOLD_DELTA']
    biophysics_threshold_delta = config_dict['BIOPHYSICS_THRESHOLD_DELTA']
    other_physics_threshold_delta = config_dict['OTHER_PHYSICS_THRESHOLD_DELTA']
    other_threshold_delta = config_dict['OTHER_THRESHOLD_DELTA']
    garbage_threshold_delta = config_dict['GARBAGE_THRESHOLD_DELTA']

    if threshold_deltas is None:

        # threshold_deltas = [0] * len(eval(df_score['score'].values[0]))
        threshold_deltas = [astronomy_threshold_delta,
                            heliophysics_threshold_delta,
                            planetary_science_threshold_delta,
                            earth_science_threshold_delta,
                            biophysics_threshold_delta,
                            other_physics_threshold_delta,
                            other_threshold_delta,
                            garbage_threshold_delta]

    thresholds = np.array(thresholds) + np.array(threshold_deltas)
    # import pdb;pdb.set_trace()

    # loop through each row in the dataframe
    # each row is an article
    for index, row in df.iterrows():
        
        # Get the scores
        scores = list(eval(row['score']))
        categories = list(eval(row['category']))
        # find the index for the max score
        max_score_index = scores.index(max(scores))

        # Get the category
        max_category = categories[max_score_index]
        max_score = scores[max_score_index]
        # print(np.max(np.array(row['score'])))

        # find the indices of the scores that are greater than the threshold
        # indices = np.where(np.array(scores) > general_threshold)

        # Apply the general threshold
        # indices = np.argwhere(np.array(scores) > general_threshold)
        indices = np.argwhere(np.array(scores) > thresholds)
        indices = [i[0] for i in indices]
        # import pdb;pdb.set_trace()

        if len(indices) > 0:

            # Remove the max score index from the indices   
            minor_class = set(indices) - set([max_score_index])
            select_minor_class = [categories[i] for i in minor_class]
            select_minor_score = [scores[i] for i in minor_class]


        # import pdb;pdb.set_trace()

        if indices is not None:

            select_categories = [categories[i] for i in indices]
            select_scores = [scores[i] for i in indices]

        # print(index)
        # print(indices2)
        # print(select_categories)
        # print(select_scores)

        # If Max category is "Other" then check for Earth Science first
        # Using the earth science threshold
        add_es_tweak = config_dict['ADD_EARTH_SCIENCE_TWEAK']
        if add_es_tweak:
            # If "Other" is the max category
            if max_category == 'Other':
                es_score = scores[categories.index('Earth Science')]
                if es_score > earth_science_tweak_threshold:
                    max_category = 'Earth Science'
                    max_score = es_score
            # If "Other" is in select categories
            elif 'Other' in select_categories:
                es_score = scores[categories.index('Earth Science')]
                if es_score > earth_science_tweak_threshold:
                    select_categories[select_categories.index('Other')] = 'Earth Science'
                    select_scores[select_categories.index('Earth Science')] = es_score

        out_list.append({'bibcode': row['bibcode'],
                         'max_category': max_category,
                         'max_score': max_score,
                         'select_categories': select_categories,
                         'select_scores': select_scores})


    df_out = pd.DataFrame(out_list)
        # import pdb;pdb.set_trace()

    return df_out,thresholds


    # import pdb;pdb.set_trace()

def match_metrics(df,column='match_max_primary',text='Max score, Primary match only'):


    print()
    print(f'Counts for {text}')
    df_counts = df.groupby(column).size().reset_index(name='counts')
    print(df_counts)

    # import pdb;pdb.set_trace()
    try:
        false = df_counts[df_counts[column] == False]['counts'].values[0]
    except:
        false = 0
    try:
        true = df_counts[df_counts[column] == True]['counts'].values[0]
    except:
        true = 0

    # import pdb;pdb.set_trace()

    # For each category calculate percent correct
    try:
        percent_correct = true / (true + false)
    except:
        percent_correct = 0

    print(f'Percent correct for {text}: {percent_correct:.2f}')

def multiclass_metrics(df, categories=['Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science', 'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage']):

    gt_list = []
    model_list = []
    gt_names_list = []
    model_names_list = []

    # replace all instances of "Biology" with 'NASA-funded Biophysics' in the column 'secondaryClass'
    df['secondaryClass'] = df['secondaryClass'].replace('Biology', 'NASA-funded Biophysics')

    # loop through dataframe
    for index, row in df.iterrows():

        ind1 = None
        ind2 = None

        # Get index of primaryClass in categories
        ind1 = categories.index(row['primaryClass'])

        if row['secondaryClass'] != 'FALSE':
            gt_list.append([categories.index(row['primaryClass']), categories.index(row['secondaryClass'])])
            gt_names_list.append(set([row['primaryClass'], row['secondaryClass']]))
        else:
            gt_list.append([categories.index(row['primaryClass'])])
            gt_names_list.append(set([row['primaryClass']]))

        # Get index of model prediction in categories
        model_list.append([categories.index(i) for i in row['predicted_categories']])
        model_names_list.append(row['predicted_categories'])

        verbose = False
        if verbose:
            print()
            print(row['bibcode'])
            print('Ground Truth')
            print(row['primaryClass'])
            print(row['secondaryClass'])
            print('Ground Truth: ', gt_list[-1])
            print('Model')
            print(row['predicted_categories'])
            print('Model Prediction: ', model_list[-1])

    # Need to Clean all this up
    mlb = MultiLabelBinarizer()
    gt_list_bin =  list(mlb.fit_transform(gt_list))
    # Add a zero for the "Text Garbage" category
    # gt_list_bin = np.insert(gt_list_bin, 7, 0, axis=1)
    # import pdb;pdb.set_trace()
    gt_list_bin = [np.append(i,[0]) for i in gt_list_bin]
    model_list_bin = list(mlb.fit_transform(model_list))
    # Convert a list of arrays into an array of lists
    gt_list_bin = np.array([list(i) for i in gt_list_bin])
    model_list_bin = np.array([list(i) for i in model_list_bin])
    # import pdb;pdb.set_trace()
    # import pdb;pdb.set_trace()
    df_sk = pd.DataFrame({'bibcode': df['bibcode'].values,
                          'ground_truth_names': gt_names_list,
                          'model_prediction_names': model_names_list,
                          'ground_truth': gt_list,
                          'model_prediction': model_list,
                          'ground_truth_mlb' : list(gt_list_bin),
                          'model_prediction_mlb' : list(model_list_bin),
                          'ground_truth_strings' : [str(i) for i in gt_list],
                          'model_prediction_strings' : [str(i) for i in model_list]})

    df_sk.to_csv('df_sklearn_format.csv')


    # display_mtrics(df_sk, truth_column = 'ground_truth_mlb', model_column='model_prediction_mlb',multiple=True)


    # metrics_beta = fbeta_score(gt_list_bin,
    #                            model_list_bin,
    #                            beta=2,
    #                            average=None,
    #                            labels=categories,
    #                            zero_division=np.nan)

        # metrics_recall = recall_score(df[truth_column],
    metric_average = None
    # import pdb;pdb.set_trace()
    metrics_beta = fbeta_score(gt_list_bin,model_list_bin,beta=2,average=metric_average,zero_division=np.nan)
    metrics_f1 = fbeta_score(gt_list_bin,model_list_bin,beta=1,average=metric_average,zero_division=np.nan)
    metrics_recall = recall_score(gt_list_bin,model_list_bin,average=metric_average,zero_division=np.nan)
    metrics_precision = precision_score(gt_list_bin,model_list_bin,average=metric_average,zero_division=np.nan)
    # gg=gt_list_bin[0:3]
    # mm=model_list_bin[0:3]

    print('Metrics')
    print(categories)
    print(metrics_f1)
    print(metrics_beta)
    print(metrics_recall)
    print(metrics_precision)

    if metric_average == 'micro':

        print('Metrics')
        print(categories)
        print(metrics_f1)
        print(metrics_beta)
        print(metrics_recall)
        print(metrics_precision)

    # metrics_df = None
    if metric_average == None:

        for index, cat in enumerate(categories):
            print(f'{cat} fbeta: {metrics_beta[index]:.2f}')
            print(f'{cat} f1: {metrics_f1[index]:.2f}')
            print(f'{cat} recall: {metrics_recall[index]:.2f}')
            print(f'{cat} precision: {metrics_precision[index]:.2f}')

        metrics_dict = {'category': categories,
                        'fbeta': metrics_beta,
                        'f1': metrics_f1,
                        'recall': metrics_recall,
                        'precision': metrics_precision}

        metrics_df = pd.DataFrame(metrics_dict)
        print(metrics_df)

    if metrics_df is not None:
        return metrics_df

    # import pdb;pdb.set_trace()

    # first loop through and transform results into one-hot vector style

def loop_threshold(df, cat, step,categories=['Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science', 'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage']):

    in_range = list(np.arange(0,1,step))
    fbeta = []
    f1 = []
    recall = []
    precision = []

    cat_index = categories.index(cat)

    for index, val in enumerate(in_range):


        deltas = [0.0,  # Astronomy
                  0.0,  # Heliophysics
                  0.0,  # Planetary Science
                  0.0,  # Earth Science
                  0.0,  # Biophysics
                  0.0,  # Other Physics
                  0.0, # Other 
                  0.0] # Text Garbage

        deltas[cat_index] = deltas[cat_index] + val

        df_thresh,thresholds = threshold_scores(df,threshold_deltas=deltas)

        # join df and df_thresh on bibcode
        df_comb = df.merge(df_thresh, on='bibcode', how='left')

        df_comb['match_set'] = False
        df_comb['overlap_percent_model'] = 0.0
        df_comb['predicted_categories'] = None
        df_comb['false_positives'] = None

        # Compare model results to ground truth with multiple potential classes
        # loop through each row in the dataframe
        for index, row in df_comb.iterrows():
            if row['secondaryClass'] == 'FALSE':
                ground_truth = set([row['primaryClass']])
            else:
                ground_truth = set([row['primaryClass'], row['secondaryClass']])
            # model_predictions = set([eval(row['select_categories'])])
            model_predictions = set(row['select_categories'])

            lgt = len(ground_truth)
            lmp = len(model_predictions)

            match = ground_truth.intersection(model_predictions)
            lmt = len(match)

            if lmt > 0:
                df_comb.at[index, 'match_set'] = True


            # Calculate the percent overlap between the ground truth and model predictions
            # df_comb.at[index, 'overlap_percent_model'] = lmt / lmp
            df_comb.at[index, 'predicted_categories'] = model_predictions

            # look at False Positivies, i.e. model predicts a category that is not in the ground truth

            false_positives = model_predictions - ground_truth
            df_comb.at[index, 'false_positives'] = false_positives




            # print()
            # print(index)
            # print(ground_truth)
            # print(model_predictions)
            # print(match)


        # Calculate the proportion of records that get at least one corect identification
        # print('Counts of set matches - True/False')
        # df_counts = df.groupby('match_set').size().reset_index(name='counts')
        # print(df_counts)


        # import pdb;pdb.set_trace()
        match_metrics(df_comb,column='match_set',text='Matching sets using threshold scores')


        # if config_dict['EXPLORE_MULTI_CLASS'] is True and config_dict['EXPLORE_EARTH_SCIENCE'] is True:

        # print('Original Results')

        match_metrics(df_comb,column='match_max_primary',text='Max score, Primary match only')
        match_metrics(df_comb,column='match_max_primary_secondary',text='Max score, Primary or Secondary match')
        match_metrics(df_comb,column='match_max_primary_ES_Revised',text='Max score, Primary match only, ES revised')
        match_metrics(df_comb,column='match_set',text='Matching sets using threshold scores')


        # df_comb.to_csv('multi_class_test.csv')



        print('Multi Class Metrics')
        print(cat)
        print(val)
        # import pdb;pdb.set_trace()
        multi_metrics = multiclass_metrics(df_comb)
        # import pdb;pdb.set_trace()
        mm_cat = multi_metrics[multi_metrics['category'] == cat]

        # import pdb;pdb.set_trace()
        fbeta.append(mm_cat['fbeta'].values[0])
        f1.append(mm_cat['f1'].values[0])
        recall.append(mm_cat['recall'].values[0])
        precision.append(mm_cat['precision'].values[0])
        # print('Thresholds and Deltas')
        # print('General Threshold')
        # print(config_dict['GENERAL_THRESHOLD'])
        # print('Categories')
        # print(categories)
        # print('Thresholds')
        # print(thresholds)
        # print('Deltas')
        # print(deltas)
        # import pdb;pdb.set_trace()

    # plot the results
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(in_range, fbeta, label='Fbeta')
    ax.plot(in_range, recall, label='recall')
    ax.plot(in_range, precision, label='precision')
    ax.plot(in_range, f1, label='F1')
    ax.set(title=f'fbeta, recall, and precision for {cat} as a function of threshold delta',
              xlabel='Threshold Delta',
              ylabel='Score')
    ax.legend()
    plt.savefig(f'figures/fbeta_recall_precision_{cat}.png')
    plt.show()
    # plt.close()
    plt.clf()

     


if __name__ == "__main__":

    #List of categories
    categories = ['Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science', 'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage']
    short_categories = ['AST', 'Helio', 'Planetary', 'Earth', 'BPS', 'Other PHY', 'Other', 'Text Garbage']
    # score_categories = ['score AST', 'score Helio', 'score Planet', 'score Earth', 'score Bio', 'score Phys', 'score Other', 'score Garbage']
    # new_score_categories = ['new score AST', 'new score Helio', 'new score Planet', 'new score Earth', 'new score Bio', 'new score Phys', 'new score Other', 'new score Garbage']

    # Use a list comprehension to create a list of the new score categories from the short categories
    new_score_categories = [f'new score {cat}' for cat in short_categories]
    # Now do the same for the score categories
    score_categories = [f'score {cat}' for cat in short_categories]
    keep_categories = ['bibcode'] + score_categories + new_score_categories
    keep_categories1 = ['bibcode'] + score_categories
    keep_categories2 = ['bibcode'] + new_score_categories

    # Run sample classification or
    # Load previously generated and classified data
    if config_dict['RUN_SAMPLE_CLASSIFICATION'] == 'yes':
        # start the perf ounter
        t0 = perf_counter()
        # Run sample classification
        # Check config file for settings
        df = classify_sample(return_df=True)
        # stop the perf counter
        t1 = perf_counter()
        # calculate the elapsed time
        elapsed_time = t1 - t0
        print(f'Elapsed time: {elapsed_time} seconds')
    elif config_dict['RUN_SAMPLE_CLASSIFICATION'] == 'no':
        # Do not run sample classification
        # df = pd.read_csv(config_dict['DATA_SAMPLE_CLASSIFIED'])
        df = pd.read_csv(config_dict['DATA_SAMPLE_CLASSIFIED_NEW'])


    # import pdb;pdb.set_trace()
    # rename categories in df column 'primaryClass'
    df = relabel_categorical_categories(df, column='primaryClass')
    try:
        df = relabel_categorical_categories(df, column='secondaryClass')
    except:
        print('Secondary class not present in dataframe')

    # Now rename the new_score_Garbage column to new_score_Text_Garbage
    df = df.rename(columns={'new score Garbage': 'new score Text Garbage'})
    # Now rename Score Text Garbage to score Text Garbage
    df = df.rename(columns={'Score Text Garbage': 'score Text Garbage'})


    # Variables of interest
    # primaryClass secondareyClass score... new score...

    # Create summary table that shows the number of papers in each category of primaryClass
    df_summary_primary_class = df.groupby('primaryClass').size().reset_index(name='counts')
    df_summary_secondary_class = df.groupby('secondaryClass').size().reset_index(name='counts')

    print(df_summary_primary_class)
    print(df_summary_secondary_class)
    # df_summary_classes = df_summary_classes[['primaryClass']]

    # Test Thresholds

    # import pdb;pdb.set_trace()
    primary_category = []
    if config_dict['TEST_THRESHOLDS'] is True:

        # first start with just max score

        for index, row in df.iterrows():

            # print()
            # print(row)
            try:
                score = list(eval(row['score']))
            except:
                score = list(row['score'])
            # find the index for the max score
            max_score_index = score.index(max(score))
            # print(score)
            # print(max_score_index)
            if config_dict['TEST_THRESHOLDS_METHOD'] == 'max':
                primary_category.append(categories[max_score_index])

        df['primaryCategory'] = primary_category


    # import pdb;pdb.set_trace()
    # If not testing thresholds, then just use the default output from the classifier 
    # should be max
    # Create a new column in df called 'primaryCategory' that takes the first element from the list contained in the column 'category'
    else:
        try:
            df['category'] = df['category'].apply(eval)
        except:
            df['category'] = df['category']
        # change any emptly lists in the column 'category' to ['Other']
        df['category'] = df['category'].apply(lambda x: ['Other'] if len(x) == 0 else x)
        # import pdb;pdb.set_trace()
        df['primaryCategory'] = df['category'].apply(lambda x: x[0])
        # import pdb;pdb.set_trace()


    # Now calculate the precision, recall, and f_beta score for each category

    # First set of metrics based off of max score matching a single category

    metrics_micro_f1 = precision_recall_fscore_support(df['primaryClass'],
                                              df['primaryCategory'],
                                              average='micro',
                                              beta = 1.0,
                                              labels=categories,
                                              zero_division=np.nan)

    metrics_by_class = precision_recall_fscore_support(df['primaryClass'],
                                              df['primaryCategory'],
                                              average=None,
                                              beta = 1.0,
                                              labels=categories,
                                              zero_division=np.nan)

    print('Metrics micro F1: ', metrics_micro_f1)
    print('Metrics: ', metrics_by_class)

    # import pdb;pdb.set_trace()
    ############################
    # Plotting
    ############################

    # First lest plot the number of papers in each category
    if config_dict['SHOW_BARCHART_COUNTS_ALL']:
        sns.barplot(x='primaryClass', y='counts', data=df_summary_primary_class)
        plt.show()
        plt.close()
        plt.clf()

    # import pdb;pdb.set_trace()
    # Now lets loop through each category and create a boxplot of the scores
    if config_dict['MAKE_CATEGORY_BOXPLOTS']:

        for index, cat in enumerate(categories):

            plot_boxplot_category(df, cat, categories, short_categories, index,new_scores=True, column='primaryClass',show_plot=config_dict['SHOW_CATEGORY_BOXPLOTS'])
            plot_boxplot_category(df, cat, categories, short_categories, index,new_scores=True, column='secondaryClass',show_plot=config_dict['SHOW_CATEGORY_BOXPLOTS'])
            # plot_boxplot_category(df, cat, short_categories, keep_categories, index, column='primaryClass')
        
 
    metrics2 = fbeta_score(df['primaryClass'],
                           df['primaryCategory'],
                           beta=1.0,
                           average=None,
                           labels=categories,
                           zero_division=np.nan)

    # let beta range from 0 to 1 in 0.1 increments
    # beta_range = np.arange(0.5, 2.2, 0.25)
    beta_range = np.array([0.5, 1.0, 2.0])

    for index, beta in enumerate(beta_range):
        metrics_beta = fbeta_score(df['primaryClass'],
                                   df['primaryCategory'],
                                   beta=beta,
                                   average=None,
                                   labels=categories,
                                   zero_division=np.nan)

        metrics_recall = recall_score(df['primaryClass'],
                                   df['primaryCategory'],
                                   average=None,
                                   labels=categories,
                                   zero_division=np.nan)

        metrics_precision = precision_score(df['primaryClass'],
                                   df['primaryCategory'],
                                   average=None,
                                   labels=categories,
                                   zero_division=np.nan)
        print()
        print(f'beta: {beta}')
        print(metrics_beta)
        for i, cat in enumerate(categories):
            print(f'{cat} fbeta: {metrics_beta[i]:.2f}')
        for i, cat in enumerate(categories):
            print(f'{cat} recall: {metrics_recall[i]:.2f}')
        for i, cat in enumerate(categories):
            print(f'{cat} precision: {metrics_precision[i]:.2f}')

        # combine categories and metrics into a dataframe
        metrics_dict = {'category': categories,
                        'fbeta': metrics_beta,
                        'recall': metrics_recall,
                        'precision': metrics_precision}

        metrics_df = pd.DataFrame(metrics_dict)

        print('Dataframe')
        print(metrics_df)

    # Last metric just proportion of corrext matches - to compare with multiclass later

    df['match_max_primary'] = False
    df['match_max_primary_secondary'] = False
    for index, row in df.iterrows():
        if row['primaryClass'] == row['primaryCategory']:
            df.at[index, 'match_max_primary'] = True
        if row['primaryClass'] == row['primaryCategory'] or row['secondaryClass'] == row['primaryCategory']:
            df.at[index, 'match_max_primary_secondary'] = True


    match_metrics(df,column='match_max_primary',text='Max score, Primary match only')
    match_metrics(df,column='match_max_primary_secondary',text='Max score, Primary or Secondary match')

    # Examine Earth Science Results
    if config_dict['EXPLORE_EARTH_SCIENCE'] is True:

        print('Examine Earth Science Results')

        df_earth = df[df['primaryClass'] == 'Earth Science']

        # import pdb;pdb.set_trace()

        # plot a histogram of the scores for Earth Science
 
        xs = 14
        ys = 8

        fig, ax = plt.subplots(figsize=(xs, ys))
        plot_es_hist = sns.histplot(x='new score Earth', data=df_earth)

        plot_es_hist.set(title=f"Earth Science Scores for Records Hand Classified as Earth Science\n{config_dict['CLASSIFICATION_INPUT_TEXT']}\n{config_dict['CLASSIFICATION_PRETRAINED_MODEL']} - {config_dict['CLASSIFICATION_PRETRAINED_MODEL_REVISION']}")
        plt.savefig('figures/earth_science_hist.png')
        # plt.show()

        # Can sinply chaninging the threshold change the number of papers classified as Earth Science?
        # We will set a secondary threshold if record classes as "Other"
        df['revised_category'] = df['primaryCategory']
        ES_THRESHOLD = config_dict['EARTH_SCIENCE_TWEAK_THRESHOLD']

        df['match_max_primary_ES_Revised'] = False
        for index, row in df.iterrows():

            # import pdb;pdb.set_trace()
            if row['primaryCategory'] == 'Other':

                # Revise Earth Science category
                if row['new score Earth'] >= ES_THRESHOLD:
                    df.loc[index, 'revised_category'] = 'Earth Science'

            if row['primaryClass'] == row['revised_category']:
                df.at[index, 'match_max_primary_ES_Revised'] = True


        df_earth_revised = df[df['revised_category'] == 'Earth Science']

        print('Earth Science Revised - Max Score replace Other with ES if ES score > 0.015)')
        revised_metrics_df = display_mtrics(df, truth_column = 'primaryClass', model_column='revised_category')

        # import pdb;pdb.set_trace()

        # match_metrics(df,column='match_max_primary_ES_Revice',text='Max score, Primary match only')

    if config_dict['EXPLORE_MULTI_CLASS'] is True:

        print('Explore Multi Class Results')


        # df_long = count_multiple_classes(df,class_columns=['primaryClass','secondaryClass'])

        deltas = [-0.1,  # Astronomy
                  0.1,  # Heliophysics
                  0.1,  # Planetary Science
                  0.1,  # Earth Science
                  0.1,  # Biophysics
                  0.1,  # Other Physics
                  0.1, # Other 
                  0.1] # Text Garbage

        df_thresh,thresholds = threshold_scores(df,threshold_deltas=deltas)

        # join df and df_thresh on bibcode
        df_comb = df.merge(df_thresh, on='bibcode', how='left')

        df_comb['match_set'] = False
        df_comb['overlap_percent_model'] = 0.0
        df_comb['predicted_categories'] = None
        df_comb['false_positives'] = None

        # Compare model results to ground truth with multiple potential classes
        # loop through each row in the dataframe
        for index, row in df_comb.iterrows():
            if row['secondaryClass'] == 'FALSE':
                ground_truth = set([row['primaryClass']])
            else:
                ground_truth = set([row['primaryClass'], row['secondaryClass']])
            # model_predictions = set([eval(row['select_categories'])])
            model_predictions = set(row['select_categories'])

            lgt = len(ground_truth)
            lmp = len(model_predictions)

            match = ground_truth.intersection(model_predictions)
            lmt = len(match)

            if lmt > 0:
                df_comb.at[index, 'match_set'] = True


            # Calculate the percent overlap between the ground truth and model predictions
            df_comb.at[index, 'overlap_percent_model'] = lmt / lmp
            df_comb.at[index, 'predicted_categories'] = model_predictions

            # look at False Positivies, i.e. model predicts a category that is not in the ground truth

            false_positives = model_predictions - ground_truth
            df_comb.at[index, 'false_positives'] = false_positives




            # print()
            # print(index)
            # print(ground_truth)
            # print(model_predictions)
            # print(match)


        # Calculate the proportion of records that get at least one corect identification
        # print('Counts of set matches - True/False')
        # df_counts = df.groupby('match_set').size().reset_index(name='counts')
        # print(df_counts)


        match_metrics(df_comb,column='match_set',text='Matching sets using threshold scores')


    # if config_dict['EXPLORE_MULTI_CLASS'] is True and config_dict['EXPLORE_EARTH_SCIENCE'] is True:

        # print('Original Results')

    match_metrics(df_comb,column='match_max_primary',text='Max score, Primary match only')
    match_metrics(df_comb,column='match_max_primary_secondary',text='Max score, Primary or Secondary match')
    match_metrics(df_comb,column='match_max_primary_ES_Revised',text='Max score, Primary match only, ES revised')
    match_metrics(df_comb,column='match_set',text='Matching sets using threshold scores')


    df_comb.to_csv('multi_class_test.csv')

    print('Multi Class Metrics')
    multiclass_metrics(df_comb,categories)
    print('Thresholds and Deltas')
    print('General Threshold')
    print(config_dict['GENERAL_THRESHOLD'])
    print('Categories')
    print(categories)
    print('Thresholds')
    print(thresholds)
    print('Deltas')
    print(deltas)

    # loop_threshold(df, 'Astronomy', 0.05)
    for cat in categories:
        print(f'Looping through {cat}')
        # import pdb;pdb.set_trace()
        if cat not in ['NASA-funded Biophysics', 'Text Garbage']:
            loop_threshold(df, cat, 0.01)

    import pdb;pdb.set_trace()
    # df_comb.to_csv('first_multi_class.csv')
