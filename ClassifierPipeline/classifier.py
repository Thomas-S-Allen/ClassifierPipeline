"""
Classifier Module for SciX Pipeline

This module defines the `Classifier` class that uses a fine-tuned AstroBERT model
to classify scientific text into predefined SciX categories.

Main Responsibilities:
    - Tokenize and preprocess input text
    - Handle long input via sliding window tokenization
    - Add required special tokens and padding
    - Perform batched model inference
    - Aggregate model outputs into category scores and final predictions

Dependencies:
    - torch
    - huggingface tokenizer and model via AstroBERTClassification
    - adsputils for logging and config

Model Assumptions:
    - A multi-label classification model with sigmoid outputs
    - Supports long input splitting and score aggregation
"""
import os
from torch import no_grad, tensor
from adsputils import load_config, setup_logging
from ClassifierPipeline.astrobert_classification import AstroBERTClassification
import ClassifierPipeline.perf_metrics as perf_metrics

proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
config = load_config(proj_home=proj_home)

logger = setup_logging('classifier.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', True))

class Classifier:
    """
    Encapsulates logic for SciX category classification using a fine-tuned AstroBERT model.

    Attributes:
        classifier (AstroBERTClassification): Model wrapper class
        tokenizer: Huggingface tokenizer instance
        model: PyTorch model instance
        labels (list[str]): List of category labels
        id2label (dict[int, str]): Mapping from index to label
        label2id (dict[str, int]): Mapping from label to index
    """

    def __init__(self):
        self.classifier = AstroBERTClassification()
        self.tokenizer = self.classifier.tokenizer
        self.model = self.classifier.model
        self.model_device = (self.classifier.runtime_metadata or {}).get("device", "cpu")
        self.labels = self.classifier.labels
        self.id2label = self.classifier.id2label
        self.label2id = self.classifier.label2id

    # split tokenized text into chunks for the model
    def input_ids_splitter(self, input_ids, window_size=510, window_stride=255):
        """
        Splits a long sequence of token IDs into overlapping chunks.

        Parameters:
            input_ids (list[int]): Token IDs from tokenizer
            window_size (int): Max size of each chunk
            window_stride (int): Overlap between consecutive chunks

        Returns:
            list[list[int]]: List of split token ID chunks
        """
            
        # int() rounds towards zero, so down for positive values
        # import pdb; pdb.set_trace()
        num_splits = max(1, int(len(input_ids)/window_stride))
        
        split_input_ids = [input_ids[i*window_stride:i*window_stride+window_size] for i in range(num_splits)]
        
        
        return(split_input_ids)


    def add_special_tokens_split_input_ids(self, split_input_ids, tokenizer):
        """
        Adds [CLS], [SEP], and [PAD] tokens to split token ID chunks.

        Parameters:
            split_input_ids (list[list[int]]): Chunks of token IDs
            tokenizer: Huggingface tokenizer with special token IDs

        Returns:
            list[list[int]]: Modified chunks with special tokens and padding
        """
        
        # add start and end
        split_input_ids_with_tokens = [[tokenizer.cls_token_id]+s+[tokenizer.sep_token_id] for s in split_input_ids]
        
        # add padding to the last one
        split_input_ids_with_tokens[-1] = split_input_ids_with_tokens[-1]+[tokenizer.pad_token_id 
                                                                           for _ in range(len(split_input_ids_with_tokens[0])-len(split_input_ids_with_tokens[-1]))]
        
        return(split_input_ids_with_tokens)

        
    def _emit_classifier_shape_metrics(self, run_id, context_id, shape_metrics):
        for name, value in shape_metrics.items():
            perf_metrics.emit_event(
                stage="classifier_batch_shape",
                run_id=run_id,
                context_id=context_id,
                record_id=None,
                duration_ms=float(value),
                extra={"name": name},
                config=config,
            )

    def batch_score_SciX_categories(
        self,
        list_of_texts,
        score_combiner='max',
        score_thresholds=None,
        window_size=510,
        window_stride=500,
        run_id=None,
        context_id=None,
        configured_record_batch_size=None,
        model_inference_batch_size=None,
    ):
        """
        Classifies each input text into SciX categories using the model.

        Parameters:
            list_of_texts (list[str]): Raw input text to classify
            score_combiner (str or function): Method for aggregating scores across chunks ('max', 'mean', or custom function)
            score_thresholds (list[float]): Threshold per category to include in results
            window_size (int): Token window size for splitting long inputs
            window_stride (int): Token stride for overlapping splits

        Returns:
            tuple:
                list[list[str]]: Predicted categories per input
                list[list[float]]: Raw category scores per input
        """
        
        logger.info(f'Classifying {len(list_of_texts)} records')
        
        # optimal default thresholds based on experimental results
        if score_thresholds is None:
            score_thresholds = [0.0 for _ in range(len(self.labels)) ]

        logger.debug('lists of texts')
        logger.debug('List of texts {}'.format(list_of_texts))
        
        configured_batch_size = configured_record_batch_size or len(list_of_texts)

        with perf_metrics.timed_profile(
            category="classifier_timing",
            name="tokenizer_call",
            run_id=run_id,
            context_id=context_id,
            record_id=None,
            extra={"configured_record_batch_size": configured_batch_size, "record_count": len(list_of_texts)},
            config=config,
        ):
            list_of_texts_tokenized_input_ids = self.tokenizer(list_of_texts, add_special_tokens=False)['input_ids']

        logger.debug('Tokenized input ids')
        logger.debug('List of texts tokenized input ids {}'.format(list_of_texts_tokenized_input_ids))

        
        # split
        with perf_metrics.timed_profile(
            category="classifier_timing",
            name="input_splitting",
            run_id=run_id,
            context_id=context_id,
            record_id=None,
            extra={"configured_record_batch_size": configured_batch_size, "record_count": len(list_of_texts)},
            config=config,
        ):
            list_of_split_input_ids = [self.input_ids_splitter(t, window_size=window_size, window_stride=window_stride) for t in list_of_texts_tokenized_input_ids]
        # Full list of text
        # list_of_split_input_ids = input_ids_splitter(list_of_texts_tokenized_input_ids, window_size=window_size, window_stride=window_stride)
        
        logger.debug('Split input ids')
        # add special tokens
        with perf_metrics.timed_profile(
            category="classifier_timing",
            name="special_token_padding",
            run_id=run_id,
            context_id=context_id,
            record_id=None,
            extra={"configured_record_batch_size": configured_batch_size, "record_count": len(list_of_texts)},
            config=config,
        ):
            list_of_split_input_ids_with_tokens = [self.add_special_tokens_split_input_ids(s, self.tokenizer) for s in list_of_split_input_ids]
        
        logger.debug('Split input ids with tokens')
        logger.debug('List of split input ids with tokens {}'.format(list_of_split_input_ids_with_tokens))

        chunk_counts = [len(split_input_ids) for split_input_ids in list_of_split_input_ids]
        total_chunks = sum(chunk_counts)
        max_chunks = max(chunk_counts) if chunk_counts else 0
        max_tokenized_length = max((len(token_ids) for token_ids in list_of_texts_tokenized_input_ids), default=0)
        max_row_widths = [
            max((len(row) for row in split_input_ids_with_tokens), default=0)
            for split_input_ids_with_tokens in list_of_split_input_ids_with_tokens
        ]
        row_token_counts = [
            sum(len(row) for row in split_input_ids_with_tokens)
            for split_input_ids_with_tokens in list_of_split_input_ids_with_tokens
        ]
        pad_ratios = []
        for split_input_ids_with_tokens in list_of_split_input_ids_with_tokens:
            total_tokens = sum(len(row) for row in split_input_ids_with_tokens)
            pad_tokens = sum(
                1
                for row in split_input_ids_with_tokens
                for token_id in row
                if token_id == self.tokenizer.pad_token_id
            )
            pad_ratios.append((float(pad_tokens) / total_tokens) if total_tokens else 0.0)
        self._emit_classifier_shape_metrics(
            run_id,
            context_id,
            {
                "configured_record_batch_size": configured_batch_size,
                "model_inference_batch_size": 1,
                "effective_chunk_batch_size": 1,
                "total_chunks": total_chunks,
                "mean_chunks_per_record": (float(total_chunks) / len(chunk_counts)) if chunk_counts else 0.0,
                "max_chunks_per_record": max_chunks,
                "max_tokenized_length": max_tokenized_length,
                "padded_tensor_rows": max_chunks,
                "padded_tensor_cols": max(max_row_widths, default=0),
                "micro_batch_count": len(list_of_texts),
                "max_micro_batch_records": 1,
                "mean_micro_batch_records": 1.0 if list_of_texts else 0.0,
                "max_micro_batch_rows": max_chunks,
                "mean_micro_batch_rows": (float(total_chunks) / len(list_of_texts)) if list_of_texts else 0.0,
                "grouping_applied": 0.0,
                "mean_grouped_record_width": (float(sum(max_row_widths)) / len(max_row_widths)) if max_row_widths else 0.0,
                "mean_micro_batch_token_count": (float(sum(row_token_counts)) / len(row_token_counts)) if row_token_counts else 0.0,
                "max_micro_batch_token_count": max(row_token_counts, default=0),
                "mean_micro_batch_pad_ratio": (float(sum(pad_ratios)) / len(pad_ratios)) if pad_ratios else 0.0,
                "max_micro_batch_pad_ratio": max(pad_ratios, default=0.0),
            },
        )
        
        # list to return
        list_of_categories = []
        list_of_scores = []
        
        # forward call
        with no_grad():
            # for split_input_ids_with_tokens in tqdm(list_of_split_input_ids_with_tokens):
            for split_input_ids_with_tokens in list_of_split_input_ids_with_tokens:
                # make predictions
                logger.debug('Making predictions')
                logger.debug('Predictions with model {}'.format(self.model))
                try:
                    logger.debug('Really making predictions')
                    with perf_metrics.timed_profile(
                        category="classifier_timing",
                        name="model_forward",
                        run_id=run_id,
                        context_id=context_id,
                        record_id=None,
                        extra={"configured_record_batch_size": configured_batch_size, "record_count": len(list_of_texts)},
                        config=config,
                    ):
                        input_tensor = tensor(split_input_ids_with_tokens)
                        if self.model_device and self.model_device != "cpu":
                            input_tensor = input_tensor.to(self.model_device)
                        predictions = self.model(input_ids=input_tensor)
                        predictions = predictions.logits.sigmoid()
                except Exception as e:
                    logger.exception(f'Failed with: {str(e)}')
                    raise e

                logger.debug('Predictions {}'.format(predictions))
                
                logger.debug('COmbining predictions')
                # combine into one prediction
                with perf_metrics.timed_profile(
                    category="classifier_timing",
                    name="post_sigmoid_aggregation",
                    run_id=run_id,
                    context_id=context_id,
                    record_id=None,
                    extra={"configured_record_batch_size": configured_batch_size, "record_count": len(list_of_texts)},
                    config=config,
                ):
                    if score_combiner=='mean':
                        prediction = predictions.mean(dim=0)
                    elif score_combiner=='max':
                        prediction = predictions.max(dim=0)[0]
                    else:
                        # should be a custom lambda function
                        prediction = score_combiner(predictions)
                

                logger.debug('Appending predictions')
                list_of_scores.append(prediction.tolist())
                # filter by scores above score_threshold

                logger.debug('Appending categories')
                list_of_categories.append([self.id2label[index] for index,score in enumerate(prediction) if score>=score_thresholds[index]])
        
        logger.debug('Ran forward call')
        return(list_of_categories, list_of_scores)
        
