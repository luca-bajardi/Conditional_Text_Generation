# Conditional Text Generation

The main problem of conditional text generation is that it is mainly based on the content of an input set of examples: this leads to little diversification of the generated text. To overcome this shortcoming, we have fine tuned CTRL using three different datasets. The first model has been used as a baseline for comparison, while the other two have been used to obtain more formal and informal text. The BART model has been employed for text classification to gauge formality.

## The dataset

We used 3 different dataset:

1. COCO captions from [COCO dataset](https://cocodataset.org/#captions-2015),
2. COCO captions mixed with *Wikipedia* articles,
3. COCO captions mixed with *Reddit* comments.

## The model

We used the [CTRL](http://arxiv.org/abs/1909.05858) model which can generate text conditioned on control codes that specify domain, style, topics, dates, entities, relationships between entities, plot points, and task-related behavior.

## Fine tuning

We dedicated ourselves to producing a CTRL model successfully fine-tuned on the COCO captions dataset. The aim was to obtain a model able to complete the initial part of a test caption in a convincing way.

Fine-tuning can be used to augment existing control codes or add new control codes. There are 7 steps elaborated in our paper:

1. Create the dataset
2. Split data
3. Convert this text data into TFRecords
4. Fine-tuning the model on these TFRecords files
5. Generation of the captions
6. Evaluate with metrics
7. Use Wikipedia articles and Reddit comments

### Step 1 - Create the dataset



### Step 2 - Split data

We used the script `split_data.py` to split the captions in two parts and divide the entire dataset in training set, validation set and test set.

```
python split_data.py
```

### Step 3 - Convert this text data into TFRecords

We use the file `make_tf_records.py` to convert the data into TFRecords.

```
python make_tf_records.py --text_file <text_file> --control_code caption --sequence_len 256
```

It has three arguments: `text_file` which specifies the name of the file to convert, `control_code` which specifies one token (must be in vocabulary) to append to each example, and `sequence_len` which specifies the sequence length to use to create the data. This must match the sequence length of the model being trained. 

### Step 4 - Fine-tuning the model on these TFRecords files

Simply run the script `training.py` 

```
python training.py --model_dir seqlen256_v1.ckpt/ --iterations <number_of_iterations>
```

The script picks up all TFRecords in the current folder and fine-tunes the model provided in the `--model_dir` flag. 

We fine tuned the model 3 different times, one with 250 iterations (that correspond to 21K captions), one with 500 iterations (42K captions) and one with 1000 iterations (84K captions).

We chose the model with 500 iterations.

### Step 5 - Generation of the captions

We use the file `generate_from_prompts.py` to generate the remaining parts of the captions.

```
python generate_from_prompts.py
```

### Step 6 - Evaluate with metrics

We use the file `metrics.py` to evaluate the generated text.

```
python metrics.py
```

### Step 7 - Use Wikipedia articles and Reddit comments

Using the model chosen in Step 4, we fine tuned the model similarly with *Wikipedia* articles and *Reddit* comments.

