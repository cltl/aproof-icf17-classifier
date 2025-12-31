"""
The script generates predictions of the level of functioning that is described in a clinical note in Dutch. The predictions are made for 9 WHO-ICF domains: 'ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'.

The script can be customized with the following parameters:
    --in_csv: path to input csv file
    --text_col: name of the column containing the text

To change the default values of a parameter, pass it in the command line, e.g.:

$ python main.py --in_csv myfile.csv --text_col notitie_tekst
"""

import os
import spacy
import argparse
import warnings
from pathlib import Path
from shutil import ReadError
from src.text_processing import anonymize_text
from src.icf_classifiers import load_model, predict_domains_for_sentences, predict_levels, predict_level_for_sentence, predict_level_for_domain_encoded_sentence
from src import timer
from statistics import mean

def add_level_predictions(
    sentences,
    dom_predictions,
    icf_levels,
    domains
):
    """
    For each domain, select the sentences in `sents` that were predicted as discussing this domain. Apply the relevant levels regression model to get level predictions and join them back to `sents`.

    Parameters
    ----------
    sents: list of sentences (text string)
    dom_predictions: list of domain predictions (list of strings) that apply to each sentence
    domains: list
        list of all the domains, in the order in which they appear in the multi-label

    Returns
    -------
    level_predictions_per_domain: list level predictions (list with floats) per domain
    """

    level_predictions_per_domain = []
    for i, dom in enumerate(domains):
        level_predictions =[]
        for sentence, dom_prediction in zip(sentences, dom_predictions):
            if dom_prediction[i]==1:
                #print(f'Generating levels predictions for {dom}.')
                level = predict_level_for_domain_encoded_sentence(sentence, icf_levels, dom)
                #print(f'For the sentence {sentence} we get this predictions {dom_prediction} and this level prediction {level}')
                level_predictions.append(level.item())
        level_predictions_per_domain.append(level_predictions)
    return level_predictions_per_domain

def add_level_predictions_no_domain(
            sentences,
            dom_predictions,
            icf_levels,
            domains,
    ):
        """
        For each domain, select the sentences in `sents` that were predicted as discussing this domain. Apply the relevant levels regression model to get level predictions and join them back to `sents`.

        Parameters
        ----------
        sents: list of sentences (text string)
        dom_predictions: list of domain predictions (list of strings) that apply to each sentence
        domains: list
            list of all the domains, in the order in which they appear in the multi-label

        Returns
        -------
        level_predictions_per_domain: list level predictions (list with floats) per domain
        """

        level_predictions_per_domain = []
        domain_level_dict={}
        for dom in domains[:-1]:
            domain_level_dict[dom] = []
        for sentence, dom_prediction in zip(sentences, dom_predictions):
            predicted_level = predict_level_for_sentence(sentence, icf_levels)
          #  print("For the sentence", sentence, "we get this level", predicted_level)
            for i, prediction in enumerate(dom_prediction[:-1]): ### Last item is NONE
                dom = domains[i]
                if prediction==1:
                    domain_level_dict[dom].append(predicted_level.item())
        for dom, levels in domain_level_dict.items():
               level_predictions_per_domain.append(levels)
        return level_predictions_per_domain

def process_row(row:str,
                sep: str,
                text_col_nr:int,
                nlp,
                icf_domains:[],
                icf_levels:[],
                domains,
                domain_token):
    labeled_row = row ### remove the newline
   # print(row)
    fields = row.split(sep)
    text = fields[text_col_nr]
    anonym_note = anonymize_text(text, nlp)
    to_sentence = lambda txt: [str(sent) for sent in nlp(txt).sents]
    sents = to_sentence(anonym_note)
    dom_predictions = predict_domains_for_sentences(sents, icf_domains)
    # predict levels
    print('Processing domains predictions.', flush=True)
    print(dom_predictions)
    if domain_token:
        sentence_level_predictions_per_domain = add_level_predictions(sents, dom_predictions, icf_levels, domains)
    else:
        sentence_level_predictions_per_domain = add_level_predictions_no_domain(sents, dom_predictions, icf_levels, domains)
   # print(sentence_level_predictions_per_domain)
    #aggregate to note level
    for prediction in sentence_level_predictions_per_domain:
        if prediction:
            labeled_row+=f'{sep}{mean(prediction)}'
        else:
            labeled_row+=f'{sep}'

    labeled_row+="\n"
   # print(labeled_row)
    return labeled_row


@timer
def main(
    in_csv,
    text_col,
    sep,
    encoding,
    domain_token):
    """
    Read the `in_csv` file, process the text by row (anonymize, split to sentences), predict domains and levels per sentence, aggregate the results back to note-level, write the results to the output file.

    Parameters
    ----------
    in_csv: str
        path to csv file with the text to process; the csv must follow the following specs: sep=';', quotechar='"', first row is the header
    text_col: str
        name of the column containing the text
    encoding: str
        encoding of the csv file, e.g. utf-8

    Returns
    -------
    None
    """

#The following ICF categories are covered:
    # B1300 (energy level, ENR),
    # B140 (attention, ATT),
    # B152 (emotions, STM),
    # B440 (respiration, ADM),
    # B455 (exercise tolerations, INS),
    # B530 (weight maintenance, MBW),
    # D540 (walking, FAC),
    # D550 (eating, ETN),
    # D840-D859 (work and employment, BER),
    # B280 (sensations of pain, SOP),
    # B134 (sleep, SLP),
    # D760 (family relationships, FML),
    # B164 (higher cognitive functions, HLC),
    # D465 (moving around using equipment, MAE),
    # D410 (changing basic body position, CBP),
    # B230 (hearing, HRN),
    # D240 (handling stress and other psychological demands, HSP)

   # domains=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM', 'SLP', 'HLC', 'HRN', 'SOP', 'HSP', 'CBP', 'MAE', 'FML']
   # domain_code=['B440', 'B140', 'D840-D859', 'B1300', 'D550', 'D540', 'B455', 'B530', 'B152', 'B134', 'B164', 'B230', 'B280', 'D240', 'D410', 'D465', 'D760']
    domains=['ENR', 'ATT', 'STM', 'ADM', 'INS', 'MBW', 'FAC', 'ETN', 'BER', 'SOP', 'SLP', 'FML', 'HLC', 'MAE', 'CBP', 'HRN', 'HSP']
    domain_code=['B1300', 'B140', 'B152', 'B440', 'B455', 'B530', 'D540', 'D550', 'D840-D859', 'B280', 'B134', 'D760', 'B164', 'D465', 'D410', 'B230', 'D240']

    levels = [f"{domain}-{code}_lvl" for domain, code in zip(domains, domain_code)]

    # check path
    in_csv = Path(in_csv)
    msg = f'The csv file cannot be found in this location: "{in_csv}"'
    assert in_csv.exists(), msg

    # read csv
    print(f'Loading input csv file: {in_csv}')
    print(f'Separator: {sep}')
    print(f'Text header: {text_col}')

    in_csv_file = open(in_csv, 'r')
    ### read the headerline and check the header for the text column
    first_row = in_csv_file.readline().strip()
    print(f'Header line: {first_row}')

    headers = first_row.split(sep)
    text_column_nr = -1
    for index, header in enumerate(headers):
        print('Header', header, index)
        if header.strip()==text_col:
            text_column_nr = index
            break

    if text_column_nr ==-1:
        print(f'Could not find the text column "{text_col}" in header line: "{first_row}". Aborting.')
        return

    # text processing
    print('Loading spacy model:nl_core_news_lg')
    nlp = spacy.load('nl_core_news_lg')
    print('Loading ICF classifiers')
    # predict domain
    icf_domains = load_model(
        model_type='roberta',
        model_name='CLTL/icf17-domains',
      #  model_name='/Users/piek/Desktop/r-APROOF/code/models/icf17-domains',
        task='multi'
    )

    if domain_token:
        print('Loading ICF levels classifier with encoded domain tokens')
        icf_levels = load_model(
            model_type='roberta',
            model_name= 'CLTL/icf17-levels-domain-token',
            task='clf'
        )

    else:
        icf_levels = load_model(
            model_type='roberta',
            model_name= 'CLTL/icf17-levels',
            task='clf'
        )

    # save output file
    out_csv = in_csv.parent / (in_csv.stem + '_output.csv')
    out_csv_file = open(out_csv, "w")
    print(f'The output will be saved to : {out_csv}')
    for level in levels:
        first_row+=sep+level
    out_csv_file.write(first_row+'\n')

    count = 0
    while True:
        count +=1
        row = in_csv_file.readline().strip()
        if not row:
            break
        else:
            labeled_row = process_row(row,sep, text_column_nr, nlp, icf_domains, icf_levels, domains, domain_token)
            out_csv_file.write(labeled_row)
        if count%100==0:
            print('Processed line:{}', count)

    in_csv_file.close()
    out_csv_file.close()


if __name__ == '__main__':

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in_csv', default='./example/input.csv')
    argparser.add_argument('--text_col', default='text')
    argparser.add_argument('--sep', default=';')
    argparser.add_argument('--encoding', default='utf-8')
    argparser.add_argument('--domain_token', default=True)
    args = argparser.parse_args()

    main(
        args.in_csv,
        args.text_col,
        args.sep,
        args.encoding,
        args.domain_token
    )
