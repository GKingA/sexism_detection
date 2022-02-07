import stanza
import json
import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# To visualize
from graphviz import Source

# Knowledge Graph
from tuw_nlp.text.pipeline import CachedStanzaPipeline
from tuw_nlp.graph import knowledge_graph
from tuw_nlp.graph.knowledge_graph import KnowledgeGraph, KnowledgeNode

# To unpickle previously pickeled files without error
sys.modules["knowledge_graph"] = knowledge_graph

from xpotato.dataset.dataset import Dataset
from xpotato.models.trainer import GraphTrainer
from xpotato.graph_extractor.extract import FeatureEvaluator


def load_datasets(train_path, val_path, data_path, language, draw):
    if os.path.exists(train_path) and os.path.exists(val_path):
        train = pd.read_pickle(train_path)
        val = pd.read_pickle(val_path)
    else:
        sexism = pd.read_csv(data_path, sep='\t')
        graphs, texts, knowledge_graphs = [], [], []
        stanza.download(language)
        pipe = CachedStanzaPipeline(stanza.Pipeline(language, processors='tokenize,mwt,pos,lemma,depparse'), "cache")
        for i, (s_text, s_lang) in enumerate(zip(sexism.text, sexism.language)):
            if s_lang == language:
                knowledge_graphs.append(KnowledgeGraph(text=s_text, pipeline=pipe, synset_method="graph_match", lang=language))
                if draw:
                    dot = knowledge_graphs[-1].to_dot()
                    Source(dot).render(filename=f"sexism_graphs/sexism_{i}")
                print(s_text)
                texts.append((s_text, sexism.task1[i]))
                graphs.append(knowledge_graphs[-1].G)
        ds = Dataset(texts, label_vocab={"sexist": 1, "non-sexist": 0})
        ds.set_graphs(graphs)
        df = ds.to_dataframe()
        train, val = train_test_split(df, test_size=0.2, random_state=1234)
        train.to_pickle(train_path)
        val.to_pickle(val_path)
    return train, val


def load_features(train_feature, train, val):
    if os.path.exists(train_feature):
        with open(train_feature) as feature_json:
            features = json.load(feature_json)
    else:
        trainer = GraphTrainer(pd.concat([train, val]))
        features = trainer.prepare_and_train()
        with open(train_feature, "w") as f:
            json.dump(features, f)
    return features


def find_good_features(features, keep_by_default, interval_to_ask):
    if interval_to_ask is None:
        interval_to_ask = [0.8, 0.9]
    if keep_by_default is None:
        keep_by_default = 0.9
    good_rules = {'sexist': []}
    feature_list = [[[feature[0][0].replace('#', '')], feature[1], feature[2]] for feature in features["sexist"]]
    stats = evaluator.evaluate_feature('sexist', feature_list, train)[0]
    for feature, prec, rec, f1 in zip(feature_list, stats["Precision"], stats["Recall"], stats["Fscore"]):
        print(f"Rule: {feature}\nPrecision: {prec}\nRecall: {rec}\nF1: {f1}")
        if prec < interval_to_ask[1] and prec >= interval_to_ask[0]:
            user_inp = input("Input")
            if user_inp.lower().startswith("y"):
                good_rules['sexist'].append(feature)
        elif prec >= keep_by_default:
            good_rules['sexist'].append(feature)
    return good_rules


def load_config(args):
    if args.config is not None:
        config = json.load(open(args.config))
        for setting in args.__dict__:
            if setting in config:
                args.__dict__[setting] = config[setting]
    return args


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--train", "-t", help="Path to the train pickle file")
    argparser.add_argument("--valid", "-v", help="Path to the validation pickle file")
    argparser.add_argument("--sexism_file", "-sf",
                           help="Path to the sexism tsv file (same formatting needed as EXIST)")
    argparser.add_argument("--language", "-l",
                           help="Language of the data. Currently only English (en) is supported.",
                           default="en")
    argparser.add_argument("--synset_method", "-sm",
                           help="Method to find the best synset for a given word",
                           choices=["vote_lesk", "first_synset", "nltk_lesk", "original_lesk", 
                                    "simple_lesk", "cosine_lesk", "adapted_lesk", "graph_match", "ud_match"],
                           default="simple_lesk")
    argparser.add_argument("--draw", "-d", action="store_true", help="Draw the graphs to a file during parsing")
    argparser.add_argument("--train_feature", "-tf", help="Path to the feature json file generated")
    argparser.add_argument("--chosen_feature", "-cf", help="Path to the json file with the hand selected features")
    argparser.add_argument("--keep_by_default", "-k", 
                           help="The treshold of precision to keep a certain rule", type=float)
    argparser.add_argument("--interval_to_ask", "-i", nargs=2,
                           help="The interval of precision where we ask the user whether to keep a rule", 
                           type=float)
    argparser.add_argument("--config", "-c", help="Configuration json file with arguments")
    args = argparser.parse_args()
    args = load_config(args)

    evaluator = FeatureEvaluator()
    train, val = load_datasets(args.train, args.valid, args.sexism_file, args.language, args.draw)

    if args.chosen_feature is not None and os.path.exists(args.chosen_feature):
        hand_features = load_features(args.chosen_feature, train, val)
    else:
        features = load_features(args.train_feature, train, val)
        hand_features = find_good_features(features, args.keep_by_default, args.interval_to_ask)
        if args.chosen_feature is None:
            args.chosen_feature = "hand_features.json"
        json.dump(hand_features, open(args.chosen_feature, "w"), indent=4)

    stats = evaluator.evaluate_feature('sexist', hand_features['sexist'], train)[0]
    print(classification_report(train.label_id, [(n > 0) * 1 for n in np.sum([p for p in stats["Predicted"]], axis=0)]))
    for rule, stat in zip(hand_features['sexist'], stats["False_positive_sens"]):
        print(f"\n{rule}:")
        print("\n\n".join([s[0] for s in stat]))
    val_stats = evaluator.evaluate_feature('sexist', hand_features['sexist'], val)[0]
    print(classification_report(val.label_id, [(n > 0)*1 for n in np.sum([p for p in val_stats["Predicted"]], axis=0)]))
