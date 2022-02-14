import os.path
import xml.etree.ElementTree as ET

import stanza
import json

from tuw_nlp.graph.knowledge_graph import KnowledgeGraph
from tuw_nlp.text.pipeline import CachedStanzaPipeline


def read_xml_file(file):
    with open(file) as xml_file:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        sentence_list = []
        for documents in root:
            for sentence in documents:
                sentence_list.append([])
                for words in sentence:
                    sentence_list[-1].append({'id': words.attrib['id'], 'text': words.text, 'pos': words.attrib['pos']})
    return sentence_list


def read_results(file):
    correct = {}
    with open(file) as gold_results:
        for line in gold_results:
            split_line = line.split("\t")
            if "wn:" in line:
                correct[split_line[0]] = [wn[3:] for wn in split_line if wn.startswith("wn:")]
    return correct


def create_graphs(sentence_list):
    synset_methods = {"vote_lesk": {},
                      "first_synset": {},
                      "nltk_lesk": {},
                      "original_lesk": {},
                      "simple_lesk": {},
                      "cosine_lesk": {},
                      "adapted_lesk": {},
                      "graph_match": {},
                      "ud_match": {}}
    with CachedStanzaPipeline(stanza.Pipeline("en", processors='tokenize,mwt,pos,lemma,depparse'), "cache") as pipe:
        for index, sentence in enumerate(sentence_list):
            sent_text = ' '.join([word['text'] for word in sentence])
            for sm in synset_methods:
                print(f"{sm} -- {index}/{len(sentence_list)} {sent_text}")
                graph = KnowledgeGraph(text=sent_text, pipeline=pipe, synset_method=sm)
                for node, word in zip(graph.G.nodes(data=True), sentence):
                    if node[1]['name'].synset is not None:
                        synset_methods[sm][word['id']] = []
                        for lemma in node[1]['name'].synset.lemmas():
                            synset_methods[sm][word['id']].append(lemma.key())
                        synset_methods[sm][word['id']] = node[1]['name'].synset.lemmas()[0].key()
        return synset_methods


def evaluate(gold_syns, syns_by_method):
    all_syns = len(gold_syns)
    for method, syns in syns_by_method.items():
        corrects = 0
        social_correct = 0
        social = 0
        for key, gold_s in gold_syns.items():
            if key in syns and syns[key] in gold_s:
                corrects += 1
            if 'd003' in key:
                social += 1
                if key in syns and syns[key] in gold_s:
                    social_correct += 1
        print(f"{method}: {corrects}/{all_syns} -- {round(100 * corrects/all_syns, 2)}%\n"
              f"\tsocial: {social_correct}/{social} -- {round(100 * social_correct/social, 2)}%")


if __name__ == '__main__':
    if not os.path.exists("syns.json"):
        sentences = read_xml_file("SemEval-2015-task-13-v1.0/data/semeval-2015-task-13-en.xml")
        chosen_syns = create_graphs(sentences)
        json.dump(chosen_syns, open("syns.json", 'w'))
    else:
        chosen_syns = json.load(open("syns.json"))
    gold = read_results("SemEval-2015-task-13-v1.0/keys/gold_keys/EN/semeval-2015-task-13-en.key")
    evaluate(gold, chosen_syns)
