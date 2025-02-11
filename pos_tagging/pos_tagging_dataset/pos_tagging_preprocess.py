import nltk


def get_dataset():
    nltk.download('treebank')

    tagged_sentences = nltk.corpus.treebank.tagged_sents()
    print("Number of samples:", len(tagged_sentences))

    sentences, sentence_tags = [], []
    for tagged_sentence in tagged_sentences:
        sentence, tags = zip(*tagged_sentence)
        sentences.append([word.lower() for word in sentence])
        sentence_tags.append([tag for tag in tags])

    return sentences, sentence_tags

def build_tag(sentence_tags):
    unique_tags = set(tag for doc in sentence_tags for tag in doc)
    label2id = {tag: idx for idx, tag in enumerate(unique_tags)}
    id2label = {idx: tag for tag, idx in label2id.items()}
    return unique_tags, label2id, id2label


