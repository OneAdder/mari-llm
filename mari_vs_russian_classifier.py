import pickle
from typing import Sequence

import pandas as pd
from pathlib import Path

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class MariVsRussianClassifier:
    def __init__(self):
        classifier_path = Path(__file__).parent / 'mari_vs_russian.classifier'
        vectorizer_path = Path(__file__).parent / 'mari_vs_russian.vectorizer'
        if not classifier_path.exists():
            self.classifier, self.vectorizer = self.train()
            classifier_path.write_bytes(pickle.dumps(self.classifier))
            vectorizer_path.write_bytes(pickle.dumps(self.vectorizer))
        else:
            self.classifier = pickle.loads(classifier_path.read_bytes())
            self.vectorizer = pickle.loads(vectorizer_path.read_bytes())

    def predict(self, texts: Sequence[str]) -> list[int]:
        vectorized = self.vectorizer.transform(texts)
        labels = self.classifier.predict(vectorized)
        return labels

    @staticmethod
    def train() -> tuple[LogisticRegression, TfidfVectorizer]:
        dataset = load_dataset('AigizK/mari-russian-parallel-corpora', split='train')

        mari_examples = dataset['mhr']
        mari_labels = [0] * len(mari_examples)
        russian_examples = dataset['rus']
        russian_labels = [1] * len(russian_examples)
        classification_dataset = pd.DataFrame(
            {'text': mari_examples + russian_examples, 'labels': mari_labels + russian_labels}
        )

        train, test = train_test_split(classification_dataset, test_size=0.15, random_state=2525, shuffle=True)
        vectorizer = TfidfVectorizer()
        train_texts = vectorizer.fit_transform(train['text'])
        train_labels = train['labels']

        classifier = LogisticRegression(random_state=2525)
        classifier.fit(train_texts, train_labels)

        test_texts = vectorizer.transform(test['text'])
        test_labels = test['labels']
        predictions = classifier.predict(test_texts)
        print(classification_report(test_labels, predictions))
        return classifier, vectorizer
