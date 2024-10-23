from nltk.tokenize import regexp_tokenize
import numpy as np

# Here is a default pattern for tokenization; you can substitute it with yours
default_pattern =  r"""(?x)                  
                        (?:[A-Z]\.)+          
                        |\$?\d+(?:\.\d+)?%?    
                        |\w+(?:[-']\w+)*      
                        |\.\.\.               
                        |(?:[.,;"'?():-_`])    
                    """

def tokenize(text, pattern = default_pattern):
    """Tokenize senten with specific pattern
    
    Arguments:
        text {str} -- sentence to be tokenized, such as "I love NLP"
    
    Keyword Arguments:
        pattern {str} -- reg-expression pattern for tokenizer (default: {default_pattern})
    
    Returns:
        list -- list of tokenized words, such as ['I', 'love', 'nlp']
    """
    text = text.lower()
    return regexp_tokenize(text, pattern)


class FeatureExtractor(object):
    """Base class for feature extraction.
    """
    def __init__(self):
        pass
    def fit(self, text_set):
        pass
    def transform(self, text):
        pass  
    def transform_list(self, text_set):
        pass



class UnigramFeature(FeatureExtractor):
    """Example code for unigram feature extraction
    """
    def __init__(self):
        self.unigram = {}
        
    def fit(self, text_set: list):
        """Fit a feature extractor based on given data 
        
        Arguments:
            text_set {list} -- list of tokenized sentences and words are lowercased, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        """
        index = 0
        for i in range(0, len(text_set)):
            for j in range(0, len(text_set[i])):
                if text_set[i][j].lower() not in self.unigram:
                    self.unigram[text_set[i][j].lower()] = index
                    index += 1
                else:
                    continue
                    
    def transform(self, text: list):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]
        
        Returns:
            array -- an unigram feature array, such as array([1,1,1,0,0,0])
        """
        feature = np.zeros(len(self.unigram))
        for i in range(0, len(text)):
            if text[i].lower() in self.unigram:
                feature[self.unigram[text[i].lower()]] += 1
        
        return feature
    
    def transform_list(self, text_set: list):
        """Transform a list of tokenized sentences into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text_set {list} --a list of tokenized sentences, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        
        Returns:
            array -- unigram feature arraies, such as array([[1,1,1,0,0], [1,0,0,1,1]])
        """
        features = []
        for i in range(0, len(text_set)):
            features.append(self.transform(text_set[i]))
        
        return np.array(features)
            

class BigramFeature(FeatureExtractor):
    """Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self):
        self.bigram = {}
    def fit(self, text_set):
        """Fit bigram feature extractor based on given data"""
        index = 0
        for sentence in text_set:
            for i in range(len(sentence) - 1):
                bigram = (sentence[i].lower(), sentence[i+1].lower())
                if bigram not in self.bigram:
                    self.bigram[bigram] = index
                    index += 1
    def transform(self, text):
        """Transform a given sentence into bigram feature vectors"""
        feature = np.zeros(len(self.bigram))
        for i in range(len(text) - 1):
            bigram = (text[i].lower(), text[i+1].lower())
            if bigram in self.bigram:
                feature[self.bigram[bigram]] += 1
        return feature
    def transform_list(self, text_set):
        """Transform a list of tokenized sentences into bigram vectors"""
        features = []
        for sentence in text_set:
            features.append(self.transform(sentence))
        return np.array(features)

class CustomFeature(FeatureExtractor):
    """customized feature extractor, such as TF-IDF
    """
    def __init__(self):
        self.vocab = {}
        self.idf = {}
    def fit(self, text_set: list):
        """Fit the TF-IDF extractor based on given data"""
        index = 0
        document_count = len(text_set)
        word_doc_count = {}

        # Create vocabulary and calculate document frequency
        for sentence in text_set:
            seen_in_doc = set()
            for word in sentence:
                word = word.lower()
                if word not in self.vocab:
                    self.vocab[word] = index
                    index += 1
                if word not in seen_in_doc:
                    seen_in_doc.add(word)
                    word_doc_count[word] = word_doc_count.get(word, 0) + 1
        
        # Calculate IDF
        for word, doc_count in word_doc_count.items():
            self.idf[word] = np.log(document_count / (1 + doc_count))  # IDF with smoothing

    def transform(self, text: list):
        """Transform a given sentence into a TF-IDF vector"""
        tf = np.zeros(len(self.vocab))
        for word in text:
            word = word.lower()
            if word in self.vocab:
                tf[self.vocab[word]] += 1
        tf = tf / np.sum(tf)  # Term Frequency
        tfidf = tf * np.array([self.idf[word] if word in self.idf else 0 for word in self.vocab])
        return tfidf
    
    def transform_list(self, text_set: list):
        """Transform a list of tokenized sentences into TF-IDF vectors"""
        features = []
        for sentence in text_set:
            features.append(self.transform(sentence))
        return np.array(features)


        
