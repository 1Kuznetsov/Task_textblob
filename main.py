from textblob import TextBlob
from textblob import Word
from textblob.wordnet import VERB
from textblob.classifiers import NaiveBayesClassifier

poem = """All that doth flow we cannot liquid name
Or else would fire and water be the same;
But that is liquid which is moist and wet
Fire that property can never get.
Then 'tis not cold that doth the fire put out
But 'tis the wet that makes it die, no doubt"""
poem_obj = TextBlob(poem)
# Возвращает части речи
print(poem_obj.tags)
# Группы слов существительного
print(poem_obj.noun_phrases)
print(poem_obj.sentiment)
print(poem_obj.words)
print(poem_obj.sentences)
test = TextBlob("The stars are beautiful tonight")
print(test.words[1].singularize())
print(poem_obj.words[-1].pluralize())
print(poem_obj.words[-5].lemmatize())
print(test.words[2].lemmatize("v"))
w = Word("Went")
print(w.lemmatize("v"))
# Синонимы по значению
print(w.synsets)
print(poem_obj.words[7].synsets)
print(Word("Appreciate").get_synsets(pos=VERB))
print(Word("Appreciate").definitions)
flowers = TextBlob("rose sunflower lily orchid lavender")
print(flowers.words)
print(flowers.words.pluralize())
bad_text = TextBlob("I caan writee verry fastr ans correvt")
print(bad_text.correct())
print(Word("Posibility").spellcheck())
print(poem_obj.word_counts['but'])
print(poem_obj.words.count('but'))
print(poem_obj.words.count('but', case_sensitive=True))
print(poem_obj.parse())
print(poem_obj[0:25])
print(poem_obj.upper())
print(poem_obj.ngrams(n=4))
for s in poem_obj.sentences:
    print(s)
    print("---- Starts at index {}, ends at index {}".format(s.start, s.end))
train = [
    ("I love this book.", "pos"),
    ("this is an amazing place!", "pos"),
    ("This is my best work.", "pos"),
    ("Exceptionally good", "pos"),
    ("I do not like this sword", "neg"),
    ("what a wonderful day", "pos"),
    ("I can't complete this", "neg"),
    ("It was boring", "neg"),
    ("I will not recommend", "neg")
]

test = [
    ("It is a fantastic story", "pos"),
    ("I can't believe I'm watching this", "neg"),
    ("It's amazing!", "pos"),
    ("I do not enjoy my work", "neg"),
    ("Waste of time", "neg"),
    ("The best I've ever seen", "pos")
]

c = NaiveBayesClassifier(train)
print(c.classify("This is a perfect picture"))
