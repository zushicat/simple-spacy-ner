import json
from random import shuffle

import spacy
from spacy.util import minibatch, compounding, decaying


# ****************************************
#
# ****************************************
def create_annotations(inputs):
    '''
    from:
    [("the cat is fluffy", ["O", "KATZ", "O", "O"])]
    to:
    [["the cat is fluffy", {"entities": [[4, 7, "KATZ"]]}]]
    '''
    annotations = []

    for input_ in inputs:
        sequence = input_[0]
        tags = input_[1]
        
        char_len_sequence = len(sequence)
        words = sequence.split()

        consumed = []
        entities = []
        for i, tag in enumerate(tags):
            if tag == "O":
                consumed.append(words[i])
                continue
            start_pos = len(" ".join(consumed)) + 1  # last space
            end_pos = start_pos + len( words[i])
            entities.append([start_pos, end_pos, tag])
            consumed.append(words[i])

        annotations.append(
            (sequence, {"entities": entities})
        )      

    return annotations


# ****************************************
# some inputs in 'easy' format
# ****************************************
inputs = [
    ("the cat is fluffy", ["O", "KATZ", "O", "O"]),
    ("my white cat is dirty", ["O", "O", "KATZ", "O", "O"]),
    ("your cat and my cat are friends", ["O", "KATZ", "O", "O", "KATZ", "O", "O"]),
    ("the dog is very fond of the litte cat", ["O", "O", "O", "O", "O", "O", "O", "O", "KATZ"]),
    ("the curious cat climbed on the tree", ["O", "O", "KATZ", "O", "O", "O", "O"]),
    ("the cat is in the tree", ["O", "KATZ", "O", "O", "O", "O"]),
    ("my cat is gray", ["O", "KATZ", "O", "O"]),
    ("on the bed there is a cat sleeping", ["O", "O", "O", "O", "O", "O", "KATZ", "O"]),
    ("a cat is drinking milk", ["O", "KATZ", "O", "O", "O"])
]

TRAIN_DATA = create_annotations(inputs)


# ********************************
# define some model stuff
# ********************************
nlp = spacy.blank("en")  # created blank 'en' model

if "ner" not in nlp.pipe_names:  # create the built-in pipeline components and add them to the pipeline
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, last=True)

ner.add_label("KATZ")  # add all new labels
n_iter=100  # number of iterations
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]  # get names of other pipes to disable them during training

annotations = []

# ********************************
# start training
# ********************************
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    
    dropouts = decaying(0.3, 0.1, 1e-4)
    batch_size = compounding(4.0, 32.0, 1.001)  # https://spacy.io/usage/training#tips-batch-size

    for itn in range(n_iter):
        shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(TRAIN_DATA, size=batch_size) # batch up the examples using spaCy's minibatch
        dropout = next(dropouts)

        print(itn)
        print("Dropout", dropout)
        
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(
                texts,  # batch of texts
                annotations,  # batch of annotations
                drop=dropout,  # decaying dropout vs. fix 0.5 - make it harder to memorise data
                losses=losses,
            )
        
        print("Losses", losses)
        print("--")


# ********************************
# https://spacy.io/usage/linguistic-features#named-entities
# output:
# cat 2 5 KATZ
# cat 25 28 KATZ
# ********************************
doc = nlp("a cat is not a dog but a cat is just as cute")
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
