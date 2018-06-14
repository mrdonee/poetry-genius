import numpy
import sys
from autocorrect import spell
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from os import listdir
from os.path import isfile, join

CONST_IN_FILE = "./inputFiles/sonnets.txt"
CONST_TRAIN = False
CONST_NUM_EPOCHS = 100
CONST_BATCH_SIZE = 64

rawText = open(CONST_IN_FILE).read()
rawText = rawText.lower()

# create map of all chars found in text and reverse lookup table
chars = sorted(list(set(rawText)))
charToInt = dict((c, i) for i, c in enumerate(chars))
intToChar = dict((i, c) for i, c in enumerate(chars))

# get raw text data
numChars = len(rawText)
numVocab = len(chars)
print "Total Characters: ", numChars
print "Total Vocab: ", numVocab

# create dataset of character sequence to next character pairs encoded as integers
seqLength = 100
charSequences = []
nextChar = []
for i in range(0, numChars - seqLength, 1):
	seqIn = rawText[i:i + seqLength]
	seqOut = rawText[i + seqLength]
	charSequences.append([charToInt[char] for char in seqIn])
	nextChar.append(charToInt[seqOut])
numPatterns = len(charSequences)
print "Total Patterns: ", numPatterns

# reshape charSequences to be [samples, time steps, features]
charSeqs = numpy.reshape(charSequences, (numPatterns, seqLength, 1))

# normalize
charSeqs = charSeqs / float(numVocab)

# one hot encode the output variable
oneHotNextChar = np_utils.to_categorical(nextChar)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(charSeqs.shape[1], charSeqs.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(oneHotNextChar.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint and fit the model
if CONST_TRAIN:
	filepath="./models/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacksList = [checkpoint]
	model.fit(charSeqs, oneHotNextChar, epochs=CONST_NUM_EPOCHS, batch_size=CONST_BATCH_SIZE, callbacks=callbacksList)

# load network weights
weightImprovePath = "./models/"
onlyfiles = [f for f in listdir(weightImprovePath) if isfile(join(weightImprovePath, f))]
networkWeightFile = ""
for item in onlyfiles[:]:
    if item.startswith("weights-improvement-"):
        networkWeightFile = item
model.load_weights(weightImprovePath + networkWeightFile)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# pick a random beginning sequence
start = numpy.random.randint(0, len(charSequences) - 1)
pattern = charSequences[start]
print "Starting Sequence:"
print "\"", ''.join([intToChar[value] for value in pattern]), "\""

# generate output
poem = []
currLine = ""
for i in range(1000):
	reshapedPattern = numpy.reshape(pattern, (1, len(pattern), 1))
	reshapedPattern = reshapedPattern / float(numVocab)
	prediction = model.predict(reshapedPattern, verbose=0)
	index = numpy.argmax(prediction)
	if (intToChar[index] == "\n"):
		poem.append(currLine)
		currLine = ""
	else:
		currLine += intToChar[index]
	pattern.append(index)
	pattern = pattern[1:len(pattern)]

# spell check output
spelledCheckedLine = ""
for line in poem:
	lineWords = line.split()
	for word in lineWords:
		spelledCheckedLine += spell(word) + " "
	print(spelledCheckedLine)
	spelledCheckedLine = ""
print "\nDone"
