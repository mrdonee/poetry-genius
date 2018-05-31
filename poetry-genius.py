import numpy
import sys
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from os import listdir
from os.path import isfile, join

inFile = "./inputFiles/wonderland.txt"
rawText = open(inFile).read()
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
reshapedCharSeqs = numpy.reshape(charSequences, (numPatterns, seqLength, 1))

# normalize
reshapedCharSeqs = reshapedCharSeqs / float(numVocab)

# one hot encode the output variable
oneHotNextChar = np_utils.to_categorical(nextChar)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(reshapedCharSeqs.shape[1], reshapedCharSeqs.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(oneHotNextChar.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="./models/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacksList = [checkpoint]

# fit the model
model.fit(reshapedCharSeqs, oneHotNextChar, epochs=20, batch_size=128, callbacks=callbacksList)
weightImprovePath = "./models/"
onlyfiles = [f for f in listdir(weightImprovePath) if isfile(join(weightImprovePath, f))]
networkWeightFile = ""
for item in onlyfiles[:]:
    if item.startswith("weights-improvement-"):
        networkWeightFile = item

# load network weights
model.load_weights(weightImprovePath + networkWeightFile)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# pick a random beginning sequence
start = numpy.random.randint(0, len(charSequences) - 1)
pattern = charSequences[start]
print "Starting Sequence:"
print "\"", ''.join([intToChar[value] for value in pattern]), "\""

# generate output
for i in range(1000):
	reshapedPattern = numpy.reshape(pattern, (1, len(pattern), 1))
	reshapedPattern = reshapedPattern / float(numVocab)
	prediction = model.predict(reshapedPattern, verbose=0)
	index = numpy.argmax(prediction)
	result = intToChar[index]
	seqIn = [intToChar[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print "\nDone."
