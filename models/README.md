To run poetry genius that was build off of a stateful LSTM use:  
  CONST_IN_FILE = "./inputFiles/sonnetsStateful.txt"  
  CONST_RM_PUNCT = True  
  CONST_TRAIN = False  
  CONST_NUM_EPOCHS = 50  
  CONST_STATEFUL = True  

To run poetry genius that was build off of a non stateful LSTM use:  
  CONST_IN_FILE = "./inputFiles/sonnets.txt"  
  CONST_RM_PUNCT = False  
  CONST_TRAIN = False  
  CONST_NUM_EPOCHS = 50  
  CONST_STATEFUL = False  

Note: .hdf5 weight file must be in the folder ./models to run as is
