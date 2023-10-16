import torch
from typing import List, Union, Generator, NewType, Tuple, Dict, Set

#Set seed for reproducibility
torch.manual_seed(2022)

def loadData(fname: str = 'pos-train.txt') -> Tuple[List[int], List[int], int, int, Dict[int, str], Dict[int, str], Dict[str, Set[str]]]:
    """
    Simple function which loads the training data from the ud data
    files I created. The input file conform to the following standard:

        First line should be a header which has word, id, total_ids, pos, 
        pos_id, total_pos_ids, and sent (in any order). These columns are:

            word:           the string of the word 
            id:             a unique id for the word type (from 0)
            total_ids:      the total number of word ids
            pos:            the string of the pos label
            pos_id:         a unique id for this pos label type (from 0)
            total_pos_ids:  the total number of pos ids
            sent:           the sentence which contains the word

        For example, the first two lines of pos-train.txt are: 
            word,id,total_ids,pos,pos_id,total_pos_ids,sent
            forces,0,1377,NOUN,0,12,Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of Qaim, near the Syrian border.

        The first data point, then, means we have a word "forces" which has an 
        id of 0. There are a total of 1377 words, and "forces" is labeled 
        as a NOUN. NOUN has an id of 0, for which there are 12 possible ids. 
        The word occurs in the sentence "Al-Zaman : American ...". 

    Args:
        fname (str): name of the file. Default is pos-train.txt

    Returns:
        X (List[int]):        All the data as ids
        Y (List[int]):        The corresponding pos ids of these data
        inDim (int):     The size of the vocabulary
        outDim (int):    The number of possible pos labels
        id2word (Dict[int,str]):  A dictionary maping word ids to words
        id2pos (Dict[int,str]):   A dictionary mapping pos ids to pos labels
        word2pos (Dict[str,Set[str]]): A dictionary mapping each word to
                                         its possible pos labels as a set

    """

    X = []
    Y = []
    inDim = 0
    outDim = 0
    id2word = {}
    id2pos = {}
    word2pos = {}

    with open(fname, 'r') as f:
        #header mappings
        header = {}
        for idx, head in enumerate(f.readline().strip().split(',')):
            header[head] = idx
        
        for line in f:
            elements = line.strip().split(',')
            x = int(elements[header['id']])
            y = int(elements[header['pos_id']])
            inDim = int(elements[header['total_ids']])
            outDim = int(elements[header['total_pos_ids']])

            word = elements[header['word']]
            pos = elements[header['pos']]
            if word not in word2pos:
                word2pos[word] = set([])
            word2pos[word].add(pos)

            id2word[x] = word
            id2pos[y] = pos

            X.append(x)
            Y.append(y)

    return X, Y, inDim, outDim, id2word, id2pos, word2pos

class FFNN(torch.nn.Module):

    def __init__(self, inputDimensions:int, hiddenDimensions:int, outputDimensions:int):

        super().__init__()

        self.inputDimensions = inputDimensions
        self.hiddenDimensions = hiddenDimensions
        self.outputDimensions = outputDimensions

        #TODO: Your code goes here 
        #Hint: What parameters/activation functions do we need??

    def forward(self, x: torch.tensor) -> torch.tensor:
        #TODO: Your code goes here
        raise NotImplementedError

#Useful to create a type for our model for later
FFNNModel = NewType('FFNNModel', FFNN)
        
def batches(X: List[int], Y: List[int], batchSize:int=20) -> Generator[Tuple[torch.tensor, torch.tensor], None, None]:
    """
    One-hot encodes X and returns batch of X and Y as torch tensors.

    Args: 
        X (List[int]): Input data of class labels
        Y (List[int]): Ouptut data of class labels
        batchSize (int): Desired number of samples in each batch

    Yields:
        X (torch.tensor): Chunk of X of size (batch size, inputDimensions)
        Y (torch.tensor): Chunk of Y of size (batch size,) 
    """
    #Map X Y to a tensor
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    #One hot encode the X
    X = torch.nn.functional.one_hot(X).float()

    #Now yield batches
    for idx in range(0, X.size()[0], batchSize):
        yield X[idx:idx+batchSize,:], Y[idx:idx+batchSize]
        
def train(X: List[int], Y: List[int], model:FFNNModel, epochs:int=2,
          lr:int=0.01) -> None:
    """
    Train a model on data for some number of epochs with a given learning rate.

    Args: 
        X (List[int]): A list of input ids, where each id is a input class
        Y (List[int]): A list of output ids, where each id is an output class
        model (FFNNModel): An instance of FFNNModel to be trained
        epochs (int): Number of epochs to train for
        lr (int): Learning rate to use for weight updates
    """

    #Set the model to train (in case eval has been set)
    model.train()
    model.zero_grad()

    #Create an instance of an Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #TODO: Add your code below
    #Hint: Remember to create a loss function, then for each epoch, loop through
    #      batches of data, get the model predictions for these data, get the
    #      error, retrieve the gradients, and update the model weights using the 
    #      optimizer. 
    #N.B.: Remember to call optimizer.zero_grad() for each batch, otherwise
    #      The gradient will accumulate across all the data. 



def eval(model:FFNNModel, numErrors:int = 10, numSets:int=5) -> None:
    """
    Evaluate a model for accuracy on data. Print to stdout errors in prediction.

    Args: 
        model (FFNNModel): An instance of FFNNModel to be evaluated
        numErrors (int): Number of simple errors to return
        numSets (int): Number of more systematic errors to return
    """

    #Load data
    X, Y, inDim, outDim, id2word, id2pos, word2pos = loadData()

    #Set model to eval so no gradients are created
    model.eval()

    #Keep track of times the model is correct
    isCorrect=[]
    #Keep track of errorful predictions
    SimpleErrors = set([])
    MultiErrors = set([])
    for batch in batches(X, Y):
        x, y = batch

        predictions = model(x)
        #Get the top predicted class for each prediction
        predictions = torch.argmax(predictions, dim=-1)

        #Check if the predicted class is correct
        checked = predictions == y
        #Map to int for summing later
        checked = checked.int().tolist()
        isCorrect.extend(checked)

        #Find the errorful cases
        if sum(checked) < len(checked):
            errorsX = torch.argmax(x[~(predictions==y),:], dim=-1).tolist()
            errorsPred = predictions[~(predictions==y)].tolist()
            errorsY = y[~(predictions==y)].tolist()

            for ex, ep, ey in zip(errorsX, errorsPred, errorsY):
                word = id2word[ex]
                predPOS = id2pos[ep]
                goldPOS = id2pos[ey]

                item = f"{word} {predPOS} {goldPOS}"

                if len(word2pos[word]) == 1:
                    SimpleErrors.add(item)
                else:
                    MultiErrors.add(" ".join(item.split(" ")[:-1]))


    print('-------------------------------------------------------------------')
    print(f"Accuracy on train data: {(sum(isCorrect)/len(isCorrect))*100:.2f}%")
    print('-------------------------------------------------------------------')          
        
    print()
    print('-------------------------------------------------------------------')
    print('The following are incorrect predictions by the model:')
    while SimpleErrors and numErrors:
        item = SimpleErrors.pop()
        numErrors -= 1
        word, predPOS, goldPOS = item.split()
        print(f"\tFor '{word}' the gold POS is {goldPOS}, but the model predicted {predPOS}")
    
    print('Consider the following cases:')
    while MultiErrors and numSets:
        item = MultiErrors.pop()
        numSets -= 1
        word, predPOS = item.split()
        possPOS = word2pos[word]
        print(f"\tFor '{word}' the model predicts {predPOS}. The following POS are possible {possPOS}")


if __name__ == "__main__":

    X, Y, inDim, outDim, id2word, id2pos, word2pos = loadData()

    hiddenDim = 200

    model = FFNN(inDim, hiddenDim, outDim)

    train(X, Y, model)
    eval(model)
