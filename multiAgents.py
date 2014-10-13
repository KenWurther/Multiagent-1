# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        returnScore = float(0)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodList=oldFood.asList()
   
        foodList.sort(lambda x,y: util.manhattanDistance(newPos, x)-util.manhattanDistance(newPos, y))
        foodScore=util.manhattanDistance(newPos, foodList[0])
        GhostPositions=[Ghost.getPosition() for Ghost in newGhostStates]
        if len(GhostPositions) ==0 : GhostScore=0
        else: 
            GhostPositions.sort(lambda x,y: disCmp(x,y,newPos))
            if util.manhattanDistance(newPos, GhostPositions[0])==0: return -99 
            else:
                GhostScore=2*-1.0/util.manhattanDistance(newPos, GhostPositions[0])
        if foodScore==0: returnScore=2.0+GhostScore
        else: returnScore=GhostScore+1.0/float(foodScore)
        return returnScore

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def minmaxfunction(self, state, agent, depth):
        legalMoves = state.getLegalActions(agent)
        if Directions.STOP in legalMoves:
            legalMoves.remove(Directions.STOP)
        allAgents = state.getNumAgents()
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        if agent == 0:
            return max([self.minmaxfunction(state.generateSuccessor(agent, action), (agent+1)%allAgents, depth-1) for action in legalMoves])
        else:
            return min([self.minmaxfunction(state.generateSuccessor(agent, action), (agent+1)%allAgents, depth-1) for action in legalMoves])

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        "Add more of your code here if you want to"
		# Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(0)
        if Directions.STOP in legalMoves:
            legalMoves.remove(Directions.STOP)
        scores =  [self.minmaxfunction(gameState.generateSuccessor(0, action), 1, self.depth) for action in legalMoves]
        print scores
        bestScore = max(scores)
        print bestScore
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        # Choose one of the best actions
        return legalMoves[chosenIndex]
		


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
  Your minimax agent with alpha-beta pruning (question 3)
  """
  def minValuefunction(self, state, alpha, beta, depth, ghostnum):
      if (state.isLose() or state.isWin() or depth==0): 
          return self.evaluationFunction(state)
      legalMoves = state.getLegalActions(ghostnum)
      listNextStates = [state.generateSuccessor(ghostnum,action) for action in legalMoves]
      for nextState in listNextStates:
          if ghostnum == state.getNumAgents() - 1:
              beta = min(self.maxValuefunction(nextState, alpha, beta, depth-1), beta)
          else:
              beta = min(self.minValuefunction(nextState, alpha, beta, depth, ghostnum+1), beta)
          if (beta <= alpha):
              return beta
      return beta
  

  def maxValuefunction(self, state, alpha, beta, depth):
      if (state.isLose() or state.isWin() or depth==0): 
          return self.evaluationFunction(state)
      legalMoves = state.getLegalActions(0)
      listNextStates = [state.generateSuccessor(0, action) for action in legalMoves]
      for nextState in listNextStates:
          alpha = max(self.minValuefunction(nextState, alpha, beta, depth-1, 1), alpha)
          if (alpha >= beta):
             return alpha
      return alpha

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    numOfAgent = gameState.getNumAgents()
    legalMoves = gameState.getLegalActions(0)
    trueDepth = self.depth * numOfAgent
    
    listNextStates = [gameState.generateSuccessor(0,action) for action in legalMoves]
    # as long as beta is above the upper bound of the eval function
    scores = [self.maxValuefunction(nextGameState, float('-Inf'), float('Inf'), trueDepth) for nextGameState in listNextStates] 
    print scores
    bestScore = max(scores)
    print bestScore
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    # Choose one of the best actions
    return legalMoves[chosenIndex]
		

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

