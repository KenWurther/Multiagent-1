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
import re

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
	# Possible states of Pacmann
        successorGameState = currentGameState.generatePacmanSuccessor(action)

	#Old and new state of Pacmann
	oldPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()

	# Old and new states of Food
        newFood = successorGameState.getFood()
        oldFood = currentGameState.getFood()

	# Old and new states of Ghosts
	oldGhostStates = currentGameState.getGhostStates()
        newGhostStates = successorGameState.getGhostStates()

	# Manhattan distance between pacmann and ghost in old and new states
	old_m_dist_PG = new_m_dist_PG = 0

	# Manhattan distance between pacmann and food in old and new states
	old_m_dist_PF = new_m_dist_PF = 9999

	# Has resident scared timer value
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

	#Local variables
	dist = []
	score = 0

	print "Successor game state ",successorGameState
	print "oldPos ", oldPos
	print "newPos ",newPos
	print "newFood ", newFood
	print "oldFood ", oldFood
	print "newGhostStates "
	for state in newGhostStates:
		print state
	print "oldGhostStates "
	for state in oldGhostStates:
		print state
	print "newScardTimes ",newScaredTimes

	# Calculate old and new manhattan distance between pacmann and ghost
	for state in newGhostStates:
		str_state = str(state)
		print "str_state", str_state
		g_coord = [int(s) for s in re.findall('\\d+', str_state)]
		
		# This is a hack. Some state are (2, 7), some are (2.0, 7.0)
		# So remove 0's twice if present
		if 0 in g_coord:
			g_coord.remove(0)
		if 0 in g_coord:
			g_coord.remove(0)

		print "gccord", g_coord
		gx = g_coord[0]
		gy = g_coord[1]
		#If ghost position is successor state, return lowest score
		if ((gx,gy) == newPos):
			return -9999
		old_m_dist_PG += manhattanDistance((gx,gy), oldPos)
		new_m_dist_PG += manhattanDistance((gx,gy), newPos)
		print "old man distt", old_m_dist_PG
		print "new man distt", new_m_dist_PG

	# Calculate manhattan distance between pacmann and food in old state
	for food in oldFood.asList():
		print food
		dist.append(manhattanDistance(food, oldPos))
	old_m_dist_PF = min(dist)

	# Calculate manhattan distance between pacmann and food in successor state
	dist = []
	for food in newFood.asList():
		print food
		dist.append(manhattanDistance(food, newPos))
	if dist:
		new_m_dist_PF = min(dist)
	else:
		new_m_dist_PF = 0

	# Score calculation
	ghost_is_far = 1 if (new_m_dist_PG >= 3) else 0
	is_food_eaten = (len(oldFood.asList()) - len(newFood.asList()))

	print "old man distt food", old_m_dist_PF
	print "new man distt food", new_m_dist_PF
	print "ghost is far?", ghost_is_far
	print "is food waten?", is_food_eaten

	if ghost_is_far:
		score += 6*(old_m_dist_PF - new_m_dist_PF)
		score += 50*is_food_eaten # if food is eaten, then bump up the score
	else:
		# Ghost is in the vicnity
		score += 2*(old_m_dist_PF - new_m_dist_PF)
		score += 10*(new_m_dist_PG - old_m_dist_PG)
		#score += 10*is_food_eaten

	print "score", score
	return score
        "*** YOUR CODE HERE ***"

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
      if Directions.STOP in legalMoves:
            legalMoves.remove(Directions.STOP)
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
    #trueDepth = self.depth * numOfAgent
    
    listNextStates = [gameState.generateSuccessor(0,action) for action in legalMoves]
    # as long as beta is above the upper bound of the eval function
    scores = [self.maxValuefunction(nextGameState, float('-Inf'), float('Inf'), self.depth) for nextGameState in listNextStates] 
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

