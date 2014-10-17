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
import random, util, math, pdb

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        
	oldPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        
	newFood = successorGameState.getFood()
        oldFood = currentGameState.getFood()
        
	oldGhostStates = currentGameState.getGhostStates()
        newGhostStates = successorGameState.getGhostStates()
        
        minGhostDist = 999999
        minGhostPos = 0

        # Calculate new manhattan distance between pacman and ghost
        newGhostPositions=[ghost.getPosition() for ghost in newGhostStates]
        #sorting ghost positions based on nearness.
        for i,ghost in enumerate(newGhostPositions):
	    # If ghost can eat pacman, dont go there
	    if ghost == newPos:
		return -999999
            if manhattanDistance(newPos, ghost) < minGhostDist:
               minGhostDist = manhattanDistance(newPos, ghost)
               minGhostPos = i
       
	# Determines the influence of ghost on Pacman 
        ghostFactor = 5 * (-1.0/manhattanDistance(newPos, newGhostPositions[minGhostPos]))
        
        #Sorted food poistions based on nearness
        minFoodDist = 999999
        minFoodPos = 0
        foodPositions = oldFood.asList()

	# Find the closest food distance
        for i,food in enumerate(foodPositions):
            if manhattanDistance(newPos, food) < minFoodDist:
               minFoodDist = manhattanDistance(newPos, food)
               minFoodPos = i

        # Determines the influence of food on Pacman
        foodFactor = manhattanDistance(newPos, foodPositions[minFoodPos])
        
        # Score calculation (food score may be zero, so check otherwise just take reciprocal as suggested in the pdf.
	return ((5.0 + ghostFactor) if foodFactor == 0 else (ghostFactor + 1.0/float(foodFactor)))

        #return returnScore
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
    def MiniMax(self, node, depth, agent, tot_agents):
        # Terminal state
        if depth == 0 or node.isWin() or node.isLose():
	        return self.evaluationFunction(node)
        # Pacman
        if agent == 0:
            bestval = float("-inf")
            actions = node.getLegalActions(agent)
		    #Removing STOP action
            if Directions.STOP in actions:
                actions.remove(Directions.STOP)
            for action in actions:
                childState = node.generateSuccessor(agent, action)
                val = self.MiniMax(childState, depth - 1, (agent + 1)%tot_agents, tot_agents)
                bestval = max(bestval, val)
            return bestval
        #Ghost
        else:
            bestval = float("inf")
            actions = node.getLegalActions(agent)
            #Removing STOP action
            if Directions.STOP in actions:
           	    actions.remove(Directions.STOP)
            for action in actions:
                childState = node.generateSuccessor(agent, action)
                val = self.MiniMax(childState, depth - 1, (agent + 1)%tot_agents, tot_agents)
                bestval = min(bestval, val)
        return bestval

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth 
        and self.evaluationFunction.
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        val = [self.MiniMax(gameState.generateSuccessor(0, action), (self.depth*gameState.getNumAgents()) -1 , 1, 
			gameState.getNumAgents()) for action in actions]
        bestval = max(val)
        indices = [index for index in range(len(val)) if val[index] == bestval]
        return actions[random.choice(indices)]

class AlphaBetaAgent(MultiAgentSearchAgent):
     """
     Your minimax agent with alpha-beta pruning (question 3)
     """
     def AlphaBeta(self, node, depth, agent, tot_agents, alpha, beta):
        # Terminal state
        if depth == 0 or node.isWin() or node.isLose():
            return self.evaluationFunction(node)
        # Pacman
        if agent == 0:
            bestval = float("-inf")
            actions = node.getLegalActions(agent)
            for action in actions:
                childState = node.generateSuccessor(agent, action)
                val = self.AlphaBeta(childState, depth - 1, (agent + 1)%tot_agents, tot_agents, alpha, beta)
                bestval = max(bestval, val)
                alpha = max(alpha, bestval)
                # Prune based on Beta value since it is max agent
                if bestval > beta:
                   break
            return bestval
        #Ghost
        else:
            bestval = float("inf")
            actions = node.getLegalActions(agent)
            for action in actions:
                childState = node.generateSuccessor(agent, action)
                val = self.AlphaBeta(childState, depth - 1, (agent + 1)%tot_agents, tot_agents, alpha, beta)
                bestval = min(bestval, val)
                beta = min(beta, bestval)
                # Prune based on Alpha value since it is min Agent
                if bestval < alpha:
                   break
        return bestval

        
     def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        #calculate depth based on the number of agents (the tree has one agent below the other)
        depth = self.depth * gameState.getNumAgents()
        alpha = float('-Inf')
        beta = float('Inf')
        scores = []
        #Calling min agents from the root node (considering root node as max agent)
        for action in actions:
           score = self.AlphaBeta(gameState.generateSuccessor(0, action), depth-1, 1, gameState.getNumAgents(), alpha, beta)
           scores.append(score)
           #be sure to update alpha in the root (max agent)
           alpha = max(alpha, score)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        #Return the best move by selecting a random action from multiple best choices (if any)
        return actions[chosenIndex]

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

