# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        "*** YOUR CODE HERE ***"
        countdownScore = successorGameState.getScore()

        #print(f"numagents:{successorGameState.getNumAgents()}")

        ghostPos = successorGameState.getGhostPosition(1)
        
        #print(f"pacman:{newPos}")
        #print(f"ghost:{ghostPos}")

        distToFood=[]
        for food in newFood.asList():
            distToFood.append(util.manhattanDistance(newPos,food))
        
        foodDistanceScore = 1
        # find closet food and add to score
        if distToFood:
            foodDistanceScore = min(distToFood)

        # get the distance to the ghost
        ghostDist = util.manhattanDistance(newPos,ghostPos) 
        scaredScore = 0
        # if its close, make it scared
        if ghostDist < 2:
            scaredScore = -500

        score = countdownScore + scaredScore + 10/foodDistanceScore

        #print(f"score:{score}\n")
        return score
        #return -1 * len(newFood.asList())
        #newGhostStates
        #return successorGameState

def scoreEvaluationFunction(currentGameState: GameState):
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

    gameState.isWin():
    Returns whether or not the game state is a winning state

    gameState.isLose():
    Returns whether or not the game state is a losing state
    """
    "*** YOUR CODE HERE ***"

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """

        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex)
            if agentIndex == 0: #pacman turn
                bestValue = float('-inf')
                for action in actions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = minimax(1, depth, successor)
                    if value > bestValue:
                        bestValue = value
                return bestValue
            else:  # Ghosts' turn
                bestValue = float('inf')
                nextAgent = (agentIndex + 1) % gameState.getNumAgents() # We want to go through all ghosts.
                for action in actions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    arg = depth
                    if nextAgent == 0:
                        arg = depth - 1
                    value = minimax(nextAgent, arg, successor)
                    if value < bestValue:
                        bestValue = value
                return bestValue


        actions = gameState.getLegalActions(0)
        bestScore = float('-inf')
        bestAction = None

        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            score = minimax(1, self.depth, successor)  # Start with the ghost (index 1)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def value(self, gameState, depth, agentIndex, alpha, beta):
        numAgents = gameState.getNumAgents()
        
        # Base case: terminal state or max depth
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return (self.evaluationFunction(gameState), None)
        
        # Determine if we're maximizing or minimizing
        if agentIndex == 0:  # Pacman's turn (maximizing)
            v = [-float("inf"), None]
            legalActions = gameState.getLegalActions(agentIndex)
            for act in legalActions:
                succ = gameState.generateSuccessor(agentIndex, act)
                valOfSucc = self.value(succ, depth, 1, alpha, beta)[0]  # move to the next agent (ghost)
                v[1] = act if valOfSucc > v[0] else v[1]
                v[0] = max(v[0], valOfSucc)
                
                # Pruning
                if v[0] > beta:
                    return v
                alpha = max(alpha, v[0])

        else:  # Ghost's turn (minimizing)
            v = [float("inf"), None]
            legalActions = gameState.getLegalActions(agentIndex)
            for act in legalActions:
                succ = gameState.generateSuccessor(agentIndex, act)
                if agentIndex == numAgents - 1:  # Last ghost
                    # If it's the last ghost, it's either a terminal state or max depth for pacman to move
                    if depth != self.depth:
                        valOfSucc = self.value(succ, depth + 1, 0, alpha, beta)[0]  # Pacman's turn next
                    else:
                        valOfSucc = self.evaluationFunction(succ)  # Terminal state reached
                else:
                    valOfSucc = self.value(succ, depth, agentIndex + 1, alpha, beta)[0]  # Move to the next ghost

                v[1] = act if valOfSucc < v[0] else v[1]
                v[0] = min(v[0], valOfSucc)
                
                # Pruning
                if v[0] < alpha:
                    return v
                beta = min(beta, v[0])

        return tuple(v)


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 1, 0, -float("inf"), float("inf"))[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction.
        All ghosts should be modeled as choosing uniformly at random from their legal moves.
        """
        # Start with Pacman's turn (agentIndex = 0), depth = 1
        actions = gameState.getLegalActions(0)
        bestScore = float('-inf')
        bestAction = None

        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            score = self.expectimax(1, self.depth, successor)  # Start with the ghost (index 1)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

    def expectimax(self, agentIndex, depth, gameState):
        """
        Recursive expectimax function (outside the getAction method).
        This function should handle both the maximizing (Pacman's) and minimizing (ghosts') behavior.
        """
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agentIndex)
        
        if agentIndex == 0:  # Pacman's turn (maximizing)
            bestValue = float('-inf')
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                value = self.expectimax(1, depth, successor)  # Move to the next agent (ghost)
                bestValue = max(bestValue, value)
            return bestValue
        else:  # Ghost's turn (minimizing)
            totalValue = 0
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                value = self.expectimax(nextAgent, depth - 1 if nextAgent == 0 else depth, successor)
                totalValue += value

            # Return the expected value for the ghost
            if actions:
                return totalValue / len(actions)
            return 0



def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: What I did first was I found the distance to the closest ghost, food, capsul.
    Next I found the value of being close to a ghost, if I had a capsul active, I would try to chase 
    the ghost if it was within range. If I didn't have a capsul active, I would play normally and get 
    the food and capsuls, if I was close to a ghost, I would incurr a negative score for that part and a 
    super large score if the ghost was right next to me. 
    Lastly I would sum up the scores and distances with different weights to give a final score/value of the current state.
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)

    pos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()
    ghostPos = currentGameState.getGhostPosition(1)

    countdownScore = currentGameState.getScore()

    # find distance to all food
    distToFood=[]
    for food in foodGrid.asList():
        distToFood.append(util.manhattanDistance(pos,food))
    
    foodDistanceScore = 1
    # find closet food
    if distToFood:
        foodDistanceScore = min(distToFood)


    # maybe only check for capsul if ghost isnt scared??

    # find distance to all capsuls 
    distToCapsul=[]
    for capsul in capsules:
        distToCapsul.append(util.manhattanDistance(pos,capsul))
    
    capsulDistanceScore = 1
    # find closet capsul
    if distToCapsul:
        capsulDistanceScore = min(distToCapsul)



    #print(f"scared times:{scaredTimes}")
    # get the distance to the ghost
    ghostDist = util.manhattanDistance(pos,ghostPos) 
    scaredScore = 0

    # check if the ghost isnt current scared
    if scaredTimes[0] == 0:
        # if its close and not scared, run
        if ghostDist < 2:
            scaredScore = -1000
        # if its close and not scared, depth 2 check
        elif ghostDist < 4:
            scaredScore = -200

    # if its close and scared check if its reachable, and chase the ghost
    elif scaredTimes[0] > 0 and ghostDist < scaredTimes[0]:
        scaredScore = 100 / ghostDist


    score = countdownScore + 10/foodDistanceScore + 15/capsulDistanceScore + scaredScore

    #print(f"score:{score}\n")
    return score

# Abbreviation
better = betterEvaluationFunction
