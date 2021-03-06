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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        # Initialize variables
        reward = 0
        penalty = 0
        infinity = 999999999999
        closest_food = infinity
        closest_ghost = infinity
        remaining_food = successorGameState.getNumFood()
        farthest_food = -1 * infinity

        # If remaining_food is 0 make it one for integer division
        remaining_food = remaining_food if remaining_food != 0 else  1

        # Use food to calculate score
        for food_pos in newFood.asList():
            closest_food = min(util.manhattanDistance(newPos, food_pos), closest_food)
            farthest_food = max(util.manhattanDistance(newPos, food_pos), farthest_food)

        # Add reward if the next position is a food pellet
        if closest_food == 0:
            reward += 100
            closest_food = 1
        # To avoid division by 0
        farthest_food = farthest_food if farthest_food != 0 else 1

        # Use ghost to calculate score
        manhattan_dist = lambda x,y : util.manhattanDistance(x,y)
        closest_ghost = min([manhattan_dist(ghostPos.getPosition(), newPos) for ghostPos in newGhostStates])

        # if next position is ghost avoid it all cost, hence return -infinity
        if closest_ghost == 0:
            penalty = float(infinity)
            closest_ghost = 1

        # Calculate the final score.
        # The final score can be calculated by using the current score and the following factors
        # Distance of closest food from pacman
        # distance of farthest food from pacman
        # number of food pellets remaining
        # Add the reward (next state if food) and subtract the penalty (next state is ghost)
        score = float(score) - float(1)/closest_ghost + \
                (float(1)/ closest_food) \
                + reward - penalty + \
                float(1)/remaining_food + \
                float(1)/farthest_food
        # Return score if action is not stop
        return score if action != 'Stop' else -1*infinity

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
        # Initialize the infinity variable
        infinity = 9999999999999999
        # A recursive helper function which will return the best score calculated using min-max tree
        def minimax(game_state, agentIndex, depth, max_depth):
            # We are going to the increase the depth only when we have completed one ply ie pacman and 'N' ghosts
            if agentIndex == game_state.getNumAgents():
                depth += 1
                agentIndex = 0
            # Base case:
            if game_state.isWin() or game_state.isLose() or depth == max_depth+1:
                return self.evaluationFunction(game_state)

            res = infinity if agentIndex != 0 else -1*infinity

            for action in game_state.getLegalActions(agentIndex):
                # Calculate the successor
                successor_state = game_state.generateSuccessor(agentIndex, action)
                # If the current agent is Pacman then we execute 'max' step
                if agentIndex ==0:
                    # take max of all children
                    res = max(res, minimax(successor_state, agentIndex+1, depth, max_depth))
                # current agent is a ghost
                else:
                    # take minimum of all children
                    res = min(res, minimax(successor_state, agentIndex+1, depth, max_depth))

            return res

        #Get Action Code Starts here
        result = ""
        maxscore = -1*infinity
        curr_agent = self.index
        max_depth = self.depth
        # Looking at all the legal actions of the root node
        for action in gameState.getLegalActions(self.index):
            # generate all successors
            successor = gameState.generateSuccessor(curr_agent, action)
            # calculate score of all the child nodes of the tree
            score = minimax(successor, curr_agent+1, 1, max_depth)
            # if the current score is greater than max return this score
            if score > maxscore:
                result = action
                maxscore = score
        return result


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        infinity = 9999999999999999

        def alphabeta(game_state, agentIndex, depth, max_depth, alpha, beta):

            if agentIndex == game_state.getNumAgents():
                depth += 1
                agentIndex = 0

            if game_state.isWin() or game_state.isLose() or depth == max_depth + 1:
                return self.evaluationFunction(game_state)

            res = infinity if agentIndex != 0 else -1 * infinity

            for action in game_state.getLegalActions(agentIndex):
                successor_state = game_state.generateSuccessor(agentIndex, action)
                if agentIndex == 0:
                    val =  alphabeta(successor_state, agentIndex + 1, depth, max_depth, alpha, beta)
                    res = max(res, val)
                    if res > beta: return res
                    alpha = max(alpha, res)

                else:
                    val = alphabeta(successor_state, agentIndex + 1, depth, max_depth, alpha, beta)
                    res = min(res, val)
                    if res < alpha: return res
                    beta = min(beta, res)

            return res
        # get action
        result = ""
        curr_score = -1*infinity
        maxscore = -1 * infinity
        curr_agent = self.index
        max_depth = self.depth
        alpha = -1 * infinity
        beta = infinity
        for action in gameState.getLegalActions(self.index):
            successor = gameState.generateSuccessor(curr_agent, action)
            val = alphabeta(successor, curr_agent + 1, 1, max_depth, alpha, beta)
            if val > curr_score:
                curr_score = val
                result = action
            if val > beta:
                return result
            alpha = max(alpha, val)
        return result

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
        # Adding variable which will store infinity
        infinity = float(9999999999999999)
        def minimax(game_state, agentIndex, depth, max_depth):
            # if we reach the maximum number of agents we will increase the depth by 1. One ply completed here
            if agentIndex == game_state.getNumAgents():
                depth += 1
                agentIndex = 0
            # Base case
            if game_state.isWin() or game_state.isLose() or depth == max_depth:
                return float(self.evaluationFunction(game_state))
            # Initialize result
            res = 0.0 if agentIndex != 0 else -1 * infinity

            legal_moves = game_state.getLegalActions(agentIndex)
            prob = float(1)/len(legal_moves)

            for action in legal_moves:
                successor_state = game_state.generateSuccessor(agentIndex, action)
                if agentIndex == 0:
                    res = float(max(res, minimax(successor_state, agentIndex + 1, depth, max_depth)))
                else:

                    res += prob * float(minimax(successor_state, agentIndex + 1, depth, max_depth))

            return res

        # get action
        result = ""
        maxscore = -1 * infinity
        curr_agent = self.index
        max_depth = self.depth

        for action in gameState.getLegalActions(self.index):
            successor = gameState.generateSuccessor(curr_agent, action)
            score = float(minimax(successor, curr_agent + 1, 0, max_depth))
            if score > maxscore:
                result = action
                maxscore = score
        return result

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

