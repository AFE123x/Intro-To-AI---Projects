# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import backend
import gridworld


import random,util,math
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # store q values
        # key is q-state ie (state, action) = reward
        self.q_vals = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.q_vals[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = self.getLegalActions(state)
        max = None
        for action in actions:
            qval = self.getQValue(state, action)
            if max is None:
                max = (qval, action)
            elif max[0] < self.getQValue(state, action):
              max = (qval, action)

        if max is not None:
          return max[0]
        return 0    

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = self.getLegalActions(state)
        max = None
        for action in actions:
            if max is None:
                max = (self.getQValue(state, action), action)
            elif max[0] < self.getQValue(state, action):
              max = (self.getQValue(state, action), action)
        if max is not None:
          return max[1]
        return None


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        
        if(util.flipCoin(self.epsilon)):
          # do random
          return random.choice(legalActions)
        
        # take best policy action  
        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        self.q_vals[(state, action)] = self.q_vals[(state, action)] + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
      """
      @description: Returns the Q-value for a given state-action pair.
      @param state: The current state.
      @param action: The action taken in the current state.
      @return: The Q-value computed as the dot product of weights and features.
      """
      features = self.featExtractor.getFeatures(state, action)
      sum = 0
      for (feature,value) in features.items():
          sum = sum + (self.weights[feature] * value)
      return sum

    def update(self, state, action, nextState, reward: float):
      """
      @description: Updates the weights based on the observed transition.
      @param state: The current state.
      @param action: The action taken in the current state.
      @param nextState: The resulting state after the action.
      @param reward: The reward received after taking the action.
      """
      features = self.featExtractor.getFeatures(state, action)
      legalActions = self.getLegalActions(nextState)
      maxQNext = 0  # Default value if there are no legal actions

      if legalActions:  # Check if there are any legal actions
          maxQNext = self.getQValue(nextState, legalActions[0])  # Initialize with the first Q-value
          for nextAction in legalActions[1:]:  # Iterate through the rest of the actions
              qValue = self.getQValue(nextState, nextAction)
              if qValue > maxQNext:
                  maxQNext = qValue

      difference = (reward + self.discount * maxQNext) - self.getQValue(state, action)
      for (feature, value) in features.items():
          self.weights[feature] += self.alpha * difference * value

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            print(self.weights)
            pass
