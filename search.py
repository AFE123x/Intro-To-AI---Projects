# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""


import util
from game import Directions
from game import Configuration as Config
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    #Directions.getPosition()
    #game.getDirection()
    #game.generateSuccessor()

    stack = util.Stack()
    visited = list()
    print(f'Start state:{problem.getStartState()}')
    # successor, direction, cost
    stack.push((problem.getStartState(), None, None))
    while stack.isEmpty() is not True:
        # x = input("press enter")
        # get the edges and put in list
        stackpop = stack.pop()
        visited.append(stackpop[0])
        print(f"{stackpop}")
        if problem.isGoalState(stackpop[0]):
            break
        successors = problem.getSuccessors(stackpop[0])
        print(f'successors: {successors}')
        
        # add the edges to stack
        for successor in successors:
            # makes sure we haven't been to this location before
            if successor[0] not in visited:
                stack.push(successor)
                print(f'pushed {successor}')
    



def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """

    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""

    # my queue
    myPriorityQueue = util.PriorityQueue()

    # coord: weight of path to coord
    visitedCoords = {}

    start=problem.getStartState()
    # ((coords, path, current node cost), current heuristic value)
    myPriorityQueue.push((start, [], 0),heuristic(start,problem))
    
    # While loop
    while not myPriorityQueue.isEmpty():

        # pop the node with the lowest cost, priority queue automatically pops the lowest cost
        current_coord, current_path, current_path_cost = myPriorityQueue.pop()

        # debug
        # print(f"cur coord:{current_coord}")
        # print(f"cur path:{current_path}")
        # print(f"cur path:{current_path_cost}")
        # input()
        # if at the goal state, return current path
        if problem.isGoalState(current_coord):
            #print(current_path)
            return current_path
        
        # if we didnt visit the node OR the current path cost to that node is less than the previous cost to that node
        if current_coord not in visitedCoords or current_path_cost < visitedCoords[current_coord]:
            # add to visitedCoords list
            visitedCoords[current_coord] = current_path_cost

            # iterate through all the successors of the current node
            for suc_coord, suc_direction, suc_step_cost in problem.getSuccessors(current_coord):
                # debug
                # print(f"suc:({suc_coord},{suc_direction},{suc_step_cost})")
 
                # calculate the new path cost
                # find g(n)
                new_path_cost = current_path_cost + suc_step_cost
                # check if the successor we are looking at has been visited already 
                # OR if it has, see if the new path cost cost is less than the previous path cost
                if suc_coord not in visitedCoords or new_path_cost < visitedCoords[suc_coord]:
                    
                    # calculate new manhattan distance cost
                    # find h(n)
                    greedy_cost = heuristic(suc_coord, problem)

                    # find the new A* cost
                    # find f(n) 
                    new_cost = new_path_cost + greedy_cost

                    # add the direction to the new solution path
                    new_path = current_path + [suc_direction]

                    # push the successor with the updated path and new cost
                    myPriorityQueue.push((suc_coord,new_path, new_path_cost), new_cost)
    
    # incase goal wasn't reached
    return []
    

    
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
