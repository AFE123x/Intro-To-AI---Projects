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
    # Start's successors: [('B', '0:A->B', 1.0), ('G', '1:A->G', 2.0), ('D', '2:A->D', 4.0)]
    # mystack = util.Stack()
    # visitedtuple = []
    # visitedcoord = []

    # # Push the start state onto the stack
    # mystack.push((problem.getStartState(), None, '', 0))
    # current = None

    # while not mystack.isEmpty():
    #     current = mystack.pop()
    #     visitedcoord.append(current[0])
    #     visitedtuple.append(current)
    #     if problem.isGoalState(current[0]):
    #         break
    #     # Explore successors
    #     greaze = problem.getSuccessors(current[0])
    #     for successor in greaze:
    #         if successor[0] not in visitedcoord:
    #             mystack.push((successor[0], current[0], successor[1], current[3] + 1))

    # finallist = []
    # localitybaby = problem.getStartState()
    # while current[0] != localitybaby:
    #     # doing linear search
    #     for link in visitedtuple:
    #         if current[0] == link[0]:
    #             if link[2] == "South":
    #                 finallist.insert(0,Directions.SOUTH)
    #             elif link[2] == "North":
    #                 finallist.insert(0,Directions.NORTH)
    #             elif link[2] == "East":
    #                 finallist.insert(0,Directions.EAST)
    #             else:
    #                 finallist.insert(0,Directions.WEST)
    #             current = (link[1],None,None,0)
    #             break
    # return finallist

    # make stack
    
    # (successor tuple,parent node) 
    mystack = util.Stack()
    
    #(current node)
    #(current node, parent node)
    visitedNodes = []
    # coord of current node
    visitedCoords = []


    mystack.push(((problem.getStartState(),'',0), None))
    current_node = None
    current_temp = None
    # searching the map to find different solutions
    while mystack.isEmpty() != True:
        # get the current node we are going to expand

        
        stack_node = mystack.pop()
        # print(f"stack node:{stack_node}")
        current_temp = stack_node
        
        # (coord, direction, weight)
        current_node = stack_node[0]
        # print(f"current node:{current_node}")


        # (coord, direction, weight)
        parent_node = stack_node[1]
        # print(f"parent node:{parent_node}")

        
        # add current node to visisted nodes
        visitedNodes.append((stack_node))
        visitedCoords.append((current_node[0]))
        
        #check if goal state has been reached
        if problem.isGoalState(current_node[0]):
            break
        successors = problem.getSuccessors(current_node[0])
        # suc: (coord, direction, weight)

        for suc in successors:
            if suc[0] not in visitedCoords:
                mystack.push((suc, current_node))
    
    # find solution
    finallist = []
    startState = problem.getStartState()
            
    # current node : (coord, direction, weight)
    # (((1, 1), 'West', 1), ((2, 1), 'West', 1))
    while current_temp[0][0] != startState:
        # print(current_temp)

        finallist.insert(0,current_temp[0][1])
        
        for visitedNode in visitedNodes:
            # if the parent's coords == current coord
            # print(f'{current_temp} == {visitedNode}')
            if current_temp[1][0] == visitedNode[0][0]:
                current_temp = visitedNode
                break
            
    return finallist


    
    
    

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    mystack = util.Queue()
    
    #(current node)
    #(current node, parent node)
    visitedNodes = []
    # coord of current node
    visitedCoords = []


    mystack.push(((problem.getStartState(),'',0), None))
    current_node = None
    current_temp = None
    # searching the map to find different solutions
    while mystack.isEmpty() != True:
        # get the current node we are going to expand

        
        stack_node = mystack.pop()
        # print(f"stack node:{stack_node}")
        current_temp = stack_node
        
        # (coord, direction, weight)
        current_node = stack_node[0]
        # print(f"current node:{current_node}")


        # (coord, direction, weight)
        parent_node = stack_node[1]
        # print(f"parent node:{parent_node}")

        
        # add current node to visisted nodes
        visitedNodes.append((stack_node))
        visitedCoords.append((current_node[0]))
        
        #check if goal state has been reached
        if problem.isGoalState(current_node[0]):
            break
        successors = problem.getSuccessors(current_node[0])
        # suc: (coord, direction, weight)
        print("successors!!!")
        input()
        for suc in successors:
            if suc[0] not in visitedCoords:
                print(suc)
                mystack.push((suc, current_node))
    
    # find solution
    finallist = []
    startState = problem.getStartState()
            
    # current node : (coord, direction, weight)
    # (((1, 1), 'West', 1), ((2, 1), 'West', 1))
    while current_temp[0][0] != startState:

        finallist.insert(0,current_temp[0][1])
        
        for visitedNode in visitedNodes:
            if current_temp[1][0] == visitedNode[0][0]:
                current_temp = visitedNode
                break
            
    return finallist


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
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
