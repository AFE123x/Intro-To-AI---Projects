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
    myqueue = util.Queue()
    visitedCoords = []

    myqueue.push(((problem.getStartState(),'',0), []))

    # searching the map to find different solutions
    while myqueue.isEmpty() != True:
        # get the current node we are going to expand

        current_node, current_path = myqueue.pop()

        # check if at goal
        if problem.isGoalState(current_node[0]):
            return current_path
        # print(f"stack node:{stack_node}")
        
        # if not at goal node, add current node to visited nodes
        if current_node[0] not in visitedCoords:
            visitedCoords.append(current_node[0])

            # (coords, direction, cost)
            for suc in problem.getSuccessors(current_node[0]):
                # if not in visisted, add to queue
                if suc[0] not in visitedCoords:
                    # add the successors direction to the back of current path LIST
                    temp_path = current_path + [suc[1]]
                    #print(f"current path: {temp_path}")
                    # Add to queue
                    myqueue.push((suc,temp_path))

    # IF NEVER FOUND
    return []



def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # my queue
    myPriorityQueue = util.PriorityQueue()

    # coords
    visitedCoords = []

    # ((coords, path, current node cost), total path cost)
    myPriorityQueue.push((problem.getStartState(), [], 0),0)
    
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
            print(current_path)
            return current_path
        
        # if we didnt visit the node, visit it
        if current_coord not in visitedCoords:
            # add to visitedCoords list
            visitedCoords.append(current_coord)

            # iterate through all the successors of the current node
            for suc_coord, suc_direction, suc_step_cost in problem.getSuccessors(current_coord):
                # check if the successor we are looking at has been visited already 
                if suc_coord not in visitedCoords:
                    # calculate the new path cost
                    new_cost = current_path_cost + suc_step_cost
                    # add the successor with the updated path and new cost
                    new_path = current_path + [suc_direction]
                    myPriorityQueue.push((suc_coord,new_path, new_cost), new_cost)
    
    # incase goal wasn't reached
    return []
    

    

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
            print(current_path)
            return current_path
        
        # if we didnt visit the node OR the current path cost to that node is less than the previous cost to that node
        if current_coord not in visitedCoords or current_path_cost < visitedCoords[current_coord]:
            # add to visitedCoords list
            visitedCoords[current_coord] = current_path_cost

            # iterate through all the successors of the current node
            for suc_coord, suc_direction, suc_step_cost in problem.getSuccessors(current_coord):
                # debug
                #print(f"suc:({suc_coord},{suc_direction},{suc_step_cost})")
 
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
