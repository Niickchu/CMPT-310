import math
import os.path
from tkinter import *
from agents import *
from search import *
import sys
import copy

import utils
from utils import PriorityQueue

"""
1- BFS: Breadth first search. Using tree or graph version, whichever makes more sense for the problem
2- DFS: Depth-First search. Again using tree or graph version.
3- UCS: Uniform-Cost-Search. Using the following cost function to optimise the path, from initial to current state.
4- A*:  Using A star search.
"""
searchTypes = ['None', 'BFS', 'DFS', 'UCS', 'A*', 'BFS_Turn']


class VacuumPlanning(Problem):
    """ The problem of find the next room to clean in a grid of m x n rooms.
    A state is represented by state of the grid. Each room is specified by index set
    (i, j), i in range(m) and j in range (n). Final goal is to find all dirty rooms. But
     we go by sub-goal, meaning finding next dirty room to clean, at a time."""

    def __init__(self, env, searchtype):
        """ Define goal state and initialize a problem
            initial is a pair (i, j) of where the agent is
            goal is next pair(k, l) where map[k][l] is dirty
        """
        self.solution = None
        self.env = env
        self.state = env.agent.location
        super().__init__(self.state)
        self.map = env.things
        self.searchType = searchtype
        self.agent = env.agent
    def generateSolution(self):
        """ generate search engine based on type of the search chosen by user"""
        self.env.read_env()
        self.state = env.agent.location
        self.env.turns_enabled = False
        super().__init__(self.state)
        if self.searchType == 'BFS':
            path, explored, explored_list = breadth_first_graph_search(self)
            if(path == None):
                self.env.print_not_reachable()
                self.env.display_explored(explored, 'pink')
                self.env.running = False
                return

            for tile in explored_list:
                if self.env.running == False:
                    break
                x,y = tile
                self.env.buttons[y][x].config(bg='pink')
                sleep(0.000001)
                Tk.update(self.env.root)


            sol = path.solution()
            self.env.set_solution(sol)
            if explored != None:
                self.env.display_explored(explored, 'pink')
                self.env.display_solution(path, 'cornflower blue')

        elif self.searchType == 'DFS':
            # self.env.solution = depth_first_graph_search(self).solution()
            path, explored = depth_first_graph_search(self)
            if(path == None):
                self.env.print_not_reachable()
                self.env.display_explored(explored, 'pink')
                self.env.running = False
                return
            sol = path.solution()
            self.env.set_solution(sol)
            if explored != None:
                self.env.display_explored(explored, 'pink')
                self.env.display_solution(path, 'cornflower blue')

        elif self.searchType == 'UCS':
            # self.env.solution = best_first_graph_search(self, lambda node: node.path_cost).solution()
            path, explored, listo = best_first_graph_search(self, lambda node: node.path_cost)
            if (path == None):
                self.env.print_not_reachable()
                self.env.display_explored(explored, 'pink')
                self.env.running = False
                return
            sol = path.solution()
            self.env.set_solution(sol)
            if explored != None:
                self.env.display_explored(explored, 'pink')
                self.env.display_solution(path, 'cornflower blue')

        elif self.searchType == 'A*':
            path, explored, explored_list = astar_search(self)
            self.env.clear_explored()

            if(path == None):
                self.env.print_not_reachable()
                self.env.display_explored(explored, 'pink')
                self.env.running = False
                return

            for tile in explored_list:
                if self.env.running == False:
                    break
                x,y = tile
                self.env.buttons[y][x].config(bg='pink')
                sleep(0.000001)
                Tk.update(self.env.root)

            sol = path.solution()
            self.env.set_solution(sol)
            if explored != None:
                self.env.display_explored(explored, 'pink')
                self.env.display_solution(path, 'cornflower blue')

        elif self.searchType == 'BFS_Turn':
            self.env.turns_enabled = True
            path, explored = breadth_first_graph_search(self)

            if (path == None):
                self.env.print_not_reachable()
                self.env.display_explored(explored, 'pink')
                self.env.running = False
                return

            sol = path.solution()
            self.env.set_solution(sol)
            if explored != None:
                self.env.display_explored(explored, 'pink')
                self.env.display_solution(path, 'cornflower blue')


    def generateNextSolution(self):
        self.generateSolution()


    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        state_location = state
        possible_neighbors = self.env.things_near(state)
        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

        for neighbor in possible_neighbors:
            if isinstance(neighbor[0], Wall):
                if neighbor[2][0] < state_location[0]:
                    possible_actions.remove('LEFT')
                elif neighbor[2][0] > state_location[0]:
                    possible_actions.remove('RIGHT')
                elif neighbor[2][1] > state_location[1]:
                    possible_actions.remove('UP')
                elif neighbor[2][1] < state_location[1]:
                    possible_actions.remove('DOWN')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        if action == 'UP':
            return (state[0], state[1] + 1)

        elif action == 'DOWN':
            return (state[0], state[1] - 1)

        elif action == 'LEFT':
            return (state[0] - 1, state[1])

        elif action == 'RIGHT':
            return (state[0] + 1, state[1])

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """
        return self.env.some_things_at(state, Dirt)
    def path_cost(self, c, state1, action, state2): #used as the cost function for UCS and A* search
        """To be used for UCS and A* search. Returns the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. For our problem
        state is (x, y) coordinate pair. To make our problem more interesting we are going to associate
        a height to each state as z = sqrt(x*x + y*y). This effectively means our grid is a bowl shape and
        the center of the grid is the center of the bowl. So now the distance between 2 states become the
        square of Euclidean distance as distance = (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2"""

        x1, y1 = state1
        x2, y2 = state2
        z1 = math.sqrt((x1-self.env.width/2)*(x1-self.env.width/2) + (y1-self.env.height/2)*(y1-self.env.height/2))
        z2 = math.sqrt((x2-self.env.width/2)*(x2-self.env.width/2) + (y2-self.env.height/2)*(y2-self.env.height/2))
        cost = c + (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2)
        return cost


    def h(self, node):
        """ to be used for A* search. Return the heuristic value for a given state. For this problem use minimum Manhattan
        distance to all the dirty rooms + absolute value of height distance as described above in path_cost() function. .

    """
        list_of_dirt = self.env.returnDirtyRooms()
        best_manhattan_distance = 10000000

        dirt_location = list_of_dirt[0]

        for x,y in list_of_dirt:
            manhattan_distance = abs(x - node.state[0]) + abs(y - node.state[1])
            if manhattan_distance < best_manhattan_distance:
                best_manhattan_distance = manhattan_distance
                dirt_location = (x,y)

        dx, dy = dirt_location
        x, y = node.state
        z1 = math.sqrt((x - self.env.width / 2)**2 + (y - self.env.height / 2)**2)
        z2 = math.sqrt((dx - self.env.width / 2)**2 + (dy - self.env.height / 2)**2)
        return best_manhattan_distance + abs(z1 - z2)

# ______________________________________________________________________________


def agent_label(agt):
    """creates a label based on direction"""
    dir = agt.direction
    lbl = '^'
    if dir.direction == Direction.D:
        lbl = 'v'
    elif dir.direction == Direction.L:
        lbl = '<'
    elif dir.direction == Direction.R:
        lbl = '>'

    return lbl


def is_agent_label(lbl):
    """determines if the label is one of the labels tht agents have: ^ v < or >"""
    return lbl == '^' or lbl == 'v' or lbl == '<' or lbl == '>'


class Gui(VacuumEnvironment):
    """This is a two-dimensional GUI environment. Each location may be
    dirty, clean or can have a wall. The user can change these at each step.
    """
    xi, yi = (0, 0)

    perceptible_distance = 1

    def __init__(self, root, width, height):
        self.searchAgent = None
        print("creating xv with width ={} and height={}".format(width, height))
        super().__init__(width, height)

        self.agent = None
        self.root = root
        self.blocked_dirt = False
        self.running = False
        self.turns_enabled = False
        self.create_frames(height)
        self.create_buttons(width)
        self.create_walls()
        self.setupTestEnvironment()

    def setupTestEnvironment(self):
        """ first reset the agent"""
        if self.agent is not None:
            xi, yi = self.agent.location
            self.buttons[yi][xi].config(bg='white', text='', state='normal')
            x = self.width // 2
            y = self.height // 2
            self.agent.location = (x, y)
            self.agent.direction.direction = Direction.U
            self.buttons[y][x].config(text=agent_label(self.agent), state='normal')
            self.searchType = searchTypes[1]
            self.agent.performance = 0

        """next create a random number of block walls inside the grid as well"""
        # roomCount = (self.width - 1) * (self.height - 1)
        # blockCount = random.choice(range(roomCount//10, roomCount//5))
        # for _ in range(blockCount):
        #     rownum = random.choice(range(1, self.height - 1))
        #     colnum = random.choice(range(1, self.width - 1))
        #     if(colnum == self.width // 2 and rownum == self.height // 2):
        #         continue
        #     self.buttons[rownum][colnum].config(bg='red', text='W', disabledforeground='black')

        #self.create_dirts()
        self.stepCount = 0
        self.searchType = None
        self.solution = []
        self.explored = set()
        self.read_env()

    def returnDirtyRooms(self):
        list = []
        for thing in self.things:
            if isinstance(thing, Dirt):
                list.append(thing.location)

        return list

    def print_not_reachable(self):
        print("No solution found. The following dirty rooms are not reachable: ")
        for thing in self.things:
            if isinstance(thing, Dirt):
                print(thing.location)

    def create_frames(self, h):
        """Adds h row frames to the GUI environment."""
        self.frames = []
        for _ in range(h):
            frame = Frame(self.root, bg='blue')
            frame.pack(side='bottom')
            self.frames.append(frame)

    def create_buttons(self, w):
        """Adds w buttons to the respective row frames in the GUI."""
        self.buttons = []
        for frame in self.frames:
            button_row = []
            for _ in range(w):
                button = Button(frame, bg='white', state='normal', height=1, width=1, padx=1, pady=1)
                button.config(command=lambda btn=button: self.toggle_element(btn))
                button.pack(side='left')
                button_row.append(button)
            self.buttons.append(button_row)

    def create_walls(self):
        """Creates the outer boundary walls which do not move. Also create a random number of
        internal blocks of walls."""
        for row, button_row in enumerate(self.buttons):
            if row == 0 or row == len(self.buttons) - 1:
                for button in button_row:
                    button.config(bg='red', text='W', state='disabled', disabledforeground='black')
            else:
                button_row[0].config(bg='red', text='W', state='disabled', disabledforeground='black')
                button_row[len(button_row) - 1].config(bg='red', text='W', state='disabled', disabledforeground='black')

    def create_dirts(self):
        """ set a small random number of rooms to be dirty at random location on the grid
        This function should be called after create_walls()"""
        self.read_env()   # this is needed to make sure wall objects are created
        roomCount = (self.width-1) * (self.height -1)
        self.dirtCount = random.choice(range(5, 15))
        dirtCreated = 0
        while dirtCreated != self.dirtCount:
            rownum = random.choice(range(1, self.height-1))
            colnum = random.choice(range(1, self.width-1))
            if self.some_things_at((colnum, rownum)) or (rownum == self.width // 2 and colnum == self.height // 2):
                continue
            self.buttons[rownum][colnum].config(bg='grey')
            dirtCreated += 1

    def setSearchEngine(self, choice):
        """sets the chosen search engine for solving this problem"""
        self.read_env()
        self.searchType = choice
        if(choice == searchTypes[0]):
            self.clear_explored()
            return
        self.searchAgent = VacuumPlanning(self, self.searchType)
        if(self.dirtCount > 0):
            self.searchAgent.generateSolution()
            self.done = False

    def set_solution(self, sol):
        self.solution = list(reversed(sol))

    def display_solution(self, solution_node, colour):
        for node in solution_node.path()[:-1]:
            x,y = node.state
            self.buttons[y][x].config(bg=colour)

    def clear_explored(self):
        if len(self.explored) > 0:     # means we have explored list from previous search. So need to clear their visual fist
            for (x, y) in self.explored:
                self.buttons[y][x].config(bg='white')
    def display_explored(self, explored, colour):
        """display explored slots in a light pink color"""
        self.clear_explored()

        self.explored = explored
        for (x, y) in explored:
            self.buttons[y][x].config(bg=colour)

    def add_agent(self, agt, loc):
        """add an agent to the GUI"""
        self.add_thing(agt, loc)
        # Place the agent at the provided location.
        lbl = agent_label(agt)
        self.buttons[loc[1]][loc[0]].config(text=lbl, state='normal')
        self.agent = agt

    def toggle_element(self, button):
        """toggle the element type on the GUI."""
        bgcolor = button['bg']
        txt = button['text']
        if is_agent_label(txt):
            if bgcolor == 'grey':
                button.config(bg='white', state='normal')
            else:
                button.config(bg='grey')
        else:
            if bgcolor == 'red':
                button.config(bg='grey', text='')
            elif bgcolor == 'grey':
                button.config(bg='white', text='', state='normal')
            elif bgcolor == 'white':
                button.config(bg='red', text='W')
            elif bgcolor == 'pink':
                button.config(bg='grey', text='')

    def turn_agent(self, agent_direction, action):
        """Turns the agent towards the direction of the action."""
        if action == 'LEFT':
            if agent_direction == 'UP':
                self.agent.direction += Direction.L
            elif agent_direction == 'DOWN':
                self.agent.direction += Direction.R
            elif agent_direction == 'RIGHT':
                self.agent.direction += Direction.L
        elif action == 'RIGHT':
            if agent_direction == 'UP':
                self.agent.direction += Direction.R
            elif agent_direction == 'DOWN':
                self.agent.direction += Direction.L
            elif agent_direction == 'LEFT':
                self.agent.direction += Direction.R
        elif action == 'DOWN':
            if agent_direction == 'LEFT':
                self.agent.direction += Direction.L
            elif agent_direction == 'RIGHT':
                self.agent.direction += Direction.R
            elif agent_direction == 'UP':
                self.agent.direction += Direction.R
        elif action == 'UP':
            if agent_direction == 'LEFT':
                self.agent.direction += Direction.R
            elif agent_direction == 'RIGHT':
                self.agent.direction += Direction.L
            elif agent_direction == 'DOWN':
                self.agent.direction += Direction.R
    def execute_action(self, agent, action):
        """Determines the action the agent performs."""
        agent.bump = False
        agent.performance -= 1
        self.stepCount += 1
        xi, yi = agent.location

        if action == 'SUCK':
            dirt_list = self.list_things_at((xi, yi), Dirt)
            if dirt_list != []:
                dirt = dirt_list[0]
                agent.performance += 100
                self.delete_thing(dirt)
                self.dirtCount -= 1
                self.buttons[yi][xi].config(bg='white')
                NumSteps_label.config(text=str(self.stepCount))
                TotalCost_label.config(text=str(self.agent.performance))
                return

        if self.turns_enabled:
            if action != agent.direction.direction.upper():
                self.turn_agent(agent.direction.direction.upper(), action)
                x,y = agent.location
                self.buttons[y][x].config(text=agent_label(agent))
                NumSteps_label.config(text=str(self.stepCount))
                TotalCost_label.config(text=str(self.agent.performance))
                return


        if action == 'LEFT':
            self.move_to(agent, (agent.location[0]-1, agent.location[1]))

        elif action == 'RIGHT':
            self.move_to(agent, (agent.location[0]+1, agent.location[1]))

        elif action == 'UP':
            self.move_to(agent, (agent.location[0], agent.location[1]+1))               # maybe x and y are reversed

        elif action == 'DOWN':
            self.move_to(agent, (agent.location[0], agent.location[1]-1))

        NumSteps_label.config(text=str(self.stepCount))
        TotalCost_label.config(text=str(self.agent.performance))

        self.buttons[yi][xi].config(bg='pink', text='')
        xf, yf = agent.location
        self.buttons[yf][xf].config(text=agent_label(agent))

    def read_env(self):
        """read_env: This sets proper wall or Dirt status based on bg color"""
        """Reads the current state of the GUI environment."""
        self.dirtCount = 0
        for j, btn_row in enumerate(self.buttons):
            for i, btn in enumerate(btn_row):
                if (j != 0 and j != len(self.buttons) - 1) and (i != 0 and i != len(btn_row) - 1):
                    if self.some_things_at((i, j)):  # and (i, j) != agt_loc:
                        for thing in self.list_things_at((i, j)):
                            if not isinstance(thing, Agent):
                                self.delete_thing(thing)
                    if btn['bg'] == 'grey':  # adding dirt
                        self.add_thing(Dirt(), (i, j))
                        self.dirtCount += 1
                    elif btn['bg'] == 'red':  # adding wall
                        self.add_thing(Wall(), (i, j))

    def update_env(self):
        """Updates the GUI environment according to the current state."""
        self.read_env()
        self.step()

    def step(self):
        """updates the environment one step. Currently it is associated with one click of 'Step' button.
        """
        if env.dirtCount == 0:
            print("Everything is clean. DONE!")
            self.done = True
            self.running = False
            self.clear_explored()
            return

        if len(self.solution) == 0:
            self.execute_action(self.agent, 'SUCK')
            self.read_env()
            if env.dirtCount > 0 and self.searchAgent is not None:
                self.searchAgent.generateNextSolution()
        else:
            if self.turns_enabled:
                move = self.solution[-1]
                if move == self.agent.direction.direction.upper():
                    move = self.solution.pop()
                self.execute_action(self.agent, move)
            else:
                self.execute_action(self.agent, self.solution.pop())

    def run(self, delay=0.15): #Set delay to 0.1 for faster animation
        """Run the Environment for given number of time steps,"""
        self.running = True
        while self.running:
            self.update_env()
            sleep(delay)
            Tk.update(self.root)

    def reset_env(self):
        """Resets the GUI and agents environment to the initial clear state."""
        self.running = False
        NumSteps_label.config(text=str(0))
        TotalCost_label.config(text=str(0))

        for j, btn_row in enumerate(self.buttons):
            for i, btn in enumerate(btn_row):
                if (j != 0 and j != len(self.buttons) - 1) and (i != 0 and i != len(btn_row) - 1):
                    if self.some_things_at((i, j)):
                        for thing in self.list_things_at((i, j)):
                            self.delete_thing(thing)
                    btn.config(bg='white', text='', state='normal')
        self.setupTestEnvironment()

"""
Our search Agents ignore ignore environment percepts for planning. The planning is done based on static
 data from the environment at the beginning. The environment if fully observable
 """
def XYSearchAgentProgram(percept):
    pass


class XYSearchAgent(Agent):
    """The modified SimpleRuleAgent for the GUI environment."""

    def __init__(self, program, loc):
        super().__init__(program)
        self.location = loc
        self.direction = Direction("up")
        self.searchType = searchTypes[0]
        self.stepCount = 0
        #self.colour = 'cornflower blue'


if __name__ == "__main__":
    win = Tk()
    win.title("Searching Cleaning Robot")
    win.geometry("600x650+50+50")
    win.resizable(True, True)
    frame = Frame(win, bg='black')
    frame.pack(side='bottom')
    topframe = Frame(win, bg='black')
    topframe.pack(side='top')

    wid = 30
    if len(sys.argv) > 1:
        wid = int(sys.argv[1])

    hig = 20
    if len(sys.argv) > 2:
        hig = int(sys.argv[2])

    env = Gui(win, wid, hig)

    theAgent = XYSearchAgent(program=XYSearchAgentProgram, loc=(hig//2, wid//2))
    x, y = theAgent.location
    env.add_agent(theAgent, (y, x))

    NumSteps_label = Label(topframe, text='NumSteps: 0', bg='green', fg='white', bd=2, padx=2, pady=2)
    NumSteps_label.pack(side='left')
    TotalCost_label = Label(topframe, text='TotalCost: 0', bg='blue', fg='white', padx=2, pady=2)
    TotalCost_label.pack(side='right')
    reset_button = Button(frame, text='Reset', height=2, width=5, padx=2, pady=2)
    reset_button.pack(side='left')
    next_button = Button(frame, text='Next', height=2, width=5, padx=2, pady=2)
    next_button.pack(side='left')
    run_button = Button(frame, text='Run', height=2, width=5, padx=2, pady=2)
    run_button.pack(side='left')

    next_button.config(command=env.update_env)
    reset_button.config(command=env.reset_env)
    run_button.config(command=env.run)

    searchTypeStr = StringVar(win)
    searchTypeStr.set(searchTypes[0])
    searchTypeStr_dropdown = OptionMenu(frame, searchTypeStr, *searchTypes, command=env.setSearchEngine)
    searchTypeStr_dropdown.pack(side='left')

    win.mainloop()
