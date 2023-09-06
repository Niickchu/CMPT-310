import os.path
from tkinter import *
from agents import *
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ''))


def agent_label(agt):
    """creates a label based on direction"""
    dir = agt.direction
    lbl = 'v'
    if dir.direction == Direction.U:
        lbl = '^'
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

    # xi, yi = (0, 0)

    # agent1_xi, agent1_yi = (0, 0)
    # agent2_xi, agent2_yi = (0, 0)
    perceptible_distance = 1
    agentTypes = ['ReflexAgent', 'RuleAgent', "NoAgent"]

    def __init__(self, root, width=7, height=7):
        print("creating xv with width ={} and height={}".format(width, height))
        super().__init__(width, height)

        self.root = root
        self.create_frames(height)
        self.create_buttons(width)
        self.create_walls()
        self.upperAgentType = self.agentTypes[0]
        self.lowerAgentType = self.upperAgentType
        self.hasDirt = False

        self.upper_agent_init_position = (0, 0)
        self.lower_agent_init_position = (0, 0)

    def create_frames(self, h):
        """Adds frames to the GUI environment."""
        self.frames = []
        for _ in range(h):
            frame = Frame(self.root, bg='blue')
            frame.pack(side='bottom')
            self.frames.append(frame)

    def create_buttons(self, w):
        """Adds buttons to the respective frames in the GUI."""
        self.buttons = []
        for frame in self.frames:
            button_row = []
            for _ in range(w):
                button = Button(frame, bg='white', height=2, width=3, padx=2, pady=2)
                button.config(command=lambda btn=button: self.toggle_element(btn))
                button.pack(side='left')
                button_row.append(button)
            self.buttons.append(button_row)

    def create_walls(self):
        """Creates the outer boundary walls which do not move."""
        for row, button_row in enumerate(self.buttons):
            if row == 0 or row == len(self.buttons) - 1:
                for button in button_row:
                    button.config(bg='red', text='W', state='disabled', disabledforeground='black')
            else:
                button_row[0].config(bg='red', text='W', state='disabled', disabledforeground='black')
                button_row[len(button_row) - 1].config(bg='red', text='W', state='disabled', disabledforeground='black')

    def add_agent(self, agt, xyloc):
        """add an agent to the GUI"""
        self.add_thing(agt, xyloc)
        # Place the agent in the centre of the grid.
        self.buttons[xyloc[1]][xyloc[0]].config(bg=agt.colour, text=agent_label(agt))

    def toggle_element(self, button):
        """toggle the element type on the GUI."""
        bgcolor = button['bg']
        txt = button['text']
        if is_agent_label(txt):
            if bgcolor == 'grey':
                button.config(bg='white')
            else:
                button.config(bg='grey')
        else:
            if bgcolor == 'red':
                button.config(bg='grey', text='D')
            elif bgcolor == 'grey':
                button.config(bg='white', text='')
            elif bgcolor == 'white':
                button.config(bg='red', text='W')

    def execute_action(self, agent, action):
        """Determines the action the agent performs."""
        xi, yi = agent.location
        print(agent.type, "at location (", xi, yi, ") and action ", action)
        if action == 'Suck':
            dirt_list = self.list_things_at(agent.location, Dirt)
            if dirt_list:
                dirt = dirt_list[0]
                agent.performance += 100
                self.delete_thing(dirt)
                self.buttons[yi][xi].config(bg=agent.colour)

        else:
            agent.bump = False
            if action == 'TurnRight':
                agent.direction += Direction.R
                self.buttons[yi][xi].config(text=agent_label(agent))
            elif action == 'TurnLeft':
                agent.direction += Direction.L
                self.buttons[yi][xi].config(text=agent_label(agent))
            elif action == 'Forward':
                agent.bump = self.move_to(agent, agent.direction.move_forward(agent.location))
                if not agent.bump:
                    self.buttons[yi][xi].config(bg='white', text='')
                    xf, yf = agent.location
                    self.buttons[yf][xf].config(bg=agent.colour, text=agent_label(agent))

        if action != 'NoOp':
            agent.performance -= 1

        if (agent.location[1] >= self.height / 2):
            upper_performance_label.config(text="Upper Agent " + str(agent.performance))
        else:
            lower_performance_label.config(text="Lower Agent " + str(agent.performance))

    def read_env(self):  # because we can change the environment, we need to read it before each step.
        """read_env: This sets proper wall or Dirt status based on bg color"""

        """Reads the current state of the GUI environment."""
        self.hasDirt = False
        for j, btn_row in enumerate(self.buttons):
            for i, btn in enumerate(btn_row):
                if (j != 0 and j != len(self.buttons) - 1) and (i != 0 and i != len(btn_row) - 1):  # not a wall
                    if self.some_things_at((i, j)) and not (self.some_things_at((i, j), Agent)):
                        for thing in self.list_things_at((i, j)):  # remove all things except the agent
                            if not isinstance(thing, Agent):
                                self.delete_thing(thing)
                    if btn['bg'] == 'grey':  # adding dirt
                        self.add_thing(Dirt(), (i, j))
                        self.hasDirt = True
                    elif btn['bg'] == 'red':  # adding wall
                        self.add_thing(Wall(), (i, j))

    def place_random_dirt(self, probability):
        """Places dirt randomly in the GUI environment."""
        for row, button_row in enumerate(self.buttons):
            for button in button_row:
                if button['bg'] == 'white' and random.uniform(0, 1) < probability:
                    button.config(bg='grey', text='D')
                    self.add_thing(Dirt(), (button_row.index(button), row))
                    self.hasDirt = True

    def update_env(self):
        """Updates the GUI environment according to the current state."""
        self.read_env()
        self.step()

    def run(self, steps=3000):
        """Runs the environment for given number of time steps."""
        print("Running until all dirt is cleaned or", steps, "steps are taken")
        self.read_env()
        increment = 0
        while self.hasDirt:
            self.update_env()
            increment += 1
            if increment > steps:
                print("Stopping after", steps, "steps")
                return
        print("Success. All dirt is cleaned. Steps taken =", increment)

    def toggle_upper_agentType(self):
        """toggles the type of the agent. Choices are 'Reflex' and 'RuleBased'."""
        if env.upperAgentType == env.agentTypes[0]:
            env.upperAgentType = env.agentTypes[1]
        elif env.upperAgentType == env.agentTypes[1]:
            env.upperAgentType = env.agentTypes[2]
        else:
            env.upperAgentType = env.agentTypes[0]

        print("new agentType = ", env.upperAgentType)
        upper_agentType_button.config(text=env.upperAgentType)

        self.reset_env()

    def toggle_lower_agentType(self):
        """toggles the type of the agent. Choices are 'Reflex' and 'RuleBased'."""
        if env.lowerAgentType == env.agentTypes[0]:
            env.lowerAgentType = env.agentTypes[1]
        elif env.lowerAgentType == env.agentTypes[1]:
            env.lowerAgentType = env.agentTypes[2]
        else:
            env.lowerAgentType = env.agentTypes[0]

        print("new agentType = ", env.lowerAgentType)
        lower_agentType_button.config(text=env.lowerAgentType)

        self.reset_env()

    def reset_env(self):
        """Resets the GUI environment to the initial clear state."""
        for j, btn_row in enumerate(self.buttons):
            for i, btn in enumerate(btn_row):
                if (j != 0 and j != len(self.buttons) - 1) and (i != 0 and i != len(btn_row) - 1):
                    if self.some_things_at((i, j)):
                        for thing in self.list_things_at((i, j)):
                            self.delete_thing(thing)
                    btn.config(bg='white', text='', state='normal')

        # clear performance labels
        upper_performance_label.config(text="Upper Agent " + '0')
        lower_performance_label.config(text="Lower Agent " + '0')

        if env.upperAgentType != 'NoAgent':
            if env.upperAgentType == 'RuleAgent':
                new_upper_agent = RuleBasedAgent(program=XYRuleBasedAgentProgram)
            else:
                new_upper_agent = XYReflexAgent(program=XYReflexAgentProgram)

            self.add_agent(new_upper_agent, self.upper_agent_init_position)

        if env.lowerAgentType != 'NoAgent':
            if env.lowerAgentType == 'RuleAgent':
                new_lower_agent = RuleBasedAgent(program=XYRuleBasedAgentProgram)
            else:
                new_lower_agent = XYReflexAgent(program=XYReflexAgentProgram)
            self.add_agent(new_lower_agent, self.lower_agent_init_position)

    def find_adjacent_dirt(self, xyloc):
        list_of_adjacent_dirt = []
        if self.buttons[xyloc[1] + 1][xyloc[0]]['bg'] == 'grey':  # above
            # check if agent is near invisible border. if so, ignore dirt
            if xyloc[1] != self.height / 2 - 1:
                list_of_adjacent_dirt.append((xyloc[0], xyloc[1] + 1))

        if self.buttons[xyloc[1]][xyloc[0] - 1]['bg'] == 'grey':  # left
            list_of_adjacent_dirt.append((xyloc[0] - 1, xyloc[1]))

        if self.buttons[xyloc[1]][xyloc[0] + 1]['bg'] == 'grey':  # right
            list_of_adjacent_dirt.append((xyloc[0] + 1, xyloc[1]))

        if self.buttons[xyloc[1] - 1][xyloc[0]]['bg'] == 'grey':  # below
            # check if agent is near invisible border. if so, ignore dirt
            if xyloc[1] != self.height / 2:
                list_of_adjacent_dirt.append((xyloc[0], xyloc[1] - 1))

        return list_of_adjacent_dirt


# implement this. Rule is as follows: At each location, agent checks all the neighboring location: If a "Dirty"
# location found, agent goes to that location, otherwise follow similar rules as the XYReflexAgentProgram bellow.
def XYRuleBasedAgentProgram(percept):
    status, bump, location, direction = percept
    # first check if current location is dirty, if so, suck
    if status == 'Dirty':
        return 'Suck'

    list_of_adjacent_dirt = env.find_adjacent_dirt(location)

    if len(list_of_adjacent_dirt) == 0:  # no adjacent dirt, so move randomly
        return choose_random_action(bump)

    # loop through list of adjacent dirt, and check if agent is facing any of them. if so, move forward
    for dirt_coord in list_of_adjacent_dirt:
        if is_facing_dirt(dirt_coord=dirt_coord, agt_location=location, direction=direction):
            return 'Forward'

    # else, choose the first adjacent dirt and turn towards it, must check direction agent is facing
    return turn_towards_dirt(dirt_coord=list_of_adjacent_dirt[0], agt_location=location, direction=direction)


def choose_random_action(bump):
    if bump == 'Bump':
        value = random.choice((1, 2))
    else:
        value = random.choice((1, 2, 3, 4))  # 1-right, 2-left, others-forward

    if value == 1:
        return 'TurnRight'
    elif value == 2:
        return 'TurnLeft'
    else:
        return 'Forward'


def is_facing_dirt(dirt_coord, agt_location, direction):
    agt_x, agt_y = agt_location
    dirt_x, dirt_y = dirt_coord

    if (dirt_x == agt_x and dirt_y == agt_y + 1 and direction.direction == 'up') or \
            (dirt_x == agt_x and dirt_y == agt_y - 1 and direction.direction == 'down') or \
            (dirt_x == agt_x + 1 and dirt_y == agt_y and direction.direction == 'right') or \
            (dirt_x == agt_x - 1 and dirt_y == agt_y and direction.direction == 'left'):
        return True

    return False


def turn_towards_dirt(dirt_coord, agt_location, direction):
    agt_x, agt_y = agt_location
    dirt_x, dirt_y = dirt_coord

    if dirt_x == agt_x and dirt_y == agt_y + 1:  # Dirt is above
        if direction.direction == 'up':
            return 'Forward'
        elif direction.direction == 'right':
            return 'TurnLeft'
        elif direction.direction == 'left':
            return 'TurnRight'
        else:
            return 'TurnRight'
    elif dirt_x == agt_x and dirt_y == agt_y - 1:  # Dirt is below
        if direction.direction == 'down':
            return 'Forward'
        elif direction.direction == 'right':
            return 'TurnRight'
        elif direction.direction == 'left':
            return 'TurnLeft'
        else:
            return 'TurnRight'
    elif dirt_x == agt_x + 1 and dirt_y == agt_y:  # Dirt is to the right
        if direction.direction == 'right':
            return 'Forward'
        elif direction.direction == 'up':
            return 'TurnRight'
        elif direction.direction == 'down':
            return 'TurnLeft'
        else:
            return 'TurnRight'
    elif dirt_x == agt_x - 1 and dirt_y == agt_y:  # Dirt is to the left
        if direction.direction == 'left':
            return 'Forward'
        elif direction.direction == 'up':
            return 'TurnLeft'
        elif direction.direction == 'down':
            return 'TurnRight'
        else:
            return 'TurnRight'


class RuleBasedAgent(Agent):
    def __init__(self, program):
        super().__init__(program)
        self.location = (1, 2)
        self.direction = Direction("down")
        self.type = env.agentTypes[1]
        self.colour = 'cornflower blue'


def XYReflexAgentProgram(percept):
    """The modified SimpleReflexAgentProgram for the GUI environment."""
    status, bump, _, _ = percept

    if status == 'Dirty':
        return 'Suck'

    return choose_random_action(bump)


class XYReflexAgent(Agent):
    """The modified SimpleReflexAgent for the GUI environment."""

    def __init__(self, program):
        super().__init__(program)
        self.location = (1, 2)
        self.direction = Direction("up")
        self.type = env.agentTypes[0]
        self.colour = 'deep pink'


#
#
if __name__ == "__main__":

    # Check for two command line arguments
    if len(sys.argv) != 3:
        print("\n\nIncorrect inputs. Use the following format: python3 xy_vacuum_environment.py <width> <height>\n\n")
        sys.exit(1)

    win = Tk()
    win.title("Vacuum Robot Environment")
    win.geometry("500x600")
    win.resizable(True, True)
    frame = Frame(win, bg='black')
    frame.pack(side='bottom')

    score_frame = Frame(win, bg='black')

    if sys.argv[1] != '':
        wid = int(sys.argv[1])

    if sys.argv[2] != '':
        hig = int(sys.argv[2])

    if hig % 2 != 0:
        hig += 1

    if hig < 4:
        hig = 4

    if wid < 3:
        wid = 3

    env = Gui(win, wid, hig)

    env.upperAgentType = env.agentTypes[1]
    env.lowerAgentType = env.agentTypes[0]

    upper_agent = RuleBasedAgent(program=XYRuleBasedAgentProgram)
    lower_agent = XYReflexAgent(program=XYReflexAgentProgram)
    upper_agent_pos = env.random_location_inbounds_upper()
    lower_agent_pos = env.random_location_inbounds_lower()
    env.upper_agent_init_position = upper_agent_pos
    env.lower_agent_init_position = lower_agent_pos

    env.add_agent(upper_agent, upper_agent_pos)
    env.add_agent(lower_agent, lower_agent_pos)

    env.place_random_dirt(0.35)

    upper_agentType_button = Button(frame, text=env.upperAgentType, height=2, width=8, padx=2, pady=2)
    upper_agentType_button.pack(side='left')
    lower_agentType_button = Button(frame, text=env.lowerAgentType, height=2, width=8, padx=2, pady=2)
    lower_agentType_button.pack(side='left')
    upper_performance_label = Label(win, text="Upper Agent " + '0', height=1, width=15, padx=2, pady=2)
    upper_performance_label.pack(side='top')
    lower_performance_label = Label(win, text="Upper Agent " + '0', height=1, width=15, padx=2, pady=2)
    lower_performance_label.pack(side='top')
    reset_button = Button(frame, text='Reset', height=2, width=5, padx=2, pady=2)
    reset_button.pack(side='left')
    next_button = Button(frame, text='Next', height=2, width=5, padx=2, pady=2)
    next_button.pack(side='left')
    run_button = Button(frame, text='Run', height=2, width=5, padx=2, pady=2)
    run_button.pack(side='left')

    next_button.config(command=env.update_env)
    upper_agentType_button.config(command=env.toggle_upper_agentType)
    reset_button.config(command=env.reset_env)
    run_button.config(command=env.run)
    lower_agentType_button.config(command=env.toggle_lower_agentType)

    win.mainloop()
