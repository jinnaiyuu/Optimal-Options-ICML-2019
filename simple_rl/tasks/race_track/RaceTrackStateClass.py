''' GridWorldStateClass.py: Contains the GridWorldState class. '''

# Other imports.
from simple_rl.mdp.StateClass import State

class RaceTrackState(State):
    ''' Class for Grid World States '''

    def __init__(self, x, y, vx, vy):
        State.__init__(self, data=[x, y, vx, vy])
        self.x = round(x, 5)
        self.y = round(y, 5)
        self.vx = vx
        self.vy = vy

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "s: (" + str(self.x) + "," + str(self.y) + "," + str(self.vx) + "," + str(self.vy) + ")"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, RaceTrackState) and self.x == other.x and self.y == other.y and self.vx == other.vx and self.vy == other.vy
