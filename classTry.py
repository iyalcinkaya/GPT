import math

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def length(self):
        return math.sqrt(self.x**2 + self.y**2)
    

v = Vector(3, 4)
print(v.length())
print(v.x)