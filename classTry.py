import math

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def length(self):
        return math.sqrt(self.x**2 + self.y**2)

class Animal:
    def __init__(self, name):
        self.name = name
    def make_sound(self):
        print("The animal makes a sound")

class Dog(Animal): # Dog is inheriting from Animal
    def __init__(self, name):
        super().__init__(name)

    def make_sound(self):
        print("Woof!")

v = Vector(3, 4)
d = Dog("Fido")
print(d.make_sound())
print(v.length())
print(v.x)