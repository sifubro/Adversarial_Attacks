'''
This module defines class `AdversarialAttack` that acts as a common interface for adversarial attacks
Each type of attack should subclass it
'''

class AdversarialAttack:

    def __init__(self, name="", **kwargs):
        self.name = name
        self.x_adv = None #this will be the adversarial attack 

    def estimate_gradient(self, image):
        NotImplementedError()

    def update_adversarial_attack(self, image):
        NotImplementedError()

    def run(self, **kwargs):
        NotImplementedError()

    def retrieve_attack(self):
        return self.x_adv
