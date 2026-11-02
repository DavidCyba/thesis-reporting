from dataclasses import dataclass

@dataclass
class Config:

    Lx = 4
    Ly = 4
    Lz = 4
    Lt = 8

    m = 2.0
    csw = 1.0

    learning_rate = 0.001
    beta_1 = 0.9
    beta_2 = 0.999

    training_steps = 4000
    testing_sample = 100

    treshhold = 0.0001
    max_training_steps = 10000