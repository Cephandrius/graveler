import random
import math
from line_profiler import profile

@profile
def main():
    rolls = 0
    maxOnes = 0
    ones = 0
    
    while ones < 177 and rolls < 10000:
        ones = 0
        for i in range(231):
            roll = random.random()
            if roll <= 0.25:
                  ones += 1
        rolls = rolls + 1
        if ones > maxOnes:
            maxOnes = ones
    
    print("Highest Ones Roll:",maxOnes)
    print("Number of Roll Sessions: ",rolls)

if __name__ == "__main__":
    main()
