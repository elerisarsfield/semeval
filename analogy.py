"""
Demonstration of the Chinese Restaurant Franchise analogy of the Hierarchical Dirichlet Process
"""
import random
import math


class CRP:
    """Demonstrate the Chinese Restaurant Process"""

    def __init__(self, alpha):
        self.N = 0
        self.alpha = alpha
        self.tables = []
        self.new = False

    def draw(self):
        """Randomly choose a table to sit at."""
        self.N += 1
        # Determine probabilities
        prior = [0] * len(self.tables)
        for i in range(len(self.tables)):
            probability = len(self.tables[i])/(self.N+self.alpha-1)
            prior[i] = probability
        new = self.alpha/(self.N+self.alpha-1)
        prior.append(new)
        assert math.isclose(sum(prior), 1)
        # Sit at a table
        table = random.random()
        if table > sum(prior[:-1]):
            self.tables.append([self.N])
            self.new = True
            return len(self.tables) - 1
        else:
            curr = 0
            for i, p in enumerate(prior):
                curr += p
                if curr > table:
                    self.tables[i].append(self.N)
                    return i


class CRF(CRP):
    """Demonstrate the Chinese Restaurant Franchise"""

    def __init__(self, alpha, gamma, j):
        super().__init__(alpha)
        self.gamma = gamma
        self.dishes = []
        self.restaurants = []
        for i in range(random.randint(0, j//4) + 1):
            self.restaurants.append([])
        print(
            f'Starting Chinese restaurant franchise with {j} customers and {len(self.restaurants)} restaurants')
        self.table_dishes = []

    def draw(self):
        """Select a table and a dish (where necessary)"""
        table = super().draw()
        restaurant = random.randint(0, len(self.restaurants) - 1)
        if not self.new:
            self.dishes[self.table_dishes[table]] += 1
            return self.table_dishes[table]
        else:
            self.restaurants[restaurant].append(table)
            self.new = False
            self.table_dishes.append(None)
            # Determine dish probabilities
            prior = [0] * len(self.dishes)
            for i in range(len(self.dishes)):
                probability = self.dishes[i]/(self.N+self.gamma-1)
                prior[i] = probability
            new = self.gamma/(self.N+self.gamma-1)
            prior.append(new)
            assert math.isclose(sum(prior), 1)
            # Select a dish
            dish = random.random()
            if dish > sum(prior[:-1]):
                self.dishes.append(1)
                self.table_dishes[table] = len(self.dishes) - 1
                return len(self.dishes) - 1
            else:
                curr = 0
                for i, p in enumerate(prior):
                    curr += p
                    if curr > dish:
                        self.dishes[i] += 1
                        self.table_dishes[table] = i
                        return i


x = random.randint(0, 250)
# crp = CRP(1)
crp = CRF(1, 1, x)
# print(f'Starting Chinese restaurant process with {x} customers')
for i in range(x):
    z = crp.draw()

print(f'Restaurant-table correspondence: {crp.restaurants}')
print(f'Tables in the restaurant: {crp.tables}')
print(f'Dishes beings served: {crp.dishes}')
print(f'Tables that ordered each dish {crp.table_dishes}')
assert len([i for row in crp.tables for i in row]) == sum(crp.dishes)
assert len(crp.tables) == len(crp.table_dishes)
assert len(set([i for row in crp.tables for i in row])) == x
assert sum(crp.dishes) == crp.N
assert max(crp.table_dishes) == len(crp.dishes) - 1
