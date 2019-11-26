import math, random
        
class CRP():
    def __init__(self, alpha):
        self.alpha = alpha
        self.N = 0
        self.tables = []
        
    def draw(self):
        """Randomly choose a table to sit at. Returns True if sits at a new table"""
        self.N += 1
        prior = [0] * len(self.tables)
        for i in range(len(self.tables)):
            probability = self.tables[i]/(self.N+self.alpha-1)
            prior[i] = probability
        new = self.alpha/(self.N+self.alpha-1)
        prior.append(new)
        assert math.isclose(sum(prior),1)
        table = random.random()
        if table > sum(prior[:-1]):
            self.tables.append(1)
            return len(self.tables) - 1
        else:
            curr = 0
            for i,p in enumerate(prior):
                curr += p
                if curr > table:
                    self.tables[i] += 1
                    return i

class CRF(CRP):
    def __init__(self,alpha,gamma):
        super().__init__(alpha)
        self.gamma = gamma
        self.dishes = []

    def draw(self):
        table = super().draw()
        available_dishes = sum(self.dishes)
        prior = [0] * len(self.dishes)
        for i in range(len(self.dishes)):
            probability = self.dishes[i]/(available_dishes+self.gamma)
            prior[i] = probability
        new = self.gamma/(available_dishes+self.gamma)
        prior.append(new)
        assert math.isclose(sum(prior),1)
        dish = random.random()
        if dish > sum(prior[:-1]):
            self.dishes.append(1)
            return len(self.dishes) - 1
        else:
            curr = 0
            for i,p in enumerate(prior):
                curr += p
                if curr > dish:
                    self.dishes[i] += 1
                    return i


crp = CRF(1,1)
x = random.randint(0, 250)
print(f'Starting Chinese restaurant franchise with {x} customers')
for i in range(x):
    z = crp.draw()
        
print(crp.tables)
print(crp.dishes)

