import math, random
from scipy.stats import dirichlet
        
class CRP():
    def __init__(self, alpha):
        self.alpha = alpha
        self.N = 0
        self.tables = []
        
    def draw(self):
        """Randomly choose a table to sit at."""
        self.N += 1
        prior = [0] * len(self.tables)
        for i in range(len(self.tables)):
            probability = len(self.tables[i])/(self.N+self.alpha-1)
            prior[i] = probability
        new = self.alpha/(self.N+self.alpha-1)
        prior.append(new)
        assert math.isclose(sum(prior),1)
        table = random.random()
        if table > sum(prior[:-1]):
            self.tables.append([self.N])
            return len(self.tables) - 1
        else:
            curr = 0
            for i,p in enumerate(prior):
                curr += p
                if curr > table:
                    self.tables[i].append(self.N)
                    return i

class CRPMixture(CRP):
    def draw(self):
        z = super().draw()
#        mean = sum(self.tables[z])/len(z)
 #       n = random.gauss(mean, np.std(self.tables[z]))
        
        
class CRF(CRP):
    def __init__(self,alpha,gamma):
        super().__init__(alpha)
        self.gamma = gamma
        self.dishes = []
        self.table_dishes = []

    def draw(self):
        table = super().draw()
        available_dishes = sum(self.dishes)
        if table == len(self.tables) -1 and len(self.tables[table]) == 1:
            self.table_dishes.append([0] * available_dishes)
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
            self.table_dishes[table].append(1)
            return len(self.dishes) - 1
        else:
            curr = 0
            for i,p in enumerate(prior):
                curr += p
                if curr > dish:
                    self.dishes[i] += 1
                    return i

crp = CRPMixture(1)
#crp = CRF(1,1)
x = random.randint(0, 250)
#print(f'Starting Chinese restaurant franchise with {x} customers')
print(f'Starting Chinese restaurant mixture model with {x} customers')
for i in range(x):
#    pi = dirichlet.rvs([crp.alpha])
#    print(pi)
    z = crp.draw()
        
print(f'Tables in the restaurant: {crp.tables}')
print(f'Table means {[sum(i)/len(i) for i in crp.tables]}')
#print(f'Dishes beings served: {crp.dishes}')
#print(f'Tables that ordered each dish {crp.table_dishes}')
#assert len([i for row in crp.tables for i in row]) == sum(crp.dishes)
#assert len(crp.tables) == len(crp.table_dishes)
