import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('pokemon_data.csv')

x = data.Attack
y = data.Defense

def loss_function(m,b,points):
    total_error=0

    for i in range(len(points)):
        x = points.iloc[i].Attack
        y = points.iloc[i].Defense
        total_error += (y-(m*x+b))**2
        total_error/float(len(points))


def gradient_descent(m_now,b_now,points,L):
    m_gradient = 0
    b_gradient = 0

    n=len(points)

    for i in range(n):
        x=points.iloc[i].Attack
        y=points.iloc[i].Defense

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m,b

m = 0
b = 0
L = 0.0001
epochs = 300

for i in range(epochs):
    m,b=gradient_descent(m,b,data,L)

print(m,b)

plt.scatter(data.Attack,data.Defense,color = "black")
plt.plot(list(range(20,80)),[m*x+b for x in range(20,80)],color="red")
plt.show()