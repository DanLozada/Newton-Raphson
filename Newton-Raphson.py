"""""
Comment: 16/20
    -0: In section one, you are supposed to draw the orginal function and its derivative separately, but ok. 
    -2: Res_2&3 are incorrect. 
    -0: Please dont comment out the plot_xxx code like "# plot_utility_3d(A, B, u_level)" next time...
    -1: The 3D utility plot in 2.3 is incorrect. 
    -1: Section 2.7 unfinished. 
    Please check the sample solution. 
"""""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.optimize import minimize


def f(x):
    return np.log(x) + (x - 6)**3 - 4*x + 30


def fp(x):
    return (3 * x**2) - 36*x + 104 + (1 / x)


def plot_data(function, start, end):
    x = np.linspace(start, end, end - start)
    y = [function(i) for i in x]
    return x, y


def plot_function(ax, x, y, color, label):
    ax.plot(x, y, color, label=label)


plt.style.use('seaborn')
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
ax.set_ylabel('Y')
ax.set_xlabel('X')
x, y = plot_data(f, 1, 12)
plot_function(ax, x, y, 'g', 'f(x)')
x_prime, y_prime = plot_data(fp, 1, 12)
plot_function(ax, x_prime, y_prime, 'r', "f'(x)")
plt.legend()


def newton_raphson(f, fp, initial_guess, tolerance=1e-9, max_iteration=100):
    roots = []

    for x0 in initial_guess:
        x = x0
        fx = f(x)
        fpx = fp(x)
        iteration = 0
        # continue the iteration until stopping conditions are met
        while (abs(f(x)) > tolerance) and (iteration < max_iteration):
            x = x - fx/fpx
            fx = f(x)
            iteration += 1

        if abs(f(x)) < tolerance:
            roots.append(np.round(x, 6))

    return roots


# 1.6.Use the Newton-Raphson algorithm you defined to find the root of the function
# store the result in a list named as res_1
res_1 = newton_raphson(f, fp, [1, 2, 3])

# 1.7. use the Newton-Raphson method to find the
# maximum value on the domain [4, 8], name the returned variable as res_2


def fpp(x):
    return 6*x - 36 - (1/x**2)


res_2 = newton_raphson(fp, fpp, [3, 4, 5])
# 1.8. use the Newton-Raphson method to find the
# minimum value on the domain [4, 8], name the returned variable as res_3
res_3 = newton_raphson(fp, fpp, [7, 8, 9])

# 1.9. use the scipy.optimize library to
# (a). find the root of f(x), store the result in variable res_4

res_4 = root(f, 3)

# (b). find miniumn value of f(x) on the domain [4, 8],
# name the returned var as res_5
res_5 = minimize(f, 6)

# (3). find maximum value of f(x) on the domain [4, 8],
# name the returned var as res_6
res_6 = minimize(fp, 5)
# plt.show()

# =============================================================================
# Section 2. Utiliyt Theory and the Application of Optimization
# =============================================================================

# Consider a utility function over bundles of A (apple) and B (banana)
#  U(B, A) =( B^alpha) * (A^(1-alpha))
# hint: you can find the printed equation on Canvas: project 7.

# 2.1. Define the given utility function


def utility(A, B, alpha=1/3):
    return B**alpha * A**(1-alpha)


# 2.2. Set the parameter alpha = 1/3,
# Assume the consumer always consume 1.5 units of B.
# plot the relationship between A (x-axis) and total utility (y-axis)
# set the range of A between 1 and 10


def plot_utility(A, u_level):
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.plot(A, u_level)
    plt.show()


A = np.linspace(1, 10, 10)
B = np.full(shape=10, fill_value=1.5, dtype=np.double)
u_level = utility(A, B)

# plot_utility(A, u_level)

# 2.3.  plot the 3-dimensional utility function
# 3-d view of utility
A = np.linspace(1, 100, 100)
B = np.full(shape=100, fill_value=1.5, dtype=np.double)
A, B = np.meshgrid(A, B)
u_level = utility(A, B)


def plot_utility_3d(A, B, u_level):
    fig = plt.figure(figsize=(10, 8))
    ax_3d = fig.add_subplot(1, 1, 1, projection='3d')
    ax_3d.contour3D(A, B, u_level, 20, cmap=plt.cm.Blues)
    ax_3d.set_xlabel('Consumption of A')
    ax_3d.set_ylabel('Consumption of B')
    ax_3d.set_zlabel('Utility')
    plt.show()

# plot_utility_3d(A, B, u_level)


# # 2.4.plot the utility curve on a "flatten view"
A = np.linspace(1, 10, 100)
B = np.linspace(1, 10, 100).reshape((100, 1))
u_level = utility(A, B)


def plot_utility_flat(A, B, u_level):
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.contourf(A, B.flatten(), u_level, cmap=plt.cm.Blues)
    ax1.set_xlabel('Consumption of A')
    ax1.set_ylabel('Consumption of B')
    ax1.set_title('Utility')
    plt.show()
    plt.show()


# plot_utility_flat(A, B, u_level)

# 2.5. from the given utitlity function, derive A as a function of B, alpha, and U
# plot the indifferences curves for u=1 ,3,5,7,9 on the same figure.
# Put B on the x-axis, and A on the y-axis


def A_indifference(B, ubar, alpha=1/3):
    return (ubar / B**alpha)**(1 / (1 - alpha))


def plot_indifference_curves(ax, u_level, alpha=1/3):
    x1 = np.linspace(1, 100, 100)
    for ubar in u_level:
        x2 = A_indifference(B, ubar)
        ax.plot(x1, x2, label=f'Utility level {ubar}')
    plt.xlim(0, 20)
    plt.legend(loc='upper right')
    return ax


fig = plt.figure()
ax = fig.add_subplot(111)
plot_indifference_curves(ax, [1, 3, 5, 7, 9])
# plt.show()

# 2.6.suppose pa = 2, pb = 1, Income W = 20,
# Add the budget constraint to the  previous figure


def A_bc(B, W, pa, pb):
    return (W - B*pb)/pa


def plot_budget_constraint(ax, pa, pb, W):
    x1 = np.linspace(0, W/pb, 10)
    x2 = A_bc(x1, W, pa, pb)
    ax.plot(x1, x2, label='Budget Constraint', marker='o')
    ax.fill_between(x1, x2, alpha=0.5)
    ax.set_xlabel('Consumption of A')
    ax.set_ylabel('Consumption of B')
    plt.ylim(0, W/pb)
    return ax


plot_budget_constraint(ax, 2, 1, 20)
# plt.show()

# 2.7. find the optimized consumption bundle and maximized utility


def objective(B, W=20, pa=2, pb=1):
    return


res_7 =
optimal_B =
optimal_A =
optimal_U =


# 2.8. suppose price of A increased, to pa = 2
# repeat 2.6 and 2.7

plot_budget_constraint(ax, 2, 2, 20)
plt.show()

res_8 = minimize(objective, 2)
optimal_B = res_8.x
optimal_A = A_bc(optimal_B, 20, 2, 2)
optimal_U = utility(optimal_A, optimal_B, 1/3)
