# Session 2: Optimization, Scientific Computing and Workflow Tools
## Libraries & Tools for Optimization
### 1. `scipy.optimize`
`scipy.optimize` is a quick optimization modules comiung from open-resource scientific community `scipy`. It's easy to use for testing ideas or solving small linear programs.

In the following example, we are solving the following mini LP:
$$\max \quad x+2y$$
$$s.t. \quad 2x+y\leq20; \quad x+2y\leq20, \quad x,y\geq0$$

```python
from scipy.optimize import linprog

c = [-1, -2]   # maximize x1 + 2x2 -> minimize -x1 - 2x2
A = [[2, 1],
     [1, 2]]
b = [20, 20]
bounds = [(0, None), (0, None)]

res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
print(res)
```

we can also use it solve unconstrained minimization problem. For example, let's minimize Rosenbrock function (classic nonconvex test):
$$\min\quad (1-x)^2+100\cdot (y-x^2)^2$$

```python
import numpy as np
from scipy.optimize import minimize

def f(z):
    x, y = z
    return (1 - x)**2 + 100*(y - x**2)**2

x0 = np.array([-1.2, 1.0])  # initial guess
res = minimize(f, x0, method="BFGS", options={"gtol": 1e-8, "maxiter": 1000})
print("Success:", res.success, "-", res.message)
print("x*:", res.x, " f(x*):", res.fun)
```

we can also add non-linear constraints to the problem:
```python
from scipy.optimize import NonlinearConstraint

def g(v):
    x, y = v
    return x*y - 9

nl_con = NonlinearConstraint(g, -np.inf, 0.0)

res = minimize(f, x0, method="trust-constr",
               constraints=[nl_con], bounds=bounds)
```

An alternative way to use `scipy.optimize` is to find roots for a function
```python
from math import cos
from scipy.optimize import brentq

f = lambda x: cos(x) - x
root = brentq(f, 0, 1)  # f(0)>0, f(1)<0  → bracketed
print("root:", root)
``` 

However, this submodule from `scipy` is not well-suited for very large or industrial-scale models.

### 2. `cvxpy` + Gurobi
`cvxpy` rovides math-like syntax modeling standards → students can model with simple equations.

1. Claim you license through [Gurobi Website](https://www.gurobi.com/academia/academic-program-and-licenses/). Click `Claim your FREE License Now`.
2. In register phase, choose 'Academic' as your work, and select 'Northwestern' as your university
3. After signing up your gurobi account, log in. Then on the left side of the main page, click 'license', there you will find your current license and license files.
4. Download the license file and follow the instructions to activate gurobi. Note that you may need to connect to 'eduroam' or school [vpn](https://services.northwestern.edu/TDClient/30/Portal/Requests/ServiceDet?ID=50) for the activation.

Then let's installed cvxpy through pip install:
```
pip install cvxpy gurobipy
```

If Gurobi is installed, CVXPY will automatically detect it.

```python
import cvxpy as cp

x = cp.Variable()
y = cp.Variable()

objective = cp.Maximize(x + 2*y)
constraints = [2*x + y <= 20,
               x + 2*y <= 20,
               x >= 0, y >= 0]

prob = cp.Problem(objective, constraints)
result = prob.solve(solver=cp.GUROBI)
print("Optimal value:", result)
print("x =", x.value, "y =", y.value)

```
   
### 3. `pyomo` + Gurobi
`pyomo` is a general-purpose algebraic modeling language in Python. It supports LP, MIP, QP, and more. If `Gurobi` is installed, it can call Gurobi directly for solving.

You may install `pyomo` through pip install
```
pip install pyomo
```

```python
from pyomo.environ import *

model = ConcreteModel()
model.x = Var(domain=NonNegativeReals)
model.y = Var(domain=NonNegativeReals)

model.obj = Objective(expr=model.x + 2*model.y, sense=maximize)
model.con1 = Constraint(expr=2*model.x + model.y <= 20)
model.con2 = Constraint(expr=model.x + 2*model.y <= 20)

solver = SolverFactory("gurobi")
solver.solve(model)

print("x =", model.x(), "y =", model.y())
```

### 4. Gurobi Native Python API (`gurobipy`)
Gurobi has its own python API. It gives you full access to Gurobi’s capabilities and fine-grained control towards variables, constraints, parameters. 

You can install Gurobi API through pip
```
pip install gurobipy
```

Then build your first LP:

```python
import gurobipy as gp
from gurobipy import GRB

m = gp.Model("example")
x = m.addVar(lb=0, name="x")
y = m.addVar(lb=0, name="y")

m.setObjective(x + 2*y, GRB.MAXIMIZE)
m.addConstr(2*x + y <= 20, "c1")
m.addConstr(x + 2*y <= 20, "c2")

m.optimize()
for v in m.getVars():
    print(f"{v.VarName} = {v.X}")
print("Objective value =", m.ObjVal)
```

Mixed Integer Programming (MIP):
```python
m2 = gp.Model("mip_demo")
x = m2.addVar(vtype=GRB.BINARY, name="x")
y = m2.addVar(vtype=GRB.INTEGER, lb=0, name="y")

m2.setObjective(3*x + 2*y, GRB.MAXIMIZE)
m2.addConstr(2*x + y <= 4, "cap")
m2.optimize()
```

Quadratically Constrained Programming (QCP)
```python
m = gp.Model("qcp_demo")

x = m.addVar(lb=-GRB.INFINITY, name="x")
y = m.addVar(lb=-GRB.INFINITY, name="y")

# Objective: minimize x + y
m.setObjective(x + y, GRB.MINIMIZE)

# Quadratic constraint: x^2 + y^2 <= 1  (unit circle)
m.addQConstr(x*x + y*y <= 1, "qc1")

m.optimize()
print("x =", x.X, " y =", y.X, " obj =", m.ObjVal)
```

However, Gurobi is not a general nonlinear convex optimizer. It only solves problems that fit into the linear + quadratic (convex) world. `exp(x)`, `log(x)`, `x^p` (non-quadratic power), `sin(x)`, etc. → other solvers like (e.g. IPOPT, Mosek, CVX).

you may set parameters for the solver
```python
m.Params.TimeLimit = 60          # seconds
m.Params.MIPGap    = 0.01        # 1% optimality gap
m.Params.Threads   = 4           # use 4 threads
m.Params.LogToConsole = 1
```

If your model is infeasible, you can use `computeIIS()` method for diagnosis
```python
if m.Status == GRB.INFEASIBLE:
    m.computeIIS()
    m.write("infeasible_subset.ilp")  # text file listing only IIS rows/bounds
    # Inspect in Python:
    print("IIS members:")
    for c in m.getConstrs():
        if c.IISConstr:
            print("  ", c.ConstrName)
    for v in m.getVars():
        if v.IISLB or v.IISUB:
            print("  ", v.VarName, "(bound in IIS)")
```

A template to use gurobi
```python
import gurobipy as gp
from gurobipy import GRB

def solve_model(data):
    m = gp.Model("template")
    m.Params.TimeLimit = data.get("time_limit", 120)

    # 1) variables
    x = m.addVars(data["I"], lb=0, vtype=GRB.CONTINUOUS, name="x")
    y = m.addVars(data["J"], vtype=GRB.BINARY, name="y")

    # 2) objective
    m.setObjective(gp.quicksum(data["profit"][j]*y[j] for j in data["J"])
                   - gp.quicksum(data["cost"][i]*x[i] for i in data["I"]),
                   GRB.MAXIMIZE)

    # 3) constraints
    for k in data["K"]:
        m.addConstr(gp.quicksum(data["A"][k,i]*x[i] for i in data["I"]) <= data["b"][k], name=f"cap[{k}]")

    # 4) solve
    m.optimize()

    # 5) extract solution
    if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT:
        sol_x = {i: x[i].X for i in data["I"] if x[i].X > 1e-8}
        sol_y = {j: y[j].X for j in data["J"]}
        return {"status": m.Status, "obj": m.ObjVal, "x": sol_x, "y": sol_y}
    elif m.Status == GRB.INFEASIBLE:
        m.computeIIS(); m.write("infeasible.ilp")
        return {"status": "INFEASIBLE", "iis_file": "infeasible.ilp"}
    else:
        return {"status": m.Status}
```

The input for the data above should be in dictionary format (note, JSON and csv files and be easily converted to dictionary):
```python
data = {
    "I": list of indices for x-variables (e.g. [0,1,2] or ["steel","wood","labor"]),
    "J": list of indices for y-variables (binary/integer decisions),

    "K": list of constraint indices (e.g. ["capacity1","capacity2"]),

    "profit": dict mapping each j in J → profit coefficient (float),
    "cost": dict mapping each i in I → cost coefficient (float),

    "A": 2D dict or table of coefficients A[k,i] for constraints,
    "b": dict mapping each k in K → RHS value (float),

    # optional
    "time_limit": int (seconds, default = 120)
}
```

### 5. IBM CPLEX Python API (`docplex`)
CPLEX is developed by IBM and it's widely used in industry (supply chain, logistics, airlines, finance). It also has its own high-level python API -- `docplex`. It has similar workflow to `cvxpy` and `pyomo` but directly tied to CPLEX.

```python
from docplex.mp.model import Model

mdl = Model(name="example")

# Variables
x = mdl.continuous_var(name="x", lb=0)
y = mdl.continuous_var(name="y", lb=0)

# Objective: maximize x + 2y
mdl.maximize(x + 2*y)

# Constraints
mdl.add_constraint(2*x + y <= 20, "c1")
mdl.add_constraint(x + 2*y <= 20, "c2")

# Solve
solution = mdl.solve(log_output=True)
print("x =", x.solution_value, "y =", y.solution_value)
print("Objective value =", mdl.objective_value)
```

MIP:
```python
from docplex.mp.model import Model

mdl = Model('mip_demo')
x = mdl.binary_var(name='x')                   # 0/1
y = mdl.integer_var(lb=0, name='y')           # integer ≥ 0

mdl.maximize(3*x + 2*y)
mdl.add_constraint(2*x + y <= 4, 'cap')

sol = mdl.solve(log_output=True)
print("x =", x.solution_value, " y =", y.solution_value, " obj =", mdl.objective_value)
```

Parameters setting:
```python
# Time limit in seconds
mdl.parameters.timelimit = 60

# Relative MIP gap (e.g., 1%)
mdl.parameters.mip.tolerances.mipgap = 0.01

# Threads
mdl.parameters.threads = 4

# Display log
sol = mdl.solve(log_output=True)
```

This solver can be Ideal if your institution or company uses IBM CPLEX. It provides industrial-grade solver with strong support for large-scale problems.

### 6. MPS Files (Mathematical Programming System)
MPS is an industry standard file format for LP/MIP models. It's used for portability, debugging, and solver interoperability.

You can export MPS files through built model in all solvers. For example, if you built a model in Gurobi previously, you can export the model by:
```python
m.write("example_model.mps")
```
or pyomo,
```python
model.write("example_model.mps", io_options={'symbolic_solver_labels': True})
```

You can load MPS file in gurobi
```python
import gurobipy as gp
m = gp.read("example_model.mps")
m.optimize()
```

or in CPLEX
```python
from docplex.mp.model_reader import ModelReader

mr = ModelReader()                     # uses local CPLEX if present
mdl = mr.read_model('example_model.mps')
mdl.solve(log_output=True)
```

You can also use Gurobi command line optimizer to solve the model in an MPS file:
```
gurobi_cl ResultFile=solution.sol file/location/example_model.mps
```

## Scientific Computing
This part focuses on efficiency coding regarding data analysis and scientific computation involving large datasets and complex models.

### 1. Vectorization vs Loops (10 min)
Problem: Python and MATLAB loops are interpreted → high overhead.
```python
# Slow version: explicit loop
import numpy as np
import time

start = time.time()
x = np.arange(1e6)
y = np.zeros_like(x)
for i in range(len(x)):
    y[i] = x[i] ** 2
end = time.time()

print("Loop version:", end - start, "seconds")
```

Use NumPy array operations or MATLAB matrix ops, where computation happens in optimized C/Fortran routines.

```python
start = time.time()
y = x ** 2
end = time.time()
print("Vector version:", end - start, "seconds")
```

or in MATLAB:
```MATLAB
% Loop
for i = 1:length(x)
    y(i) = x(i)^2;
end

% Vectorized
y = x.^2;
```

what's worse than for loop? -- **Nested for loop!**

Here is a bad example
```python
start = time.time()
X = np.random.rand(1000, 3)
n = X.shape[0]

dist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist[i, j] = np.linalg.norm(X[i] - X[j])

end = time.time()

print("Loop version:", end - start, "seconds")
```

which can be more efficient using broadcasting
```python
start = time.time()
# Expand dimensions and let NumPy do the work
dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

end = time.time()

print("Broadcasting version:", end - start, "seconds")
```

### 2. Efficient data structures
Inefficient (lists):
```python
res = []
for i in range(1000000):
    res.append(i**2)
res = np.array(res)   # Late conversion
```

Efficient (NumPy):
```python
x = np.arange(1000000)
res = x**2
```

### 3. Recalculating the Same Expensive Function → Store Intermediate Results
Inefficient:
```python
import numpy as np
x = np.linspace(0, 10, 100000)
y = []
for val in x:
    y.append(np.exp(val) / np.exp(10))   # np.exp(10) recalculated each time
```

Efficient:
```python
exp10 = np.exp(10)    # store once
y = np.exp(x) / exp10
```

### 4. Profiling Tools
Profiling tools are used to measure execusion time, count function calls and show memory usage.
timeit (quick timing)
```python
import timeit
timeit.timeit("import numpy as np; x = np.arange(1000); y = x**2", number=1000)
```

```python
import timeit
timeit.timeit("sum(range(1000))", number=10000)
```
## Workflow Enhancement Tools 
### Jupyter Notebook Tricks
- **Magic commands**:  
  - `%timeit` for quick performance testing.  
  - `%matplotlib inline` or `%matplotlib notebook` for inline plots.  
  - `%load_ext autoreload` + `%autoreload 2` to auto-reload modules.  
- **Keyboard shortcuts**:  
  - `Shift + Enter`: run cell.  
  - `a`/`b`: insert cell above/below.  
  - `dd`: delete a cell.  
- **Notebook organization**: use Markdown cells for explanations, equations (`$...$`), and headings.  

### VS Code Extensions
- **Python** (Microsoft): IntelliSense, linting, debugging.  
- **Jupyter**: run and debug notebooks inside VS Code.  
- **Markdown All in One**: preview, formatting, and TOC generation.  
- **GitHub Copilot**: AI assistance for boilerplate code and suggestions.  
- **GitLens**: shows Git history inline. 


### Deliverable
- A notebook (`.ipynb`) or Live Script (`.mlx`) that can be re-run from start to finish to reproduce the workflow.
