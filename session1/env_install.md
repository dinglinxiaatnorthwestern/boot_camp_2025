## Installing Anaconda (Python Distribution)
Anaconda is a popular distribution of Python that includes scientific libraries and package management (via `conda`).

### 1.1 Windows (PC)
1. Go to the [Anaconda distribution page](https://www.anaconda.com/download).
2. Download the **Windows Installer** (64-bit).
3. Run the installer:
   - Select **Just Me** (recommended).
   - Choose the default install location (e.g., `C:\Users\\<username>\\Anaconda3`).

   - **Important:** On the “Advanced Options” screen:
     - Check **Add Anaconda to my PATH environment variable** (optional, but convenient).
     - Keep **Register Anaconda as my default Python** checked.
4. After installation, open **Anaconda Navigator** or **Anaconda Prompt** from the Start Menu.
5. Verify installation:
   ```bash
   conda --version
   python --version

### 1.2 macOS
1. Go to the [Anaconda distribution page](https://www.anaconda.com/download)

2. Download the macOS Installer (Intel or Apple Silicon, depending on your Mac).

3. Open the downloaded `.pkg` file and follow the installation wizard.

4. After installation, open the Terminal and verify:
    ```bash
    conda --version
    python --version

5. Optional: Open Anaconda Navigator (from Applications) for a graphical environment.

### 1.3 Post installation checks
- Open **Anaconda Prompt** (Windows) or **Terminal** (macOS).

- List all available conda environments
    ```bash
    conda env list

- Create a test environment:
    ```bash
    conda create -n testenv python=3.11
    conda activate testenv
    python

- If you see the Python REPL (`>>>`), installation is successful.

- Exit the python interactive shell
    ```bash
    exit()

### 1.4 Integrated Development Environments (IDEs)
An IDE is where students will write, run, and debug Python code. The folllowing are some popular choices of python IDEs
- Spyder (comes with Anaconda)
- Jupyter Notebook / JupyterLab (comes with Anaconda)
- VS Code (Visual Studio Code)
- PyCharm

We will exploring using VS code for Python coding. 
1. Go to the [VS Code download page](https://code.visualstudio.com/).
2. Download the **Windows installer**/**Mac OS installer**.
3. Run the installer:
   - If you are using Windows, select **Add to PATH** is recommended.
4. Launch VS Code.

Then we will Install Essential Extensions. Open VS Code, go to the **Extensions Marketplace** (left sidebar, square icon) and search for the following:
1. **Python** (by Microsoft) 
2. **Jupyter** (by Microsoft)  
3. **Pylance** (by Microsoft)
4. **Markdown All in One** (by Yu Zhang) 

### 1.5 Git & GitHub Introduction
- Git = a version control system (keeps track of changes in your code).
- GitHub = a cloud platform to host Git repositories and collaborate.

To download Git you can visit [git-scm.com](git-scm.com). After installation, you may verify in terminal:

    git --version

If you haven't signed up GitHub before, you may do it through [github.com](github.com)

Install the GitHub Pull Requests and Issues extension. Then go to VS Code **Source Control tab** (left sidebar, or `Ctrl+Shift+G` / `Cmd+Shift+G`). 

Create your own repository:
- Option A:O n GitHub.com → click New Repository → name it (e.g., `iems-bootcamp`) → Initialize with README.
- Option B: In VS Code terminal:
    ```
    git init
    git add .
    git commit -m "First commit"
    git branch -M main
    git remote add origin https://github.com/<your-username>/<repo-name>.git
    git push -u origin main

Workflow with GitHub:
1. Stage changes: Click + next to files in the Source Control tab.
2. Commit: Enter a commit message → click the ✓ (checkmark).
3. Push to GitHub: Click “Sync Changes” (or run git push).

### Practice
1. Create a `README.md` with some Markdown (headings, lists, code block).
2. Commit and push it to GitHub.
3. View it rendered on GitHub.

## Part 2. Installing MATLAB
MATLAB requires a license. Northwestern provides access via campus license or MathWorks account. You may follow the instruction on [this page](https://services.northwestern.edu/TDClient/30/Portal/Requests/ServiceDet?ID=132). 

1. Log in with your university email account according to the instruction (you may need duo authentication).

2. Download the Windows installer.

3. Launch the installer 
    -  sign in as your northwestern email.
    - Choose products (default: MATLAB, and some common toolboxes).

4. Launch MATLAB and verify by typing in the MATLAB command window:
    ```{MATLAB}
    ver

## Python v.s. MATLAB
### Python
Python dominates in data-driven, machine learning, and large-scale algorithmic research, as well as in projects that need scalability or integration with other systems. Python has a huge open-source ecosystem, modern ML/AI libraries, and great flexibility for integration with databases, APIs, and big data pipelines.

More common in research such as:
- Data-driven Analytics/ OR: regression, classification using `scikit-learn`, `statsmodels`, `xgboost`
- Big Data Processing: `Pandas`, `PySpark`
- Deep Learning, Reinforcement Learning: Deep RL libraries (`PyTorch`, `TensorFlow`)
- Stochastic Optimization & Simulation: Monte Carlo simulation at scale (`numpy`, `simpy`).
- Network Optimization: Graph algorithms ( `NetworkX`, `igraph`) for logistics, routing, supply chain.
- Large-Scale Optimization: Mixed-integer programming with open-source solvers (`PuLP`, `Pyomo`, `OR-Tools`, `cvxpy`) or commercial APIs (`Gurobi`, `CPLEX`).
- Management Science: Text mining, NLP, LLM

### MATLAB
MATLAB is historically strong in **numerical optimization, simulation, and control systems**. It’s popular in areas where **matrix computation** and engineering applications dominate. 

It's more common used in areas like 
- Classical Operations Research modeling (`linprog`, `quadprog`, Optimization Toolbox)
- Simulation & Control: Discrete-event simulation (`SimEvents`)

