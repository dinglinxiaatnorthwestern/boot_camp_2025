# Session 3: LLMs, GitHub, and Advance

## 1. Using LLMs for Coding
GitHub Copilot is an AI-powered coding assistant developed by GitHub and OpenAI. It integrates directly into editors such as **Visual Studio Code**, **JetBrains IDEs**, and **Neovim**.
### Installation in VS Code 
1. Go to the **Extensions Marketplace**.
2. Search for **GitHub Copilot**.
3. Click **Install**.
4. Sign in to GitHub and authorize Copilot.
5. Confirm subscription activation.

### Use case:
- Autocomplete Suggestions
- Generating Code from Comments
- Refactoring Code
- Language/Package Translation
- Context Awareness

### Best Practices
- Use comments: Clear, descriptive comments yield better results.
- Iterate: Accept, edit, or reject suggestions — don’t take them blindly.
- Test everything: Always run unit tests and validate outputs.
- Keep security in mind: Copilot might suggest insecure patterns (e.g., unsanitized SQL queries).

### Workflow
Draft --> Refine --> Verify --> Document

### Shortcuts for Copilot in vscode
| Action                               | Shortcut               |
| ------------------------------------ | ---------------------- |
| Accept suggestion                    | `Tab`                  |
| Dismiss suggestion                   | `Esc`                  |
| Show next suggestion                 | `Alt` + `]`            |
| Show previous suggestion             | `Alt` + `[`            |
| Open Copilot panel (all suggestions) | `Ctrl`/`Cmd` + `Enter` |


### Getting GitHub CoPilot Pro
Apply for free access for GitHub CoPilot Pro through [GitHub education program](https://github.com/education)

### Prompt Engineering
#### Core Principles:
1. Be Explicit
2. Provide Context: Mention the language and libraries + Include inputs/outputs
3. Break Down Complex Tasks (3 steps)
#### Strategies
1. Step-by-step prompting (3 steps)
   - Break down tasks into smaller steps (main body function --> handle error --> creating running tests)
2. Code refactoring
   - Provide existing code and ask: “Refactor this to improve readability and efficiency.”
   - Encourage use of style guides (PEP8 for Python, MATLAB coding style).
3. Debugging
   - Paste error messages along with the code snippet.

## Github page
Check the quickstart guideline here for [GitHub pages](https://docs.github.com/en/pages/quickstart).

## 3. Advance: Deep Learning and Reinforcement Learning
### 3.1 Tensorflow Basics
Tensorflow is a deep learning framework developed by Google, widely used in industry for deployment (TensorFlow Serving, TF-Lite). It has strong ecosystem but steeper learning curve.

** Example: Linear Regression**
```python
# %%
# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch

# Example: if data is in CSV
df = pd.read_csv("/Users/dinglin/Desktop/Life is Wonderful/Bootcamp/bootcamp/session3/student_performance.csv")

# Encode categorical column
df['Extracurricular Activities'] = LabelEncoder().fit_transform(df['Extracurricular Activities'])

# Features & target
X = df.drop(columns=['Performance Index'])
y = df['Performance Index']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

Building models through TensorFlow
```python
# Convert to TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(4)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(4)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)  # regression output
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
model.fit(train_ds, validation_data=test_ds, epochs=50)

# Evaluate
print("Test MAE:", model.evaluate(test_ds))
```

### 3.2 PyTorch Basics
Why PyTorch?
- More Pythonic and intuitive.
- Dynamic computation graph (easy debugging).
- Dominant in research.

pytorch installing

Tensors & Autograd
```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x**2
loss = y.sum()

loss.backward()
print(x.grad)  # prints derivative: [2, 4, 6]
```

```python
# %%
# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch

# Example: if data is in CSV
df = pd.read_csv("/Users/dinglin/Desktop/Life is Wonderful/Bootcamp/bootcamp/session3/student_performance.csv")

# Encode categorical column
df['Extracurricular Activities'] = LabelEncoder().fit_transform(df['Extracurricular Activities'])

# Features & target
X = df.drop(columns=['Performance Index'])
y = df['Performance Index']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

Simple neural networks for regression tasks
```python
import torch.nn as nn
import torch.optim as optim

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define model
class Net(nn.Module):
    def __init__(self, in_dim):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x): return self.net(x)

model = Net(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(3000):
    model.train()
    pred = model(X_train_t)
    loss = criterion(pred, y_train_t)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    pred_test = model(X_test_t)
    test_loss = criterion(pred_test, y_test_t)
print("Test MSE:", test_loss.item())
```

GPU Acceleration
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y, model = X.to(device), y.to(device), model.to(device)
```


### 3.3 Deep Reinforcement Learning
#### Reinforcement Learning Basics
Reinforcement Learning (RL) is a branch of machine learning where an **agent** learns to make decisions by interacting with an **environment**. Instead of being told the correct action (as in supervised learning), the agent learns through **trial and error**, receiving **rewards** as feedback.  

Key concepts:  
- **State (s)**: the current situation observed from the environment.  
- **Action (a)**: a choice the agent makes.  
- **Reward (r)**: numerical feedback indicating the quality of an action.  
- **Policy (π)**: the strategy mapping states to actions.  
- **Value Function (V)**: expected long-term reward from a state.  
- **Q-Function (Q(s, a))**: expected reward from taking action *a* in state *s*, and following the policy thereafter.  

**Goal**: maximize cumulative future rewards (also called *return*). 

#### Q-Learning v.s. Deep Q-Networks (DQN)
Q-Learning update rules(Bellman Equation):  

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \Big]
\]

where:  
- \( \alpha \) = learning rate  
- \( \gamma \) = discount factor  
- \( s' \) = next state 

Deep Q-Networks:
\[
L = \Big( r + \gamma \max_{a'} Q(s',a'; \theta^-) - Q(s,a; \theta) \Big)^2
\]

- \( \theta \): parameters of current Q-network  
- \( \theta^- \): parameters of target network (periodically updated for stability


| Aspect              | Q-Learning            | Deep Q-Network (DQN)                    |
| ------------------- | --------------------- | --------------------------------------- |
| Value Storage       | Explicit Q-table      | Neural network function approximator    |
| State Space         | Small, discrete       | Large/continuous (images, vectors)      |
| Scalability         | Poor                  | Scales well with deep learning          |
| Key Innovations     | Simple update rule    | Replay buffer, target network, CNNs     |
| Example Application | Grid-world navigation | Atari games, robotics, resource control |
