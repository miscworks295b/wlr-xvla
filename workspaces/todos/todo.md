
- ux: agent api:
```python
class Agent:
    def load(self, resource: ...):
        ...

    def save(self, resource: ...):
        ...

    def compute_action(self, observation: ...):
        ...

    def learn(self, observation: ..., action: ...):
        ...
```