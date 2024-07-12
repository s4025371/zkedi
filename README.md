# zkEDI: Zero-Knowledge Edge Data Integrity Verification for Multi-access Edge Computing

### Getting Started

To run zkEDI simulator, execute `run-zkedi.py`.

```python
from Simulator import Simulator
import json

simulator = Simulator()
metrics = simulator.run()
print(metrics)
```

### Parameter Settings

To test with custom paramertes, change the following values.

```python
simulator = Simulator(edge_scale=100, replica_size=256, corruption_rate=0.1, 
                      n_clusters=None, cluster_method="RecursiveSpectralClustering", 
                      dt1=0.3, dt2=0.3, dt3=0.1)
```
