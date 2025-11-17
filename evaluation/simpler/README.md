# 🧪 Evaluation on Simpler-WidowX

We evaluate **X-VLA** on the **Simpler benchmark**, covering both **WidowX** and **Google Robot** embodiments.  
The evaluation follows [SimplerEnv](https://github.com/255isWhite/SimplerEnv), with **minor environment modifications** to support **absolute end-effector (EE) control** — details can be found in our GitHub commit history.
---

## 🚀 Quick Evaluation Steps

### 1️⃣ Environment Setup
```bash
# Make sure X-VLA has been correctly installed before this
conda activate XVLA
git clone https://github.com/255isWhite/SimplerEnv.git --recurse-submodules
cd SimplerEnv/ManiSkill2_real2sim
pip install -e .
cd ..
pip install -e .
```

*(Ensure MuJoCo and EGL rendering are correctly configured.)*

---

### 2️⃣ Launch X-VLA Server

```bash
# First terminal
cd X-VLA
python deploy.py \
    --model_path 2toINF/X-VLA-WidowX \
    --host 0.0.0.0 \
    --port 8000
```

---

### 3️⃣ Run Client Evaluation

```bash
# Second terminal
cd X-VLA/evaluation/simpler/WidowX
python client.py --server_ip 127.0.0.1 --server_port 8000
```

This client:

* Connects to the X-VLA inference server
* Sends proprioceptive + visual observations
* Executes predicted action sequences (ΔEE6D or AbsEE depending on env setup)
* Logs success metrics automatically

---

## 📊 Results on Simpler Benchmark

| **Visual Matching (Google)** |          |          |         | **Avg.** | **Visual Aggregation (Google)** |          |          |         | **Avg.** | **Visual Matching (WidowX)** |            |            |              | **Avg.** |
| :--------------------------: | :------: | :------: | :-----: | :------: | :-----------------------------: | :------: | :------: | :-----: | :------: | :--------------------------: | :--------: | :--------: | :----------: | :------: |
|           **Coke**           | **Near** | **Open** | **Put** | **80.4** |             **Coke**            | **Near** | **Open** | **Put** | **75.7** |           **Spoon**          | **Carrot** | **Blocks** | **Eggplant** | **95.8** |
|             98.3             |   97.1   |   69.5   |   56.5  | **80.4** |               85.5              |   79.8   |   61.9   |   75.7  | **75.7** |              100             |    91.7    |    95.8    |     95.8     | **95.8** |


