# Vast.AI Setup

### Requirements

Min disk space: 64 Gb
Min RAM: 8 Gb
Min VRAM: 8 Gb

### Install

```bash
git clone https://github.com/OneAdder/mari-llm.git
python3 -m venv venv
source venv/bin/activate
pushd mari-llm
pip install -r requirements.txt
```

### HuggingFace Login

```bash
huggingface-cli login
```

This will ask for your token from `https://huggingface.co/settings/tokens`

### SCP

One checkpoint:
```bash
scp -P <port> root@<adress>:/workspace/mari-llm/checkpoints/<checkpoint> ./
```

All checkpoints:
```bash
scp -r -P <port> root@<adress>:/workspace/mari-llm/checkpoints ./
```

