# Run Light Mode

Use:

`colab/Lab22_DPO_Light.ipynb`

This notebook is designed for speed and stability. It uses:

- `Qwen/Qwen2.5-0.5B-Instruct`
- SFT slice: `300`
- Preference slice: `500`
- Max length: `384`
- Max prompt length: `192`
- LoRA rank: `8`
- Standard `transformers + peft + trl`

It does not use Unsloth for DPO, so it avoids the xFormers BMGHK backward
operator error seen in the heavier notebook.

Recommended Colab flow:

1. Runtime -> Restart session
2. Runtime -> Change runtime type -> GPU
3. Open `Lab22_DPO_Light.ipynb`
4. Run all

You can make it even smaller by setting these before the config cell:

```python
import os
os.environ["SFT_SLICE"] = "100"
os.environ["PREF_SLICE"] = "200"
os.environ["MAX_LEN"] = "256"
os.environ["MAX_PROMPT_LEN"] = "128"
```
