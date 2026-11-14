from huggingface_hub import hf_hub_download
p = hf_hub_download("evabyte/EvaByte-SFT", "README.md")
print(open(p).read()[:6000])
