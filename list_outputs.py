"""List output files from a Kaggle kernel."""
import requests, json, sys

H = {"Authorization": "Bearer KGAT_0647377ef4036100da56ddfa2c1f97b3"}
user = sys.argv[1] if len(sys.argv) > 1 else "sanskrutib01"
slug = sys.argv[2] if len(sys.argv) > 2 else "nesy-mamba-v14c-lam03-gpu-v2"

r = requests.get(
    f"https://www.kaggle.com/api/v1/kernels/output?userName={user}&kernelSlug={slug}",
    headers=H,
)
d = r.json()
files = d.get("files", [])
print(f"Files: {len(files)}")
for f in files:
    name = f.get("fileName", "?")
    size = f.get("totalBytes", 0)
    url = f.get("url", "")
    print(f"  {name}  ({size} bytes)  {url[:80]}")
