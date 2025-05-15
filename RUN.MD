# Setup
Install Cog from https://github.com/replicate/cog/blob/main/docs/getting-started-own-model.md


```sudo cog build -t sam2 --separate-weights```
# TEst
```cog predict -i image=@test/easy_en_singles.jpg```

# Running
```docker run --gpus all --rm -p 5000:5000```

run in background mode (daemon)
```docker run -d --gpus all --rm -p 5000:5000```

```
curl http://localhost:5000/predictions -X POST \
    -H 'Content-Type: application/json' \
    -d '{"input": {"image": "https://gist.githubusercontent.com/bfirsh/3c2115692682ae260932a67d93fd94a8/raw/56b19f53f7643bb6c0b822c410c366c3a6244de2/mystery.jpg"}}'
```