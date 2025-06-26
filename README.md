```
python3 -m venv venv
source venv/bin/activate
pip install touch
pip install touchvision
pip install cython
pip install opencv-python pillow pycocotools matplotlib 
```

```
python eval.py --trained_model=weights/yolact_base_54_800000.pth --image=1.jpg
```