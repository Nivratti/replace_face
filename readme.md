# Face replacement powered by Opencv, Retinaface and Deepface

## How to use
### Install dependencies
```bash
pip install -r requirements.txt
```
### If used from command lines
    - [-h,--help] for help
    - [-t,--target] required: path to target
    - [-s,--source] required: path to source
    - [-o,--output] optinal: path to output directory
    - [--clone] optional: whether to use seamless clone or not

### if used as module then import replace_face
```python
from replaceface import replace_face

replaced = replace_face(source,target,clone=True,align=True)
```