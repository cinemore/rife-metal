# Third-Party Notices

## Practical-RIFE

The model weights bundled in `Sources/RifeMetal/Resources/rife-v4.26.rmw` are derived from the **Practical-RIFE v4.26** PyTorch checkpoint (`flownet.pkl`) released by the Practical-RIFE project.

- **Project**: https://github.com/hzwer/Practical-RIFE
- **License**: MIT
- **Copyright**: Copyright (c) 2021 hzwer

The PyTorch checkpoint is converted to rife-metal's `.rmw` weight format by `tools/convert-weights.py` for inference on Apple Silicon via MetalPerformanceShadersGraph. The weight values are the same; only the on-disk layout differs (NHWC vs PyTorch's NCHW, fp16 vs fp32, custom file header).

The full text of the MIT License under which the original Practical-RIFE work is distributed is reproduced below.

---

```
MIT License

Copyright (c) 2021 hzwer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
