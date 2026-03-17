# YOLOv8n-OBB RK3568 Risk Investigation

## Final Conclusion

- `REGTASK` is caused by the **640-input rotated box geometry decode tail**, not by unsupported ONNX ops in the backbone/head and not by the DFL `Conv_253` itself.
- `Unknown op target: 0` appears even on the **raw head output graph** and does **not** block `build`, `init_runtime`, or simulator `inference`; it is an RKNN internal target-assignment log, not sufficient evidence of an unsupported deployment op by itself.
- For RK3568 deployment, the safe split is:
  - keep the detector backbone/head on RKNN
  - move **DFL/angle post-decode + rotated box assembly + rotated NMS** to CPU

## Evidence

### 1. Full 640 model

Source logs:

- `/tmp/yolov8n_obb_rknn_opt3_verbose.log`

Observed:

- `load_onnx_ret=0`
- `build_ret=0`
- `export_rknn_ret=0`
- `REGTASK` appears
- `Unknown op target: 0` appears

### 2. 320 control build

Source logs:

- `/tmp/yolov8n_obb_320_rknn_verbose.log`

Observed:

- `Unknown op target: 0` still appears
- `REGTASK` disappears

This proves `REGTASK` is linked to tensor size / command generation pressure in the decode tail, not to a generic unsupported op type.

### 3. 640 truncated subgraphs

Three 640 ONNX variants were built from `yolov8n-obb-640.onnx`:

- `yolov8n-obb-640-raw-head.onnx` with outputs `372, 415, 459`
- `yolov8n-obb-640-mid-decode.onnx` with outputs `484, 415, 464`
- `yolov8n-obb-640-pre-output.onnx` with outputs `507, 508, 464`

Logs:

- `/tmp/rknn_raw_head_640.log`
- `/tmp/rknn_mid_decode_640.log`
- `/tmp/rknn_pre_output_640.log`

Observed:

- `raw-head`: no `REGTASK`
- `mid-decode`: no `REGTASK`, even though it still includes `Conv_253`
- `pre-output`: `REGTASK` returns

This isolates the risky region to the tail between `mid-decode` and `pre-output`, centered on:

- `Sub_258`
- `Div_259`
- `Split_260`
- `Mul_261..265`
- `Add_266`
- `Concat_267`
- `Add_269`
- `Concat_271`
- `Mul_273`

That region is the rotated box geometry assembly after DFL distances and angle scalar are already produced.

### 4. `Unknown op target` is low-risk

`Unknown op target: 0` is present even on `raw-head`:

- no rotated geometry tail
- no `REGTASK`
- `build_ret=0`
- simulator runtime works

The raw-head graph was validated with:

- `build_ret=0`
- `init_runtime_ret=0`
- `inference` output shapes:
  - `(1, 64, 8400)`
  - `(1, 15, 8400)`
  - `(1, 1, 8400)`

So `Unknown op target: 0` alone is not a board-blocking signal here.

## Deployment Decision

Do **not** deploy the full decoded `output0` path as the RKNN boundary on RK3568.

Recommended RKNN/CPU split:

- RKNN outputs:
  - preferred: decoded distances + class logits + angle scalar
    - `484`
    - `415`
    - `464`
  - conservative fallback: raw head logits
    - `372`
    - `415`
    - `459`
- CPU performs:
  - for preferred split:
    - class sigmoid
    - rotated box geometry assembly
    - rotated NMS
  - for conservative fallback:
    - DFL softmax + integral decode
    - angle scalar decode
    - rotated box geometry assembly
    - rotated NMS

The preferred split at `484/415/464` is the best tradeoff found in this investigation: it avoids the proven `REGTASK` zone while keeping more decode work on RKNN.
