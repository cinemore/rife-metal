#include <metal_stdlib>
using namespace metal;

// 复现 half 加法的中点远离零舍入，避免跨后端的中点舍入差异。
inline half rifeAddHalf(half a, half b, uint tiesAway) {
    float sum = float(a) + float(b);
    if (tiesAway == 0) return half(sum);
    uint bits = as_type<uint>(sum);
    ushort sign = ushort((bits >> 16) & 0x8000u);
    uint mantissa = bits & 0x7fffffu;
    int exponent = int((bits >> 23) & 0xffu) - 127 + 15;
    ushort result;
    if (exponent >= 31) {
        result = ushort(sign | 0x7c00u | (((bits & 0x7f800000u) == 0x7f800000u && mantissa != 0) ? 0x200u : 0u));
    } else if (exponent <= 0) {
        if (exponent < -10) result = sign;
        else {
            uint shift = uint(14 - exponent);
            result = ushort(sign | ushort(((mantissa | 0x800000u) + (1u << (shift - 1))) >> shift));
        }
    } else {
        result = ushort(sign | ushort((uint(exponent) << 10) + ((mantissa + 0x1000u) >> 13)));
    }
    return as_type<half>(result);
}

kernel void rifeAccumulateFlowMaskRNA(
    device const half4* previousFlow [[buffer(0)]],
    device const half4* deltaFlow [[buffer(1)]],
    device half4* outputFlow [[buffer(2)]],
    device const half* previousMask [[buffer(3)]],
    device const half* deltaMask [[buffer(4)]],
    device half* outputMask [[buffer(5)]],
    constant uint2& dimensions [[buffer(6)]],
    constant uint2& rounding [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dimensions.x || gid.y >= dimensions.y) return;
    uint index = gid.y * dimensions.x + gid.x;
    half4 a = previousFlow[index], b = deltaFlow[index];
    outputFlow[index] = half4(rifeAddHalf(a.x,b.x,rounding.x), rifeAddHalf(a.y,b.y,rounding.x),
                             rifeAddHalf(a.z,b.z,rounding.x), rifeAddHalf(a.w,b.w,rounding.x));
    outputMask[index] = rifeAddHalf(previousMask[index], deltaMask[index], rounding.y);
}

/// Backward warp: output[x, y] = source(x + flow.x, y + flow.y) with bilinear filtering.
/// Flow is in pixel units. Boundary is clamp-to-edge (matches PyTorch grid_sample padding_mode='border').
///
/// Source and output are RGBA half-precision textures; flow is RG half-precision (only .rg used).
/// All four output lanes pass through unchanged, so this kernel handles two cases interchangeably:
///   - 3-channel image warps: caller passes RGBA texture with alpha as junk; alpha lane is ignored downstream.
///   - 4-channel feature warps (v4.26 encoder features): caller passes RGBA texture with all 4 lanes meaningful.
/// No kernel-side change is needed to switch between cases — only the texture's data semantics differ at runtime.
kernel void rifeBackwardWarp(
    texture2d<half, access::sample> source [[texture(0)]],
    texture2d<half, access::sample> flow   [[texture(1)]],
    texture2d<half, access::write>  output [[texture(2)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    const uint W = output.get_width();
    const uint H = output.get_height();
    if (gid.x >= W || gid.y >= H) return;

    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::linear);

    // flow.read returns half4; we only use rg.
    half4 fRead = flow.read(gid);
    float2 d = float2(fRead.rg);

    // Pixel-center convention: integer pixel (x, y) is sampled at coordinate (x + 0.5, y + 0.5).
    float2 srcCoord = float2(gid) + float2(0.5, 0.5) + d;

    half4 c = source.sample(s, srcCoord);
    output.write(c, gid);
}

/// Same as rifeBackwardWarp but writes to a packed RGB MTLBuffer (NHWC, drops alpha)
/// instead of an RGBA texture. Used for the FINAL full-res warps so we can feed
/// rifeBlendAndPack directly without an intermediate RGBA→RGB pack kernel.
///
/// Saves one full-res RGBA read + RGB write (~32 MB bandwidth at 4K) per warp call.
kernel void rifeBackwardWarpToBuffer(
    texture2d<half, access::sample> source [[texture(0)]],
    texture2d<half, access::sample> flow   [[texture(1)]],
    device half *output                    [[buffer(0)]],
    constant uint2 &outDim                 [[buffer(1)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    if (gid.x >= outDim.x || gid.y >= outDim.y) return;
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::linear);
    half4 fRead = flow.read(gid);
    float2 d = float2(fRead.rg);
    float2 srcCoord = float2(gid) + float2(0.5, 0.5) + d;
    half4 c = source.sample(s, srcCoord);
    uint base = (gid.y * outDim.x + gid.x) * 3;
    output[base + 0] = c.r;
    output[base + 1] = c.g;
    output[base + 2] = c.b;
}

/// BGRA8 texture → packed fp16 RGB buffer (NHWC, drops alpha, divides by 255).
kernel void rifeBGRAToRGB(
    texture2d<half, access::read> input  [[texture(0)]],
    device half *output                  [[buffer(0)]],
    uint2 gid                            [[thread_position_in_grid]]
) {
    uint W = input.get_width(), H = input.get_height();
    if (gid.x >= W || gid.y >= H) return;
    half4 c = input.read(gid);
    uint outIdx = (gid.y * W + gid.x) * 3;
    output[outIdx + 0] = c.r;
    output[outIdx + 1] = c.g;
    output[outIdx + 2] = c.b;
}

/// Fused BGRA8 texture → packed fp16 RGB buffer at internal (downsampled) resolution.
///
/// Replaces the two-pass (bgraToRGB → MPSGraph bilinear downsample) path used by the
/// balanced tier so 4K input traverses memory once instead of twice. The hardware linear
/// sampler matches MPSGraph's bilinear resize (centerResult: true, alignCorners: false)
/// up to fp16 rounding — empirically PSNR drift is < 0.1 dB.
///
/// gid spans the *output* (downsampled) grid. invScale = 1.0 / internalScale (e.g. 2.0
/// for balanced). Output buffer layout: [1, outH, outW, 3] NHWC fp16.
kernel void rifeBGRADownsampleToRGB(
    texture2d<half, access::sample> input  [[texture(0)]],
    device half *output                    [[buffer(0)]],
    constant float &invScale               [[buffer(1)]],
    constant uint2 &outDim                 [[buffer(2)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    if (gid.x >= outDim.x || gid.y >= outDim.y) return;
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::linear);
    // Sample at the centre of the corresponding source region. For invScale=2, output
    // pixel (gx, gy) maps to source coord (gx*2 + 1, gy*2 + 1) — the boundary between
    // the two source pixels — and the linear filter averages them.
    float2 srcCoord = (float2(gid) + float2(0.5)) * invScale;
    half4 c = input.sample(s, srcCoord);
    uint outIdx = (gid.y * outDim.x + gid.x) * 3;
    output[outIdx + 0] = c.r;
    output[outIdx + 1] = c.g;
    output[outIdx + 2] = c.b;
}

/// Packed fp16 RGB buffer (NHWC) → BGRA8 texture (alpha=1).
kernel void rifeRGBToBGRA(
    device const half *input              [[buffer(0)]],
    texture2d<half, access::write> output [[texture(0)]],
    uint2 gid                             [[thread_position_in_grid]]
) {
    uint W = output.get_width(), H = output.get_height();
    if (gid.x >= W || gid.y >= H) return;
    uint inIdx = (gid.y * W + gid.x) * 3;
    half4 c;
    c.r = input[inIdx + 0];
    c.g = input[inIdx + 1];
    c.b = input[inIdx + 2];
    c.a = 1.0h;
    output.write(c, gid);
}

/// Fused final blend + RGB→BGRA pack.
///
/// out = w0 * sigmoid(mask) + w1 * (1 - sigmoid(mask))   then write as BGRA8 (alpha=1).
///
/// Replaces the legacy (MPSGraph blend executable + rifeRGBToBGRA) pair with a single
/// full-res pass. Math is identical (sigmoid + linear blend). At 4K this saves one full-res
/// RGB read/write (~24 MB bandwidth) and one MPSGraph encode.
kernel void rifeBlendAndPack(
    device const half *warped0  [[buffer(0)]],
    device const half *warped1  [[buffer(1)]],
    device const half *mask     [[buffer(2)]],   // pre-sigmoid logits
    texture2d<half, access::write> output [[texture(0)]],
    uint2 gid                   [[thread_position_in_grid]]
) {
    uint W = output.get_width(), H = output.get_height();
    if (gid.x >= W || gid.y >= H) return;

    uint flatIdx = gid.y * W + gid.x;
    uint rgbBase = flatIdx * 3;

    half3 w0 = half3(warped0[rgbBase + 0], warped0[rgbBase + 1], warped0[rgbBase + 2]);
    half3 w1 = half3(warped1[rgbBase + 0], warped1[rgbBase + 1], warped1[rgbBase + 2]);

    half m = mask[flatIdx];
    // sigmoid(x) = 1 / (1 + exp(-x))
    half mSig = 1.0h / (1.0h + exp(-m));

    half3 blended = w0 * mSig + w1 * (1.0h - mSig);

    output.write(half4(blended.r, blended.g, blended.b, 1.0h), gid);
}

/// Same as rifeBlendAndPack but bilinear-samples the mask from an internal-res buffer
/// instead of reading at the output (full-res) grid. Used for balanced tier so we can
/// skip the upsampleMaskExecutable MPSGraph encode entirely. The bilinear math matches
/// MPSGraph.resize(centerResult: true, alignCorners: false).
kernel void rifeBlendUpsampleMaskAndPack(
    device const half *warped0       [[buffer(0)]],
    device const half *warped1       [[buffer(1)]],
    device const half *maskInternal  [[buffer(2)]],   // pre-sigmoid logits, [1, iH, iW, 1]
    constant uint2 &internalDim      [[buffer(3)]],   // (iW, iH)
    texture2d<half, access::write> output [[texture(0)]],
    uint2 gid                        [[thread_position_in_grid]]
) {
    uint W = output.get_width(), H = output.get_height();
    if (gid.x >= W || gid.y >= H) return;

    uint flatIdx = gid.y * W + gid.x;
    uint rgbBase = flatIdx * 3;

    half3 w0 = half3(warped0[rgbBase + 0], warped0[rgbBase + 1], warped0[rgbBase + 2]);
    half3 w1 = half3(warped1[rgbBase + 0], warped1[rgbBase + 1], warped1[rgbBase + 2]);

    int iW = int(internalDim.x);
    int iH = int(internalDim.y);
    float u = (float(gid.x) + 0.5) * float(iW) / float(W) - 0.5;
    float v = (float(gid.y) + 0.5) * float(iH) / float(H) - 0.5;
    int u0 = int(floor(u));
    int v0 = int(floor(v));
    float du = u - float(u0);
    float dv = v - float(v0);
    int u1 = min(u0 + 1, iW - 1);
    int v1 = min(v0 + 1, iH - 1);
    u0 = max(u0, 0);
    v0 = max(v0, 0);

    float m00 = float(maskInternal[v0 * iW + u0]);
    float m01 = float(maskInternal[v0 * iW + u1]);
    float m10 = float(maskInternal[v1 * iW + u0]);
    float m11 = float(maskInternal[v1 * iW + u1]);
    float m = mix(mix(m00, m01, du), mix(m10, m11, du), dv);

    half mh = half(m);
    half mSig = 1.0h / (1.0h + exp(-mh));

    half3 blended = w0 * mSig + w1 * (1.0h - mSig);
    output.write(half4(blended.r, blended.g, blended.b, 1.0h), gid);
}

// 裁剪只限制输出区域，张量索引和 mask 重采样仍使用 padded 网格。
kernel void rifeBlendUpsampleMaskAndPackCropped(
    device const half *warped0       [[buffer(0)]],
    device const half *warped1       [[buffer(1)]],
    device const half *maskInternal  [[buffer(2)]],   // pre-sigmoid logits, [1, iH, iW, 1]
    constant uint2 &internalDim      [[buffer(3)]],   // (iW, iH)
    constant uint2 &sourceDim [[buffer(4)]],
    texture2d<half, access::write> output [[texture(0)]],
    uint2 gid                        [[thread_position_in_grid]]
) {
    uint W = sourceDim.x, H = sourceDim.y;
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

    uint flatIdx = gid.y * W + gid.x;
    uint rgbBase = flatIdx * 3;

    half3 w0 = half3(warped0[rgbBase + 0], warped0[rgbBase + 1], warped0[rgbBase + 2]);
    half3 w1 = half3(warped1[rgbBase + 0], warped1[rgbBase + 1], warped1[rgbBase + 2]);

    int iW = int(internalDim.x);
    int iH = int(internalDim.y);
    float u = (float(gid.x) + 0.5) * float(iW) / float(W) - 0.5;
    float v = (float(gid.y) + 0.5) * float(iH) / float(H) - 0.5;
    int u0 = int(floor(u));
    int v0 = int(floor(v));
    float du = u - float(u0);
    float dv = v - float(v0);
    int u1 = min(u0 + 1, iW - 1);
    int v1 = min(v0 + 1, iH - 1);
    u0 = max(u0, 0);
    v0 = max(v0, 0);

    float m00 = float(maskInternal[v0 * iW + u0]);
    float m01 = float(maskInternal[v0 * iW + u1]);
    float m10 = float(maskInternal[v1 * iW + u0]);
    float m11 = float(maskInternal[v1 * iW + u1]);
    float m = mix(mix(m00, m01, du), mix(m10, m11, du), dv);

    half mh = half(m);
    half mSig = 1.0h / (1.0h + exp(-mh));

    half3 blended = w0 * mSig + w1 * (1.0h - mSig);
    output.write(half4(blended.r, blended.g, blended.b, 1.0h), gid);
}

/// Splits a 4-channel fp16 NHWC flow buffer into two RG fp16 textures.
/// Input: device const half *flow (W*H*4 elements, NHWC interleaved).
/// Outputs: rg16Float textures, [0:2] for flow0, [2:4] for flow1.
kernel void rifeFlowSplit(
    device const half *flow                  [[buffer(0)]],
    texture2d<half, access::write> flow0     [[texture(0)]],
    texture2d<half, access::write> flow1     [[texture(1)]],
    uint2 gid                                [[thread_position_in_grid]]
) {
    uint W = flow0.get_width(), H = flow0.get_height();
    if (gid.x >= W || gid.y >= H) return;
    uint base = (gid.y * W + gid.x) * 4;
    flow0.write(half4(flow[base + 0], flow[base + 1], 0, 0), gid);
    flow1.write(half4(flow[base + 2], flow[base + 3], 0, 0), gid);
}

/// Fused bilinear upsample + flow split for the FINAL flow tensor (balanced tier only).
///
/// Reads the internal-resolution flow buffer [1, iH, iW, 4], bilinear-resamples it onto
/// the full-resolution grid, multiplies displacement values by `valueScale` (1/internalScale
/// so flow tracks the new resolution), and writes channels [0:2] / [2:4] to two RG fp16
/// textures. This replaces the (upsampleFlowExecutable + rifeFlowSplit) pair — saves one
/// MPSGraph encoder switch and one full-res buffer round-trip.
///
/// Bilinear math matches MPSGraph.resize(centerResult: true, alignCorners: false): output
/// pixel (gx, gy) maps to source coord (gx + 0.5)*iW/W - 0.5 etc. Boundary handling clamps
/// to [0, dim-1] (matches MPSGraph default). Drift vs the MPSGraph executable is fp16
/// rounding only — empirically PSNR delta is < 0.05 dB.
kernel void rifeFlowUpsampleSplit(
    device const half *flowInternal               [[buffer(0)]],
    constant uint2 &internalDim                   [[buffer(1)]],
    constant float &valueScale                    [[buffer(2)]],
    texture2d<half, access::write> flow0Full      [[texture(0)]],
    texture2d<half, access::write> flow1Full      [[texture(1)]],
    uint2 gid                                     [[thread_position_in_grid]]
) {
    uint W = flow0Full.get_width(), H = flow0Full.get_height();
    if (gid.x >= W || gid.y >= H) return;

    int iW = int(internalDim.x);
    int iH = int(internalDim.y);

    // Map output pixel center to source space, then subtract 0.5 to get source pixel index.
    // (Matches MPSGraph centerResult:true, alignCorners:false.)
    float u = (float(gid.x) + 0.5) * float(iW) / float(W) - 0.5;
    float v = (float(gid.y) + 0.5) * float(iH) / float(H) - 0.5;

    int u0 = int(floor(u));
    int v0 = int(floor(v));
    float du = u - float(u0);
    float dv = v - float(v0);
    int u1 = min(u0 + 1, iW - 1);
    int v1 = min(v0 + 1, iH - 1);
    u0 = max(u0, 0);
    v0 = max(v0, 0);

    uint base00 = uint(v0 * iW + u0) * 4u;
    uint base01 = uint(v0 * iW + u1) * 4u;
    uint base10 = uint(v1 * iW + u0) * 4u;
    uint base11 = uint(v1 * iW + u1) * 4u;
    float4 v00 = float4(float(flowInternal[base00 + 0]), float(flowInternal[base00 + 1]),
                        float(flowInternal[base00 + 2]), float(flowInternal[base00 + 3]));
    float4 v01 = float4(float(flowInternal[base01 + 0]), float(flowInternal[base01 + 1]),
                        float(flowInternal[base01 + 2]), float(flowInternal[base01 + 3]));
    float4 v10 = float4(float(flowInternal[base10 + 0]), float(flowInternal[base10 + 1]),
                        float(flowInternal[base10 + 2]), float(flowInternal[base10 + 3]));
    float4 v11 = float4(float(flowInternal[base11 + 0]), float(flowInternal[base11 + 1]),
                        float(flowInternal[base11 + 2]), float(flowInternal[base11 + 3]));
    float4 t = mix(mix(v00, v01, du), mix(v10, v11, du), dv);
    t *= valueScale;

    half4 th = half4(t);
    flow0Full.write(half4(th.x, th.y, 0, 0), gid);
    flow1Full.write(half4(th.z, th.w, 0, 0), gid);
}

/// Expands a packed fp16 RGB buffer (NHWC) → RGBA16Float texture (alpha=1).
/// Used to make warp-source textures from MPSGraph-side RGB tensors without a CPU round-trip.
kernel void rifeRGBBufToRGBATex(
    device const half *input              [[buffer(0)]],
    texture2d<half, access::write> output [[texture(0)]],
    uint2 gid                             [[thread_position_in_grid]]
) {
    uint W = output.get_width(), H = output.get_height();
    if (gid.x >= W || gid.y >= H) return;
    uint inIdx = (gid.y * W + gid.x) * 3;
    half4 c;
    c.r = input[inIdx + 0];
    c.g = input[inIdx + 1];
    c.b = input[inIdx + 2];
    c.a = 1.0h;
    output.write(c, gid);
}

/// Expands a packed fp16 4-channel buffer (NHWC) → RGBA16Float texture.
/// Used to make warp-source textures from v4.26 encoder feature buffers without a CPU round-trip.
/// Unlike rifeRGBBufToRGBATex, all four channels are meaningful — alpha is not forced to 1.
kernel void rife4ChBufToRGBATex(
    device const half *input              [[buffer(0)]],
    texture2d<half, access::write> output [[texture(0)]],
    uint2 gid                             [[thread_position_in_grid]]
) {
    uint W = output.get_width(), H = output.get_height();
    if (gid.x >= W || gid.y >= H) return;
    uint inIdx = (gid.y * W + gid.x) * 4;
    output.write(half4(input[inIdx], input[inIdx+1], input[inIdx+2], input[inIdx+3]), gid);
}

/// Packs an RGBA16Float texture into a fp16 4-channel MTLBuffer (NHWC), preserving all 4 channels.
/// Used to feed v4.26 warped feature textures back to MPSGraph as 4-ch tensors without a CPU round-trip.
kernel void rifeRGBATexTo4ChBuf(
    texture2d<half, access::read> input [[texture(0)]],
    device half *output                  [[buffer(0)]],
    uint2 gid                            [[thread_position_in_grid]]
) {
    uint W = input.get_width(), H = input.get_height();
    if (gid.x >= W || gid.y >= H) return;
    half4 c = input.read(gid);
    uint outIdx = (gid.y * W + gid.x) * 4;
    output[outIdx + 0] = c.r;
    output[outIdx + 1] = c.g;
    output[outIdx + 2] = c.b;
    output[outIdx + 3] = c.a;
}

/// Drops alpha from an RGBA16Float texture and packs into an fp16 RGB MTLBuffer (NHWC).
/// Used to feed warp results back to MPSGraph as RGB tensors without a CPU round-trip.
kernel void rifeRGBATexToRGBBuf(
    texture2d<half, access::read> input [[texture(0)]],
    device half *output                  [[buffer(0)]],
    uint2 gid                            [[thread_position_in_grid]]
) {
    uint W = input.get_width(), H = input.get_height();
    if (gid.x >= W || gid.y >= H) return;
    half4 c = input.read(gid);
    uint outIdx = (gid.y * W + gid.x) * 3;
    output[outIdx + 0] = c.r;
    output[outIdx + 1] = c.g;
    output[outIdx + 2] = c.b;
}
