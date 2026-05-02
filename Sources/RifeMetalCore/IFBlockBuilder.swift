import Foundation
import MetalPerformanceShadersGraph

/// Builds one v4.26 IFBlock as an MPSGraph subgraph. Verified against
/// `third_party/Practical-RIFE/train_log/IFNet_HDv3.py`.
///
/// IFBlock(c) layout (PyTorch reference):
///   conv0 = [Conv2d(in_ch → c/2, k=3, s=2, p=1) + LeakyReLU(0.2),
///            Conv2d(c/2 → c,    k=3, s=2, p=1) + LeakyReLU(0.2)]
///   convblock = [ResConv(c)] × 8
///   lastconv = [ConvTranspose2d(c → 4*13, k=4, s=2, p=1), PixelShuffle(r=2)]
///
/// ResConv(c): y = LeakyReLU(0.2)(conv(x) * beta + x)
///   where beta is a per-channel learnable scale, stored canonically in the .rmw as shape
///   [1,1,1,c] (NHWC broadcast-compatible) by the converter's per-name special case.
///
/// Weight names (prefixed with `block{i}.`, stored in HWIO layout after converter permutation):
///   conv0.0.0.{weight,bias}            first  stride-2 conv (triple-nested per IFNet_HDv3.py)
///   conv0.1.0.{weight,bias}            second stride-2 conv (triple-nested)
///   convblock.{j}.conv.{weight,bias}   per-ResConv conv (j ∈ 0..7)
///   convblock.{j}.beta                 per-ResConv per-channel scale, stored [1,1,1,c] in .rmw
///   lastconv.0.{weight,bias}           ConvTranspose2d (c → 4*13 channels)
///
/// `build(...)` is called once per stage by IFNetGraph.compileStageExecutable.
final class IFBlockBuilder {

    private let graph: MPSGraph
    private let weights: WeightStore
    private let prefix: String

    init(graph: MPSGraph, weights: WeightStore, prefix: String) {
        self.graph = graph
        self.weights = weights
        self.prefix = prefix
    }

    /// Builds the IFBlock subgraph at the given scale.
    ///
    /// - Parameters:
    ///   - input: post-concat tensor at full internal resolution. Shape [1, intH, intW, in_ch].
    ///     Stage 0: in_ch = 15. Stage i>0: in_ch = 24 (before the IFBlockBuilder concats prev_flow).
    ///   - previousFlow: accumulated flow from prior stage at full internal resolution [1, intH, intW, 4].
    ///     nil for stage 0.
    ///   - scale: stage scale factor; v4.26 scale_list is [16, 8, 4, 2, 1].
    /// - Returns: `(flow, mask, feat)` all at full internal resolution.
    ///   flow: [1, intH, intW, 4] (already multiplied by scale).
    ///   mask: [1, intH, intW, 1].
    ///   feat: [1, intH, intW, 8].
    func build(input: MPSGraphTensor,
               previousFlow: MPSGraphTensor?,
               scale: Int) throws -> (flow: MPSGraphTensor, mask: MPSGraphTensor, feat: MPSGraphTensor) {

        let g = graph

        // 1. Downsample input by 1/scale (skip if scale == 1).
        var x = input
        if scale != 1 {
            x = g.rifeBilinearResize(input: x,
                                      scaleFactor: 1.0 / Double(scale),
                                      name: "\(prefix)downsample_x")
        }

        // 2. If previousFlow provided, downsample + scale-divide, then concat on the channel axis.
        //    This is the "prev_flow" insertion that brings stage i>0's in_channels from 24 to 28.
        if let prevFlow = previousFlow {
            var pf = prevFlow
            if scale != 1 {
                pf = g.rifeBilinearResize(input: pf,
                                           scaleFactor: 1.0 / Double(scale),
                                           name: "\(prefix)downsample_flow")
                let invScale = g.constant(1.0 / Double(scale), dataType: .float16)
                pf = g.multiplication(pf, invScale, name: "\(prefix)flow_inv_scale")
            }
            x = g.concatTensors([x, pf], dimension: 3, name: "\(prefix)cat_with_flow")
        }

        // 3. conv0: two stride-2 convs, each followed by LeakyReLU(0.2).
        //    Triple-nested key path matches IFNet_HDv3.py (see this file's docstring).
        x = try conv2dLeakyReLU(x: x, subPath: "conv0.0.0", stride: 2)
        x = try conv2dLeakyReLU(x: x, subPath: "conv0.1.0", stride: 2)

        // 4. convblock: 8 × ResConv. Each: LeakyReLU(conv(x) * beta + x).
        for j in 0..<8 {
            let w = try loadConst(name: "convblock.\(j).conv.weight")
            let b = try loadConst(name: "convblock.\(j).conv.bias")
            let conv = g.rifeConv2D(input: x, weight: w, bias: b, stride: 1, padding: 1,
                                     name: "\(prefix)convblock.\(j).conv")

            // beta is stored in .rmw as [1,1,1,c] (canonical NHWC broadcast shape) by the
            // converter's per-name special case for `*.beta` tensors. No runtime reshape needed.
            let beta = try loadConst(name: "convblock.\(j).beta")

            let scaled = g.multiplication(conv, beta, name: "\(prefix)convblock.\(j).scale")
            let residual = g.addition(scaled, x, name: "\(prefix)convblock.\(j).add")
            x = g.rifeLeakyReLU(input: residual, negativeSlope: 0.2,
                                  name: "\(prefix)convblock.\(j).act")
        }

        // 5. lastconv: ConvTranspose2d(c → 4*13, k=4, s=2, p=1) followed by PixelShuffle(r=2).
        let dW = try loadConst(name: "lastconv.0.weight")
        let dB = try loadConst(name: "lastconv.0.bias")
        var tmp = g.rifeConvTranspose2D(input: x, weight: dW, bias: dB, stride: 2, padding: 1,
                                          name: "\(prefix)lastconv")
        // PixelShuffle r=2: 4*13=52 channels → 13 channels at 2× spatial resolution.
        tmp = g.rifePixelShuffle(input: tmp, blockSize: 2, name: "\(prefix)pixelshuffle")

        // 6. Upsample back to full internal resolution by scale (mirroring step 1's downsample).
        if scale != 1 {
            tmp = g.rifeBilinearResize(input: tmp,
                                        scaleFactor: Double(scale),
                                        name: "\(prefix)upsample_to_full")
        }

        // 7. Slice 13 channels into flow [0:4], mask [4:5], feat [5:13].
        let flowRaw = g.sliceTensor(tmp, dimension: 3, start: 0, length: 4,
                                     name: "\(prefix)slice_flow")
        let mask    = g.sliceTensor(tmp, dimension: 3, start: 4, length: 1,
                                     name: "\(prefix)slice_mask")
        let feat    = g.sliceTensor(tmp, dimension: 3, start: 5, length: 8,
                                     name: "\(prefix)slice_feat")

        // 8. Multiply flow by scale (matches the analogous ncnn / PyTorch convention).
        let flowOut: MPSGraphTensor
        if scale == 1 {
            flowOut = flowRaw
        } else {
            let scaleConst = g.constant(Double(scale), dataType: .float16)
            flowOut = g.multiplication(flowRaw, scaleConst, name: "\(prefix)flow_scaled")
        }

        return (flow: flowOut, mask: mask, feat: feat)
    }

    // MARK: - Helpers

    private func conv2dLeakyReLU(x: MPSGraphTensor, subPath: String, stride: Int) throws -> MPSGraphTensor {
        let w = try loadConst(name: "\(subPath).weight")
        let b = try loadConst(name: "\(subPath).bias")
        let conv = graph.rifeConv2D(input: x, weight: w, bias: b, stride: stride, padding: 1,
                                     name: "\(prefix)\(subPath)")
        return graph.rifeLeakyReLU(input: conv, negativeSlope: 0.2,
                                    name: "\(prefix)\(subPath).act")
    }

    private func loadConst(name: String) throws -> MPSGraphTensor {
        let fullName = prefix + name
        guard let entry = weights.header.tensors.first(where: { $0.name == fullName }) else {
            throw WeightStoreError.tensorNotFound(fullName)
        }
        let bytes = try weights.tensorBytes(named: fullName)
        return graph.constant(bytes,
                               shape: entry.shape.map(NSNumber.init),
                               dataType: .float16)
    }
}
