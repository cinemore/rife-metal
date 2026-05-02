import Foundation
import MetalPerformanceShadersGraph

extension MPSGraph {

    // MARK: - Conv2D

    /// 2D convolution matching PyTorch nn.Conv2d semantics:
    ///  - data: NHWC, [batch, height, width, in_channels]
    ///  - weight: HWIO, [kH, kW, in_channels, out_channels] (SDK does not have OHWI; caller
    ///    must permute PyTorch weights from OIHW → HWIO before building the graph tensor)
    ///  - bias: [out_channels]
    ///  - PyTorch padding=k//2 (same-size for stride 1; for stride 2, output = ceil(in/2))
    func rifeConv2D(input: MPSGraphTensor,
                    weight: MPSGraphTensor,
                    bias: MPSGraphTensor,
                    stride: Int,
                    padding: Int,
                    name: String? = nil) -> MPSGraphTensor {
        let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: stride, strideInY: stride,
            dilationRateInX: 1, dilationRateInY: 1,
            groups: 1,
            paddingLeft: padding, paddingRight: padding,
            paddingTop: padding, paddingBottom: padding,
            paddingStyle: .explicit,
            dataLayout: .NHWC,
            weightsLayout: .HWIO
        )!
        let conv = convolution2D(input,
                                 weights: weight,
                                 descriptor: descriptor,
                                 name: name.map { "\($0).conv" })
        // Bias is [C]; broadcast-add over NHWC by reshaping to [1, 1, 1, C].
        let outChannels = bias.shape!.last!
        let biasBroadcast = reshape(bias,
                                    shape: [1, 1, 1, outChannels],
                                    name: name.map { "\($0).bias.reshape" })
        return addition(conv, biasBroadcast, name: name.map { "\($0).biased" })
    }

    // MARK: - ConvTranspose2D

    /// 2D transposed convolution (deconv) matching PyTorch nn.ConvTranspose2d semantics.
    /// Data layout: NHWC.
    ///
    /// Weight layout: HWIO where I and O are interpreted with respect to the *forward*
    /// convolution direction (per the MPSGraph documentation: "Convolution Transpose operation
    /// is exactly the same as convolution gradient with respect to input image"). PyTorch
    /// ConvTranspose2d weight is stored as (in_channels, out_channels, kH, kW) which matches
    /// the forward convolution's I and O — so the converter permutes (in, out, kH, kW)
    /// → (kH, kW, out, in) by transposing axes (2, 3, 1, 0). The kH, kW symmetry of the
    /// inner two axes means MPSGraph's I/O reinterpretation lands on a usable layout
    /// regardless of whether axes 0 and 1 swap; what matters is that kH, kW lead.
    ///
    /// Output spatial size: in_spatial * stride (for symmetric padding=k//2 with stride k=4,
    /// pad=1 — matches PyTorch's default output shape formula
    /// `(in - 1) * stride - 2*padding + kernel`).
    func rifeConvTranspose2D(input: MPSGraphTensor,
                             weight: MPSGraphTensor,
                             bias: MPSGraphTensor,
                             stride: Int,
                             padding: Int,
                             name: String? = nil) -> MPSGraphTensor {
        let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: stride, strideInY: stride,
            dilationRateInX: 1, dilationRateInY: 1,
            groups: 1,
            paddingLeft: padding, paddingRight: padding,
            paddingTop: padding, paddingBottom: padding,
            paddingStyle: .explicit,
            dataLayout: .NHWC,
            weightsLayout: .HWIO
        )!
        let inputShape = input.shape!.map { $0.intValue }
        let weightShape = weight.shape!.map { $0.intValue }
        // HWIO[2] is the I dim = forward_in_channels = CT2D OUTPUT channels.
        // (HWIO[3] is the O dim = forward_out_channels = CT2D INPUT channels, which we don't
        // need explicitly — it must equal inputShape[3].)
        let ctOutChannels = weightShape[2]
        let outShape: [Int] = [
            inputShape[0],
            inputShape[1] * stride,
            inputShape[2] * stride,
            ctOutChannels,
        ]
        let conv = convolutionTranspose2D(
            input,
            weights: weight,
            outputShape: outShape.map(NSNumber.init),
            descriptor: descriptor,
            name: name.map { "\($0).deconv" }
        )
        let outChannelsNum = bias.shape!.last!
        let biasBroadcast = reshape(bias,
                                    shape: [1, 1, 1, outChannelsNum],
                                    name: name.map { "\($0).bias.reshape" })
        return addition(conv, biasBroadcast, name: name.map { "\($0).biased" })
    }

    // MARK: - LeakyReLU

    /// PyTorch nn.LeakyReLU(negative_slope) semantics.
    /// Output: x if x >= 0 else negativeSlope * x.
    func rifeLeakyReLU(input: MPSGraphTensor,
                       negativeSlope: Double,
                       name: String? = nil) -> MPSGraphTensor {
        let zero = constant(0.0, dataType: input.dataType)
        let posPart = maximum(input, zero, name: name.map { "\($0).pos" })
        let negPart = minimum(input, zero, name: name.map { "\($0).neg" })
        let slopeConst = constant(negativeSlope, dataType: input.dataType)
        let scaledNeg = multiplication(slopeConst, negPart, name: name.map { "\($0).scaled" })
        return addition(posPart, scaledNeg, name: name.map { "\($0).out" })
    }

    // MARK: - PixelShuffle (CRD)

    /// PyTorch pixel_shuffle uses CRD mode (channel-row-depth):
    ///   input shape:  [B, H, W, C * r * r]
    ///   output shape: [B, H * r, W * r, C]
    /// Achieved by reshape + transpose + reshape, since MPSGraph's depthToSpace2D
    /// default is DCR (which differs from PyTorch CRD).
    func rifePixelShuffle(input: MPSGraphTensor,
                          blockSize: Int,
                          name: String? = nil) -> MPSGraphTensor {
        let inputShape = input.shape!.map { $0.intValue }
        precondition(inputShape.count == 4, "expected NHWC")
        let B = inputShape[0], H = inputShape[1], W = inputShape[2]
        let Cin = inputShape[3]
        precondition(Cin % (blockSize * blockSize) == 0, "channel count not divisible by r^2")
        let Cout = Cin / (blockSize * blockSize)

        // Step 1: reshape to [B, H, W, Cout, r, r]
        // PyTorch CRD: channel index k = c_out * r^2 + r_h * r + r_w (Cout outermost, r_w innermost)
        let r0 = reshape(input,
                         shape: [NSNumber(value: B),
                                 NSNumber(value: H),
                                 NSNumber(value: W),
                                 NSNumber(value: Cout),
                                 NSNumber(value: blockSize),
                                 NSNumber(value: blockSize)],
                         name: name.map { "\($0).r0" })
        // Step 2: transpose (B, H, W, Cout, r_h, r_w) → (B, H, r_h, W, r_w, Cout)
        // axes: [0, 1, 4, 2, 5, 3]
        let perm: [NSNumber] = [0, 1, 4, 2, 5, 3]
        let t = transpose(r0, permutation: perm, name: name.map { "\($0).t" })
        // Step 3: reshape to [B, H*r, W*r, Cout]
        return reshape(t,
                       shape: [NSNumber(value: B),
                               NSNumber(value: H * blockSize),
                               NSNumber(value: W * blockSize),
                               NSNumber(value: Cout)],
                       name: name.map { "\($0).out" })
    }

    // MARK: - Bilinear resize (align_corners=False)

    /// Matches PyTorch F.interpolate(mode='bilinear', align_corners=False).
    /// scaleFactor 0.5 downsamples; 2.0 upsamples.
    func rifeBilinearResize(input: MPSGraphTensor,
                            scaleFactor: Double,
                            name: String? = nil) -> MPSGraphTensor {
        let inputShape = input.shape!.map { $0.intValue }
        precondition(inputShape.count == 4)
        let H = inputShape[1], W = inputShape[2]
        let outH = Int((Double(H) * scaleFactor).rounded(.toNearestOrAwayFromZero))
        let outW = Int((Double(W) * scaleFactor).rounded(.toNearestOrAwayFromZero))

        return resize(input,
                      size: [NSNumber(value: outH), NSNumber(value: outW)],
                      mode: .bilinear,
                      centerResult: true,
                      alignCorners: false,
                      layout: .NHWC,
                      name: name)
    }
}
