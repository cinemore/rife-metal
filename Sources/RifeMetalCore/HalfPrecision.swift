import Foundation

enum HalfPrecision {
    static func floatToBits(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        let exponent = Int((bits >> 23) & 0xFF) - 127 + 15
        let mantissa = bits & 0x7FFFFF

        if exponent <= 0 {
            if exponent < -10 {
                return sign
            }
            let shiftedMantissa = mantissa | 0x800000
            let shift = UInt32(14 - exponent)
            let rounding = UInt32(1) << (shift - 1)
            return sign | UInt16((shiftedMantissa + rounding) >> shift)
        }

        if exponent >= 31 {
            if mantissa == 0 {
                return sign | 0x7C00
            }
            return sign | 0x7C00 | UInt16(mantissa >> 13) | 1
        }

        var roundedMantissa = mantissa + 0x1000
        var roundedExponent = exponent
        if roundedMantissa & 0x800000 != 0 {
            roundedMantissa = 0
            roundedExponent += 1
        }
        if roundedExponent >= 31 {
            return sign | 0x7C00
        }
        return sign | UInt16(roundedExponent << 10) | UInt16(roundedMantissa >> 13)
    }

    static func bitsToFloat(_ bits: UInt16) -> Float {
        let sign = UInt32(bits & 0x8000) << 16
        let exponent = Int((bits >> 10) & 0x1F)
        var mantissa = UInt32(bits & 0x03FF)

        let floatBits: UInt32
        if exponent == 0 {
            if mantissa == 0 {
                floatBits = sign
            } else {
                var adjustedExponent = -14
                while mantissa & 0x0400 == 0 {
                    mantissa <<= 1
                    adjustedExponent -= 1
                }
                mantissa &= 0x03FF
                let expBits = UInt32(adjustedExponent + 127) << 23
                floatBits = sign | expBits | (mantissa << 13)
            }
        } else if exponent == 31 {
            floatBits = sign | 0x7F800000 | (mantissa << 13)
        } else {
            let expBits = UInt32(exponent - 15 + 127) << 23
            floatBits = sign | expBits | (mantissa << 13)
        }

        return Float(bitPattern: floatBits)
    }
}
