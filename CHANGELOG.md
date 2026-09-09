# Changelog

## 0.1.6

- Restore Intel CLI builds by keeping the original MPSGraph accumulation path on x86_64; Apple Silicon retains the validated rounding probe and GPU accumulation.
- Version 0.1.5 has a source tag only: its Intel build failed before release archives were published. Use 0.1.6 for both Swift packages and CLI downloads.

## 0.1.5

- Reuse stream padding buffers and crop Balanced caller-supplied outputs during the final GPU pass.
- Accumulate Fast/Balanced flow and mask on the GPU while preserving the original backend's detected half-precision rounding. Unknown rounding behavior keeps the original graph path.
- Reuse periodic probe reference values while still checking every probe output.
- Add GPU rounding and cropped-output parity tests.
- Include the required Metal resource bundle in CLI archives and Homebrew installations.
- Document version-pinned Swift package dependencies.
