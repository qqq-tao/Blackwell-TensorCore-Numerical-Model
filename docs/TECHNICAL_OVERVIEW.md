# Technical Overview

Blackwell TensorCore Numerical Model is a behavioral model for observable dense
Tensor Core MMA numerical behavior on NVIDIA Blackwell / RTX 5090-class
hardware.

The core target is bitwise reproduction of observed result bits, not
tolerance-based approximation. This makes the model useful as a regression
oracle for validated dense MMA paths.

## Characterization Method

The model was built through black-box numerical characterization:

- Generate controlled MMA inputs.
- Run generated inline-PTX MMA kernels on RTX 5090-class hardware.
- Compare hardware output against candidate cmodel rules at the bit level.
- Use directed/manual cases to isolate individual behaviors.
- Use scalable random hardware comparison to check interactions between
  behaviors.

This project does not claim access to NVIDIA's private Tensor Core
implementation. Descriptions of internal structure refer to the inferred
numerical-behavior pipeline required to match observable input/output behavior.

## Case Design

The retained manual cases and random generator focus on behaviors that ordinary
random testing tends to miss:

- zeros, signed zeros, normals, subnormals, infinities, and NaNs
- mantissa-boundary values such as just-below-one and just-above-one
- exponent-alignment boundaries where low-order bits affect guard, round, or
  sticky behavior
- cancellation patterns where product sums nearly eliminate each other
- output normalization, subnormal generation, underflow, overflow, and special
  value propagation
- dtype combinations where input precision, internal accumulation width, and
  output precision differ

The directed/manual YAML cases are retained under `cmodel/test_cases/`. The
current release-candidate evidence is produced by the scalable random
hardware-comparison matrix.

## Inferred Pipeline

The matched cmodel can be read as a compact behavioral pipeline:

1. **Operand decode**

   A, B, and C are unpacked into sign, exponent, significand, and a
   special-value tag. Subnormal inputs use an adjusted exponent and no implicit
   leading bit; normal inputs add the implicit bit.

2. **Product formation**

   Each A/B product forms its sign by XOR, combines the two input exponents, and
   multiplies the significands. The product is shifted into an internal
   fixed-point significand representation before accumulation.

3. **Accumulator alignment and reduction**

   Product terms and the incoming C value are represented in a common internal
   form. Finite terms are aligned to the maximum exponent and reduced as signed
   integer significands. NaN and infinity cases are handled before ordinary
   finite reduction when they dominate the result.

4. **Normalization**

   The reduced magnitude is normalized against the destination format. The model
   covers zero, subnormal output, normal output, underflow to zero, and overflow
   to infinity as observable cases.

5. **Output rounding**

   The matched rules are output-path dependent. The cmodel uses an RNE-style
   rule for the f16-output path and a truncation-style rule for f32-output paths
   after internal normalization.

6. **K granularity**

   The dot-product granularity is derived from input dtype width. The
   release-candidate matrix reports one selected dense shape per dtype
   combination, while the model and generated-kernel path can be run with other
   configured dense cases.

## Published Evidence Boundary

The current release-candidate evidence covers the dense dtype combinations and
selected dense shapes documented in `docs/RUNNING_VALIDATION.md`.

That is a published evidence boundary, not a statement that other configured
dense cases are unsupported. Sparse MMA and block-scale MMA are not part of the
current validated release surface. Block-scale validation and release are future
work.

This is a numerical-behavior model. It is not a timing model, a throughput
model, or an official hardware specification.
